"""
Raw tracking + heatmap API router.

Serves what backend/pipeline/runner.py::_persist_frames_and_tracking()
actually wrote to PlayerTracking/BallDetection -- nothing here is
aggregated the way tactical.py/player_intelligence.py are, this is the
per-frame data those endpoints don't expose. Two endpoints:

  - GET /api/matches/{match_id}/tracking: a frame-range window of raw
    pixel positions, for the video overlay. Windowed on purpose -- a real
    match is tens of thousands of frames, so returning "the whole match"
    in one payload isn't a viable API shape.
  - GET /api/matches/{match_id}/heatmap/{player_id}: server-aggregated
    pitch-coordinate density grid, gated on homography_confidence the same
    honesty pattern as TeamMetric/PlayerMetric (see backend/api/tactical.py's
    DEFAULT_TEAM_SCOPE comment and TabTeamIntelligence.jsx's PitchVisual) --
    never presents a heatmap as trustworthy when the upstream pitch
    coordinates aren't.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from ai.computer_vision.tactical_analysis.constants import (
    HOMOGRAPHY_CONFIDENCE_MIN,
    MIN_SAMPLE_EVENTS,
    PITCH_LENGTH_M,
    PITCH_WIDTH_M,
)
from backend.api.schemas import HeatmapResponse, TrackingWindowResponse
from backend.database.models import BallDetection, Frame, Match, PlayerTracking
from backend.database.session import get_db

router = APIRouter(prefix="/api/matches", tags=["tracking"])

# A real match clip is tens of thousands of frames; PlayerTracking holds
# one row per player per frame (see runner.py's comment on why it isn't
# stride-sampled like Frame/PlayerDetection/BallDetection are). Shipping
# an unbounded frame range would mean a single request could pull the
# entire match's tracking table. DEFAULT_WINDOW_FRAMES is ~6s at a typical
# 25fps clip -- enough for the frontend's rolling playback cache (see
# TabMatchAnalysis.jsx) without either endpoint needing to guess an fps
# ahead of time.
DEFAULT_WINDOW_FRAMES = 150
MAX_WINDOW_FRAMES = 500

# 20x13 bins over the standard pitch (105m x 68m) = ~5.25m x ~5.23m per
# cell -- coarse enough that a few dozen tracked points per player produce
# a legible density map, without shipping every raw pitch coordinate to
# the browser (see this module's docstring).
HEATMAP_GRID_COLS = 20
HEATMAP_GRID_ROWS = 13


def _get_match_or_404(db: Session, match_id: str) -> Match:
    match = db.get(Match, match_id)
    if not match:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No match found for match_id={match_id}.",
        )
    return match


@router.get("/{match_id}/tracking", response_model=TrackingWindowResponse)
def get_tracking_window(
    match_id: str,
    start_frame: int = Query(default=0, ge=0),
    end_frame: int | None = Query(default=None, ge=0),
    db: Session = Depends(get_db),
):
    _get_match_or_404(db, match_id)

    if end_frame is None:
        end_frame = start_frame + DEFAULT_WINDOW_FRAMES
    if end_frame < start_frame:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="end_frame must be >= start_frame.",
        )
    if end_frame - start_frame > MAX_WINDOW_FRAMES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Requested window ({end_frame - start_frame} frames) exceeds the "
                   f"{MAX_WINDOW_FRAMES}-frame max per request. Page through with "
                   f"start_frame/end_frame instead of requesting the whole match.",
        )

    # fps is stored per-Frame (sampled rows only, see FRAME_PERSIST_STRIDE
    # in backend/pipeline/runner.py) -- any row for this match carries the
    # same fps the pipeline measured, so the first one is enough to convert
    # PlayerTracking.frame_id (a real frame *number*, not a Frame FK -- see
    # models.py's PlayerTracking.frame_id comment) into a timestamp.
    frame_row = db.query(Frame).filter(Frame.match_id == match_id).first()
    fps = frame_row.fps if frame_row else 25.0

    tracking_rows = (
        db.query(PlayerTracking)
        .filter(
            PlayerTracking.match_id == match_id,
            PlayerTracking.frame_id >= start_frame,
            PlayerTracking.frame_id <= end_frame,
        )
        .order_by(PlayerTracking.frame_id)
        .all()
    )

    # BallDetection only exists for the stride-sampled Frame rows (unlike
    # PlayerTracking, it isn't backfilled for every frame), so it's joined
    # through Frame.frame_number rather than assumed to line up 1:1 with
    # every frame_number in the requested window.
    ball_rows = (
        db.query(Frame.frame_number, BallDetection)
        .join(BallDetection, BallDetection.frame_id == Frame.frame_id)
        .filter(
            Frame.match_id == match_id,
            Frame.frame_number >= start_frame,
            Frame.frame_number <= end_frame,
        )
        .all()
    )
    ball_by_frame = {frame_number: ball for frame_number, ball in ball_rows}

    players_by_frame: dict[int, list] = {}
    for t in tracking_rows:
        players_by_frame.setdefault(t.frame_id, []).append({
            "player_id": t.player_id,
            "team_id": t.team_id,
            "pixel_x": t.pixel_x,
            "pixel_y": t.pixel_y,
            "pitch_x_m": t.pitch_x_m,
            "pitch_y_m": t.pitch_y_m,
        })

    frame_numbers = sorted(set(players_by_frame) | set(ball_by_frame))
    frames = []
    for fn in frame_numbers:
        ball = ball_by_frame.get(fn)
        frames.append({
            "frame_number": fn,
            "timestamp": fn / fps,
            "players": players_by_frame.get(fn, []),
            "ball": {"pixel_x": ball.ball_x, "pixel_y": ball.ball_y} if ball else None,
        })

    return {
        "match_id": match_id,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "fps": fps,
        "frames": frames,
    }


@router.get("/{match_id}/heatmap/{player_id}", response_model=HeatmapResponse)
def get_player_heatmap(match_id: str, player_id: int, db: Session = Depends(get_db)):
    _get_match_or_404(db, match_id)

    rows = (
        db.query(PlayerTracking)
        .filter(PlayerTracking.match_id == match_id, PlayerTracking.player_id == player_id)
        .all()
    )
    sample_size = len(rows)

    usable = [
        r for r in rows
        if r.pitch_x_m is not None
        and r.pitch_y_m is not None
        and r.homography_confidence is not None
        and r.homography_confidence >= HOMOGRAPHY_CONFIDENCE_MIN
    ]
    usable_sample_size = len(usable)

    # Same honesty gate as PlayerMetric/TeamMetric (backend/database/models.py's
    # MetricConfidence): "not computed" isn't representable by this enum, so
    # zero-row and zero-usable-row cases both honestly report
    # low_upstream_confidence with an empty grid -- never a fabricated
    # density map. sample_size/usable_sample_size tell the caller which of
    # the two actually happened.
    if usable_sample_size == 0:
        confidence = "low_upstream_confidence"
    elif usable_sample_size < MIN_SAMPLE_EVENTS:
        confidence = "low_sample"
    else:
        confidence = "normal"

    counts: dict[tuple[int, int], int] = {}
    if confidence == "normal" or confidence == "low_sample":
        for r in usable:
            gx = min(HEATMAP_GRID_COLS - 1, max(0, int((r.pitch_x_m / PITCH_LENGTH_M) * HEATMAP_GRID_COLS)))
            gy = min(HEATMAP_GRID_ROWS - 1, max(0, int((r.pitch_y_m / PITCH_WIDTH_M) * HEATMAP_GRID_ROWS)))
            counts[(gx, gy)] = counts.get((gx, gy), 0) + 1

    max_count = max(counts.values()) if counts else 0
    cells = [
        {
            "grid_x": gx,
            "grid_y": gy,
            "count": count,
            "density": (count / max_count) if max_count else 0.0,
        }
        for (gx, gy), count in sorted(counts.items())
    ]

    return {
        "match_id": match_id,
        "player_id": player_id,
        "grid_cols": HEATMAP_GRID_COLS,
        "grid_rows": HEATMAP_GRID_ROWS,
        "pitch_length_m": PITCH_LENGTH_M,
        "pitch_width_m": PITCH_WIDTH_M,
        "cells": cells,
        "confidence": confidence,
        "sample_size": sample_size,
        "usable_sample_size": usable_sample_size,
    }
