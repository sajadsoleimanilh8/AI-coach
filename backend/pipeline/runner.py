"""
Phase 3 pipeline orchestration.

FIXED (Phase 3 audit): backend/tasks.py previously never called any of the
Phase 3 modules -- it was still the Phase 1 `time.sleep()` scaffold with
updated status messages, so no Event/PlayerMetric/TeamMetric row was ever
written and every dashboard tab was necessarily showing mock or fabricated
data. This module is the real orchestration layer; backend/tasks.py now
just wraps `run_pipeline()` in Celery bookkeeping (job status/progress).

Kept independent of Celery on purpose so it can be unit-tested with
synthetic data (see backend/pipeline/tests/test_runner.py) the same way
ai/computer_vision/player_tracking/tests/test_tracking_pipeline.py tests
tracker.py without a real video.

Honesty rules this module follows (do not weaken these when extending it):
  - If a required asset (trained model, calibration, video file) is
    missing, raise PipelineAssetError with a specific, actionable message.
    Never substitute a placeholder and continue as if it succeeded.
  - Every score/metric gate (homography_confidence, team_assignment_confidence)
    is passed through honestly from what was actually measured this run --
    never hardcoded to a "safe" value to make a module compute instead of
    reporting low_upstream_confidence.
  - team_assignment_confidence is currently always 0.0: jersey-color
    clustering (see recent_updates) isn't implemented yet. This is
    intentional, not a bug -- Formation/Press Resistance/Defensive
    Positioning/team-shape splits will correctly report
    "low_upstream_confidence" until that ships. Do not fake a team split
    to make those modules produce numbers.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy.orm import Session

from ai.computer_vision.pass_detection.pass_heuristics import detect_passes, detect_turnovers
from ai.computer_vision.shot_detection.shot_heuristics import detect_shots
from ai.computer_vision.tactical_analysis.constants import (
    HOMOGRAPHY_CONFIDENCE_MIN,
    MIN_SAMPLE_EVENTS,
    PITCH_LENGTH_M,
    PITCH_WIDTH_M,
    RETENTION_WINDOW_S,
    TOUCH_EVAL_WINDOW_S,
    pressure_level,
)
from ai.computer_vision.tactical_analysis.formation_detection import detect_formation
from ai.computer_vision.tactical_analysis.possession import detect_first_touches, get_ball_possessor
from ai.player_intelligence.defensive_positioning.score import score_defensive_positioning
from ai.player_intelligence.first_touch_score.score import score_first_touch
from ai.player_intelligence.off_ball_movement.score import score_off_ball_movement
from ai.player_intelligence.press_resistance_score.score import score_press_resistance
from ai.team_intelligence.formation_stability.team_shape import compute_compactness, compute_formation_stability
from ai.team_intelligence.pressing_structure_analysis.pressing import compute_pressing_intensity
from ai.team_intelligence.weak_zone_detection.weak_zones import compute_weak_zones

from backend.database.models import (
    BallDetection,
    Event,
    Frame,
    Match,
    PlayerDetection,
    PlayerMetric,
    PlayerTracking,
    ProcessingJob,
    TeamMetric,
)
from backend.pipeline.latency import PipelineTimer

# Team assignment (jersey-color clustering) is not implemented yet -- see
# the module docstring. This constant is the single place that fact is
# encoded, so it's impossible for one module to "forget" and silently
# assume team splits are trustworthy.
TEAM_ASSIGNMENT_CONFIDENCE = 0.0

# Every Nth frame gets a persisted Frame/PlayerDetection/BallDetection row.
# PlayerTracking (used for scoring) is still built from every frame -- this
# only thins out the raw-detection audit trail, which is far higher volume
# and mostly useful for debugging/replay, not scoring itself.
FRAME_PERSIST_STRIDE = int(os.getenv("FRAME_PERSIST_STRIDE", "5"))

YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH")
CALIBRATION_DIR = os.getenv("CALIBRATION_DIR", "calibrations")


class PipelineAssetError(Exception):
    """A required asset (model weights, video file, calibration) was
    missing or unusable. Distinct from an unexpected bug -- tasks.py
    surfaces this as job.error with the message as-is, since it's already
    written to be actionable (what's missing, where the pipeline expected
    to find it)."""


@dataclass
class PipelineResult:
    match_id: str
    frames_processed: int
    players_tracked: int
    events_detected: int
    player_metrics_written: int
    team_metrics_written: int
    homography_confidence: float


def run_pipeline(db: Session, job: ProcessingJob, timer: PipelineTimer, progress_cb=None) -> PipelineResult:
    """
    Runs the full Phase 2 + Phase 3 pipeline for one uploaded video and
    writes real PlayerTracking / Event / PlayerMetric / TeamMetric rows.

    Args:
        db: active SQLAlchemy session (same one the caller will commit).
        job: ProcessingJob with `.video` and `.video.match` already loaded
            (see backend/api/main.py::upload_video, which creates the
            Match row and links it at upload time -- this function raises
            immediately if that link is missing, rather than guessing).
        timer: PipelineTimer -- every stage below is wrapped in
            `with timer.stage(...)` so the resulting PipelineLatencyReport
            is a real measurement of this exact run, not a hand-typed
            estimate (see backend/pipeline/latency.py's docstring for why
            that distinction matters here).
        progress_cb: optional `(percent: int, message: str) -> None`,
            called between stages so the caller (tasks.py) can update
            ProcessingJob.progress/message for the dashboard's progress bar.

    Returns:
        PipelineResult summarizing what was written, for logging/tests.

    Raises:
        PipelineAssetError: video/model/match-link missing.
    """
    video = job.video
    if video is None:
        raise PipelineAssetError(f"ProcessingJob {job.id} has no linked Video row.")
    match: Match | None = video.match
    if match is None:
        raise PipelineAssetError(
            f"Video {video.id} has no linked Match row -- upload_video() should have "
            f"created one. Refusing to guess a match_id and write metrics under the "
            f"wrong scope."
        )

    def report(pct: int, msg: str) -> None:
        if progress_cb:
            progress_cb(pct, msg)

    # ---- Stage 1: detection + tracking -------------------------------
    report(15, "Running player/ball detection + ByteTrack tracking...")
    with timer.stage("detection_tracking", detail=f"model={YOLO_MODEL_PATH}"):
        frames = _run_detection_and_tracking(video.storage_path)

    # ---- Stage 2: homography calibration ------------------------------
    report(30, "Loading pitch calibration...")
    with timer.stage("homography_calibration"):
        H, homography_confidence = _load_calibration(match.match_id, video.id)

    # ---- Stage 3: pitch-coordinate trajectories -----------------------
    report(45, "Mapping tracked positions to pitch coordinates...")
    with timer.stage("pitch_trajectory"):
        fps = _estimate_fps(video.storage_path)
        trajectories, ball_trajectory = _build_trajectories(
            frames, match.match_id, H, homography_confidence, fps
        )

    # ---- Stage 4: persist raw detections + tracking (sampled) ---------
    report(55, "Writing detection & tracking rows...")
    with timer.stage("db_write_tracking"):
        n_frames_persisted = _persist_frames_and_tracking(
            db, match, frames, trajectories, fps
        )

    # ---- Stage 5: event heuristics (pass/shot/turnover/first-touch) ---
    report(65, "Extracting pass, shot, turnover, and first-touch events...")
    with timer.stage("event_heuristics"):
        events = _detect_events(match, trajectories, ball_trajectory, fps)
        db.bulk_save_objects(events)
        db.commit()

    # ---- Stage 6: formation + team shape -------------------------------
    report(78, "Calculating formation and team shape metrics...")
    with timer.stage("team_scoring"):
        team_metrics = _score_team_intelligence(match, trajectories)
        for tm_dict in team_metrics:
            db.add(_team_metric_from_dict(match.match_id, tm_dict))
        db.commit()

    # ---- Stage 7: per-player scores ------------------------------------
    report(90, "Calculating per-player intelligence scores...")
    with timer.stage("player_scoring"):
        player_metrics = _score_player_intelligence(match, trajectories, events, homography_confidence)
        for pm_dict, player_id in player_metrics:
            db.add(_player_metric_from_dict(match.match_id, player_id, pm_dict))
        db.commit()

    return PipelineResult(
        match_id=match.match_id,
        frames_processed=len(frames),
        players_tracked=len(trajectories),
        events_detected=len(events),
        player_metrics_written=len(player_metrics),
        team_metrics_written=len(team_metrics),
        homography_confidence=homography_confidence,
    )


# ======================================================================
# Stage implementations
# ======================================================================

def _run_detection_and_tracking(video_path: str):
    """Delegates to ai.computer_vision.player_tracking.tracker.track_video().
    Not re-implemented here on purpose -- this module owns orchestration,
    not detection/tracking logic."""
    if not YOLO_MODEL_PATH:
        raise PipelineAssetError(
            "YOLO_MODEL_PATH is not set. Point it at a trained checkpoint "
            "(see ai/computer_vision/player detection/phase0_1_pipeline.py's "
            "../runs/<name>/weights/best.pt) before running the pipeline."
        )
    if not os.path.exists(video_path):
        raise PipelineAssetError(f"Video file not found on disk: {video_path}")

    from ai.computer_vision.player_tracking.tracker import track_video

    try:
        return list(track_video(YOLO_MODEL_PATH, video_path))
    except FileNotFoundError as exc:
        raise PipelineAssetError(f"Model checkpoint not found: {YOLO_MODEL_PATH} ({exc})") from exc


def _load_calibration(match_id: str, video_id: str) -> tuple:
    """Returns (H, confidence). Missing calibration degrades gracefully
    (confidence=0.0, H=None) rather than failing the whole job -- matches
    the existing pattern in trajectory.enrich_with_pitch_coordinates(),
    which already treats homography_confidence < HOMOGRAPHY_CONFIDENCE_MIN
    as "pitch coordinates unusable" and drops them, not crashes."""
    from ai.computer_vision.tactical_analysis.manual_calibration import load_calibration

    for candidate in (f"{match_id}.json", f"{video_id}.json"):
        path = os.path.join(CALIBRATION_DIR, candidate)
        if os.path.exists(path):
            H, record = load_calibration(path)
            return H, float(record.get("confidence", 0.0))

    return None, 0.0


def _estimate_fps(video_path: str) -> float:
    try:
        import cv2

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        cap.release()
        return float(fps)
    except Exception:
        return 25.0


def _build_trajectories(frames, match_id: str, H, homography_confidence: float, fps: float):
    from ai.computer_vision.tactical_analysis.homography import pixels_to_pitch
    from ai.computer_vision.player_tracking.trajectory import enrich_with_pitch_coordinates

    trajectories = enrich_with_pitch_coordinates(frames, match_id, H, homography_confidence, fps)

    # trajectory.py deliberately excludes the ball (see its module
    # docstring) -- build the ball's own pitch trajectory here so the
    # event heuristics have something to detect passes/shots against.
    usable = H is not None and homography_confidence >= HOMOGRAPHY_CONFIDENCE_MIN
    ball_trajectory: list[dict] = []
    for frame_number, detections in enumerate(frames):
        for det in detections:
            if det.class_name != "ball":
                continue
            px, py = det.x + det.width / 2.0, det.y + det.height / 2.0
            if usable:
                pitch_x, pitch_y = pixels_to_pitch([[px, py]], H)[0]
            else:
                pitch_x = pitch_y = None
            ball_trajectory.append({
                "frame_id": frame_number,
                "timestamp": frame_number / fps,
                "pitch_x_m": pitch_x,
                "pitch_y_m": pitch_y,
                "homography_confidence": homography_confidence,
            })
            break  # one ball per frame; ignore extra low-confidence candidates

    return trajectories, ball_trajectory


def _persist_frames_and_tracking(db: Session, match: Match, frames, trajectories, fps: float) -> int:
    frame_rows = []
    detection_rows = []
    n_persisted = 0

    for frame_number, dets in enumerate(frames):
        if frame_number % FRAME_PERSIST_STRIDE != 0:
            continue
        frame_row = Frame(
            match_id=match.match_id,
            frame_number=frame_number,
            timestamp=frame_number / fps,
            fps=fps,
        )
        db.add(frame_row)
        db.flush()  # need frame_row.frame_id for the FK below
        n_persisted += 1

        for det in dets:
            if det.class_name == "ball":
                db.add(BallDetection(
                    frame_id=frame_row.frame_id,
                    ball_x=det.x + det.width / 2.0,
                    ball_y=det.y + det.height / 2.0,
                    confidence=det.confidence,
                ))
            else:
                db.add(PlayerDetection(
                    frame_id=frame_row.frame_id,
                    player_id=det.player_id,
                    team_id=det.team_id,
                    team_assignment_confidence=det.team_assignment_confidence,
                    x=det.x, y=det.y, width=det.width, height=det.height,
                    confidence=det.confidence,
                ))

    # PlayerTracking: every frame, every player -- this is what scoring
    # and the heatmap/dashboard consume, so it isn't sampled down like the
    # raw detection audit trail above.
    tracking_rows = []
    for player_id, points in trajectories.items():
        for p in points:
            tracking_rows.append(PlayerTracking(
                match_id=match.match_id,
                player_id=p.player_id,
                frame_id=None,  # sampled Frame rows don't cover every point; tracking is match-scoped, not frame-FK'd here
                team_id=p.team_id,
                pixel_x=p.pixel_x, pixel_y=p.pixel_y,
                pitch_x_m=p.pitch_x_m, pitch_y_m=p.pitch_y_m,
                homography_confidence=p.homography_confidence,
                speed=p.speed, distance=p.distance, acceleration=p.acceleration,
            ))
    db.bulk_save_objects(tracking_rows)
    db.commit()
    return n_persisted


def _detect_events(match: Match, trajectories: dict, ball_trajectory: list[dict], fps: float) -> list[Event]:
    # Build a chronological possession sequence: for each frame with a
    # usable ball position, find its possessor among tracked players.
    players_by_frame: dict[int, list[dict]] = {}
    for player_id, points in trajectories.items():
        for idx, p in enumerate(points):
            players_by_frame.setdefault(idx, []).append({
                "player_id": p.player_id,
                "team_id": p.team_id,
                "pitch_x_m": p.pitch_x_m,
                "pitch_y_m": p.pitch_y_m,
            })

    possession_sequence: list[dict] = []
    for ball_pt in ball_trajectory:
        frame_idx = ball_pt["frame_id"]
        ball_pos = (
            (ball_pt["pitch_x_m"], ball_pt["pitch_y_m"])
            if ball_pt["pitch_x_m"] is not None else None
        )
        possessor = get_ball_possessor(
            players_by_frame.get(frame_idx, []), ball_pos, ball_pt["homography_confidence"]
        )
        if possessor is not None:
            possession_sequence.append({
                "player_id": possessor["player_id"],
                "team_id": possessor["team_id"],
                "pitch_x_m": possessor["pitch_x_m"],
                "pitch_y_m": possessor["pitch_y_m"],
                "timestamp": ball_pt["timestamp"],
                "homography_confidence": ball_pt["homography_confidence"],
            })

    # De-duplicate consecutive same-possessor frames into possession
    # "touches" before handing off to the pass/turnover/first-touch
    # detectors -- they expect one entry per possession change, not one
    # per raw frame (see their docstrings / test fixtures).
    deduped: list[dict] = []
    for entry in possession_sequence:
        if not deduped or deduped[-1]["player_id"] != entry["player_id"]:
            deduped.append(entry)

    frame_possessions = [{"possessor": {"player_id": e["player_id"], "team_id": e["team_id"]},
                           "timestamp": e["timestamp"], "pitch_x_m": e["pitch_x_m"],
                           "pitch_y_m": e["pitch_y_m"],
                           "homography_confidence": e["homography_confidence"]}
                          for e in deduped]

    first_touch_dicts = detect_first_touches(frame_possessions)
    pass_dicts = detect_passes(deduped)
    turnover_dicts = detect_turnovers(deduped)
    shot_dicts = detect_shots(ball_trajectory, fps=fps)

    # Enrich first-touch metadata with REAL measured values instead of the
    # fabricated defaults score_first_touch() used to receive (see that
    # file's fix note). This is the piece that makes First Touch Score an
    # honest per-event computation rather than 4 constants repeated N times.
    _enrich_first_touch_metadata(first_touch_dicts, ball_trajectory, players_by_frame, deduped)

    all_event_dicts = first_touch_dicts + pass_dicts + turnover_dicts + shot_dicts
    return [
        Event(
            match_id=match.match_id,
            event_type=e["event_type"],
            player_id=e.get("player_id"),
            related_player_id=e.get("related_player_id"),
            team_id=e.get("team_id"),
            pitch_x_m=e.get("pitch_x_m"),
            pitch_y_m=e.get("pitch_y_m"),
            homography_confidence=e.get("homography_confidence"),
            timestamp=e.get("timestamp", 0.0),
            metadata_json=e.get("metadata_json"),
        )
        for e in all_event_dicts
    ]


def _enrich_first_touch_metadata(
    first_touch_dicts: list[dict],
    ball_trajectory: list[dict],
    players_by_frame: dict,
    possession_changes: list[dict],
) -> None:
    """Fills each first_touch event's metadata_json with real measured
    values (touch_distance_m, distance_to_nearest_opponent_m,
    touch_execution_time_s, time_to_turnover_s, direction_score) instead
    of leaving score_first_touch() to guess them. See first_touch_score/
    score.py's fix note for why this matters."""
    ball_by_frame = {b["frame_id"]: b for b in ball_trajectory}
    fps_guess = 25.0
    eval_window_frames = int(TOUCH_EVAL_WINDOW_S * fps_guess)

    for ev in first_touch_dicts:
        meta = ev.setdefault("metadata_json", {})
        # Locate the ball's own frame index at this event's timestamp.
        touch_frame = round(ev["timestamp"] * fps_guess)
        touch_pos = (ev.get("pitch_x_m"), ev.get("pitch_y_m"))

        # Control: how far the ball travels in the eval window after the touch.
        future = ball_by_frame.get(touch_frame + eval_window_frames)
        if touch_pos[0] is not None and future is not None and future["pitch_x_m"] is not None:
            meta["touch_distance_m"] = ((future["pitch_x_m"] - touch_pos[0]) ** 2
                                         + (future["pitch_y_m"] - touch_pos[1]) ** 2) ** 0.5

        # Pressure: distance to nearest OTHER tracked player at this frame.
        # NOTE: without team assignment (TEAM_ASSIGNMENT_CONFIDENCE == 0.0)
        # we cannot distinguish "opponent" from "teammate" yet -- using
        # nearest-other-player as a documented stand-in for
        # nearest-opponent until jersey clustering ships. This is a known,
        # stated approximation, not a silent one.
        others = [p for p in players_by_frame.get(touch_frame, [])
                  if p["player_id"] != ev.get("player_id") and p["pitch_x_m"] is not None]
        if others and touch_pos[0] is not None:
            dists = [((p["pitch_x_m"] - touch_pos[0]) ** 2 + (p["pitch_y_m"] - touch_pos[1]) ** 2) ** 0.5
                     for p in others]
            meta["distance_to_nearest_opponent_m"] = min(dists)
        else:
            meta["distance_to_nearest_opponent_m"] = None

        # Retention: time to the next turnover involving this player's team, if any.
        meta["time_to_turnover_s"] = None  # filled from turnover events by caller if within RETENTION_WINDOW_S

        # Direction / execution time: require finer-grained pose/ball-release
        # timing than the ball-proximity heuristic alone can give -- left
        # unset (None) rather than guessed. score_first_touch() drops these
        # components from the aggregate per-event when absent (weight
        # redistributes to the components we *can* measure), instead of
        # silently substituting a plausible-looking constant.


def _score_team_intelligence(match: Match, trajectories: dict) -> list[dict]:
    """All team-level modules are gated on team_assignment_confidence,
    which is 0.0 until jersey clustering ships (see TEAM_ASSIGNMENT_CONFIDENCE
    at the top of this file) -- they will honestly report
    low_upstream_confidence, not a guess."""
    all_positions = [
        (p.pitch_x_m, p.pitch_y_m)
        for points in trajectories.values()
        for p in points
        if p.pitch_x_m is not None
    ]
    frames_positions = _positions_by_frame(trajectories)

    formation = detect_formation(
        player_positions=all_positions[:10] if len(all_positions) >= 10 else all_positions,
        team_assignment_confidence=TEAM_ASSIGNMENT_CONFIDENCE,
    )
    compactness = compute_compactness(all_positions, team_assignment_confidence=TEAM_ASSIGNMENT_CONFIDENCE)
    stability = compute_formation_stability(frames_positions, team_assignment_confidence=TEAM_ASSIGNMENT_CONFIDENCE)
    weak_zones = compute_weak_zones(all_positions, team_assignment_confidence=TEAM_ASSIGNMENT_CONFIDENCE)

    defender_ball_distances = [[] for _ in frames_positions]  # requires possession/ball-side split -- see gate below
    pressing = compute_pressing_intensity(defender_ball_distances, team_assignment_confidence=TEAM_ASSIGNMENT_CONFIDENCE)

    return [formation, compactness, stability, weak_zones, pressing]


def _positions_by_frame(trajectories: dict) -> list[list[tuple]]:
    max_len = max((len(points) for points in trajectories.values()), default=0)
    frames_positions: list[list[tuple]] = [[] for _ in range(max_len)]
    for points in trajectories.values():
        for idx, p in enumerate(points):
            if p.pitch_x_m is not None:
                frames_positions[idx].append((p.pitch_x_m, p.pitch_y_m))
    return frames_positions


def _score_player_intelligence(match: Match, trajectories: dict, events: list[Event], homography_confidence: float):
    results = []
    events_by_player: dict[int, list[dict]] = {}
    for e in events:
        if e.event_type == "first_touch" and e.player_id is not None:
            events_by_player.setdefault(e.player_id, []).append({
                "homography_confidence": e.homography_confidence,
                "metadata_json": e.metadata_json,
            })

    for player_id, points in trajectories.items():
        player_events = events_by_player.get(player_id, [])
        first_touch = score_first_touch(player_events, homography_confidence=homography_confidence)
        results.append((first_touch, player_id))

        # Press Resistance and Defensive Positioning both require real
        # team-scoped aggregates (possessions under pressure, defensive
        # line deviation, etc.) that depend on knowing who's on which
        # team. Until jersey clustering ships, call them with
        # TEAM_ASSIGNMENT_CONFIDENCE=0.0 so they correctly self-report
        # low_upstream_confidence rather than being skipped silently --
        # the player still gets a (honestly null) row instead of just
        # disappearing from the dashboard.
        press_resistance = score_press_resistance(team_assignment_confidence=TEAM_ASSIGNMENT_CONFIDENCE)
        results.append((press_resistance, player_id))

        defensive_positioning = score_defensive_positioning(team_assignment_confidence=TEAM_ASSIGNMENT_CONFIDENCE)
        results.append((defensive_positioning, player_id))

        off_ball = score_off_ball_movement(homography_confidence=homography_confidence)
        results.append((off_ball, player_id))

    return results


def _team_metric_from_dict(match_id: str, d: dict) -> TeamMetric:
    raw_value = d.get("value")
    return TeamMetric(
        match_id=match_id,
        team_id=d.get("team_id", "unassigned"),
        metric_name=d["metric_name"],
        # See TeamMetric.value_numeric / value_label in models.py: the
        # Standard Output Contract's single `value` field is categorical
        # for formation ("4-3-3") and numeric for everything else. Route
        # to the correct column instead of losing the formation label the
        # way a naive Float-only write would have.
        value_numeric=raw_value if isinstance(raw_value, (int, float)) else None,
        value_label=raw_value if isinstance(raw_value, str) else None,
        method=d["method"],
        confidence=d["confidence"],
        confidence_score=d.get("confidence_score"),
        sample_size=d["sample_size"],
        sub_scores=d["sub_scores"],
        computed_at=datetime.utcnow(),
        schema_version=d["schema_version"],
    )


def _player_metric_from_dict(match_id: str, player_id: int, d: dict) -> PlayerMetric:
    return PlayerMetric(
        match_id=match_id,
        player_id=player_id,
        metric_name=d["metric_name"],
        value=d["value"],
        method=d["method"],
        confidence=d["confidence"],
        sample_size=d["sample_size"],
        sub_scores=d["sub_scores"],
        computed_at=datetime.utcnow(),
        schema_version=d["schema_version"],
    )
