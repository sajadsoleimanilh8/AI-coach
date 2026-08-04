"""
Possession and First Touch tracking helper.
Implementation Spec §2.
"""

from __future__ import annotations
import math

from ai.computer_vision.tactical_analysis.constants import (
    BALL_CONTROL_RADIUS_M,
    HOMOGRAPHY_CONFIDENCE_MIN,
)


def get_ball_possessor(
    player_positions: list[dict],
    ball_position: tuple[float, float] | None,
    homography_confidence: float = 1.0,
    control_radius: float = BALL_CONTROL_RADIUS_M,
) -> dict | None:
    """
    Finds the tracked player whose foot point is within control_radius of ball pitch position.

    Args:
        player_positions: list of dicts with keys: player_id, team_id, pitch_x_m, pitch_y_m
        ball_position: (x, y) pitch meter position of ball
        homography_confidence: confidence score for this frame's homography
        control_radius: max distance in meters to declare control (default 1.5m)

    Returns:
        dict of closest player within control_radius, or None if no player in radius or homography_confidence low.
    """
    if homography_confidence < HOMOGRAPHY_CONFIDENCE_MIN or ball_position is None:
        return None

    bx, by = ball_position
    closest_player = None
    min_dist = float("inf")

    for p in player_positions:
        px = p.get("pitch_x_m")
        py = p.get("pitch_y_m")
        if px is None or py is None:
            continue

        dist = math.hypot(px - bx, py - by)
        if dist <= control_radius and dist < min_dist:
            min_dist = dist
            closest_player = p

    return closest_player


def detect_first_touches(
    frame_possessions: list[dict],
) -> list[dict]:
    """
    Emits a 'first_touch' event on the frame a player gains possession after a frame with no controller or different controller.

    Args:
        frame_possessions: list of dicts with keys: frame_id, timestamp, possessor (dict or None), homography_confidence, pitch_x_m, pitch_y_m

    Returns:
        list of Event dicts for first touches.
    """
    events = []
    prev_possessor_id = None

    for entry in frame_possessions:
        curr = entry.get("possessor")
        curr_id = curr.get("player_id") if curr else None

        if curr_id is not None and curr_id != prev_possessor_id:
            # First touch event!
            events.append({
                "event_type": "first_touch",
                "player_id": curr_id,
                "team_id": curr.get("team_id"),
                "timestamp": entry.get("timestamp", 0.0),
                "pitch_x_m": entry.get("pitch_x_m"),
                "pitch_y_m": entry.get("pitch_y_m"),
                "homography_confidence": entry.get("homography_confidence", 1.0),
                "metadata_json": {"distance_to_ball": entry.get("distance_to_ball", 0.0)},
            })

        prev_possessor_id = curr_id

    return events
