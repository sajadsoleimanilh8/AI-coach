"""
Shot detection heuristics.
Implementation Spec §2.
"""

from __future__ import annotations
import math

from ai.computer_vision.tactical_analysis.constants import (
    GOAL_WIDTH_M,
    PITCH_LENGTH_M,
    PITCH_WIDTH_M,
    SHOT_VELOCITY_MIN_MS,
)


def detect_shots(
    ball_positions: list[dict],
    fps: float = 25.0,
    attacking_direction: str = "left_to_right",
) -> list[dict]:
    """
    Detects shots based on ball velocity, goal-mouth heading intersection, and attacking half origin.

    Args:
        ball_positions: list of dicts with frame_id, timestamp, pitch_x_m, pitch_y_m,
            homography_confidence, player_id (last possessor)

    Returns:
        list of shot Event dicts.
    """
    events = []
    if len(ball_positions) < 2:
        return events

    dt = 1.0 / fps
    half_goal = GOAL_WIDTH_M / 2.0
    goal_y_min = (PITCH_WIDTH_M / 2.0) - half_goal
    goal_y_max = (PITCH_WIDTH_M / 2.0) + half_goal

    for i in range(len(ball_positions) - 1):
        b1 = ball_positions[i]
        b2 = ball_positions[i + 1]

        x1, y1 = b1.get("pitch_x_m"), b1.get("pitch_y_m")
        x2, y2 = b2.get("pitch_x_m"), b2.get("pitch_y_m")

        if x1 is None or y1 is None or x2 is None or y2 is None:
            continue

        dx = x2 - x1
        dy = y2 - y1
        dist = math.hypot(dx, dy)
        velocity = dist / dt

        if velocity >= SHOT_VELOCITY_MIN_MS:
            # Check attacking half and goal-mouth intersection
            target_goal_x = PITCH_LENGTH_M if attacking_direction == "left_to_right" else 0.0
            in_attacking_half = (x1 >= PITCH_LENGTH_M / 2.0) if attacking_direction == "left_to_right" else (x1 <= PITCH_LENGTH_M / 2.0)

            if in_attacking_half and abs(dx) > 1e-5:
                # Extrapolate Y at goal line X
                t = (target_goal_x - x1) / dx
                if t > 0:  # Moving towards goal
                    y_at_goal = y1 + t * dy
                    if goal_y_min - 2.0 <= y_at_goal <= goal_y_max + 2.0:
                        shooter_id = b1.get("player_id") or b2.get("player_id")
                        events.append({
                            "event_type": "shot",
                            "player_id": shooter_id,
                            "team_id": b1.get("team_id"),
                            "timestamp": b2.get("timestamp", 0.0),
                            "pitch_x_m": x2,
                            "pitch_y_m": y2,
                            "homography_confidence": b2.get("homography_confidence", 1.0),
                            "metadata_json": {
                                "ball_velocity_ms": round(velocity, 2),
                                "projected_y_at_goal": round(y_at_goal, 2),
                            },
                        })

    return events
