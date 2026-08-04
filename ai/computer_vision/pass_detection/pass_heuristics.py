"""
Pass detection heuristics.
Implementation Spec §2.
"""

from __future__ import annotations
import math

from ai.computer_vision.tactical_analysis.constants import PASS_MIN_DISTANCE_M


def detect_passes(possession_sequence: list[dict]) -> list[dict]:
    """
    Pass detection: possession transfers from player A to player B on the same team,
    ball travels a minimum distance (>= 3m) with no opponent touch in between.

    Args:
        possession_sequence: chronological list of possession events with keys:
            player_id, team_id, pitch_x_m, pitch_y_m, timestamp, homography_confidence

    Returns:
        list of pass Event dicts.
    """
    events = []
    if len(possession_sequence) < 2:
        return events

    for i in range(len(possession_sequence) - 1):
        p1 = possession_sequence[i]
        p2 = possession_sequence[i + 1]

        pid1, team1 = p1.get("player_id"), p1.get("team_id")
        pid2, team2 = p2.get("player_id"), p2.get("team_id")

        if pid1 is None or pid2 is None:
            continue

        # Same team, different players
        if team1 == team2 and pid1 != pid2:
            x1, y1 = p1.get("pitch_x_m"), p1.get("pitch_y_m")
            x2, y2 = p2.get("pitch_x_m"), p2.get("pitch_y_m")

            if x1 is not None and y1 is not None and x2 is not None and y2 is not None:
                dist = math.hypot(x2 - x1, y2 - y1)
                if dist >= PASS_MIN_DISTANCE_M:
                    events.append({
                        "event_type": "pass",
                        "player_id": pid1,
                        "related_player_id": pid2,
                        "team_id": team1,
                        "timestamp": p2.get("timestamp", 0.0),
                        "pitch_x_m": x2,
                        "pitch_y_m": y2,
                        "homography_confidence": p2.get("homography_confidence", 1.0),
                        "metadata_json": {
                            "pass_distance_m": round(dist, 2),
                            "start_x": x1,
                            "start_y": y1,
                        },
                    })

    return events


def detect_turnovers(possession_sequence: list[dict]) -> list[dict]:
    """
    Turnover detection: possession transfers to a player on the opposing team.

    Returns list of turnover Event dicts.
    """
    events = []
    if len(possession_sequence) < 2:
        return events

    for i in range(len(possession_sequence) - 1):
        p1 = possession_sequence[i]
        p2 = possession_sequence[i + 1]

        pid1, team1 = p1.get("player_id"), p1.get("team_id")
        pid2, team2 = p2.get("player_id"), p2.get("team_id")

        if pid1 is None or pid2 is None:
            continue

        # Different teams
        if team1 != team2 and team1 is not None and team2 is not None:
            events.append({
                "event_type": "turnover",
                "player_id": pid1,  # lost by
                "related_player_id": pid2,  # won by
                "team_id": team1,
                "timestamp": p2.get("timestamp", 0.0),
                "pitch_x_m": p2.get("pitch_x_m"),
                "pitch_y_m": p2.get("pitch_y_m"),
                "homography_confidence": p2.get("homography_confidence", 1.0),
                "metadata_json": {"lost_by": pid1, "won_by": pid2},
            })

    return events
