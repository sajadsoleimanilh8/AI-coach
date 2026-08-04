"""
Unit tests for Event Heuristics (Pass, Shot, First Touch, Turnover).
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

from ai.computer_vision.pass_detection.pass_heuristics import detect_passes, detect_turnovers
from ai.computer_vision.shot_detection.shot_heuristics import detect_shots
from ai.computer_vision.tactical_analysis.possession import detect_first_touches, get_ball_possessor


def test_get_ball_possessor_and_first_touch():
    players = [
        {"player_id": 10, "team_id": "team-A", "pitch_x_m": 20.0, "pitch_y_m": 30.0},
        {"player_id": 7, "team_id": "team-B", "pitch_x_m": 50.0, "pitch_y_m": 30.0},
    ]

    # Ball near player 10
    ball = (20.5, 30.2)
    possessor = get_ball_possessor(players, ball)
    assert possessor is not None
    assert possessor["player_id"] == 10

    # Ball far from everyone
    ball_far = (80.0, 10.0)
    assert get_ball_possessor(players, ball_far) is None


def test_pass_detection_synthetic():
    possessions = [
        {"player_id": 10, "team_id": "team-A", "pitch_x_m": 20.0, "pitch_y_m": 30.0, "timestamp": 1.0},
        {"player_id": 8, "team_id": "team-A", "pitch_x_m": 30.0, "pitch_y_m": 32.0, "timestamp": 3.0},
    ]

    passes = detect_passes(possessions)
    assert len(passes) == 1
    assert passes[0]["event_type"] == "pass"
    assert passes[0]["player_id"] == 10
    assert passes[0]["related_player_id"] == 8


def test_shot_detection_synthetic():
    # Ball traveling fast towards goal at x=105.0, y=34.0
    ball_positions = [
        {"frame_id": 1, "timestamp": 0.0, "pitch_x_m": 80.0, "pitch_y_m": 34.0, "player_id": 9, "team_id": "team-A"},
        {"frame_id": 2, "timestamp": 0.04, "pitch_x_m": 82.0, "pitch_y_m": 34.0, "player_id": 9, "team_id": "team-A"},
    ]  # dx = 2m in 0.04s -> 50 m/s

    shots = detect_shots(ball_positions, fps=25.0)
    assert len(shots) == 1
    assert shots[0]["event_type"] == "shot"
    assert shots[0]["player_id"] == 9


if __name__ == "__main__":
    test_get_ball_possessor_and_first_touch()
    test_pass_detection_synthetic()
    test_shot_detection_synthetic()
    print("ALL EVENT HEURISTICS TESTS PASSED!")
