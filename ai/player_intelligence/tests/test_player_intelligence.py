"""
Unit tests for Player Intelligence scoring modules.
"""

import os
import sys

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from ai.player_intelligence.defensive_positioning.score import score_defensive_positioning
from ai.player_intelligence.first_touch_score.score import score_first_touch
from ai.player_intelligence.off_ball_movement.score import score_off_ball_movement
from ai.player_intelligence.press_resistance_score.score import score_press_resistance


def test_press_resistance_never_pressured():
    result = score_press_resistance(
        possessions_under_pressure=0, pressure_events=0, team_assignment_confidence=0.9
    )
    assert result["value"] is None
    assert result["confidence"] == "low_sample"


def test_press_resistance_low_team_assignment_confidence():
    result = score_press_resistance(
        possessions_under_pressure=10, pressure_events=10, team_assignment_confidence=0.0
    )
    assert result["value"] is None
    assert result["confidence"] == "low_upstream_confidence"


def test_off_ball_movement_space_creation():
    result = score_off_ball_movement(
        avg_distance_gained_from_marker=5.0,  # Max gain (5.0m)
        off_ball_frames_in_attacking_third=100,
        total_off_ball_frames_while_team_in_possession=200,
        net_displacement_m=50.0,
        total_path_length_m=50.0,
        sample_size_phases=10,
    )
    assert result["value"] is not None
    assert result["sub_scores"]["space_creation"] == 100.0


def test_defensive_positioning_drifted_player():
    # Player drifted 15m out of line (max acceptable = 8m)
    result_drifted = score_defensive_positioning(
        avg_deviation_from_defensive_line_m=15.0,
        nearest_teammate_distance_m=10.0,
        closing_speed_toward_threat_ms=2.0,
        sample_size_defensive_phases=10,
        team_assignment_confidence=0.9,
    )
    assert result_drifted["sub_scores"]["line_discipline"] == 0.0

    # Player in line (0m deviation)
    result_in_line = score_defensive_positioning(
        avg_deviation_from_defensive_line_m=0.0,
        nearest_teammate_distance_m=10.0,
        closing_speed_toward_threat_ms=2.0,
        sample_size_defensive_phases=10,
        team_assignment_confidence=0.9,
    )
    assert result_in_line["sub_scores"]["line_discipline"] == 100.0


if __name__ == "__main__":
    test_press_resistance_never_pressured()
    test_press_resistance_low_team_assignment_confidence()
    test_off_ball_movement_space_creation()
    test_defensive_positioning_drifted_player()
    print("ALL PLAYER INTELLIGENCE SCORE TESTS PASSED!")
