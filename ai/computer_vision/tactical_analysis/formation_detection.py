"""
Formation Detection module.
Uses Hungarian algorithm (scipy.optimize.linear_sum_assignment) to match average outfield player positions
against candidate formation templates.
Analysis Logic Design v3 (docs/data analysis.md §3).
"""

from __future__ import annotations
import math
import numpy as np
from scipy.optimize import linear_sum_assignment

from ai.computer_vision.tactical_analysis.constants import (
    PITCH_LENGTH_M,
    PITCH_WIDTH_M,
    SCHEMA_VERSION,
    TEAM_ASSIGNMENT_CONFIDENCE_MIN,
)
from ai.computer_vision.tactical_analysis.formation_templates import FORMATION_TEMPLATES

NORMALIZATION_CONSTANT = 8.0  # Documented placeholder constant from §3


def detect_formation(
    player_positions: list[tuple[float, float]],
    team_assignment_confidence: float = 0.0,
    low_confidence_player_count: int = 0,
    normalization_constant: float = NORMALIZATION_CONSTANT,
    attacking_direction: str = "left_to_right",
) -> dict:
    """
    Detects the best matching formation for a given set of average outfield player positions (pitch_x_m, pitch_y_m).

    Args:

        player_positions: list of (x, y) pitch meter positions for tracked outfield players.
        team_assignment_confidence: confidence level of team assignment for the group.
        low_confidence_player_count: number of players with team_assignment_confidence < TEAM_ASSIGNMENT_CONFIDENCE_MIN.
        normalization_constant: calibration constant for distance score exponential decay.
        attacking_direction: "left_to_right" or "right_to_left".

    Returns:
        dict matching TeamMetric Standard Output Contract.
    """
    # 1. Gate check: team_assignment_confidence / low confidence players
    if team_assignment_confidence < TEAM_ASSIGNMENT_CONFIDENCE_MIN or low_confidence_player_count > 2:
        return {
            "metric_name": "formation",
            "value": None,
            "method": "deterministic",
            "confidence": "low_upstream_confidence",
            "confidence_score": None,
            "sample_size": len(player_positions),
            "sub_scores": {"reason": "low_team_assignment_confidence"},
            "schema_version": SCHEMA_VERSION,
        }

    # 2. Sample size check (< 8 outfield players)
    n_players = len(player_positions)
    if n_players < 8:
        return {
            "metric_name": "formation",
            "value": None,
            "method": "deterministic",
            "confidence": "low_sample",
            "confidence_score": None,
            "sample_size": n_players,
            "sub_scores": {"reason": "insufficient_outfield_players"},
            "schema_version": SCHEMA_VERSION,
        }

    player_pts = np.array(player_positions, dtype=np.float64)

    best_template = None
    best_confidence_score = -1.0
    best_distance_score = float("inf")

    for name, norm_slots in FORMATION_TEMPLATES.items():
        # Scale slots from normalized pitch fraction to actual meters
        slots_m = np.array(
            [(x_frac * PITCH_LENGTH_M, y_frac * PITCH_WIDTH_M) for x_frac, y_frac in norm_slots],
            dtype=np.float64,
        )

        if attacking_direction == "right_to_left":
            slots_m[:, 0] = PITCH_LENGTH_M - slots_m[:, 0]

        # Truncate or pad slots if player count != 10
        target_slots = slots_m[:n_players] if len(slots_m) >= n_players else slots_m

        # Compute cost matrix (Euclidean distances)
        # cost_matrix shape: (n_players, n_slots)
        cost_matrix = np.linalg.norm(player_pts[:, None, :] - target_slots[None, :, :], axis=2)

        # Hungarian algorithm assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        distance_score = float(cost_matrix[row_ind, col_ind].sum())

        # Confidence formula: 100 * exp(-distance_score / (n_players * C))
        confidence_score = 100.0 * math.exp(-distance_score / (n_players * normalization_constant))

        if confidence_score > best_confidence_score:
            best_confidence_score = confidence_score
            best_template = name
            best_distance_score = distance_score

    # FIXED (Phase 3 audit): a poor template match isn't a sample-size
    # problem -- all 10 outfield players were tracked fine, the shape
    # just doesn't fit any known template well (e.g. mid-transition,
    # broken shape after a corner). Mislabeling that as "low_sample"
    # conflates two different failure modes the MetricConfidence enum is
    # meant to distinguish (see docs/data analysis.md §0.3's severity
    # order). The categorical badge stays "normal" -- a low-confidence
    # *number* is still an honest, real result -- and the float
    # `confidence_score` in the response already carries the fit-quality
    # signal for the frontend to badge/color as it sees fit.
    confidence_enum = "normal"

    return {
        "metric_name": "formation",
        "value": best_template,
        "method": "deterministic",
        "confidence": confidence_enum,
        "confidence_score": round(best_confidence_score / 100.0, 4),
        "sample_size": n_players,
        "sub_scores": {
            "template_matching_distance": round(best_distance_score, 2),
            "matching_confidence_pct": round(best_confidence_score, 2),
        },
        "schema_version": SCHEMA_VERSION,
    }
