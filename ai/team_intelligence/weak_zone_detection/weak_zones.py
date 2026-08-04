"""
Weak-Zone Detection module.
Divides pitch into 6x4 coarse grid and computes defensive-player occupancy density.
Analysis Logic Design v3 & Implementation Spec §1.2.
"""

from __future__ import annotations
import numpy as np

from ai.computer_vision.tactical_analysis.constants import (
    PITCH_LENGTH_M,
    PITCH_WIDTH_M,
    SCHEMA_VERSION,
    TEAM_ASSIGNMENT_CONFIDENCE_MIN,
)


def compute_weak_zones(
    player_positions_m: list[tuple[float, float]],
    grid_x: int = 6,
    grid_y: int = 4,
    team_assignment_confidence: float = 0.0,
) -> dict:
    """
    Computes occupancy density per grid zone (grid_x x grid_y).
    Zones below bottom quartile coverage threshold are flagged as weak.

    Returns TeamMetric dict with metric_name="weak_zone_map".
    """
    if team_assignment_confidence < TEAM_ASSIGNMENT_CONFIDENCE_MIN or len(player_positions_m) == 0:
        # Generate default grid with None/low_upstream_confidence
        sub_scores = {f"zone_{x+1}_{y+1}": 0.0 for x in range(grid_x) for y in range(grid_y)}
        return {
            "metric_name": "weak_zone_map",
            "value": None,
            "method": "deterministic",
            "confidence": "low_upstream_confidence" if team_assignment_confidence < TEAM_ASSIGNMENT_CONFIDENCE_MIN else "low_sample",
            "sample_size": len(player_positions_m),
            "sub_scores": sub_scores,
            "schema_version": SCHEMA_VERSION,
        }

    dx = PITCH_LENGTH_M / grid_x
    dy = PITCH_WIDTH_M / grid_y

    counts = np.zeros((grid_x, grid_y), dtype=np.float64)

    for x, y in player_positions_m:
        gx = int(min(grid_x - 1, max(0, x // dx)))
        gy = int(min(grid_y - 1, max(0, y // dy)))
        counts[gx, gy] += 1.0

    total_pts = len(player_positions_m)
    density = counts / total_pts

    sub_scores = {}
    for x in range(grid_x):
        for y in range(grid_y):
            key = f"zone_{x+1}_{y+1}"
            sub_scores[key] = round(float(density[x, y]), 3)

    # Average occupancy per zone as baseline
    q25 = float(np.percentile(density, 25))
    weak_zone_count = int((density <= q25).sum())

    return {
        "metric_name": "weak_zone_map",
        "value": float(weak_zone_count),
        "method": "deterministic",
        "confidence": "normal",
        "sample_size": total_pts,
        "sub_scores": sub_scores,
        "schema_version": SCHEMA_VERSION,
    }
