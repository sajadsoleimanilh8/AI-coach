"""
Shared constants for tactical analysis and scoring functions.
Analysis Logic Design v3 (docs/data analysis.md §0.4).
"""

PRESSURE_RADIUS_M = 5.0  # opponent within this = "applying pressure"
PRESSURE_THRESHOLD = 0.5  # pressure_level above this = "under pressure"
RETENTION_WINDOW_S = 3.0  # min time to count possession as "retained"
TOUCH_EVAL_WINDOW_S = 2.0  # window after first touch to evaluate outcome
MAX_TOUCH_DISTANCE_M = 5.0  # ball travel beyond this after touch = poor control
DECISION_TIME_MIN_S = 0.3  # fastest realistic decision
DECISION_TIME_MAX_S = 2.0  # slowest before treated as "too slow"
MIN_SAMPLE_EVENTS = 5  # below this, flag score as "low_sample"
HOMOGRAPHY_CONFIDENCE_MIN = 0.6  # below this, pitch coordinates are unusable
TEAM_ASSIGNMENT_CONFIDENCE_MIN = 0.5  # below this, treat team_id as unassigned
SCHEMA_VERSION = "v3"

# Day 9 & Day 10 new constants
BALL_CONTROL_RADIUS_M = 1.5  # player within 1.5m of ball = possession/touch
PASS_MIN_DISTANCE_M = 3.0  # min ball distance to qualify as a pass
SHOT_VELOCITY_MIN_MS = 12.0  # min ball velocity to qualify as a shot
MAX_USEFUL_SEPARATION_GAIN_M = 5.0  # Off-ball movement space creation max
MAX_ACCEPTABLE_LINE_DEVIATION_M = 8.0  # Defensive line max deviation
IDEAL_DEFENSIVE_SPACING_M = 10.0  # Ideal distance between adjacent defenders
SPACING_TOLERANCE_M = 3.0  # Spacing Gaussian tolerance
REACTION_SPEED_REFERENCE_MS = 3.0  # Closing speed reference for threat reaction

# Pitch Dimensions (Standard FIFA Pitch in meters)
PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0
GOAL_WIDTH_M = 7.32
SIX_YARD_DEPTH_M = 5.5
SIX_YARD_WIDTH_M = 18.32
PENALTY_AREA_DEPTH_M = 16.5
PENALTY_AREA_WIDTH_M = 40.32
PENALTY_SPOT_DISTANCE_M = 11.0
CENTER_CIRCLE_RADIUS_M = 9.15

_L = PITCH_LENGTH_M
_W = PITCH_WIDTH_M
_CY = _W / 2.0

# FIXED (Phase 3 audit): the six-yard-box and center-circle-top/bottom
# entries below were dropped when Phase 3's constants were merged into
# this file. tests/test_homography.py (already in the repo, not part of
# this diff) asserts against exactly these keys -- losing them silently
# breaks that test suite the next time it's run. Restored here, unchanged
# from their original values.
REFERENCE_POINTS = {
    "corner_bottom_left": (0.0, 0.0),
    "corner_top_left": (0.0, _W),
    "corner_bottom_right": (_L, 0.0),
    "corner_top_right": (_L, _W),
    "halfway_bottom": (_L / 2.0, 0.0),
    "halfway_top": (_L / 2.0, _W),
    "center_spot": (_L / 2.0, _CY),
    "center_circle_top": (_L / 2.0, _CY + CENTER_CIRCLE_RADIUS_M),
    "center_circle_bottom": (_L / 2.0, _CY - CENTER_CIRCLE_RADIUS_M),
    "left_penalty_area_bottom_near": (0.0, _CY - PENALTY_AREA_WIDTH_M / 2.0),
    "left_penalty_area_top_near": (0.0, _CY + PENALTY_AREA_WIDTH_M / 2.0),
    "left_penalty_area_bottom_far": (PENALTY_AREA_DEPTH_M, _CY - PENALTY_AREA_WIDTH_M / 2.0),
    "left_penalty_area_top_far": (PENALTY_AREA_DEPTH_M, _CY + PENALTY_AREA_WIDTH_M / 2.0),
    "left_six_yard_bottom_near": (0.0, _CY - SIX_YARD_WIDTH_M / 2.0),
    "left_six_yard_top_near": (0.0, _CY + SIX_YARD_WIDTH_M / 2.0),
    "left_six_yard_bottom_far": (SIX_YARD_DEPTH_M, _CY - SIX_YARD_WIDTH_M / 2.0),
    "left_six_yard_top_far": (SIX_YARD_DEPTH_M, _CY + SIX_YARD_WIDTH_M / 2.0),
    "left_penalty_spot": (PENALTY_SPOT_DISTANCE_M, _CY),
    "right_penalty_area_bottom_near": (_L, _CY - PENALTY_AREA_WIDTH_M / 2.0),
    "right_penalty_area_top_near": (_L, _CY + PENALTY_AREA_WIDTH_M / 2.0),
    "right_penalty_area_bottom_far": (_L - PENALTY_AREA_DEPTH_M, _CY - PENALTY_AREA_WIDTH_M / 2.0),
    "right_penalty_area_top_far": (_L - PENALTY_AREA_DEPTH_M, _CY + PENALTY_AREA_WIDTH_M / 2.0),
    "right_six_yard_bottom_near": (_L, _CY - SIX_YARD_WIDTH_M / 2.0),
    "right_six_yard_top_near": (_L, _CY + SIX_YARD_WIDTH_M / 2.0),
    "right_six_yard_bottom_far": (_L - SIX_YARD_DEPTH_M, _CY - SIX_YARD_WIDTH_M / 2.0),
    "right_six_yard_top_far": (_L - SIX_YARD_DEPTH_M, _CY + SIX_YARD_WIDTH_M / 2.0),
    "right_penalty_spot": (_L - PENALTY_SPOT_DISTANCE_M, _CY),
}

CALIBRATION_POINT_ORDER = [
    "corner_bottom_left", "corner_top_left", "corner_bottom_right", "corner_top_right",
    "halfway_bottom", "halfway_top", "center_spot",
    "left_penalty_area_bottom_near", "left_penalty_area_top_near",
    "left_penalty_area_bottom_far", "left_penalty_area_top_far", "left_penalty_spot",
    "right_penalty_area_bottom_near", "right_penalty_area_top_near",
    "right_penalty_area_bottom_far", "right_penalty_area_top_far", "right_penalty_spot",
]
MIN_CALIBRATION_POINTS = 4

def pressure_level(distance_to_nearest_opponent_m: float | None) -> float | None:
    """Computes pressure level (0.0 to 1.0) based on opponent proximity."""
    if distance_to_nearest_opponent_m is None:
        return None
    return max(0.0, min(1.0, 1.0 - (distance_to_nearest_opponent_m / PRESSURE_RADIUS_M)))
