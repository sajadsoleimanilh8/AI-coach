"""
Standard FIFA full-size pitch geometry (meters) and named reference points.

Coordinate system: origin at the bottom-left corner flag, x running along the
length of the pitch (0 -> 105), y running along the width (0 -> 68). This is
the single source of truth for pitch-space coordinates -- the manual
calibration tool and the sanity-check tests both import from here so nobody
hand-types these numbers twice and lets them drift apart.

These match docs/data analysis.md (Analysis Logic Design v3) conventions:
pitch_x_m / pitch_y_m, HOMOGRAPHY_CONFIDENCE_MIN = 0.6.
"""

# ---- Pitch dimensions (meters) ----
# FIFA allows a range (100-110m x 64-75m); 105 x 68 is the standard used by
# most professional stadiums and by StatsBomb/Opta-style coordinate systems.
PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0

CENTER_CIRCLE_RADIUS_M = 9.15
PENALTY_AREA_DEPTH_M = 16.5     # goal line -> edge of the box
PENALTY_AREA_WIDTH_M = 40.32
SIX_YARD_DEPTH_M = 5.5
SIX_YARD_WIDTH_M = 18.32
PENALTY_SPOT_DISTANCE_M = 11.0
GOAL_WIDTH_M = 7.32

# Same threshold used throughout the analysis-logic doc (Analysis Logic
# Design v3, §0.2/§0.4) -- kept identical here on purpose so a
# homography_confidence computed by this module can be compared directly
# against that threshold without a units/convention mismatch.
HOMOGRAPHY_CONFIDENCE_MIN = 0.6

_L = PITCH_LENGTH_M
_W = PITCH_WIDTH_M
_CY = _W / 2.0  # halfway line's y (center of pitch, width-wise)

# ---- Named reference points a human can reliably click in a broadcast frame ----
# Every value is (pitch_x_m, pitch_y_m). Left-hand penalty box is the one
# nearest x=0, right-hand box is nearest x=105.
REFERENCE_POINTS = {
    # Corner flags
    "corner_bottom_left": (0.0, 0.0),
    "corner_top_left": (0.0, _W),
    "corner_bottom_right": (_L, 0.0),
    "corner_top_right": (_L, _W),

    # Halfway line
    "halfway_bottom": (_L / 2.0, 0.0),
    "halfway_top": (_L / 2.0, _W),
    "center_spot": (_L / 2.0, _CY),
    "center_circle_top": (_L / 2.0, _CY + CENTER_CIRCLE_RADIUS_M),
    "center_circle_bottom": (_L / 2.0, _CY - CENTER_CIRCLE_RADIUS_M),

    # Left penalty area (near x = 0)
    "left_penalty_area_bottom_near": (0.0, _CY - PENALTY_AREA_WIDTH_M / 2.0),
    "left_penalty_area_top_near": (0.0, _CY + PENALTY_AREA_WIDTH_M / 2.0),
    "left_penalty_area_bottom_far": (PENALTY_AREA_DEPTH_M, _CY - PENALTY_AREA_WIDTH_M / 2.0),
    "left_penalty_area_top_far": (PENALTY_AREA_DEPTH_M, _CY + PENALTY_AREA_WIDTH_M / 2.0),
    "left_six_yard_bottom_near": (0.0, _CY - SIX_YARD_WIDTH_M / 2.0),
    "left_six_yard_top_near": (0.0, _CY + SIX_YARD_WIDTH_M / 2.0),
    "left_six_yard_bottom_far": (SIX_YARD_DEPTH_M, _CY - SIX_YARD_WIDTH_M / 2.0),
    "left_six_yard_top_far": (SIX_YARD_DEPTH_M, _CY + SIX_YARD_WIDTH_M / 2.0),
    "left_penalty_spot": (PENALTY_SPOT_DISTANCE_M, _CY),

    # Right penalty area (near x = 105), mirrored
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

# Ordered list of point names shown to the operator by the calibration tool.
# Corners + halfway + one penalty box first (the points most likely to be
# visible in a single broadcast/tactical-cam frame), rest are optional extras.
CALIBRATION_POINT_ORDER = [
    "corner_bottom_left",
    "corner_top_left",
    "corner_bottom_right",
    "corner_top_right",
    "halfway_bottom",
    "halfway_top",
    "center_spot",
    "left_penalty_area_bottom_near",
    "left_penalty_area_top_near",
    "left_penalty_area_bottom_far",
    "left_penalty_area_top_far",
    "left_penalty_spot",
    "right_penalty_area_bottom_near",
    "right_penalty_area_top_near",
    "right_penalty_area_bottom_far",
    "right_penalty_area_top_far",
    "right_penalty_spot",
]

MIN_CALIBRATION_POINTS = 4
