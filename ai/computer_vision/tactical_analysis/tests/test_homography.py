"""
Sanity checks for the pixel <-> pitch homography.

No real video needed: we synthesize a plausible broadcast-camera view by
applying a known forward (pitch -> pixel) perspective transform to the
pitch reference points, feed the resulting "clicked pixels" into
compute_homography(), and then check that:

  1. Reprojection error / confidence behave sanely (low error -> high conf,
     added click noise -> lower confidence).
  2. Real-world distances (penalty box width ~40.32m, center circle
     diameter ~18.3m, six-yard box depth ~5.5m) measured on *held-out*
     points (not used to fit H) come back close to their known values --
     this is the actual "did we get a usable transform" test, since
     reprojection error alone can look fine while distances are still off
     if the point set is degenerate.

Run: python3 test_homography.py
"""

from __future__ import annotations

import math
import os
import sys

import cv2
import numpy as np

# tests/ is one level below the package root -- add the parent dir so
# `from constants import ...` / `from homography import ...` resolve when
# this file is run directly (python3 tests/test_homography.py) or via pytest.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from constants import CENTER_CIRCLE_RADIUS_M, PENALTY_AREA_WIDTH_M, REFERENCE_POINTS, SIX_YARD_DEPTH_M
from homography import compute_homography, homography_confidence, pixel_to_pitch, pixels_to_pitch


def make_synthetic_forward_homography() -> np.ndarray:
    """
    A hand-built pitch (meters) -> pixel perspective transform standing in
    for a real broadcast/tactical camera: pitch recedes into the distance,
    so far points compress toward a vanishing region, near points spread out.
    This is the *inverse* of what compute_homography() will fit.
    """
    # Map 4 known pitch corners of the left half of the pitch to plausible
    # pixel positions in a 1920x1080 frame, with the standard "camera looking
    # down the touchline" perspective compression on the far side.
    pitch_corners = np.array(
        [
            [0.0, 0.0],     # near-left corner  -> bottom-left of frame, spread out
            [0.0, 68.0],    # near-right corner -> bottom-right of frame
            [52.5, 0.0],    # halfway near      -> compressed toward center, higher up
            [52.5, 68.0],   # halfway far
        ],
        dtype=np.float64,
    )
    pixel_corners = np.array(
        [
            [200.0, 1000.0],
            [1720.0, 1000.0],
            [760.0, 300.0],
            [1160.0, 300.0],
        ],
        dtype=np.float64,
    )
    H_forward, _ = cv2.findHomography(pitch_corners, pixel_corners, method=0)
    return H_forward


def pitch_to_pixel(points_m: np.ndarray, H_forward: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_m, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, H_forward).reshape(-1, 2)


def euclidean(p1, p2) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def run():
    H_forward = make_synthetic_forward_homography()

    # --- Calibration set: points an operator would actually click ---
    calib_names = [
        "corner_bottom_left",
        "corner_top_left",
        "halfway_bottom",
        "halfway_top",
        "left_penalty_area_bottom_near",
        "left_penalty_area_top_near",
    ]
    calib_pitch_pts = np.array([REFERENCE_POINTS[n] for n in calib_names])
    calib_pixel_pts = pitch_to_pixel(calib_pitch_pts, H_forward)

    # --- Held-out set: NOT used to fit H, used only to check distances ---
    holdout_names = [
        "left_penalty_area_bottom_far",
        "left_penalty_area_top_far",
        "left_six_yard_bottom_near",
        "left_six_yard_bottom_far",
        "center_circle_top",
        "center_circle_bottom",
    ]
    holdout_pitch_pts = np.array([REFERENCE_POINTS[n] for n in holdout_names])
    holdout_pixel_pts = pitch_to_pixel(holdout_pitch_pts, H_forward)

    print("=" * 70)
    print("TEST 1: clean click points -> low reprojection error, high confidence")
    print("=" * 70)
    result = compute_homography(calib_pixel_pts, calib_pitch_pts, method=0)
    print(f"  reprojection_error_m = {result.reprojection_error_m:.4f}")
    print(f"  confidence           = {result.confidence:.4f}")
    assert result.reprojection_error_m < 0.01, "clean synthetic points should reproject almost exactly"
    assert result.confidence > 0.99, "near-zero error should yield near-1.0 confidence"
    print("  PASS")

    print()
    print("=" * 70)
    print("TEST 2: known real-world distances on HELD-OUT points")
    print("=" * 70)
    H = result.H
    recovered_pitch = pixels_to_pitch(holdout_pixel_pts, H)
    recovered = dict(zip(holdout_names, recovered_pitch))

    # Penalty box width: near/far edges share the box-width span across y
    box_width_recovered = euclidean(
        recovered["left_penalty_area_bottom_far"], recovered["left_penalty_area_top_far"]
    )
    print(f"  Penalty box width:   recovered={box_width_recovered:.3f}m  known={PENALTY_AREA_WIDTH_M}m")
    assert abs(box_width_recovered - PENALTY_AREA_WIDTH_M) < 0.05, "penalty box width off by >5cm"

    # Center circle diameter (top point to bottom point, straight through center)
    circle_diameter_recovered = euclidean(recovered["center_circle_top"], recovered["center_circle_bottom"])
    known_diameter = 2 * CENTER_CIRCLE_RADIUS_M
    print(f"  Center circle diam:  recovered={circle_diameter_recovered:.3f}m  known={known_diameter}m")
    assert abs(circle_diameter_recovered - known_diameter) < 0.05, "center circle diameter off by >5cm"

    # Six-yard box depth
    six_yard_depth_recovered = euclidean(
        recovered["left_six_yard_bottom_near"], recovered["left_six_yard_bottom_far"]
    )
    print(f"  Six-yard box depth:  recovered={six_yard_depth_recovered:.3f}m  known={SIX_YARD_DEPTH_M}m")
    assert abs(six_yard_depth_recovered - SIX_YARD_DEPTH_M) < 0.05, "six-yard depth off by >5cm"
    print("  PASS")

    print()
    print("=" * 70)
    print("TEST 3: noisy clicks (simulating imprecise manual point-picking)")
    print("=" * 70)
    rng = np.random.default_rng(42)
    noisy_pixel_pts = calib_pixel_pts + rng.normal(scale=3.0, size=calib_pixel_pts.shape)  # +-3px jitter
    noisy_result = compute_homography(noisy_pixel_pts, calib_pitch_pts, method=0)
    print(f"  reprojection_error_m = {noisy_result.reprojection_error_m:.4f}")
    print(f"  confidence           = {noisy_result.confidence:.4f}")
    assert noisy_result.reprojection_error_m > result.reprojection_error_m, (
        "noisy clicks should produce strictly higher error than clean clicks"
    )
    assert noisy_result.confidence < result.confidence, "noisy clicks should reduce confidence"
    print("  PASS (noise correctly degrades confidence)")

    print()
    print("=" * 70)
    print("TEST 4: single-point convenience function matches batch function")
    print("=" * 70)
    x, y = holdout_pixel_pts[0]
    single = pixel_to_pitch(float(x), float(y), H)
    batch = pixels_to_pitch(holdout_pixel_pts, H)[0]
    assert abs(single[0] - batch[0]) < 1e-9 and abs(single[1] - batch[1]) < 1e-9
    print(f"  pixel_to_pitch({x:.1f}, {y:.1f}) = {single}")
    print("  PASS")

    print()
    print("=" * 70)
    print("TEST 5: too few points raises ValueError")
    print("=" * 70)
    try:
        compute_homography(calib_pixel_pts[:3], calib_pitch_pts[:3])
        raise AssertionError("expected ValueError for <4 points")
    except ValueError as e:
        print(f"  correctly raised: {e}")
    print("  PASS")

    print()
    print("=" * 70)
    print("TEST 6: confidence formula spot-check against HOMOGRAPHY_CONFIDENCE_MIN")
    print("=" * 70)
    from constants import HOMOGRAPHY_CONFIDENCE_MIN

    # A calibration with ~1.8m average error should fall right around the
    # "unusable" threshold used everywhere else in the pipeline (0.6).
    c = homography_confidence(1.8)
    print(f"  homography_confidence(1.8m) = {c:.3f}  (threshold={HOMOGRAPHY_CONFIDENCE_MIN})")
    c_good = homography_confidence(0.3)
    c_bad = homography_confidence(5.0)
    assert c_good > HOMOGRAPHY_CONFIDENCE_MIN
    assert c_bad < HOMOGRAPHY_CONFIDENCE_MIN
    print(f"  homography_confidence(0.3m) = {c_good:.3f}  (above threshold, usable)")
    print(f"  homography_confidence(5.0m) = {c_bad:.3f}  (below threshold, unusable)")
    print("  PASS")

    print()
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    run()
