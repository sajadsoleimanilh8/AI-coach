"""
Camera (pixel) -> pitch (meters) homography.

Implements the contract required by docs/data analysis.md (Analysis Logic
Design v3) §0.2: every homography transform must carry a confidence score
derived from reprojection error, not just report coordinates. Downstream
consumers (First Touch, Press Resistance, Formation Detection, Injury Risk)
treat pitch_x_m/pitch_y_m as unusable when homography_confidence is below
HOMOGRAPHY_CONFIDENCE_MIN (0.6) -- see constants.py.

Typical flow:
    1. Run calibrate_pitch.py once per match/camera-angle to click 4-8+
       reference points in a representative frame -> saves a calibration
       JSON with pixel_pts, pitch_pts, H, reprojection_error, confidence.
    2. Load that JSON at analysis time, call pixel_to_pitch() (or the batch
       variant) on every tracked player/ball pixel position for that match.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class HomographyResult:
    """Everything downstream code needs to trust (or distrust) this transform."""

    H: np.ndarray                     # 3x3 homography matrix, pixel -> pitch meters
    reprojection_error_m: float       # mean per-point error, in pitch meters
    max_reprojection_error_m: float   # worst single-point error, in pitch meters
    confidence: float                 # 0-1, see homography_confidence()
    n_points: int
    per_point_errors_m: list = field(default_factory=list)


def compute_homography(
    pixel_pts: np.ndarray | list,
    pitch_pts: np.ndarray | list,
    method: int = 0,
    ransac_reproj_threshold: float = 3.0,
) -> HomographyResult:
    """
    Fit a pixel -> pitch-meters homography from corresponding point pairs.

    Args:
        pixel_pts: (N, 2) array of (x, y) pixel coordinates in the source frame.
        pitch_pts: (N, 2) array of (x, y) pitch coordinates in meters
            (use constants.REFERENCE_POINTS for known landmarks).
        method: 0 for exact least-squares/DLT fit (use when N == 4 and every
            point is trusted, e.g. clean corner-flag picks). Pass
            cv2.RANSAC for N > 4 with possibly-noisy clicks, which will
            down-weight outlier points automatically.
        ransac_reproj_threshold: only used when method=cv2.RANSAC; max
            allowed reprojection error (in pitch meters) for a point to be
            treated as an inlier.

    Returns:
        HomographyResult with the fitted matrix and an honest error/confidence
        estimate computed by reprojecting every input point and comparing
        against its known pitch position (not cv2's internal RANSAC mask,
        which can hide how bad the "inliers" actually are).

    Raises:
        ValueError: fewer than 4 point correspondences given (a homography
            has 8 degrees of freedom and needs at least 4 non-collinear
            point pairs to be well-posed).
    """
    pixel_pts = np.asarray(pixel_pts, dtype=np.float64).reshape(-1, 2)
    pitch_pts = np.asarray(pitch_pts, dtype=np.float64).reshape(-1, 2)

    if len(pixel_pts) != len(pitch_pts):
        raise ValueError(
            f"pixel_pts ({len(pixel_pts)}) and pitch_pts ({len(pitch_pts)}) "
            "must have the same number of points"
        )
    if len(pixel_pts) < 4:
        raise ValueError(
            f"Need at least 4 point correspondences to fit a homography, got {len(pixel_pts)}"
        )

    kwargs = {}
    if method == cv2.RANSAC:
        kwargs["ransacReprojThreshold"] = ransac_reproj_threshold

    H, _mask = cv2.findHomography(pixel_pts, pitch_pts, method=method, **kwargs)

    if H is None:
        raise ValueError(
            "cv2.findHomography failed to fit a matrix -- check that points "
            "aren't collinear or duplicated"
        )

    # Reproject every input pixel point through H and compare to its known
    # pitch position. This is the ground truth for confidence, independent
    # of whatever RANSAC internally decided was an inlier.
    projected = _apply_homography(pixel_pts, H)
    per_point_errors = np.linalg.norm(projected - pitch_pts, axis=1)

    mean_error = float(np.mean(per_point_errors))
    max_error = float(np.max(per_point_errors))

    return HomographyResult(
        H=H,
        reprojection_error_m=mean_error,
        max_reprojection_error_m=max_error,
        confidence=homography_confidence(mean_error),
        n_points=len(pixel_pts),
        per_point_errors_m=per_point_errors.tolist(),
    )


def _apply_homography(points: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Apply a 3x3 homography to an (N, 2) array of points."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(points, H)
    return transformed.reshape(-1, 2)


def pixel_to_pitch(x: float, y: float, H: np.ndarray) -> tuple[float, float]:
    """
    Transform a single pixel coordinate to pitch meters.

    Returns (pitch_x_m, pitch_y_m). Matches the field naming used by
    PlayerTracking / Event in docs/database_schema.md.
    """
    result = _apply_homography(np.array([[x, y]]), H)
    px, py = result[0]
    return float(px), float(py)


def pixels_to_pitch(points: np.ndarray | list, H: np.ndarray) -> np.ndarray:
    """
    Batch version of pixel_to_pitch. Use this for whole tracking arrays
    (e.g. all player positions in a frame, or a full trajectory) instead of
    looping pixel_to_pitch() point by point -- one cv2 call instead of N.

    Args:
        points: (N, 2) array of (x, y) pixel coordinates.
        H: 3x3 homography matrix from compute_homography().

    Returns:
        (N, 2) array of (pitch_x_m, pitch_y_m).
    """
    return _apply_homography(np.asarray(points, dtype=np.float64), H)


def homography_confidence(reprojection_error_m: float, scale_m: float = 2.0) -> float:
    """
    Map reprojection error (pitch meters) to a 0-1 confidence score.

    Uses the same exp(-error / scale) shape as Formation Detection's
    similarity confidence in docs/data analysis.md §3, for consistency
    across the codebase's confidence formulas: 0 error -> confidence 1.0,
    error growing relative to `scale_m` decays confidence toward 0.

    `scale_m` = 2.0 means: a 2m average reprojection error yields ~0.37
    confidence, comfortably below HOMOGRAPHY_CONFIDENCE_MIN (0.6), which a
    professional broadcast-camera calibration should never actually hit if
    the clicked points are accurate. Retune scale_m only after checking
    real calibration data, same caveat as NORMALIZATION_CONSTANT in the
    Formation Detection calibration procedure.

    Reference points, for intuition:
        error = 0.0m  -> confidence = 1.00
        error = 0.5m  -> confidence = 0.78
        error = 1.0m  -> confidence = 0.61
        error = 2.0m  -> confidence = 0.37
        error = 4.0m  -> confidence = 0.14
    """
    if reprojection_error_m < 0:
        raise ValueError("reprojection_error_m cannot be negative")
    confidence = math.exp(-reprojection_error_m / scale_m)
    return max(0.0, min(1.0, confidence))
