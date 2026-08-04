"""
MediaPipe Pose on player crops -> shoulder-line body orientation.
Mirrors homography.py's pattern: every result carries its own confidence,
never just a bare angle -- see constants.py's HOMOGRAPHY_CONFIDENCE_MIN
precedent for why silent-low-confidence numbers are the failure mode to avoid here.
"""
import math
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose

def estimate_body_orientation(frame_crop: np.ndarray) -> dict:
    """
    Args: frame_crop -- BGR image cropped tightly to one player's bbox.
    Returns: {"orientation_deg": float | None, "confidence": float}
        orientation_deg: angle of the shoulder line (landmark 11 <-> 12)
        relative to the pitch's x-axis, 0-360. None if landmarks not visible.
        confidence: min visibility of the two shoulder landmarks (0-1).
    """
    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        results = pose.process(frame_crop)
        if not results.pose_landmarks:
            return {"orientation_deg": None, "confidence": 0.0}

        lm = results.pose_landmarks.landmark
        left_shoulder, right_shoulder = lm[11], lm[12]
        confidence = min(left_shoulder.visibility, right_shoulder.visibility)

        dx = right_shoulder.x - left_shoulder.x
        dy = right_shoulder.y - left_shoulder.y
        angle = math.degrees(math.atan2(dy, dx)) % 360

        return {"orientation_deg": angle, "confidence": confidence}