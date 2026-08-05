"""
Builds per-player trajectories from tracked detections and computes
speed / distance / acceleration in pitch meters.

Implements the formulas from docs/tracking_system_design.md §7 exactly:
    distance     = sqrt((x2-x1)^2 + (y2-y1)^2)
    speed        = distance / time
    acceleration = (speed2 - speed1) / time

...but computed in PITCH meters (via tactical_analysis/homography.py), not
pixels -- pixel-space "speed" is meaningless (a player far from camera
moves fewer pixels/frame than the same real speed up close). This is the
concrete link between player_tracking (this module) and tactical_analysis
(homography) that docs/database_schema.md's PlayerTracking table assumes.

Per Analysis Logic Design v3 §0.2/§0.6: any frame where homography_confidence
is below HOMOGRAPHY_CONFIDENCE_MIN is dropped from the trajectory entirely
(pitch coordinates unusable), not imputed -- a speed computed across a
dropped frame would silently use a bad position.
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass

# trajectory.py lives at ai/computer_vision/player_tracking/trajectory.py.
# tactical_analysis/ is a SIBLING package at ai/computer_vision/tactical_analysis/,
# not a subpackage of player_tracking/ -- so Python won't find it via a plain
# `from tactical_analysis... import ...` unless ai/computer_vision/ itself is
# on sys.path. Add it explicitly, relative to this file's own location, so
# this works no matter where the script is invoked from (repo root, this
# folder, an IDE, etc.) instead of depending on the caller's cwd.
_AI_COMPUTER_VISION_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _AI_COMPUTER_VISION_DIR not in sys.path:
    sys.path.insert(0, _AI_COMPUTER_VISION_DIR)

# FIXED (found while wiring pose estimation into backend/pipeline/runner.py
# and actually running the import path it uses -- not caught before
# because no test previously imported this module the way runner.py does).
# The block above puts ai/computer_vision/ on sys.path so `tactical_analysis`
# resolves, but `from tracker import TrackedDetection` below needs THIS
# file's OWN directory (player_tracking/) on sys.path -- that's only true
# by accident when a script imports this file directly from within
# player_tracking/ (Python auto-adds a run script's own directory), which
# is exactly what every existing test here does
# (`sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))`
# from inside player_tracking/tests/). It is NOT true when something
# imports this module via its full package path, e.g.
# `from ai.computer_vision.player_tracking.trajectory import ...`
# -- which is exactly how backend/pipeline/runner.py imports it. That
# import previously raised ModuleNotFoundError: No module named 'tracker'
# the first time the real pipeline ran, since nothing had exercised this
# import path before (there is still no backend/pipeline/tests/test_runner.py --
# see that gap noted elsewhere). Adding this file's own directory fixes both
# import styles at once.
_PLAYER_TRACKING_DIR = os.path.dirname(os.path.abspath(__file__))
if _PLAYER_TRACKING_DIR not in sys.path:
    sys.path.insert(0, _PLAYER_TRACKING_DIR)

# Import the tactical_analysis package built earlier.
from tactical_analysis.constants import HOMOGRAPHY_CONFIDENCE_MIN
from tactical_analysis.homography import pixel_to_pitch

from tracker import TrackedDetection


@dataclass
class TrackingPoint:
    """One row of docs/database_schema.md's PlayerTracking table."""

    match_id: str
    player_id: int
    frame_id: int
    team_id: str | None
    pixel_x: float
    pixel_y: float
    pitch_x_m: float | None
    pitch_y_m: float | None
    homography_confidence: float | None
    speed: float | None          # m/s
    distance: float | None       # meters, delta since previous point
    acceleration: float | None   # m/s^2
    body_orientation_deg: float | None = None          # shoulder-line angle, 0-360; None if not sampled/not visible
    body_orientation_confidence: float | None = None   # min shoulder-landmark visibility, 0-1


def enrich_with_pitch_coordinates(
    frames: list[list["TrackedDetection"]],
    match_id: str,
    H,
    homography_confidence: float,
    fps: float,
) -> dict[int, list[TrackingPoint]]:
    """
    Convert tracked pixel detections into pitch-meter TrackingPoints, then
    compute speed/distance/acceleration per player along the way.

    Args:
        frames: output of tracker.track_video() / tracker.track_detections()
            -- one list[TrackedDetection] per frame, in frame order.
        match_id: Match.match_id this video belongs to.
        H: 3x3 homography matrix from tactical_analysis.homography.compute_homography().
        homography_confidence: the single confidence score for this H
            (one calibration per match/camera-angle, per the manual_calibration
            tool -- not recomputed per frame).
        fps: video frame rate, used to convert frame deltas to seconds.

    Returns:
        {player_id: [TrackingPoint, ...]} -- one ordered trajectory per
        tracked player_id (ball included, under its own track id, if the
        input frames contained ball detections).
    """
    dt = 1.0 / fps
    usable = homography_confidence >= HOMOGRAPHY_CONFIDENCE_MIN

    trajectories: dict[int, list[TrackingPoint]] = {}
    last_point_by_player: dict[int, TrackingPoint] = {}
    last_speed_by_player: dict[int, float] = {}

    for frame_number, detections in enumerate(frames):
        for det in detections:
            if det.class_name not in ("player", "goalkeeper"):
                continue  # ball/referee trajectories, if needed, go through the same
                          # function separately -- kept out here so player speed/
                          # distance aggregates (ACWR, sprint counts) aren't polluted

            px, py = det.foot_point()

            if not usable:
                point = TrackingPoint(
                    match_id=match_id, player_id=det.player_id, frame_id=frame_number,
                    team_id=det.team_id, pixel_x=px, pixel_y=py,
                    pitch_x_m=None, pitch_y_m=None,
                    homography_confidence=homography_confidence,
                    speed=None, distance=None, acceleration=None,
                )
                trajectories.setdefault(det.player_id, []).append(point)
                continue

            pitch_x, pitch_y = pixel_to_pitch(px, py, H)

            prev = last_point_by_player.get(det.player_id)
            distance = speed = acceleration = None
            if prev is not None and prev.pitch_x_m is not None:
                distance = math.hypot(pitch_x - prev.pitch_x_m, pitch_y - prev.pitch_y_m)
                speed = distance / dt
                prev_speed = last_speed_by_player.get(det.player_id)
                if prev_speed is not None:
                    acceleration = (speed - prev_speed) / dt

            point = TrackingPoint(
                match_id=match_id, player_id=det.player_id, frame_id=frame_number,
                team_id=det.team_id, pixel_x=px, pixel_y=py,
                pitch_x_m=pitch_x, pitch_y_m=pitch_y,
                homography_confidence=homography_confidence,
                speed=speed, distance=distance, acceleration=acceleration,
            )
            trajectories.setdefault(det.player_id, []).append(point)
            last_point_by_player[det.player_id] = point
            if speed is not None:
                last_speed_by_player[det.player_id] = speed

    return trajectories


def attach_body_orientation(
    trajectories: dict[int, list[TrackingPoint]],
    orientation_by_player_frame: dict[tuple[int, int], "OrientationResult"],
) -> None:
    """
    Mutates `trajectories` in place, filling in body_orientation_deg/
    body_orientation_confidence on the TrackingPoints that have a pose
    reading.

    Kept as a separate pass rather than computed inline inside
    enrich_with_pitch_coordinates() on purpose: pose estimation
    (ai/computer_vision/pose_estimation/pose.py) is expensive enough that
    it's only run on a stride-sampled subset of frames (see
    POSE_SAMPLE_STRIDE in backend/pipeline/runner.py), not every frame
    like the pixel->pitch homography above -- mixing a sparse data source
    into the main per-frame loop would make that loop's "is this frame
    usable" logic harder to reason about for no benefit, since orientation
    readings are sparse by design, not by failure.

    Args:
        trajectories: output of enrich_with_pitch_coordinates(), mutated
            in place.
        orientation_by_player_frame: {(player_id, frame_number): OrientationResult}
            for whichever (player, frame) pairs were actually sampled --
            most (player_id, frame_number) combinations in a trajectory
            will simply have no entry here, and those TrackingPoints keep
            their default None/None (dataclass defaults above), which is
            the correct "not measured" state, not an error.
    """
    for player_id, points in trajectories.items():
        for point in points:
            key = (player_id, point.frame_id)
            reading = orientation_by_player_frame.get(key)
            if reading is None:
                continue
            point.body_orientation_deg = reading.orientation_deg
            point.body_orientation_confidence = reading.confidence


def total_distance_covered(trajectory: list[TrackingPoint]) -> float:
    """Sum of per-frame distance deltas, ignoring frames with no distance
    (first frame of the trajectory, or frames dropped for low homography
    confidence)."""
    return sum(p.distance for p in trajectory if p.distance is not None)


def sprint_count(trajectory: list[TrackingPoint], sprint_threshold_ms: float = 7.0) -> int:
    """
    Number of frames where instantaneous speed exceeds sprint_threshold_ms
    (7.0 m/s ~= 25.2 km/h is a commonly used sprint-speed threshold in
    sports-science literature). Used as `sprint_load` input for the MVP
    Injury Risk fallback in docs/data analysis.md §4.
    """
    return sum(1 for p in trajectory if p.speed is not None and p.speed > sprint_threshold_ms)
