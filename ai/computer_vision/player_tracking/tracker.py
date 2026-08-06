"""
Player & ball tracking layer: assigns stable identities to per-frame YOLO
detections using ByteTrack.

Bridges `ai/computer_vision/player detection` (per-frame boxes, no
identity -- see that folder's test.py / detections.json) to:
  - docs/database_schema.md's PlayerDetection / PlayerTracking tables
    (continuous trajectories need a stable player_id per frame)
  - ai/computer_vision/tactical_analysis/homography.py (pixel -> pitch,
    needs to know *which* pixel belongs to *which* player across frames
    to compute speed/distance)

IMPORTANT (matches docs/database_schema.md's own note): the `player_id`
produced here is a ByteTrack tracking ID, valid only within one continuous
video/match. It is NOT a persistent player identity across matches -- that
requires jersey-number OCR or Re-ID, which is out of scope here (flagged as
a Phase 3+ item in tracking_system_design.md §13).

Two entry points, pick based on where your detections come from:

  track_video(model_path, video_path)
      Runs YOLO + ByteTrack together via ultralytics' built-in `.track()`.
      Preferred path for a single, already-trained checkpoint (e.g. the
      one produced by phase0_1_pipeline.py) -- one pass over the video,
      no intermediate JSON needed.

  track_detections(frames)
      Re-tracks an ALREADY-COMPUTED per-frame detection list -- the exact
      shape `ai/computer_vision/player detection/test.py` writes to
      detections.json -- using supervision's ByteTrack implementation.
      Useful when detections came from a different/merged source (e.g.
      ensembled across multiple checkpoints) and re-running YOLO would be
      wasteful.

Both return the same canonical per-frame shape (see `TrackedDetection`),
matching PlayerDetection field names 1:1 so the output can be written
straight to that table with no renaming/reshaping step.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterator


# Classes produced by the Phase 0-1 detector (see merge.py / data.yaml):
# 0: ball, 1: goalkeeper, 2: player, 3: referee
TRACKABLE_CLASS_NAMES = {"player", "goalkeeper"}  # referees excluded from player tracking by default
BALL_CLASS_NAME = "ball"


@dataclass
class TrackedDetection:
    """One tracked box in one frame. Field names match PlayerDetection
    (docs/database_schema.md) so this can be inserted with no remapping."""

    frame_number: int
    player_id: int          # ByteTrack ID -- stable within this video only
    class_name: str         # "player" | "goalkeeper" | "ball" | "referee"
    team_id: str | None     # always None here -- team assignment (jersey
                             # clustering) is a separate, not-yet-implemented
                             # module (see recent_updates: hardcoded "neutral")
    team_assignment_confidence: float | None
    x: float                # pixel-space, top-left corner
    y: float
    width: float
    height: float
    confidence: float       # YOLO detection confidence, 0-1

    def foot_point(self) -> tuple[float, float]:
        """
        Pixel position to feed into homography.pixel_to_pitch().

        Use bottom-center of the box, not the box center: the homography
        maps points that lie ON the pitch plane (z=0), and a standing
        player's *feet* are on that plane -- the bbox center is roughly at
        chest height and will project to the wrong pitch location,
        especially for players near the edges of the frame where the
        camera angle is shallow. This is a common, easy-to-miss source of
        several-meter position error that reprojection-error checks alone
        won't catch (see tactical_analysis/README.md's calibration notes).
        """
        return (self.x + self.width / 2.0, self.y + self.height)

    def to_dict(self) -> dict:
        return asdict(self)


def _xyxy_to_xywh(x1: float, y1: float, x2: float, y2: float) -> tuple[float, float, float, float]:
    return float(x1), float(y1), float(x2 - x1), float(y2 - y1)


def track_video(
    model_path: str,
    video_path: str,
    tracker_config: str = "bytetrack.yaml",
    conf: float = 0.25,
    classes: list[str] | None = None,
) -> Iterator[list[TrackedDetection]]:
    """
    Run YOLO detection + ByteTrack tracking together, frame by frame.

    Args:
        model_path: path to a trained .pt checkpoint (e.g. from
            phase0_1_pipeline.py's `../runs/<name>/weights/best.pt`).
        video_path: path to the match video.
        tracker_config: ultralytics tracker config name. "bytetrack.yaml"
            (default) needs no Re-ID model and is fast; "botsort.yaml" adds
            appearance-based Re-ID and handles longer occlusions better at
            higher compute cost -- worth trying in Phase 2 if ID switches
            during player pile-ups (corners, goal celebrations) turn out
            to be a problem with ByteTrack alone.
        conf: YOLO confidence threshold. Kept at 0.25 (matches
            phase0_1_pipeline.py's suggested inference conf) rather than
            YOLO's 0.5 default, since the ball class is small and easy to
            under-detect -- see phase0_1_pipeline.py's imgsz note.
        classes: optional whitelist of class names to keep (e.g.
            ["player", "ball"] to drop referees). None = keep everything
            the model detects.

    Yields:
        One list[TrackedDetection] per video frame, in frame order.
    """
    from ultralytics import YOLO

    model = YOLO(model_path)
    class_name_by_id = model.names  # {0: "ball", 1: "goalkeeper", ...}

    class_filter_ids = None
    if classes is not None:
        wanted = set(classes)
        class_filter_ids = [i for i, name in class_name_by_id.items() if name in wanted]

    results = model.track(
        source=video_path,
        tracker=tracker_config,
        conf=conf,
        classes=class_filter_ids,
        persist=True,   # keep ByteTrack state across the stream call
        stream=True,    # generator, don't load the whole video into memory
        verbose=False,
    )

    for frame_number, result in enumerate(results):
        frame_detections: list[TrackedDetection] = []

        if result.boxes is None or result.boxes.id is None:
            # No detections, or ByteTrack hasn't assigned IDs yet this frame
            # (can happen on the very first frames) -- yield an empty frame
            # rather than skipping it, so frame_number stays contiguous.
            yield frame_detections
            continue

        boxes = result.boxes
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i])
            class_name = class_name_by_id[cls_id]
            track_id = int(boxes.id[i])
            conf_score = float(boxes.conf[i])
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
            x, y, w, h = _xyxy_to_xywh(x1, y1, x2, y2)

            frame_detections.append(
                TrackedDetection(
                    frame_number=frame_number,
                    player_id=track_id,
                    class_name=class_name,
                    team_id=None,
                    team_assignment_confidence=None,
                    x=x, y=y, width=w, height=h,
                    confidence=conf_score,
                )
            )

        yield frame_detections


def track_detections(
    frames: list[dict],
    classes: list[str] | None = None,
) -> list[list[TrackedDetection]]:
    """
    Re-track an already-computed per-frame detection list using our own
    ByteTrack implementation (bytetrack.py -- adapted from the uploaded
    reference implementation, with two bugfixes: a missing frame_id
    assignment in STrack.activate() that crashed on any track seen for
    only a single frame, and an always-True is_activated flag that skipped
    ByteTrack's one-frame confirmation delay for brand-new tracks. See
    bytetrack.py's inline BUGFIX 1 / BUGFIX 2 comments for details.)

    Args:
        frames: same shape as detections.json from
            `ai/computer_vision/player detection/test.py`:
            [{"frame": int, "detections": [{"class": str, "confidence": float,
              "bbox": [x1, y1, x2, y2]}, ...]}, ...]
        classes: optional whitelist of class names to keep.

    Returns:
        list of list[TrackedDetection], same length and frame order as
        `frames`.
    """
    import os
    import sys

    import numpy as np

    # FIXED (same bug class found in trajectory.py and manual_calibration.py
    # -- see their fix comments): this bare `from bytetrack import ...` only
    # resolves when player_tracking/ is already on sys.path. Currently dormant
    # in production (run_pipeline() only calls track_video(), never
    # track_detections()), but fixing it here too so this doesn't become the
    # next silent landmine if something starts calling this function via its
    # full package path.
    _PLAYER_TRACKING_DIR = os.path.dirname(os.path.abspath(__file__))
    if _PLAYER_TRACKING_DIR not in sys.path:
        sys.path.insert(0, _PLAYER_TRACKING_DIR)

    from bytetrack import BYTETracker, STrack

    # Fixed class-name <-> id mapping for the WHOLE call, not recomputed per
    # frame. The previous supervision-based version did
    # `sorted(set(class_names))` inside the per-frame loop, which is only
    # self-consistent because class_id round-trips through the same dict
    # within that one frame -- but if frame 3 has only {"player","ball"}
    # while frame 4 has {"player","referee"}, "player" would silently get
    # a different numeric id in each frame. Doesn't break tracking itself
    # (IoU distance doesn't use class_id), but is a correctness footgun for
    # anything downstream that reads class_id directly instead of the
    # class_name we already reattach below.
    class_name_order = TRACKABLE_CLASS_NAMES | {BALL_CLASS_NAME, "referee"}
    class_name_order = sorted(class_name_order)
    name_to_id = {name: i for i, name in enumerate(class_name_order)}
    id_to_name = {i: name for name, i in name_to_id.items()}

    byte_tracker = BYTETracker(track_thresh=0.5, track_buffer=30, match_thresh=0.8)
    byte_tracker.reset()  # fresh player_id counter for this call, see reset_id_counter()

    output: list[list[TrackedDetection]] = []

    for frame_entry in frames:
        frame_number = frame_entry["frame"]
        raw_detections = frame_entry["detections"]

        if classes is not None:
            raw_detections = [d for d in raw_detections if d["class"] in classes]

        # bytetrack.py's BYTETracker.update() expects an (N, 6) array of
        # [x1, y1, x2, y2, score, class_id] -- see BYTETracker.update()'s
        # docstring-equivalent unpacking (`d[0]..d[5]`) in bytetrack.py.
        if raw_detections:
            output_results = np.array(
                [
                    [*d["bbox"], d["confidence"], name_to_id.get(d["class"], -1)]
                    for d in raw_detections
                    if d["class"] in name_to_id
                ],
                dtype=np.float64,
            )
        else:
            output_results = np.empty((0, 6), dtype=np.float64)

        tracked_stracks = byte_tracker.update(output_results)

        frame_detections: list[TrackedDetection] = []
        for strack in tracked_stracks:
            x1, y1, x2, y2 = strack.tlbr
            x, y, w, h = _xyxy_to_xywh(x1, y1, x2, y2)
            frame_detections.append(
                TrackedDetection(
                    frame_number=frame_number,
                    player_id=strack.track_id,
                    class_name=id_to_name.get(strack.class_id, "unknown"),
                    team_id=None,
                    team_assignment_confidence=None,
                    x=float(x), y=float(y), width=float(w), height=float(h),
                    confidence=float(strack.score),
                )
            )
        output.append(frame_detections)

    return output
