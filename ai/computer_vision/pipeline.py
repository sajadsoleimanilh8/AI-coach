"""
Unified Phase 2 Video Pipeline Runner.
Runs YOLOv8/v11 detection + ByteTrack tracking + Pitch Homography calibration
and outputs structured JSON suitable for the backend AnalysisResult table.
"""
from pathlib import Path
import cv2
import numpy as np

from computer_vision.player_tracking.tracker import BYTETracker
from computer_vision.player_tracking.trajectory import TrackingPoint, enrich_with_pitch_coordinates
from computer_vision.tactical_analysis.homography import pixels_to_pitch, HomographyResult
from computer_vision.tactical_analysis.manual_calibration import load_calibration


def run_cv_pipeline(
    video_path: str,
    calibration_path: str | None = None,
    model_path: str = "yolov8n.pt",
    max_frames: int | None = None,
    progress_callback=None,
) -> dict:
    """
    Executes detection, tracking, and homography projection on a video file.
    Returns analysis summary and spatial coordinates for heatmaps.
    """
    video_file = Path(video_path)
    if not video_file.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # 1. Load Calibration (if available)
    H = None
    homography_conf = 0.0
    if calibration_path and Path(calibration_path).exists():
        try:
            H, record = load_calibration(calibration_path)
            homography_conf = record.get("confidence", 0.0)
        except Exception:
            H = None

    # 2. Load Models & Trackers
    try:
        from ultralytics import YOLO
        model = YOLO(model_path)
    except ImportError:
        model = None  # Fallback to dummy mode if ultralytics not installed in env

    tracker = BYTETracker(track_thresh=0.5, track_buffer=30)
    tracker.reset()

    cap = cv2.VideoCapture(str(video_file))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 100

    frame_idx = 0
    all_raw_frames = []
    heatmap_pitch_points = []  # [(pitch_x, pitch_y), ...]

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if max_frames and frame_idx > max_frames:
            break

        # A. Detect
        dets = []
        if model is not None:
            results = model(frame, verbose=False)[0]
            if hasattr(results, "boxes") and len(results.boxes) > 0:
                for box in results.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    score = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    if cls_id == 0:  # Class 0: Person / Player
                        dets.append([x1, y1, x2, y2, score, cls_id])

        dets_arr = np.array(dets) if len(dets) > 0 else np.empty((0, 6))

        # B. Track (ByteTrack)
        tracked_stracks = tracker.update(dets_arr)

        frame_detections = []
        for strack in tracked_stracks:
            x1, y1, x2, y2 = strack.tlbr
            foot_x = (x1 + x2) / 2.0
            foot_y = y2  # Bottom-center (feet)

            # C. Map to Pitch (if Homography H is available)
            pitch_x, pitch_y = None, None
            if H is not None and homography_conf >= 0.6:
                px_pitch, py_pitch = pixels_to_pitch(np.array([[foot_x, foot_y]]), H)[0]
                pitch_x, pitch_y = float(px_pitch), float(py_pitch)
                if 0.0 <= pitch_x <= 105.0 and 0.0 <= pitch_y <= 68.0:
                    heatmap_pitch_points.append({"x": round(pitch_x, 2), "y": round(pitch_y, 2)})

            frame_detections.append({
                "track_id": strack.track_id,
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "confidence": float(strack.score),
                "pitch_x_m": pitch_x,
                "pitch_y_m": pitch_y,
            })

        all_raw_frames.append({"frame": frame_idx, "detections": frame_detections})

        # Update progress callback every 15 frames
        if progress_callback and frame_idx % 15 == 0:
            progress_pct = int((frame_idx / total_frames) * 100)
            progress_callback(min(progress_pct, 95), f"Processing frame {frame_idx}/{total_frames}...")

    cap.release()

    unique_track_ids = set(
        d["track_id"] for f in all_raw_frames for d in f["detections"]
    )

    return {
        "frames_processed": frame_idx,
        "fps": fps,
        "total_tracks": len(unique_track_ids),
        "homography_confidence": homography_conf,
        "heatmap_points_count": len(heatmap_pitch_points),
        "heatmap_sample": heatmap_pitch_points[:: max(1, len(heatmap_pitch_points) // 500)],  # Max 500 points for UI render
        "tracking_summary": {
            "unique_players": len(unique_track_ids),
            "total_detections": sum(len(f["detections"]) for f in all_raw_frames),
        },
    }