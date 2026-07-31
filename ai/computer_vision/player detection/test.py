from __future__ import annotations

import argparse
import json
import os
import sys
import time

import cv2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=r"D:\SportsStrategyCoachAI\runs\runs\player_ball_v1-4\weights\best.pt", help="Path to best.pt")
    parser.add_argument("--video", default=r"D:\SportsStrategyCoachAI\SportsStrategyCoachAI\test.mp4", help="Path to match video")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"ERROR: model not found: {args.model}")
        sys.exit(1)
    if not os.path.exists(args.video):
        print(f"ERROR: video not found: {args.video}")
        sys.exit(1)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("Missing dependency. Run: pip install ultralytics --break-system-packages")
        sys.exit(1)

    video_stem = os.path.splitext(os.path.basename(args.video))[0]
    out_dir = os.path.dirname(os.path.abspath(args.video))
    out_video_path = os.path.join(out_dir, f"{video_stem}_quicktest.mp4")
    out_json_path = os.path.join(out_dir, f"{video_stem}_quicktest.json")

    print(f"Loading model: {args.model}")
    model = YOLO(args.model)
    class_names = model.names
    print(f"Classes: {class_names}")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"ERROR: could not open video: {args.video}")
        sys.exit(1)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {width}x{height} @ {fps:.1f}fps, ~{total_frames} frames\n")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))

    colors = {
        "player": (255, 140, 0),
        "goalkeeper": (0, 215, 255),
        "referee": (0, 255, 255),
        "ball": (255, 255, 255),
    }

    results_stream = model.track(
        source=args.video,
        tracker="bytetrack.yaml",
        conf=args.conf,
        persist=True,
        stream=True,
        verbose=False,
    )

    json_output = []
    class_counts: dict[str, int] = {}
    unique_ids_by_class: dict[str, set] = {}

    start_time = time.time()
    frame_number = 0

    for result in results_stream:
        frame = result.orig_img.copy()
        frame_dets = []

        if result.boxes is not None and result.boxes.id is not None:
            boxes = result.boxes
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i])
                cname = class_names[cls_id]
                track_id = int(boxes.id[i])
                conf_score = float(boxes.conf[i])
                x1, y1, x2, y2 = [int(v) for v in boxes.xyxy[i].tolist()]

                class_counts[cname] = class_counts.get(cname, 0) + 1
                unique_ids_by_class.setdefault(cname, set()).add(track_id)

                frame_dets.append({
                    "class": cname, "track_id": track_id,
                    "confidence": round(conf_score, 3),
                    "bbox": [x1, y1, x2, y2],
                })

                color = colors.get(cname, (200, 200, 200))
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"#{track_id} {cname} {conf_score:.2f}"
                (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x1, y1 - lh - 8), (x1 + lw + 4, y1), color, -1)
                cv2.putText(frame, label, (x1 + 2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        json_output.append({"frame": frame_number, "detections": frame_dets})
        writer.write(frame)

        frame_number += 1
        if frame_number % 50 == 0:
            print(f"  ...processed {frame_number} frames")
        if args.max_frames is not None and frame_number >= args.max_frames:
            break

    cap.release()
    writer.release()
    elapsed = time.time() - start_time

    with open(out_json_path, "w") as f:
        json.dump(json_output, f, indent=2)

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Frames processed: {frame_number}  ({elapsed:.1f}s, {frame_number/max(elapsed,1e-6):.1f} fps)")
    print()
    print("Detections per class (total across all frames):")
    for cname, count in sorted(class_counts.items()):
        n_ids = len(unique_ids_by_class.get(cname, set()))
        print(f"  {cname:12s}  {count:6d} detections   {n_ids:3d} distinct track IDs")
        if cname in ("player", "goalkeeper") and n_ids > 30:
            print(f"    ^ heads up: {n_ids} distinct IDs is a lot for ~22 players+subs, "
                  f"probably means the tracker is losing and re-creating IDs, not that many people appeared")
    print()
    print(f"Annotated video: {out_video_path}")
    print(f"JSON output:     {out_json_path}")


if __name__ == "__main__":
    main()
