"""
Extract a frame range from a video into a new, shorter video file.
Uses only OpenCV (which you already have installed) -- no ffmpeg needed.

Usage:
    python extract_clip.py --video 19.mp4 --start 6181 --end 6551 --out clip1.mp4

Defaults below are pre-filled for your case (19.mp4, frames 6181-6551, the
longest clean continuous segment found in the tracking JSON analysis).
Just run it with no arguments, or edit the defaults / pass --start/--end
to grab a different segment.
"""
from __future__ import annotations
 
import argparse
import os
import sys
 
import cv2
 
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default=r"D:\SportsStrategyCoachAI\SportsStrategyCoachAI\test.mp4",
                         help="Path to the source video")
    # FIXED: old defaults (6181 / 6551) were the frame range identified for
    # 19.mp4's fragmented footage. test.mp4's own fragmentation analysis
    # (analyze_fragmentation.py + sliding-window scan) found its best clean
    # window at frames 270-570 (10.1s, 0% empty frames, ~17.6 avg
    # players/frame) -- that is the correct default for THIS video.
    parser.add_argument("--start", type=int, default=270, help="Start frame (inclusive)")
    parser.add_argument("--end", type=int, default=570, help="End frame (exclusive)")
    parser.add_argument("--out", default=None,
                         help="Output path. Default: <video_name>_clip_<start>_<end>.mp4 next to the source video")
    parser.add_argument("--still-frame-offset", type=int, default=0,
                         help="Which frame within [start, end) to also save as a .jpg, "
                              "for manual_calibration.py's headless mode (default: first frame of range)")
    args = parser.parse_args()
 
    if not os.path.exists(args.video):
        print(f"ERROR: video not found: {args.video}")
        sys.exit(1)
 
    if args.out is None:
        video_stem = os.path.splitext(os.path.basename(args.video))[0]
        out_dir = os.path.dirname(os.path.abspath(args.video))
        args.out = os.path.join(out_dir, f"{video_stem}_clip_{args.start}_{args.end}.mp4")
 
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"ERROR: could not open video: {args.video}")
        sys.exit(1)
 
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    reported_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
 
    # NOTE: cv2.CAP_PROP_FRAME_COUNT reads container metadata, which can be
    # WRONG (under- or over-counted) for variable-frame-rate or re-encoded
    # mp4 files -- this has been observed to disagree with the actual
    # number of frames ultralytics can decode from the same file. So this
    # is printed for information only and is NOT used to validate --end,
    # and frames are read sequentially below (not via cap.set seeking,
    # which relies on the same unreliable index).
    print(f"Source: {args.video}  ({width}x{height} @ {fps:.1f}fps, "
          f"container reports {reported_total} frames -- may be inaccurate, see note in script)")
 
    if args.start < 0 or args.start >= args.end:
        print(f"ERROR: invalid range --start {args.start} --end {args.end}")
        sys.exit(1)
 
    print(f"Extracting frames {args.start} to {args.end} ({args.end - args.start} frames, "
          f"~{(args.end - args.start) / fps:.1f}s)")
 
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.out, fourcc, fps, (width, height))
 
    still_frame_index = args.start + args.still_frame_offset
    still_path = os.path.splitext(args.out)[0] + ".jpg"
    saved_still = False
 
    # Sequential read from frame 0, skipping until --start, instead of
    # cap.set(CAP_PROP_POS_FRAMES, start) -- seeking can land on the wrong
    # frame (or fail silently) on files with inconsistent internal
    # indexing, which is exactly the kind of file this metadata mismatch
    # points to.
    frame_index = 0
    frame_count = 0
    while frame_index < args.end:
        ok, frame = cap.read()
        if not ok:
            if frame_index < args.start:
                print(f"ERROR: video ended at frame {frame_index}, before reaching "
                      f"--start {args.start}. This clip range doesn't exist in the file.")
                sys.exit(1)
            print(f"NOTE: video ended early at frame {frame_index} "
                  f"(wanted to go to {args.end})")
            break
        if frame_index >= args.start:
            writer.write(frame)
            frame_count += 1
            if frame_index == still_frame_index and not saved_still:
                cv2.imwrite(still_path, frame)
                saved_still = True
        frame_index += 1
 
    cap.release()
    writer.release()
 
    print(f"\nDone. Wrote {frame_count} frames to: {args.out}")
    if saved_still:
        print(f"Representative still frame saved to: {still_path}  <- upload this for calibration")
    else:
        print("WARNING: still frame was not saved (range may have ended before the offset frame).")
 
 
if __name__ == "__main__":
    main()