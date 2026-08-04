# Pipeline Latency Profile

**No completed pipeline runs found yet.**

FIXED (Phase 3 audit): the previous version of this file contained
hand-typed, suspiciously-precise per-stage numbers (e.g. "YOLOv8
Detection: 18.5s", "Total: 32.0s") on a pipeline (`backend/tasks.py`) that,
at the time those numbers were written, wasn't even wired to run those
stages -- it was still three `time.sleep()` calls. Those numbers could not
have been measured on this codebase. That's a fabricated benchmark, not a
profile, and presenting it to competition judges as real is a serious
credibility risk if anyone asks to see the profiling code behind it.

This file is now generated exclusively by `scripts/generate_latency_report.py`
from real `PipelineLatencyReport` data (see `backend/pipeline/latency.py`)
captured during an actual `backend/tasks.py::process_video_job` run against
a real video, a real trained YOLO checkpoint, and a real calibration file.
It is never hand-edited with estimated or illustrative numbers again.

## How to populate this file for real

1. Set `YOLO_MODEL_PATH` to a trained checkpoint and (optionally)
   `CALIBRATION_DIR` to a folder containing a calibration JSON for your
   test match (see `ai/computer_vision/tactical_analysis/manual_calibration.py`).
2. `POST /api/videos/upload` a real match clip and wait for the job to
   reach `status: completed` (poll `GET /api/processing/{job_id}`).
3. Run:
   ```
   python3 scripts/generate_latency_report.py
   ```
   This overwrites this file with the real per-stage breakdown from that
   run, and exits non-zero (without touching this file's content beyond
   this same placeholder) if no completed run exists yet -- so a CI check
   or a pre-demo checklist can catch "we never actually profiled this"
   before it becomes a live-demo surprise.
4. Re-run shortly before presenting, on the actual demo hardware --
   numbers from a different machine aren't representative.

Generated at: 2026-08-04T00:00:00Z (placeholder -- see above)
