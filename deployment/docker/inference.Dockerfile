# GPU inference worker for the CV pipeline (backend/tasks.py::process_video_job).
#
# This is a SEPARATE image from backend/Dockerfile. That one serves the
# FastAPI app and is deliberately CPU-only/lightweight (no torch,
# ultralytics, or mediapipe). This one runs ONLY the Celery worker that
# consumes jobs backend/api/main.py::upload_video() enqueues -- it needs a
# real GPU + NVIDIA drivers on the host to be worth running at all. See
# deployment/docs.md for the one step this file can't automate: standing
# up that host.
#
# Build from the repo root (paths below assume that build context, same
# convention as backend/Dockerfile):
#   docker build -f deployment/docker/inference.Dockerfile -t sports-strategy-inference .

# Base image ships CUDA 12.1 + cuDNN 8 + a matching torch/torchvision
# build already installed. Deliberately not "pip install torch" from
# inference-requirements.txt -- that would risk silently resolving a
# CPU-only wheel instead of one built against this image's CUDA. If you
# change the CUDA version here, confirm the GPU host's NVIDIA driver
# supports it first (see deployment/docs.md's driver compatibility note).
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

WORKDIR /app

# libgl1/libglib2.0-0: opencv-python-headless still dlopens libGL at
# import time on Debian-based images even though it never opens a window
# -- without these, `import cv2` fails at container startup.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt /app/backend/requirements.txt
COPY deployment/docker/inference-requirements.txt /app/deployment/docker/inference-requirements.txt
RUN pip install --no-cache-dir -r /app/deployment/docker/inference-requirements.txt

# ai/computer_vision/pose_estimation/pose.py's own docstring flags that
# mediapipe releases vary on whether mediapipe.solutions.pose still exists
# (legacy Solutions API vs newer Tasks API, unpinned mediapipe is not
# guaranteed to give you one that has it). Fail the build loudly here
# instead of shipping an image where pose estimation -- Stage 4 of every
# pipeline run, not optional -- breaks on the first real job.
RUN python -c "\
import mediapipe as mp; \
assert hasattr(mp.solutions, 'pose'), \
    'mediapipe.solutions.pose is missing from this mediapipe build -- pin a ' \
    'different mediapipe version in inference-requirements.txt (see ' \
    'ai/computer_vision/pose_estimation/pose.py _load_legacy_pose_solution()).'"

COPY backend /app/backend
COPY ai /app/ai

ENV PYTHONPATH=/app

# --pool=solo: CUDA contexts don't survive fork() cleanly, which is what
# Celery's default 'prefork' pool does per task -- solo runs one task at a
# time in-process instead, avoiding that failure mode entirely. This also
# means the worker is intentionally single-concurrency: one video's GPU
# inference at a time, matching a single-GPU demo host (see
# deployment/docs.md). Scale by running more instances of this container
# against more GPUs, not by raising concurrency on one.
CMD ["celery", "-A", "backend.celery_app:celery_app", "worker", "--loglevel=info", "--pool=solo"]
