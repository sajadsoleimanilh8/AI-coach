import time
from datetime import datetime

from backend.celery_app import celery_app
from backend.database.models import ProcessingJob, ProcessingStatus
from backend.database.session import SessionLocal


@celery_app.task(name="backend.tasks.process_video_job", bind=True)
def process_video_job(self, job_id: str):
    """
    Async entry point for the video processing pipeline.

    Phase 1 scope: prove the queue mechanics end-to-end (job picked up by a
    real Celery worker, status/progress updated in Postgres, dashboard reads
    it back via GET /api/processing/{job_id}).

    Phase 3 TODO: replace the placeholder steps below with the real
    pipeline call — YOLO detection -> ByteTrack -> homography -> event
    detection -> PlayerMetric/TeamMetric writes (see
    ai/computer_vision/player_detection/phase0_1_pipeline.py and
    docs/data analysis.md for the formulas each step should feed).
    """
    db = SessionLocal()
    try:
        job = db.get(ProcessingJob, job_id)
        if job is None:
            return {"job_id": job_id, "status": "failed", "error": "job not found"}

        job.status = ProcessingStatus.processing
        job.progress = 10
        job.message = "Worker picked up job. Running detection pipeline..."
        job.started_at = datetime.utcnow()
        job.updated_at = datetime.utcnow()
        db.commit()

        # --- Placeholder pipeline steps (Phase 1 scaffold only) ---
        time.sleep(2)
        job.progress = 50
        job.message = "Running tracking + homography..."
        job.updated_at = datetime.utcnow()
        db.commit()

        time.sleep(2)
        job.progress = 85
        job.message = "Extracting player/team metrics..."
        job.updated_at = datetime.utcnow()
        db.commit()
        # --- End placeholder steps ---

        job.status = ProcessingStatus.completed
        job.progress = 100
        job.message = "Analysis complete."
        job.completed_at = datetime.utcnow()
        job.updated_at = datetime.utcnow()
        db.commit()

        return {"job_id": job_id, "status": "completed"}

    except Exception as error:
        job = db.get(ProcessingJob, job_id)
        if job is not None:
            job.status = ProcessingStatus.failed
            job.error = str(error)
            job.updated_at = datetime.utcnow()
            job.completed_at = datetime.utcnow()
            db.commit()
        raise

    finally:
        db.close()