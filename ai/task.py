import os
from datetime import datetime
from pathlib import Path

from backend.celery_app import celery_app
from backend.database.models import AnalysisResult, ProcessingJob, ProcessingStatus, Video
from backend.database.session import SessionLocal

# Import Computer Vision Pipeline Runner
from ai.computer_vision.pipeline import run_cv_pipeline


@celery_app.task(name="backend.tasks.process_video_job", bind=True)
def process_video_job(self, job_id: str):
    """
    Async entry point for video processing.
    Executes YOLOv8 detection + ByteTrack player tracking + Homography pitch projection.
    """
    db = SessionLocal()
    try:
        job = db.get(ProcessingJob, job_id)
        if job is None:
            return {"job_id": job_id, "status": "failed", "error": "Job not found"}

        video = db.get(Video, job.video_id)
        if video is None or not Path(video.storage_path).exists():
            raise FileNotFoundError(f"Video file missing at path: {video.storage_path if video else 'None'}")

        # 1. Update job status to Processing
        job.status = ProcessingStatus.processing
        job.progress = 5
        job.message = "Initializing Computer Vision models & worker stream..."
        job.started_at = datetime.utcnow()
        job.updated_at = datetime.utcnow()
        db.commit()

        # Progress Callback Function
        def update_progress(pct: int, msg: str):
            job.progress = pct
            job.message = msg
            job.updated_at = datetime.utcnow()
            db.commit()

        # 2. Run Computer Vision Pipeline
        pipeline_output = run_cv_pipeline(
            video_path=video.storage_path,
            calibration_path=None,  # Pass custom homography calibration path if available
            max_frames=300,         # Process up to 300 frames for quick demo execution
            progress_callback=update_progress,
        )

        # 3. Formulate Summary Text
        summary_text = (
            f"Phase 2 Complete: Processed {pipeline_output['frames_processed']} frames at {pipeline_output['fps']:.1f} FPS. "
            f"Identified {pipeline_output['total_tracks']} distinct tracked entities across pitch."
        )

        # 4. Save to Database AnalysisResult Table
        existing_result = db.query(AnalysisResult).filter(AnalysisResult.job_id == job_id).first()
        if existing_result:
            existing_result.result_json = pipeline_output
            existing_result.summary = summary_text
        else:
            db.add(AnalysisResult(job_id=job_id, result_json=pipeline_output, summary=summary_text))

        # 5. Mark Job Completed
        job.status = ProcessingStatus.completed
        job.progress = 100
        job.message = "Analysis complete. Player tracks & spatial heatmap generated."
        job.completed_at = datetime.utcnow()
        job.updated_at = datetime.utcnow()
        db.commit()

        return {
            "job_id": job_id,
            "status": "completed",
            "tracks": pipeline_output["total_tracks"],
            "heatmap_points": pipeline_output["heatmap_points_count"],
        }

    except Exception as error:
        job = db.get(ProcessingJob, job_id)
        if job is not None:
            job.status = ProcessingStatus.failed
            job.error = str(error)
            job.message = f"CV Pipeline execution failed: {str(error)}"
            job.updated_at = datetime.utcnow()
            job.completed_at = datetime.utcnow()
            db.commit()
        raise
    finally:
        db.close()