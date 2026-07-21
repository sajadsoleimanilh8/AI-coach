import shutil
import json
from datetime import datetime
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from backend.api.schemas import (
    AnalysisResultCreate,
    AnalysisResultResponse,
    JobStatusUpdate,
    ProcessingStatusResponse,
    VideoUploadResponse,
)
from backend.database.models import AnalysisResult, ProcessingJob, ProcessingStatus, Video, new_id
from backend.database.session import Base, engine, get_db


PROJECT_ROOT = Path(__file__).resolve().parents[2]
UPLOAD_DIR = PROJECT_ROOT / "storage" / "uploads"
ALLOWED_VIDEO_TYPES = {
    "video/mp4",
    "video/mpeg",
    "video/quicktime",
    "video/x-msvideo",
    "application/octet-stream",
}

app = FastAPI(
    title="SportsStrategyCoachAI Backend",
    version="0.1.0",
    description="Video upload, processing status, and analysis JSON API.",
)


@app.on_event("startup")
def startup():
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(bind=engine)


@app.get("/health")
def health():
    return {"status": "ok", "service": "sports-strategy-coach-ai"}


@app.post("/api/videos/upload", response_model=VideoUploadResponse, status_code=status.HTTP_201_CREATED)
def upload_video(
    file: Annotated[UploadFile, File()],
    metadata: Annotated[str | None, Form()] = None,
    db: Session = Depends(get_db),
):
    if file.content_type not in ALLOWED_VIDEO_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type: {file.content_type}",
        )

    metadata_json = None
    if metadata:
        try:
            metadata_json = json.loads(metadata)
        except json.JSONDecodeError as error:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"metadata must be valid JSON: {error.msg}",
            ) from error

    file_id = new_id()
    extension = Path(file.filename or "video.mp4").suffix or ".mp4"
    stored_filename = f"{file_id}{extension.lower()}"
    storage_path = UPLOAD_DIR / stored_filename

    with storage_path.open("wb") as output:
        shutil.copyfileobj(file.file, output)

    file_size = storage_path.stat().st_size
    video = Video(
        id=file_id,
        original_filename=file.filename or stored_filename,
        stored_filename=stored_filename,
        content_type=file.content_type,
        file_size=file_size,
        storage_path=str(storage_path),
        metadata_json=metadata_json,
    )
    job = ProcessingJob(
        video_id=video.id,
        status=ProcessingStatus.queued,
        progress=0,
        message="Video uploaded and queued for processing.",
    )

    db.add(video)
    db.add(job)
    db.commit()
    db.refresh(job)

    return VideoUploadResponse(
        video_id=video.id,
        job_id=job.id,
        filename=video.original_filename,
        status=job.status.value,
        message=job.message or "Video uploaded.",
    )


@app.get("/api/processing/{job_id}", response_model=ProcessingStatusResponse)
def get_processing_status(job_id: str, db: Session = Depends(get_db)):
    job = db.get(ProcessingJob, job_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Processing job not found")

    return ProcessingStatusResponse(
        job_id=job.id,
        video_id=job.video_id,
        status=job.status.value,
        progress=job.progress,
        message=job.message,
        error=job.error,
        created_at=job.created_at,
        updated_at=job.updated_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
    )


@app.patch("/api/processing/{job_id}", response_model=ProcessingStatusResponse)
def update_processing_status(job_id: str, payload: JobStatusUpdate, db: Session = Depends(get_db)):
    job = db.get(ProcessingJob, job_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Processing job not found")

    next_status = ProcessingStatus(payload.status)
    job.status = next_status
    job.progress = payload.progress
    job.message = payload.message
    job.error = payload.error
    job.updated_at = datetime.utcnow()

    if next_status == ProcessingStatus.processing and job.started_at is None:
        job.started_at = datetime.utcnow()
    if next_status in {ProcessingStatus.completed, ProcessingStatus.failed}:
        job.completed_at = datetime.utcnow()

    db.commit()
    db.refresh(job)

    return ProcessingStatusResponse(
        job_id=job.id,
        video_id=job.video_id,
        status=job.status.value,
        progress=job.progress,
        message=job.message,
        error=job.error,
        created_at=job.created_at,
        updated_at=job.updated_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
    )


@app.post("/api/processing/{job_id}/result", response_model=AnalysisResultResponse)
def save_analysis_result(job_id: str, payload: AnalysisResultCreate, db: Session = Depends(get_db)):
    job = db.get(ProcessingJob, job_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Processing job not found")

    existing = db.query(AnalysisResult).filter(AnalysisResult.job_id == job_id).first()
    if existing:
        existing.result_json = payload.result
        existing.summary = payload.summary
    else:
        db.add(AnalysisResult(job_id=job_id, result_json=payload.result, summary=payload.summary))

    job.status = ProcessingStatus.completed
    job.progress = 100
    job.message = "Analysis result saved."
    job.error = None
    job.completed_at = datetime.utcnow()
    job.updated_at = datetime.utcnow()

    db.commit()
    db.refresh(job)

    return AnalysisResultResponse(
        job_id=job.id,
        video_id=job.video_id,
        status=job.status.value,
        summary=payload.summary,
        result=payload.result,
    )


@app.get("/api/processing/{job_id}/result", response_model=AnalysisResultResponse)
def get_analysis_result(job_id: str, db: Session = Depends(get_db)):
    job = db.get(ProcessingJob, job_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Processing job not found")

    result = db.query(AnalysisResult).filter(AnalysisResult.job_id == job_id).first()
    return AnalysisResultResponse(
        job_id=job.id,
        video_id=job.video_id,
        status=job.status.value,
        summary=result.summary if result else None,
        result=result.result_json if result else None,
    )
