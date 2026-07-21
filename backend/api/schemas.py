from datetime import datetime
from typing import Literal
from typing import Any

from pydantic import BaseModel, Field


class VideoUploadResponse(BaseModel):
    video_id: str
    job_id: str
    filename: str
    status: str
    message: str


class ProcessingStatusResponse(BaseModel):
    job_id: str
    video_id: str
    status: str
    progress: int
    message: str | None = None
    error: str | None = None
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None


class JobStatusUpdate(BaseModel):
    status: Literal["queued", "processing", "completed", "failed"]
    progress: int = Field(default=0, ge=0, le=100)
    message: str | None = None
    error: str | None = None


class AnalysisResultCreate(BaseModel):
    result: dict[str, Any]
    summary: str | None = None


class AnalysisResultResponse(BaseModel):
    job_id: str
    video_id: str
    status: str
    summary: str | None = None
    result: dict[str, Any] | None = None
