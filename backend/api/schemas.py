from datetime import datetime
from typing import Any, Literal
from pydantic import BaseModel, Field

class VideoUploadResponse(BaseModel):
    video_id: str
    job_id: str
    match_id: str
    filename: str
    status: str
    message: str

class ProcessingStatusResponse(BaseModel):
    job_id: str
    video_id: str
    match_id: str | None = None
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

class PlayerMetricResponse(BaseModel):
    metric_id: str
    match_id: str
    player_id: int
    metric_name: str
    value: float | None = None
    method: str
    confidence: str
    sample_size: int
    sub_scores: dict[str, Any]
    computed_at: datetime
    schema_version: str

class TeamMetricResponse(BaseModel):
    metric_id: str
    match_id: str
    team_id: str
    metric_name: str
    value: Any = None
    method: str
    confidence: str
    confidence_score: float | None = None
    sample_size: int
    sub_scores: dict[str, Any]
    computed_at: datetime
    schema_version: str

class PlayerIntelligenceResponse(BaseModel):
    player_id: int
    player_name: str | None = None
    metrics: list[PlayerMetricResponse]

class TrackedPlayerPoint(BaseModel):
    player_id: int
    team_id: str | None = None
    pixel_x: float
    pixel_y: float
    pitch_x_m: float | None = None
    pitch_y_m: float | None = None

class TrackedBallPoint(BaseModel):
    pixel_x: float
    pixel_y: float

class TrackingFrame(BaseModel):
    frame_number: int
    timestamp: float
    players: list[TrackedPlayerPoint]
    ball: TrackedBallPoint | None = None

class TrackingWindowResponse(BaseModel):
    match_id: str
    start_frame: int
    end_frame: int
    fps: float
    frames: list[TrackingFrame]

class HeatmapCell(BaseModel):
    grid_x: int
    grid_y: int
    count: int
    density: float  # count / max_count in this grid, 0..1

class HeatmapResponse(BaseModel):
    match_id: str
    player_id: int
    grid_cols: int
    grid_rows: int
    pitch_length_m: float
    pitch_width_m: float
    cells: list[HeatmapCell]
    confidence: str
    sample_size: int
    usable_sample_size: int

class PipelineStageTiming(BaseModel):
    """One row of a real, measured pipeline run -- see backend/pipeline/latency.py.
    Never hand-typed: every instance of this model in the system is produced
    by an actual PipelineTimer.stage() context manager wrapping real work."""
    stage: str
    seconds: float
    detail: str | None = None

class PipelineLatencyReport(BaseModel):
    job_id: str
    match_id: str | None = None
    total_seconds: float
    stages: list[PipelineStageTiming]
    generated_at: datetime
    source: Literal["measured"] = "measured"
