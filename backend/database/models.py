import enum
import uuid
from datetime import datetime
from sqlalchemy import Column, DateTime, Enum, Float, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.orm import relationship
from backend.database.session import Base

def new_id() -> str:
    return str(uuid.uuid4())

class ProcessingStatus(str, enum.Enum):
    queued = "queued"
    processing = "processing"
    completed = "completed"
    failed = "failed"

class MetricMethod(str, enum.Enum):
    ml_trained = "ml_trained"
    deterministic = "deterministic"
    heuristic_proxy = "heuristic_proxy"

class MetricConfidence(str, enum.Enum):
    normal = "normal"
    low_sample = "low_sample"
    low_upstream_confidence = "low_upstream_confidence"

class Video(Base):
    __tablename__ = "videos"
    id = Column(String(36), primary_key=True, default=new_id)
    original_filename = Column(String(255), nullable=False)
    stored_filename = Column(String(255), nullable=False, unique=True)
    content_type = Column(String(120), nullable=True)
    file_size = Column(Integer, nullable=False)
    storage_path = Column(Text, nullable=False)
    metadata_json = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # FIXED (Phase 3 audit): previously there was no link at all between an
    # uploaded Video/ProcessingJob and the Match row its PlayerMetric/
    # TeamMetric/Event rows get written under. That made it impossible for
    # the pipeline (backend/tasks.py) or the dashboard (GET /api/*_intelligence
    # /{match_id}) to know which match a given upload corresponds to.
    # Nullable because the Match row is created immediately after the Video
    # row in the same upload request (see backend/api/main.py::upload_video)
    # -- there's a brief instant where the Video exists and the Match
    # doesn't yet, not because this link is meant to stay unset long-term.
    match_id = Column(String(36), ForeignKey("matches.match_id"), nullable=True, index=True)

    jobs = relationship("ProcessingJob", back_populates="video", cascade="all, delete-orphan")
    match = relationship("Match", back_populates="videos")

class ProcessingJob(Base):
    __tablename__ = "processing_jobs"
    id = Column(String(36), primary_key=True, default=new_id)
    video_id = Column(String(36), ForeignKey("videos.id"), nullable=False, index=True)
    status = Column(Enum(ProcessingStatus), default=ProcessingStatus.queued, nullable=False)
    progress = Column(Integer, default=0, nullable=False)
    message = Column(Text, nullable=True)
    error = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    video = relationship("Video", back_populates="jobs")
    result = relationship("AnalysisResult", back_populates="job", uselist=False, cascade="all, delete-orphan")

class AnalysisResult(Base):
    __tablename__ = "analysis_results"
    id = Column(String(36), primary_key=True, default=new_id)
    job_id = Column(String(36), ForeignKey("processing_jobs.id"), nullable=False, unique=True, index=True)
    result_json = Column(JSON, nullable=False)
    summary = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    job = relationship("ProcessingJob", back_populates="result")

class Match(Base):
    """General match information. Root of the football analysis schema."""
    __tablename__ = "matches"
    match_id = Column(String(36), primary_key=True, default=new_id)
    home_team = Column(String(255), nullable=False)
    away_team = Column(String(255), nullable=False)
    video_path = Column(Text, nullable=False)
    duration = Column(Float, nullable=False)  # seconds
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    frames = relationship("Frame", back_populates="match", cascade="all, delete-orphan")
    player_trackings = relationship("PlayerTracking", back_populates="match", cascade="all, delete-orphan")
    events = relationship("Event", back_populates="match", cascade="all, delete-orphan")
    player_metrics = relationship("PlayerMetric", back_populates="match", cascade="all, delete-orphan")
    team_metrics = relationship("TeamMetric", back_populates="match", cascade="all, delete-orphan")
    videos = relationship("Video", back_populates="match")

    def __repr__(self) -> str:
        return f"<Match id={self.match_id} {self.home_team} vs {self.away_team}>"

class Frame(Base):
    """A processed video frame belonging to a match."""
    __tablename__ = "frames"
    frame_id = Column(Integer, primary_key=True, autoincrement=True)
    match_id = Column(String(36), ForeignKey("matches.match_id"), nullable=False, index=True)
    frame_number = Column(Integer, nullable=False)
    timestamp = Column(Float, nullable=False)
    fps = Column(Float, nullable=False)
    match = relationship("Match", back_populates="frames")

class PlayerDetection(Base):
    __tablename__ = "player_detections"
    detection_id = Column(String(36), primary_key=True, default=new_id)
    frame_id = Column(Integer, ForeignKey("frames.frame_id"), nullable=False, index=True)
    player_id = Column(Integer, nullable=False)
    team_id = Column(String(64), nullable=True)
    team_assignment_confidence = Column(Float, nullable=True)
    x = Column(Float, nullable=False)
    y = Column(Float, nullable=False)
    width = Column(Float, nullable=False)
    height = Column(Float, nullable=False)
    confidence = Column(Float, nullable=False)

class BallDetection(Base):
    __tablename__ = "ball_detections"
    detection_id = Column(String(36), primary_key=True, default=new_id)
    frame_id = Column(Integer, ForeignKey("frames.frame_id"), nullable=False, index=True)
    ball_x = Column(Float, nullable=False)
    ball_y = Column(Float, nullable=False)
    confidence = Column(Float, nullable=False)

class PlayerTracking(Base):
    __tablename__ = "player_tracking"
    tracking_id = Column(String(36), primary_key=True, default=new_id)
    match_id = Column(String(36), ForeignKey("matches.match_id"), nullable=False, index=True)
    player_id = Column(Integer, nullable=False)

    # FIXED (found via a real pipeline run on ~35k tracking rows): this was
    # declared as ForeignKey("frames.frame_id") with nullable=False, but
    # `frames` rows are only persisted for a FRAME_PERSIST_STRIDE-sampled
    # subset (see backend/pipeline/runner.py), with an autoincrement PK
    # unrelated to the actual video frame number -- while every
    # TrackingPoint naturally carries the real frame *number* (0, 1, 2, ...),
    # not a Frame.frame_id row reference. These two ID spaces don't
    # correspond, so this could never be populated as a real FK. The
    # previous code's workaround (writing None here) hit the nullable=False
    # constraint the first time bulk_save_objects() ran on real data,
    # crashing the pipeline outright. Nothing elsewhere in the codebase
    # joins on this column as a real FK (checked), so dropping the FK
    # constraint and storing the real frame number is the correct fix --
    # not a fake reference to a table row that doesn't exist for most frames.
    frame_id = Column(Integer, nullable=False, index=True)  # actual video frame number, NOT a frames.frame_id FK
    team_id = Column(String(64), nullable=True)
    pixel_x = Column(Float, nullable=False)
    pixel_y = Column(Float, nullable=False)
    pitch_x_m = Column(Float, nullable=True)
    pitch_y_m = Column(Float, nullable=True)
    homography_confidence = Column(Float, nullable=True)
    speed = Column(Float, nullable=True)
    distance = Column(Float, nullable=True)
    acceleration = Column(Float, nullable=True)

    # Added for Phase 2 Day 7 (pose / body orientation). Nullable for the
    # same reason pitch_x_m/homography_confidence are nullable: a frame
    # can legitimately have no reading (pose not sampled this frame --
    # see POSE_SAMPLE_STRIDE in backend/pipeline/runner.py -- or MediaPipe
    # found no visible shoulder landmarks). None means "not measured",
    # never silently defaulted to 0.0 -- a 0deg orientation is a real,
    # different fact from "we don't know". Downstream consumers must
    # check body_orientation_confidence before trusting the angle, same
    # pattern as homography_confidence gating pitch_x_m/pitch_y_m.
    body_orientation_deg = Column(Float, nullable=True)         # shoulder-line angle, 0-360
    body_orientation_confidence = Column(Float, nullable=True)  # min visibility of the two shoulder landmarks, 0-1

    match = relationship("Match", back_populates="player_trackings")

class Event(Base):
    __tablename__ = "events"
    event_id = Column(String(36), primary_key=True, default=new_id)
    match_id = Column(String(36), ForeignKey("matches.match_id"), nullable=False, index=True)
    frame_id = Column(Integer, ForeignKey("frames.frame_id"), nullable=True, index=True)
    event_type = Column(String(64), nullable=False)
    player_id = Column(Integer, nullable=True)
    related_player_id = Column(Integer, nullable=True)
    team_id = Column(String(64), nullable=True)
    pitch_x_m = Column(Float, nullable=True)
    pitch_y_m = Column(Float, nullable=True)
    homography_confidence = Column(Float, nullable=True)
    timestamp = Column(Float, nullable=False)
    metadata_json = Column("metadata", JSON, nullable=True)
    match = relationship("Match", back_populates="events")

class PlayerMetric(Base):
    __tablename__ = "player_metrics"
    metric_id = Column(String(36), primary_key=True, default=new_id)
    match_id = Column(String(36), ForeignKey("matches.match_id"), nullable=False, index=True)
    player_id = Column(Integer, nullable=False)
    metric_name = Column(String(120), nullable=False)
    value = Column(Float, nullable=True)
    method = Column(Enum(MetricMethod), nullable=False)
    confidence = Column(Enum(MetricConfidence), nullable=False)
    sample_size = Column(Integer, nullable=False)
    sub_scores = Column(JSON, nullable=False)
    computed_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    schema_version = Column(String(16), nullable=False)
    match = relationship("Match", back_populates="player_metrics")

class TeamMetric(Base):
    __tablename__ = "team_metrics"
    metric_id = Column(String(36), primary_key=True, default=new_id)
    match_id = Column(String(36), ForeignKey("matches.match_id"), nullable=False, index=True)
    team_id = Column(String(64), nullable=False)
    metric_name = Column(String(120), nullable=False)

    # FIXED (Phase 3 audit): TeamMetric.value was a single Float column,
    # but Formation Detection's own contract
    # (ai/computer_vision/tactical_analysis/formation_detection.py,
    # docs/data analysis.md §3) stores a template name like "4-3-3" here
    # -- a plain string. SQLite silently tolerates that (dynamic type
    # affinity), which is exactly why this went unnoticed in dev; the
    # project's own docker-compose.yml target is Postgres, where writing
    # a string into a Float column raises outright. Split into two
    # columns instead of loosening the type to something untyped (e.g.
    # JSON/Text for everything) so numeric metrics -- team_rating, xG,
    # compactness_score, etc. -- keep real Float semantics (sorting,
    # aggregation, range queries) rather than becoming string comparisons.
    # Exactly one of the two should be set per row; the API layer
    # (backend/api/tactical.py) picks whichever is non-null when building
    # the `value` field of the response, so this split is invisible to
    # the frontend/API contract.
    value_numeric = Column(Float, nullable=True)
    value_label = Column(String(64), nullable=True)

    method = Column(Enum(MetricMethod), nullable=False)
    confidence = Column(Enum(MetricConfidence), nullable=False)
    confidence_score = Column(Float, nullable=True)
    sample_size = Column(Integer, nullable=False)
    sub_scores = Column(JSON, nullable=False)
    computed_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    schema_version = Column(String(16), nullable=False)
    match = relationship("Match", back_populates="team_metrics")

    @property
    def value(self):
        """Convenience accessor mirroring the pre-split single-`value`
        shape for any code (or future migration) that just wants
        whichever value is set, without caring which column it lives in."""
        return self.value_label if self.value_label is not None else self.value_numeric

    def __repr__(self) -> str:
        return f"<TeamMetric id={self.metric_id} team_id={self.team_id} metric={self.metric_name} value={self.value}>"
