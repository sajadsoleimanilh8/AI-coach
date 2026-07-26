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

    jobs = relationship("ProcessingJob", back_populates="video", cascade="all, delete-orphan")


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

    def __repr__(self) -> str:
        return f"<Match id={self.match_id} {self.home_team} vs {self.away_team}>"


class Frame(Base):
    """A processed video frame belonging to a match."""

    __tablename__ = "frames"

    frame_id = Column(Integer, primary_key=True, autoincrement=True)
    match_id = Column(String(36), ForeignKey("matches.match_id"), nullable=False, index=True)
    frame_number = Column(Integer, nullable=False)
    timestamp = Column(Float, nullable=False)  # seconds from match start
    fps = Column(Float, nullable=False)

    match = relationship("Match", back_populates="frames")
    player_detections = relationship("PlayerDetection", back_populates="frame", cascade="all, delete-orphan")
    ball_detections = relationship("BallDetection", back_populates="frame", cascade="all, delete-orphan")
    player_trackings = relationship("PlayerTracking", back_populates="frame", cascade="all, delete-orphan")
    events = relationship("Event", back_populates="frame")

    def __repr__(self) -> str:
        return f"<Frame id={self.frame_id} match_id={self.match_id} number={self.frame_number}>"


class PlayerDetection(Base):
    """Raw YOLO detection output, one row per player per frame."""

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

    frame = relationship("Frame", back_populates="player_detections")

    def __repr__(self) -> str:
        return f"<PlayerDetection id={self.detection_id} frame_id={self.frame_id} player_id={self.player_id}>"


class BallDetection(Base):
    """Ball detection output, one row per frame (or per candidate, if multiple)."""

    __tablename__ = "ball_detections"

    detection_id = Column(String(36), primary_key=True, default=new_id)
    frame_id = Column(Integer, ForeignKey("frames.frame_id"), nullable=False, index=True)
    ball_x = Column(Float, nullable=False)  # pixel-space
    ball_y = Column(Float, nullable=False)
    confidence = Column(Float, nullable=False)

    frame = relationship("Frame", back_populates="ball_detections")

    def __repr__(self) -> str:
        return f"<BallDetection id={self.detection_id} frame_id={self.frame_id}>"


class PlayerTracking(Base):
    """Continuous player trajectories, one row per player per frame, in pixel and pitch coordinates."""

    __tablename__ = "player_tracking"

    tracking_id = Column(String(36), primary_key=True, default=new_id)
    match_id = Column(String(36), ForeignKey("matches.match_id"), nullable=False, index=True)
    player_id = Column(Integer, nullable=False)
    frame_id = Column(Integer, ForeignKey("frames.frame_id"), nullable=False, index=True)
    team_id = Column(String(64), nullable=True)
    pixel_x = Column(Float, nullable=False)
    pixel_y = Column(Float, nullable=False)
    pitch_x_m = Column(Float, nullable=True)
    pitch_y_m = Column(Float, nullable=True)
    homography_confidence = Column(Float, nullable=True)

    speed = Column(Float, nullable=True)
    distance = Column(Float, nullable=True)
    acceleration = Column(Float, nullable=True)

    match = relationship("Match", back_populates="player_trackings")
    frame = relationship("Frame", back_populates="player_trackings")

    def __repr__(self) -> str:
        return f"<PlayerTracking id={self.tracking_id} match_id={self.match_id} player_id={self.player_id}>"


class Event(Base):
    """Detected match events (pass, shot, touch, turnover, press, etc.)."""

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
    frame = relationship("Frame", back_populates="events")

    def __repr__(self) -> str:
        return f"<Event id={self.event_id} match_id={self.match_id} type={self.event_type}>"


class PlayerMetric(Base):
    """Analysis engine output, one row per player per metric per match.

    Shape is fixed to the Standard Output Contract in docs/data analysis.md
    (Analysis Logic Design v3) §0.3 -- do not add ad hoc fields per metric
    type, everything metric-specific goes in sub_scores.
    """

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
    schema_version = Column(String(16), nullable=False)  # e.g. "v3"

    match = relationship("Match", back_populates="player_metrics")

    def __repr__(self) -> str:
        return f"<PlayerMetric id={self.metric_id} player_id={self.player_id} metric={self.metric_name} value={self.value}>"


class TeamMetric(Base):
    """Team-level analysis output, one row per team per metric per match.

    Structurally identical to PlayerMetric except player_id -> team_id, by
    design, so the DB layer and API serialization code can be shared between
    the two (see docs/database_schema.md, fixes #4 and #5).
    """

    __tablename__ = "team_metrics"

    metric_id = Column(String(36), primary_key=True, default=new_id)
    match_id = Column(String(36), ForeignKey("matches.match_id"), nullable=False, index=True)
    team_id = Column(String(64), nullable=False)
    metric_name = Column(String(120), nullable=False)
    value = Column(Float, nullable=True)
    method = Column(Enum(MetricMethod), nullable=False)
    confidence = Column(Enum(MetricConfidence), nullable=False)
    confidence_score = Column(Float, nullable=True)
    sample_size = Column(Integer, nullable=False)
    sub_scores = Column(JSON, nullable=False)
    computed_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    schema_version = Column(String(16), nullable=False)

    match = relationship("Match", back_populates="team_metrics")

    def __repr__(self) -> str:
        return f"<TeamMetric id={self.metric_id} team_id={self.team_id} metric={self.metric_name} value={self.value}>"
