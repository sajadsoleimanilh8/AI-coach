"""
Player Intelligence API router.
Implementation Spec §5.1.
"""

from __future__ import annotations
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.api.cache import get_cached_metrics, set_cached_metrics
from backend.api.schemas import PlayerIntelligenceResponse, PlayerMetricResponse
from backend.database.models import PlayerMetric
from backend.database.session import get_db

router = APIRouter(prefix="/api/player_intelligence", tags=["player_intelligence"])

# FIXED: PLAYER_NAME_MAP used to map raw ByteTrack player_id -> a
# celebrity name (Alex Morgan, Marcus Rashford, etc.) for ANY real match,
# not just demo data. player_id is a ByteTrack tracking ID scoped to a
# single video, not a resolved player identity (see
# docs/database_schema.md's own note on PlayerDetection.player_id, and
# ai/computer_vision/player_tracking/tracker.py's TrackedDetection
# docstring) -- there is no jersey-number OCR or Re-ID in this pipeline,
# so there is no legitimate way to know that track_id 10 is actually
# Alex Morgan in a real uploaded match. Presenting a fabricated identity
# as if it were resolved is exactly the "confident-looking garbage"
# failure mode docs/data analysis.md's own honesty principles warn
# against -- it previously showed a real, unidentified tracked player
# under a real athlete's name on the dashboard for any real match. Demo
# data (frontend/web/src/api/mockClient.js) is free to use illustrative
# names since the UI clearly labels it "Demo data" (see
# TabPlayerIntelligence.jsx's isDemo banner) -- that's disclosed
# fiction, this endpoint serving it as fact for real matches was not.


@router.get("/{match_id}/{player_id}", response_model=list[PlayerMetricResponse])
def get_player_metrics(match_id: str, player_id: int, db: Session = Depends(get_db)):
    scope = f"player:{player_id}"
    cached = get_cached_metrics(match_id, scope)
    if cached:
        return cached

    metrics = (
        db.query(PlayerMetric)
        .filter(PlayerMetric.match_id == match_id, PlayerMetric.player_id == player_id)
        .all()
    )

    results = [
        {
            "metric_id": m.metric_id,
            "match_id": m.match_id,
            "player_id": m.player_id,
            "metric_name": m.metric_name,
            "value": m.value,
            "method": m.method.value,
            "confidence": m.confidence.value,
            "sample_size": m.sample_size,
            "sub_scores": m.sub_scores,
            "computed_at": m.computed_at,
            "schema_version": m.schema_version,
        }
        for m in metrics
    ]

    set_cached_metrics(match_id, scope, results)
    return results


@router.get("/{match_id}", response_model=list[PlayerIntelligenceResponse])
def get_all_player_intelligence(match_id: str, db: Session = Depends(get_db)):
    scope = "all_players"
    cached = get_cached_metrics(match_id, scope)
    if cached:
        return cached

    metrics = db.query(PlayerMetric).filter(PlayerMetric.match_id == match_id).all()

    grouped: dict[int, list] = {}
    for m in metrics:
        grouped.setdefault(m.player_id, []).append({
            "metric_id": m.metric_id,
            "match_id": m.match_id,
            "player_id": m.player_id,
            "metric_name": m.metric_name,
            "value": m.value,
            "method": m.method.value,
            "confidence": m.confidence.value,
            "sample_size": m.sample_size,
            "sub_scores": m.sub_scores,
            "computed_at": m.computed_at,
            "schema_version": m.schema_version,
        })

    results = [
        {
            "player_id": pid,
            # No name resolution exists for real tracked players -- see the
            # module-level comment above for why this must not be a
            # celebrity name lookup. "Player #N" honestly reflects that
            # this is a tracking ID, not a resolved identity.
            "player_name": f"Player #{pid}",
            "metrics": m_list,
        }
        for pid, m_list in grouped.items()
    ]

    set_cached_metrics(match_id, scope, results)
    return results
