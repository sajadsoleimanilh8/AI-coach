# SportsStrategyCoachAI Backend

FastAPI backend for video upload, processing status, database storage, and JSON analysis results.

## Database

Tables are defined in SQLAlchemy models and documented in `backend/database/schema.sql`.

- `videos`: uploaded video file records
- `processing_jobs`: queue/status/progress for each video
- `analysis_results`: JSON output from computer vision or tactical analysis

## Run locally

```powershell
pip install -r backend/requirements.txt
uvicorn backend.api.main:app --reload --host 0.0.0.0 --port 8000
```

The local default database is SQLite:

```text
backend/database/sports_strategy.db
```

## Run with Docker

```powershell
docker compose up --build
```

This starts:

- FastAPI backend on `http://localhost:8000`
- PostgreSQL on `localhost:5432`

## API examples

Upload a video:

```powershell
curl.exe -X POST "http://localhost:8000/api/videos/upload" `
  -F "file=@D:\SportsStrategyCoachAI\SportsStrategyCoachAI\test.mp4" `
  -F "metadata={\"team\":\"home\",\"opponent\":\"away\",\"match_date\":\"2026-07-20\"}"
```

Check processing status:

```powershell
curl.exe "http://localhost:8000/api/processing/<job_id>"
```

Update processing status:

```powershell
curl.exe -X PATCH "http://localhost:8000/api/processing/<job_id>" `
  -H "Content-Type: application/json" `
  -d "{\"status\":\"processing\",\"progress\":45,\"message\":\"Detecting players and ball\"}"
```

Save analysis JSON:

```powershell
curl.exe -X POST "http://localhost:8000/api/processing/<job_id>/result" `
  -H "Content-Type: application/json" `
  -d "{\"summary\":\"High press created weak-side space.\",\"result\":{\"players_detected\":22,\"ball_tracks\":184,\"tactical_notes\":[\"pressing intensity dropped after minute 60\"]}}"
```

Read analysis JSON:

```powershell
curl.exe "http://localhost:8000/api/processing/<job_id>/result"
```
