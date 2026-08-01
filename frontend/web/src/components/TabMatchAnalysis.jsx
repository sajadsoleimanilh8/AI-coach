import React, { useState, useEffect, useRef } from 'react';
import { realApi } from '../api/realClient';

export default function TabMatchAnalysis() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [job, setJob] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const canvasRef = useRef(null);
  const pollRef = useRef(null);

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setSelectedFile(e.target.files[0]);
    }
  };

  const handleUploadAndRun = async () => {
    if (!selectedFile) {
      setError('Please choose a video file first.');
      return;
    }

    try {
      setLoading(true);
      setError(null);
      setResult(null);

      // 1. Upload the REAL file to FastAPI (multipart/form-data)
      const uploadRes = await realApi.uploadVideoFile(selectedFile);
      const initialStatus = await realApi.getProcessingStatus(uploadRes.job_id);
      setJob({ ...initialStatus, video_id: uploadRes.video_id });

      // 2. Poll the REAL backend every 1.5s until completed/failed.
      //    Celery is running the actual YOLO + ByteTrack + Homography
      //    pipeline in the background -- this polls its real progress.
      if (pollRef.current) clearInterval(pollRef.current);

      pollRef.current = setInterval(async () => {
        try {
          const currentStatus = await realApi.getProcessingStatus(uploadRes.job_id);
          setJob({ ...currentStatus, video_id: uploadRes.video_id });

          if (currentStatus.status === 'completed') {
            clearInterval(pollRef.current);
            const resultData = await realApi.getAnalysisResult(uploadRes.job_id);
            setResult(resultData.result);
            setLoading(false);
          } else if (currentStatus.status === 'failed') {
            clearInterval(pollRef.current);
            setError(currentStatus.error || 'Processing job failed');
            setLoading(false);
          }
        } catch (e) {
          clearInterval(pollRef.current);
          setError(e.message);
          setLoading(false);
        }
      }, 1500);
    } catch (err) {
      setError(err.message);
      setLoading(false);
    }
  };

  // Cleanup polling on unmount
  useEffect(() => {
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, []);

  // Draw 2D Pitch Canvas (105m x 68m) & Heatmap Points
  useEffect(() => {
    if (!result || !result.heatmap_sample || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    ctx.clearRect(0, 0, width, height);

    // Pitch Background
    ctx.fillStyle = '#1b381b';
    ctx.fillRect(0, 0, width, height);

    // Pitch Boundary Lines
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 2;
    const pad = 15;
    const pitchW = width - 2 * pad;
    const pitchH = height - 2 * pad;

    ctx.strokeRect(pad, pad, pitchW, pitchH);

    // Halfway Line
    ctx.beginPath();
    ctx.moveTo(pad + pitchW / 2, pad);
    ctx.lineTo(pad + pitchW / 2, pad + pitchH);
    ctx.stroke();

    // Center Circle
    ctx.beginPath();
    ctx.arc(pad + pitchW / 2, pad + pitchH / 2, (9.15 / 68.0) * pitchH, 0, 2 * Math.PI);
    ctx.stroke();

    // Render Heatmap Spatial Points
    result.heatmap_sample.forEach((pt) => {
      const px = pad + (pt.x / 105.0) * pitchW;
      const py = pad + (pt.y / 68.0) * pitchH;

      ctx.beginPath();
      ctx.arc(px, py, 7, 0, 2 * Math.PI);
      ctx.fillStyle = 'rgba(56, 189, 248, 0.45)'; // Cyan Spatial Glow
      ctx.fill();
    });
  }, [result]);

  return (
    <div>
      <h2>Match Analysis & Spatial Tracking Pipeline</h2>
      <p>Upload a match clip to execute YOLO detection, ByteTrack tracking, and Pitch Homography calibration.</p>

      <div className="card" style={{ marginBottom: '1.5rem' }}>
        <div style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
          <input type="file" accept="video/*" onChange={handleFileChange} disabled={loading} />
          <button className="btn" onClick={handleUploadAndRun} disabled={loading}>
            {loading ? 'Processing Pipeline...' : 'Run CV Pipeline'}
          </button>
        </div>
      </div>

      {error && <p className="status-error">Error: {error}</p>}

      {job && (
        <div className="card">
          <h4>Job Status: <span style={{ color: '#38bdf8' }}>{job.status.toUpperCase()}</span></h4>
          <p>{job.message}</p>
          <div className="progress-bar">
            <div className="progress-fill" style={{ width: `${job.progress}%` }}></div>
          </div>
          <small>Pipeline Progress: {job.progress}%</small>

          {/* In-Browser Video Player -- served from the real backend's /storage mount */}
          {job.video_id && (
            <div style={{ marginTop: '1.5rem' }}>
              <h4>Uploaded Match Video Player</h4>
              <video
                controls
                width="100%"
                style={{ maxHeight: '380px', borderRadius: '8px', border: '1px solid #334155' }}
                src={`http://localhost:8000/storage/uploads/${job.video_id}.mp4`}
              />
            </div>
          )}
        </div>
      )}

      {result && (
        <div className="card">
          <h3>Phase 2 Pipeline Deliverables</h3>
          <p>
            Frames Processed: <strong>{result.frames_processed}</strong> |{' '}
            Unique Player Tracks: <strong>{result.total_tracks}</strong> |{' '}
            Homography Confidence: <strong>{((result.homography_confidence || 0) * 100).toFixed(1)}%</strong>
          </p>

          <h4 style={{ marginTop: '1.5rem' }}>2D Pitch Spatial Heatmap Overlay (105m x 68m)</h4>
          <canvas
            ref={canvasRef}
            width={640}
            height={400}
            style={{ border: '1px solid #334155', borderRadius: '8px', marginTop: '0.5rem' }}
          />
        </div>
      )}
    </div>
  );
}
