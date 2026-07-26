import React, { useState } from 'react';
import { mockApi } from '../api/mockClient';

export default function TabMatchAnalysis() {
  const [job, setJob] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const startUpload = async () => {
    try {
      setLoading(true);
      setError(null);
      
      // Step 1: Upload
      const uploadRes = await mockApi.uploadVideo("match_clip_first_half.mp4");
      const statusRes = await mockApi.getProcessingStatus(uploadRes.job_id);
      setJob(statusRes);

      // Step 2: Simulate Queued -> Processing -> Completed over timer
      setTimeout(async () => {
        const p1 = await mockApi.advanceJobState(uploadRes.job_id, "processing", 45, "Running YOLO detection and homography calibration...");
        setJob({ ...p1 });
      }, 1500);

      setTimeout(async () => {
        const p2 = await mockApi.advanceJobState(uploadRes.job_id, "processing", 85, "Extracting player metrics & running event detection...");
        setJob({ ...p2 });
      }, 3500);

      setTimeout(async () => {
        const p3 = await mockApi.advanceJobState(uploadRes.job_id, "completed", 100, "Analysis complete. Player/Team metrics generated.");
        setJob({ ...p3 });
        setLoading(false);
      }, 5500);

    } catch (err) {
      setError(err.message);
      setLoading(false);
    }
  };

  return (
    <div>
      <h2>Match Analysis (Video Processing)</h2>
      <p className="metric-meta">Upload a match clip to trigger spatial tracking, event extraction, and pipeline analysis.</p>

      <div className="card">
        <button className="btn" onClick={startUpload} disabled={loading || (job && job.status === "processing")}>
          {loading ? "Processing..." : "Simulate Video Upload & Run Pipeline"}
        </button>

        {error && <p className="status-error">Error: {error}</p>}

        {job && (
          <div style={{ marginTop: '1.5rem' }}>
            <div className="metric-header">
              <strong>Job ID:</strong> <code>{job.job_id}</code>
              <span className={`badge ${job.status === 'completed' ? 'badge-normal' : job.status === 'failed' ? 'badge-low_upstream_confidence' : 'badge-low_sample'}`}>
                {job.status}
              </span>
            </div>

            <div className="progress-bar">
              <div className="progress-fill" style={{ width: `${job.progress}%` }}></div>
            </div>

            <div style={{ display: 'flex', justifyContent: 'space-between' }} className="metric-meta">
              <span>{job.message}</span>
              <span>{job.progress}%</span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}