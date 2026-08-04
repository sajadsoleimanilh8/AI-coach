import React, { useEffect, useRef, useState } from 'react';
import { api } from '../api/client';
import { mockApi } from '../api/mockClient';

/**
 * FIXED (Phase 3 audit): this tab previously only called mockApi and never
 * touched the real backend, so a real upload could never happen and
 * matchId was never threaded anywhere. It now does a real upload against
 * POST /api/videos/upload, polls the real job status, and hands the real
 * match_id up to App once the job is created (not once it's *complete* --
 * the Player/Team Intelligence tabs handle "not computed yet" honestly on
 * their own, see their empty states).
 *
 * "Load Demo Data" is kept as a clearly separate, clearly labeled path
 * (matchId === 'demo') rather than a silent fallback -- see
 * TabPlayerIntelligence.jsx / TabTeamIntelligence.jsx's demo-banner
 * handling. Silently swapping to mock data on failure is exactly what
 * made the match_id wiring bug invisible during the team's own testing.
 */
export default function TabMatchAnalysis({ onMatchReady }) {
  const [file, setFile] = useState(null);
  const [job, setJob] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const pollRef = useRef(null);

  useEffect(() => () => clearInterval(pollRef.current), []);

  const pollStatus = (jobId) => {
    clearInterval(pollRef.current);
    pollRef.current = setInterval(async () => {
      try {
        const status = await api.getProcessingStatus(jobId);
        setJob(status);
        if (status.status === 'completed' || status.status === 'failed') {
          clearInterval(pollRef.current);
          setLoading(false);
        }
      } catch (err) {
        setError(err.message);
        clearInterval(pollRef.current);
        setLoading(false);
      }
    }, 2000);
  };

  const startRealUpload = async () => {
    if (!file) {
      setError('Choose a video file first.');
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const uploadRes = await api.uploadVideo(file);
      setJob({
        job_id: uploadRes.job_id,
        match_id: uploadRes.match_id,
        status: uploadRes.status,
        progress: 0,
        message: uploadRes.message,
      });
      onMatchReady?.(uploadRes.match_id, uploadRes.job_id);
      pollStatus(uploadRes.job_id);
    } catch (err) {
      setError(err.message);
      setLoading(false);
    }
  };

  const loadDemoData = async () => {
    setError(null);
    setLoading(true);
    const uploadRes = await mockApi.uploadVideo('match_clip_first_half.mp4');
    const statusRes = await mockApi.getProcessingStatus(uploadRes.job_id);
    setJob(statusRes);
    setLoading(false);
    // 'demo' is a sentinel matchId every tab recognizes as "show mock
    // data behind an explicit demo-mode banner" -- never confused with a
    // real match_id (real ones are UUIDs from the Match table).
    onMatchReady?.('demo', null);
  };

  return (
    <div>
      <h2>Match Analysis (Video Processing)</h2>
      <p className="metric-meta">
        Upload a real match clip to run the actual pipeline (detection, tracking, homography,
        event heuristics, tactical &amp; player scoring), or load demo data to explore the
        dashboard without waiting on a full pipeline run.
      </p>

      <div className="card">
        <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'center', flexWrap: 'wrap' }}>
          <input
            type="file"
            accept="video/mp4,video/quicktime,video/x-msvideo"
            onChange={(e) => setFile(e.target.files?.[0] ?? null)}
            disabled={loading}
          />
          <button className="btn" onClick={startRealUpload} disabled={loading || !file}>
            {loading ? 'Processing...' : 'Upload & Run Pipeline'}
          </button>
          <button
            className="btn"
            style={{ backgroundColor: 'var(--bg-hover)', color: 'var(--text-main)' }}
            onClick={loadDemoData}
            disabled={loading}
          >
            Load Demo Data
          </button>
        </div>

        {error && <p className="status-error" style={{ marginTop: '1rem' }}>Error: {error}</p>}

        {job && (
          <div style={{ marginTop: '1.5rem' }}>
            <div className="metric-header">
              <strong>Job ID:</strong> <code>{job.job_id ?? '(demo)'}</code>
              <span className={`badge ${job.status === 'completed' ? 'badge-normal' : job.status === 'failed' ? 'badge-low_upstream_confidence' : 'badge-low_sample'}`}>
                {job.status}
              </span>
            </div>

            <div className="progress-bar">
              <div className="progress-fill" style={{ width: `${job.progress}%` }} />
            </div>

            <div style={{ display: 'flex', justifyContent: 'space-between' }} className="metric-meta">
              <span>{job.message}</span>
              <span>{job.progress}%</span>
            </div>

            {job.status === 'failed' && job.error && (
              <p className="status-error" style={{ marginTop: '0.5rem' }}>
                {job.error}
              </p>
            )}

            {job.status === 'completed' && job.match_id && (
              <p className="metric-meta" style={{ marginTop: '0.75rem' }}>
                Match <code>{job.match_id}</code> is ready — check the Player Intelligence and
                Team Intelligence tabs.
              </p>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
