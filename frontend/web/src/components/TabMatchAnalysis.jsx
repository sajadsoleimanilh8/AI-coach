import React, { useEffect, useRef, useState } from 'react';
import { api } from '../api/client';
import { mockApi } from '../api/mockClient';
import MetricBadge from './MetricBadge';

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

// A window of frames pulled per /tracking request. Kept in sync with the
// server's own DEFAULT_WINDOW_FRAMES (backend/api/tracking.py) only in
// spirit -- the server enforces the real cap, this just avoids firing an
// oversized request from the client.
const TRACKING_WINDOW = 150;

// Ball position is only persisted on the stride-sampled Frame rows (see
// backend/pipeline/runner.py's FRAME_PERSIST_STRIDE, default 5), not every
// frame like PlayerTracking. Holding the last-seen ball position for a few
// extra frames avoids the dot flickering on/off every frame while still
// dropping it once the last reading is stale enough to be misleading.
const BALL_STALE_FRAMES = 8;

// Real team_id values ("team-home"/"team-away") aren't produced yet --
// jersey-color clustering isn't implemented (see backend/pipeline/runner.py's
// TEAM_ASSIGNMENT_CONFIDENCE = 0.0), so every tracked player currently has
// team_id=null and renders gray below. These two entries exist so that once
// team assignment ships, real values get real (not invented) colors instead
// of falling through to the "unknown team_id we've never seen" fallback.
const TEAM_COLORS = { 'team-home': '#3b82f6', 'team-away': '#ef4444' };
const UNASSIGNED_COLOR = '#9ca3af';
const OTHER_TEAM_COLOR = '#a855f7';
const BALL_COLOR = '#facc15';

function colorForTeam(teamId) {
  if (!teamId) return UNASSIGNED_COLOR;
  return TEAM_COLORS[teamId] || OTHER_TEAM_COLOR;
}

function PitchOutline({ pitchWidth, pitchHeight }) {
  return (
    <>
      <line x1={pitchWidth / 2} y1="0" x2={pitchWidth / 2} y2={pitchHeight} stroke="#ffffff" strokeWidth="2" />
      <circle cx={pitchWidth / 2} cy={pitchHeight / 2} r="45" fill="none" stroke="#ffffff" strokeWidth="2" />
      <circle cx={pitchWidth / 2} cy={pitchHeight / 2} r="3" fill="#ffffff" />
      <rect x="0" y={(pitchHeight - 160) / 2} width="80" height="160" fill="none" stroke="#ffffff" strokeWidth="2" />
      <rect x={pitchWidth - 80} y={(pitchHeight - 160) / 2} width="80" height="160" fill="none" stroke="#ffffff" strokeWidth="2" />
    </>
  );
}

function HeatmapPitch({ heatmap, loading, error }) {
  const pitchWidth = 525;
  const pitchHeight = 340;

  if (loading) return <div className="pitch-empty-state">Loading heatmap...</div>;
  if (error) return <div className="status-error">Error loading heatmap: {error}</div>;
  if (!heatmap) return null;

  // Same honesty gate as PitchVisual's weak-zone overlay in
  // TabTeamIntelligence.jsx: only render density as if it were trustworthy
  // when confidence is 'normal'. low_upstream_confidence (homography never
  // usable for enough points) and low_sample both get the explicit empty
  // state instead of a heatmap that looks complete but isn't.
  const isComputed = heatmap.confidence === 'normal';
  const cellWidth = pitchWidth / heatmap.grid_cols;
  const cellHeight = pitchHeight / heatmap.grid_rows;

  return (
    <div className="pitch-wrap">
      <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', gap: '0.75rem', marginBottom: '0.75rem' }}>
        <h4 style={{ margin: 0 }}>Player #{heatmap.player_id} Position Density</h4>
        <MetricBadge confidence={heatmap.confidence} />
      </div>

      {!isComputed ? (
        <div className="pitch-empty-state">
          Not enough tracking data to show a heatmap for this player (
          {heatmap.usable_sample_size} of {heatmap.sample_size} tracked frames had usable pitch
          coordinates -- homography confidence may be below the threshold, or too few frames
          were tracked yet).
        </div>
      ) : (
        <svg width="100%" viewBox={`0 0 ${pitchWidth} ${pitchHeight}`} className="pitch-svg">
          {heatmap.cells.map((c) => (
            <rect
              key={`${c.grid_x}_${c.grid_y}`}
              x={c.grid_x * cellWidth}
              y={c.grid_y * cellHeight}
              width={cellWidth}
              height={cellHeight}
              fill="#ef4444"
              fillOpacity={Math.max(0.08, c.density) * 0.85}
            />
          ))}
          <PitchOutline pitchWidth={pitchWidth} pitchHeight={pitchHeight} />
        </svg>
      )}

      <div className="metric-meta" style={{ marginTop: '0.5rem' }}>
        {isComputed
          ? `* Density from ${heatmap.usable_sample_size} tracked positions (of ${heatmap.sample_size} total frames).`
          : '* Heatmap not shown: pitch coordinates for this player are not confident enough yet.'}
      </div>
    </div>
  );
}

function TrackingOverlayPlayer({ matchId, videoId }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const rafRef = useRef(null);
  const framesRef = useRef(new Map()); // frame_number -> { players, ball }
  const loadedRangesRef = useRef([]); // [[start, end], ...]
  const pendingRangesRef = useRef(new Set()); // "start-end" keys in flight
  const fpsRef = useRef(25);
  const lastBallRef = useRef(null); // { frame_number, pixel_x, pixel_y }
  const knownPlayerIdsRef = useRef(new Set());

  const [mode, setMode] = useState('overlay');
  const [playerIds, setPlayerIds] = useState([]);
  const [selectedPlayerId, setSelectedPlayerId] = useState(null);
  const [hasTrackingData, setHasTrackingData] = useState(false);
  const [trackingError, setTrackingError] = useState(null);
  const [heatmap, setHeatmap] = useState(null);
  const [heatmapLoading, setHeatmapLoading] = useState(false);
  const [heatmapError, setHeatmapError] = useState(null);

  // Reset all per-match caches when the match/video actually changes --
  // otherwise a second upload in the same session would keep drawing the
  // previous match's cached frames until its own windows overwrite them.
  useEffect(() => {
    framesRef.current = new Map();
    loadedRangesRef.current = [];
    pendingRangesRef.current = new Set();
    lastBallRef.current = null;
    knownPlayerIdsRef.current = new Set();
    fpsRef.current = 25;
    setPlayerIds([]);
    setSelectedPlayerId(null);
    setHasTrackingData(false);
    setTrackingError(null);
    setHeatmap(null);
  }, [matchId, videoId]);

  const ensureWindow = async (frameNumber) => {
    const covered = loadedRangesRef.current.some(([s, e]) => frameNumber >= s && frameNumber <= e);
    if (covered) return;
    const start = Math.max(0, frameNumber - 10);
    const end = start + TRACKING_WINDOW;
    const key = `${start}-${end}`;
    if (pendingRangesRef.current.has(key)) return;
    pendingRangesRef.current.add(key);

    try {
      const data = await api.getTracking(matchId, start, end);
      fpsRef.current = data.fps || fpsRef.current;
      loadedRangesRef.current.push([start, end]);

      let sawData = false;
      data.frames.forEach((f) => {
        framesRef.current.set(f.frame_number, f);
        if (f.players.length > 0 || f.ball) sawData = true;
        f.players.forEach((p) => knownPlayerIdsRef.current.add(p.player_id));
      });

      if (sawData) setHasTrackingData(true);

      // Read/write player ids through the ref + functional updaters rather
      // than the `playerIds` state closed over above -- this function is
      // called from inside the draw-loop effect's rAF callback, which
      // captures whatever closure existed when that effect last committed
      // (see its [mode, matchId, videoId] deps), so a direct read of
      // `playerIds` here could compare against a stale value instead of
      // the latest state.
      const sorted = Array.from(knownPlayerIdsRef.current).sort((a, b) => a - b);
      setPlayerIds((prev) => (prev.length === sorted.length ? prev : sorted));
      setSelectedPlayerId((prev) => (prev == null ? sorted[0] ?? null : prev));
    } catch (err) {
      setTrackingError(err.message);
    } finally {
      pendingRangesRef.current.delete(key);
    }
  };

  // Load the first window immediately so the player selector (for the
  // heatmap toggle) has something to show without waiting for playback.
  useEffect(() => {
    if (matchId && videoId) ensureWindow(0);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [matchId, videoId]);

  // Canvas internal resolution is set to the video's native pixel
  // resolution (not its displayed CSS size) so pixel_x/pixel_y from the
  // tracking endpoint -- measured against the raw video frame -- map 1:1
  // onto canvas drawing coordinates regardless of how large the video is
  // rendered on screen.
  useEffect(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return undefined;
    const onLoadedMetadata = () => {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
    };
    video.addEventListener('loadedmetadata', onLoadedMetadata);
    return () => video.removeEventListener('loadedmetadata', onLoadedMetadata);
  }, [videoId]);

  // Draw loop: reads video.currentTime every animation frame and paints
  // whichever tracked frame is closest, rather than relying on the
  // low-resolution 'timeupdate' event -- that's what keeps the overlay
  // visually in sync with playback instead of stepping every ~250ms.
  useEffect(() => {
    if (mode !== 'overlay') return undefined;
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return undefined;
    let active = true;

    const draw = () => {
      if (!active) return;
      const ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      const frameNumber = Math.round(video.currentTime * fpsRef.current);
      ensureWindow(frameNumber);
      const frameData = framesRef.current.get(frameNumber);

      if (frameData) {
        frameData.players.forEach((p) => {
          ctx.beginPath();
          ctx.arc(p.pixel_x, p.pixel_y, Math.max(6, canvas.width / 160), 0, Math.PI * 2);
          ctx.fillStyle = colorForTeam(p.team_id);
          ctx.fill();
          ctx.lineWidth = Math.max(1, canvas.width / 640);
          ctx.strokeStyle = '#ffffff';
          ctx.stroke();
        });
        if (frameData.ball) {
          lastBallRef.current = { frame_number: frameNumber, ...frameData.ball };
        }
      }

      const ball = lastBallRef.current;
      if (ball && frameNumber - ball.frame_number >= 0 && frameNumber - ball.frame_number <= BALL_STALE_FRAMES) {
        ctx.beginPath();
        ctx.arc(ball.pixel_x, ball.pixel_y, Math.max(4, canvas.width / 250), 0, Math.PI * 2);
        ctx.fillStyle = BALL_COLOR;
        ctx.fill();
        ctx.lineWidth = 1;
        ctx.strokeStyle = '#020617';
        ctx.stroke();
      }

      rafRef.current = requestAnimationFrame(draw);
    };

    rafRef.current = requestAnimationFrame(draw);
    return () => {
      active = false;
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mode, matchId, videoId]);

  useEffect(() => {
    if (mode !== 'heatmap' || selectedPlayerId == null) return;
    setHeatmapLoading(true);
    setHeatmapError(null);
    api.getHeatmap(matchId, selectedPlayerId)
      .then((data) => { setHeatmap(data); setHeatmapLoading(false); })
      .catch((err) => { setHeatmapError(err.message); setHeatmapLoading(false); });
  }, [mode, selectedPlayerId, matchId]);

  return (
    <div className="card">
      <div className="metric-header">
        <h3 style={{ margin: 0 }}>Tracking Overlay</h3>
        <div style={{ display: 'flex', gap: '0.5rem' }}>
          <button
            className="btn"
            style={mode !== 'overlay' ? { backgroundColor: 'var(--bg-hover)', color: 'var(--text-main)' } : undefined}
            onClick={() => setMode('overlay')}
          >
            Video Overlay
          </button>
          <button
            className="btn"
            style={mode !== 'heatmap' ? { backgroundColor: 'var(--bg-hover)', color: 'var(--text-main)' } : undefined}
            onClick={() => setMode('heatmap')}
            disabled={playerIds.length === 0}
          >
            Heatmap
          </button>
        </div>
      </div>

      {trackingError && (
        <p className="status-error">Error loading tracking data: {trackingError}</p>
      )}

      {mode === 'overlay' && (
        <>
          {!hasTrackingData && (
            <p className="metric-meta">
              No tracking data loaded yet for this window -- the pipeline may still be running,
              or hasn't reached the tracking stage for this part of the clip. The video will
              still play; player/ball dots appear once tracking data is available.
            </p>
          )}
          <div style={{ position: 'relative', width: '100%', maxWidth: 720, margin: '0 auto' }}>
            {/* eslint-disable-next-line jsx-a11y/media-has-caption */}
            <video
              ref={videoRef}
              src={api.getVideoUrl(videoId)}
              controls
              style={{ width: '100%', display: 'block', borderRadius: 6, backgroundColor: '#000' }}
            />
            <canvas
              ref={canvasRef}
              style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', pointerEvents: 'none' }}
            />
          </div>
          <div className="metric-meta" style={{ marginTop: '0.5rem', textAlign: 'center' }}>
            <span style={{ color: UNASSIGNED_COLOR }}>&#9679;</span> unassigned player (team assignment not implemented yet)
            {' '}&nbsp;<span style={{ color: BALL_COLOR }}>&#9679;</span> ball
          </div>
        </>
      )}

      {mode === 'heatmap' && (
        <>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.75rem' }}>
            <label className="metric-meta" htmlFor="heatmap-player-select">Player:</label>
            <select
              id="heatmap-player-select"
              value={selectedPlayerId ?? ''}
              onChange={(e) => setSelectedPlayerId(Number(e.target.value))}
              style={{ backgroundColor: '#0f172a', color: 'var(--text-main)', border: '1px solid var(--border-color)', borderRadius: 4, padding: '0.3rem 0.5rem' }}
            >
              {playerIds.map((pid) => (
                <option key={pid} value={pid}>Player #{pid}</option>
              ))}
            </select>
          </div>
          <HeatmapPitch heatmap={heatmap} loading={heatmapLoading} error={heatmapError} />
        </>
      )}
    </div>
  );
}

export default function TabMatchAnalysis({ onMatchReady }) {
  const [file, setFile] = useState(null);
  const [job, setJob] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [isDemo, setIsDemo] = useState(false);
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
    setIsDemo(false);
    try {
      const uploadRes = await api.uploadVideo(file);
      setJob({
        job_id: uploadRes.job_id,
        video_id: uploadRes.video_id,
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
    setIsDemo(true);
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

      {isDemo && (
        <div className="demo-banner">
          <span>
            Demo data has no underlying video file or frame-level tracking -- the tracking
            overlay and heatmap below only work for a real uploaded &amp; processed match.
          </span>
        </div>
      )}

      {!isDemo && job?.match_id && job?.video_id && (
        <TrackingOverlayPlayer matchId={job.match_id} videoId={job.video_id} />
      )}
    </div>
  );
}
