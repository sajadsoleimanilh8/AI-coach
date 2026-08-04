import React, { useEffect, useState } from 'react';
import { api } from '../api/client';
import { mockApi } from '../api/mockClient';
import MetricBadge from './MetricBadge';

const RADAR_KEYS = [
  { key: 'first_touch_score', label: 'First Touch' },
  { key: 'press_resistance_score', label: 'Press Resistance' },
  { key: 'off_ball_movement_score', label: 'Off-Ball Movement' },
  { key: 'defensive_positioning_score', label: 'Defensive Position' },
];

function PlayerRadarChart({ metrics, playerName }) {
  const scores = RADAR_KEYS.map(({ key, label }) => {
    const m = metrics.find((item) => item.metric_name === key);
    return {
      label,
      value: m && m.value !== null ? m.value : 0,
      // FIXED (Phase 3 audit): a metric absent from the list (pipeline
      // never computed it for this player) is now 'not_computed', not
      // 'low_sample' -- those mean different things (see MetricBadge.jsx).
      confidence: m ? m.confidence : 'not_computed',
    };
  });

  const size = 280;
  const center = size / 2;
  const radius = 95;
  const numAngles = scores.length;

  const coords = (index, value) => {
    const angle = ((Math.PI * 2) / numAngles) * index - Math.PI / 2;
    const r = (value / 100) * radius;
    return { x: center + r * Math.cos(angle), y: center + r * Math.sin(angle) };
  };

  const points = scores.map((s, i) => { const { x, y } = coords(i, s.value); return `${x},${y}`; }).join(' ');

  return (
    <div className="radar-wrap">
      <svg width={size} height={size} style={{ overflow: 'visible' }}>
        {[0.25, 0.5, 0.75, 1.0].map((frac, idx) => (
          <circle key={idx} cx={center} cy={center} r={radius * frac} fill="none"
                  stroke="var(--border-color)" strokeDasharray={frac === 1.0 ? '0' : '3 3'} />
        ))}
        {scores.map((s, i) => {
          const { x, y } = coords(i, 100);
          const label = coords(i, 122);
          return (
            <g key={i}>
              <line x1={center} y1={center} x2={x} y2={y} stroke="var(--border-color)" />
              <text x={label.x} y={label.y} textAnchor="middle" fontSize="10.5"
                    fill="var(--text-muted)" fontWeight="600">
                {s.label} {s.confidence === 'not_computed' ? '' : `(${Math.round(s.value)})`}
              </text>
            </g>
          );
        })}
        <polygon points={points} fill="rgba(56, 189, 248, 0.30)" stroke="var(--accent-blue)" strokeWidth="2" />
        {scores.map((s, i) => {
          const { x, y } = coords(i, s.value);
          return <circle key={i} cx={x} cy={y} r="4" fill="var(--accent-blue)" />;
        })}
      </svg>
    </div>
  );
}

export default function TabPlayerIntelligence({ matchId }) {
  const [playersData, setPlayersData] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [isDemo, setIsDemo] = useState(false);

  useEffect(() => {
    if (!matchId) {
      setPlayersData([]);
      setError(null);
      return;
    }

    setLoading(true);
    setError(null);

    if (matchId === 'demo') {
      setIsDemo(true);
      mockApi.getPlayerIntelligence()
        .then((data) => { setPlayersData(data); setLoading(false); })
        .catch((err) => { setError(err.message); setLoading(false); });
      return;
    }

    setIsDemo(false);
    api.getPlayerIntelligence(matchId)
      .then((data) => { setPlayersData(data); setLoading(false); })
      .catch((err) => {
        // FIXED (Phase 3 audit): no more silent mockApi fallback on
        // failure here -- a failed real fetch is shown as a real error,
        // not quietly swapped for indistinguishable mock data. That
        // silence is exactly what let the match_id/pipeline wiring bugs
        // in this project pass casual testing unnoticed.
        setError(err.message);
        setLoading(false);
      });
  }, [matchId]);

  if (!matchId) {
    return (
      <div>
        <h2>Player Intelligence</h2>
        <div className="card">
          <p className="metric-meta">
            No match selected yet. Upload and process a video (or load demo data) on the
            Match Analysis tab first.
          </p>
        </div>
      </div>
    );
  }

  if (loading) return <div>Loading Player Intelligence metrics...</div>;
  if (error) return <div className="status-error">Error loading player intelligence: {error}</div>;

  return (
    <div>
      <h2>Player Intelligence</h2>
      <p className="metric-meta">Per-player tactical & physical scores computed via rule-based heuristics.</p>

      {isDemo && (
        <div className="demo-banner">
          <span>Showing demo data — not from a real processed match.</span>
        </div>
      )}

      {playersData.length === 0 && (
        <div className="card">
          <p className="metric-meta">
            No player metrics computed yet for this match. The pipeline may still be
            processing, or no players were detected/tracked.
          </p>
        </div>
      )}

      {playersData.map((player) => (
        <div key={player.player_id} className="card">
          <h3 className="card-title">{player.player_name || `Player #${player.player_id}`}</h3>

          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1.25rem', alignItems: 'flex-start' }}>
            <div style={{ flex: '1 1 300px' }}>
              <PlayerRadarChart metrics={player.metrics} playerName={player.player_name} />
            </div>

            <div style={{ flex: '2 1 380px' }} className="metrics-grid">
              {player.metrics.map((m) => (
                <div key={m.metric_id} className="metric-card">
                  <div className="metric-header">
                    <span className="metric-name">{m.metric_name.replace(/_/g, ' ')}</span>
                    <MetricBadge confidence={m.confidence} method={m.method} />
                  </div>

                  <div className="metric-value">
                    {m.value !== null ? m.value : 'N/A'}
                  </div>

                  <div className="metric-meta">
                    Sample Size: {m.sample_size} | Schema: {m.schema_version}
                  </div>

                  {m.sub_scores && Object.keys(m.sub_scores).length > 0 && (
                    <div className="subscores-list">
                      {Object.entries(m.sub_scores).map(([k, v]) => (
                        <div key={k} className="subscore-item">
                          <span>{k}:</span>
                          <strong>{v !== null && v !== undefined ? v : 'N/A'}</strong>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}
