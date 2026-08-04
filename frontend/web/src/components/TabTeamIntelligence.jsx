import React, { useEffect, useState } from 'react';
import { api } from '../api/client';
import { mockApi } from '../api/mockClient';
import { getFormationSlots } from '../lib/formationTemplates';
import MetricBadge from './MetricBadge';

const ROLE_LABELS_BY_COUNT = {
  10: ['LB', 'LCB', 'RCB', 'RB', 'LCM', 'CM', 'RCM', 'LW', 'ST', 'RW'], // 4-3-3 default labeling
};

function PitchVisual({ formationMetric, weakZoneMetric }) {
  const pitchWidth = 525;
  const pitchHeight = 340;
  const gridX = 6;
  const gridY = 4;
  const cellWidth = pitchWidth / gridX;
  const cellHeight = pitchHeight / gridY;

  // FIXED (Phase 3 audit): this used to hardcode a 4-3-3 slot layout
  // (`slots433`) regardless of what formationMetric.value actually said
  // -- a real "4-4-2" result would show the correct header text while
  // still drawing 4-3-3 dots underneath, a visible contradiction. Slots
  // are now derived from the ACTUAL detected formation name, via the JS
  // mirror of the Python template dict (see lib/formationTemplates.js).
  // If nothing's been detected yet, render an empty pitch instead of
  // defaulting to a shape that was never measured.
  const formationName = formationMetric?.value;
  const isComputed = formationMetric && formationMetric.confidence !== 'low_upstream_confidence'
    && formationMetric.confidence !== 'low_sample' && formationName;
  const slots = isComputed ? getFormationSlots(formationName) : null;

  // Same honesty rule for the heatmap: weak_zone_map's fallback (no
  // upstream data) fills every zone with density=0.0, which under the
  // old always-on overlay rendered as an all-red pitch -- implying "every
  // zone is weak" when really "nothing was measured." Only render the
  // overlay when the metric was actually computed.
  const weakZoneComputed = weakZoneMetric?.confidence === 'normal';
  const weakZoneScores = weakZoneComputed ? (weakZoneMetric.sub_scores || {}) : {};

  return (
    <div className="pitch-wrap">
      <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', gap: '0.75rem', marginBottom: '0.75rem' }}>
        <h4 style={{ margin: 0 }}>
          Detected Formation: {formationName || 'Not yet computed'}
        </h4>
        {formationMetric && <MetricBadge confidence={formationMetric.confidence} method={formationMetric.method} />}
      </div>

      {!slots ? (
        <div className="pitch-empty-state">
          No formation has been detected yet for this match (pipeline may still be running,
          team assignment isn't confident enough, or too few players are tracked in the window).
        </div>
      ) : (
        <svg width="100%" viewBox={`0 0 ${pitchWidth} ${pitchHeight}`} className="pitch-svg">
          <line x1={pitchWidth / 2} y1="0" x2={pitchWidth / 2} y2={pitchHeight} stroke="#ffffff" strokeWidth="2" />
          <circle cx={pitchWidth / 2} cy={pitchHeight / 2} r="45" fill="none" stroke="#ffffff" strokeWidth="2" />
          <circle cx={pitchWidth / 2} cy={pitchHeight / 2} r="3" fill="#ffffff" />
          <rect x="0" y={(pitchHeight - 160) / 2} width="80" height="160" fill="none" stroke="#ffffff" strokeWidth="2" />
          <rect x={pitchWidth - 80} y={(pitchHeight - 160) / 2} width="80" height="160" fill="none" stroke="#ffffff" strokeWidth="2" />

          {weakZoneComputed && Array.from({ length: gridX }).map((_, gx) =>
            Array.from({ length: gridY }).map((_, gy) => {
              const zoneKey = `zone_${gx + 1}_${gy + 1}`;
              const density = weakZoneScores[zoneKey] ?? 0.0;
              const opacity = density < 0.03 ? 0.45 : density < 0.08 ? 0.2 : 0.05;
              const color = density < 0.03 ? '#d32f2f' : '#ffa000';
              return (
                <rect key={zoneKey} x={gx * cellWidth} y={gy * cellHeight} width={cellWidth} height={cellHeight}
                      fill={color} fillOpacity={opacity} stroke="rgba(255,255,255,0.2)" strokeWidth="1" />
              );
            })
          )}

          {slots.map(([xFrac, yFrac], idx) => {
            const cx = xFrac * pitchWidth;
            const cy = yFrac * pitchHeight;
            const label = (ROLE_LABELS_BY_COUNT[slots.length] || [])[idx] || `#${idx + 1}`;
            return (
              <g key={idx}>
                <circle cx={cx} cy={cy} r="12" fill="var(--accent-blue)" stroke="#ffffff" strokeWidth="2" />
                <text x={cx} y={cy + 4} textAnchor="middle" fill="#020617" fontSize="10" fontWeight="bold">
                  {label}
                </text>
              </g>
            );
          })}
        </svg>
      )}

      <div className="metric-meta" style={{ marginTop: '0.5rem' }}>
        {weakZoneComputed
          ? '* Red overlay indicates weak defensive occupancy zones (< 3% density).'
          : '* Weak-zone heatmap not shown: not yet computed for this match.'}
      </div>
    </div>
  );
}

export default function TabTeamIntelligence({ matchId }) {
  const [metrics, setMetrics] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [isDemo, setIsDemo] = useState(false);

  useEffect(() => {
    if (!matchId) {
      setMetrics([]);
      setError(null);
      return;
    }

    setLoading(true);
    setError(null);

    if (matchId === 'demo') {
      setIsDemo(true);
      mockApi.getTeamIntelligence()
        .then((data) => { setMetrics(data); setLoading(false); })
        .catch((err) => { setError(err.message); setLoading(false); });
      return;
    }

    setIsDemo(false);
    api.getTeamIntelligence(matchId)
      .then((data) => { setMetrics(data); setLoading(false); })
      .catch((err) => {
        // FIXED (Phase 3 audit): no silent mockApi fallback here either --
        // see TabPlayerIntelligence.jsx's fix note for why that matters.
        setError(err.message);
        setLoading(false);
      });
  }, [matchId]);

  if (!matchId) {
    return (
      <div>
        <h2>Team Intelligence</h2>
        <div className="card">
          <p className="metric-meta">
            No match selected yet. Upload and process a video (or load demo data) on the
            Match Analysis tab first.
          </p>
        </div>
      </div>
    );
  }

  if (loading) return <div>Loading Team Intelligence metrics...</div>;
  if (error) return <div className="status-error">Error loading team intelligence: {error}</div>;

  const formationMetric = metrics.find((m) => m.metric_name === 'formation');
  const weakZoneMetric = metrics.find((m) => m.metric_name === 'weak_zone_map');

  return (
    <div>
      <h2>Team Intelligence</h2>
      <p className="metric-meta">Collective tactical metrics, pitch formation diagram, and weak-zone heatmap.</p>

      {isDemo && (
        <div className="demo-banner">
          <span>Showing demo data — not from a real processed match.</span>
        </div>
      )}

      <PitchVisual formationMetric={formationMetric} weakZoneMetric={weakZoneMetric} />

      <h3 style={{ marginTop: '1.5rem' }}>Team Metric Breakdown</h3>

      {metrics.length === 0 && (
        <div className="card">
          <p className="metric-meta">No team metrics computed yet for this match.</p>
        </div>
      )}

      <div className="metrics-grid">
        {metrics.map((m) => (
          <div key={m.metric_id} className="metric-card">
            <div className="metric-header">
              <span className="metric-name">{m.metric_name.replace(/_/g, ' ')}</span>
              <MetricBadge confidence={m.confidence} method={m.method} />
            </div>

            <div className="metric-value">{m.value !== null && m.value !== undefined ? m.value : 'N/A'}</div>

            <div className="metric-meta">
              Conf Score: {m.confidence_score ?? 'N/A'} | Sample Size: {m.sample_size} | Schema: {m.schema_version}
            </div>

            {m.sub_scores && Object.keys(m.sub_scores).length > 0 && m.metric_name !== 'weak_zone_map' && (
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
  );
}
