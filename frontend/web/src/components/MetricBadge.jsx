import React from 'react';

/**
 * FIXED (Phase 3 audit): the previous version only applied the
 * `badge-{confidence}` class, dropping the shared base `.badge` class
 * that supplies padding/border-radius/uppercase styling -- badges were
 * rendering as bare colored text. Also adds a `not_computed` state,
 * distinct from `low_sample`: a metric that was never attempted (e.g. no
 * pipeline run yet for this match) is a different situation from one that
 * *was* computed but on too few events, and conflating them (a radar
 * chart previously defaulted missing metrics to a `low_sample` badge) can
 * misrepresent "we haven't tried" as "we tried and weren't confident."
 */
const CONFIG = {
  normal: { badgeClass: 'badge-normal', label: 'Normal' },
  low_sample: { badgeClass: 'badge-low_sample', label: 'Low Sample' },
  low_upstream_confidence: { badgeClass: 'badge-low_upstream_confidence', label: 'Low Upstream Conf.' },
  not_computed: { badgeClass: 'badge-not_computed', label: 'Not Computed Yet' },
};

export default function MetricBadge({ confidence, method }) {
  const cfg = CONFIG[confidence] || { badgeClass: 'badge-normal', label: confidence };

  return (
    <span style={{ display: 'inline-flex', gap: '0.4rem', alignItems: 'center' }}>
      <span className={`badge ${cfg.badgeClass}`}>{cfg.label}</span>
      {method && <span className="metric-meta">({method})</span>}
    </span>
  );
}
