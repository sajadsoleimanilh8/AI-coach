import React from 'react';

export default function MetricBadge({ confidence, method }) {
  let badgeClass = 'badge-normal';
  let label = confidence;

  if (confidence === 'low_sample') {
    badgeClass = 'badge-low_sample';
    label = 'Low Sample';
  } else if (confidence === 'low_upstream_confidence') {
    badgeClass = 'badge-low_upstream_confidence';
    label = 'Low Upstream Conf.';
  } else if (confidence === 'normal') {
    label = 'Normal';
  }

  return (
    <div style={{ display: 'inline-flex', gap: '0.4rem', alignItems: 'center' }}>
      <span className={`badge ${badgeClass}`}>{label}</span>
      <span className="metric-meta">({method})</span>
    </div>
  );
}
