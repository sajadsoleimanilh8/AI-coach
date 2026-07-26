import React, { useEffect, useState } from 'react';
import { mockApi } from '../api/mockClient';
import MetricBadge from './MetricBadge';

export default function TabTeamIntelligence() {
  const [metrics, setMetrics] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    mockApi.getTeamIntelligence()
      .then(data => {
        setMetrics(data);
        setLoading(false);
      })
      .catch(err => {
        setError(err.message);
        setLoading(false);
      });
  }, []);

  if (loading) return <div>Loading Team Intelligence metrics...</div>;
  if (error) return <div className="status-error">Error: {error}</div>;

  return (
    <div>
      <h2>Team Intelligence</h2>
      <p className="metric-meta">Collective tactical metrics and formation tracking.</p>

      <div className="metrics-grid">
        {metrics.map(m => (
          <div key={m.metric_id} className="metric-card">
            <div className="metric-header">
              <span className="metric-name">{m.metric_name}</span>
              <MetricBadge confidence={m.confidence} method={m.method} />
            </div>

            <div className="metric-value">
              {m.value}
            </div>

            <div className="metric-meta">
              Conf Score: {m.confidence_score ?? 'N/A'} | Sample Size: {m.sample_size}
            </div>

            {m.sub_scores && (
              <div className="subscores-list">
                {Object.entries(m.sub_scores).map(([k, v]) => (
                  <div key={k} className="subscore-item">
                    <span>{k}:</span>
                    <strong>{v}</strong>
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