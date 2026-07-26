import React, { useEffect, useState } from 'react';
import { mockApi } from '../api/mockClient';
import MetricBadge from './MetricBadge';

export default function TabPlayerIntelligence() {
  const [players, setPlayers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    mockApi.getPlayerIntelligence()
      .then(data => {
        setPlayers(data);
        setLoading(false);
      })
      .catch(err => {
        setError(err.message);
        setLoading(false);
      });
  }, []);

  if (loading) return <div>Loading Player Intelligence metrics...</div>;
  if (error) return <div className="status-error">Error: {error}</div>;

  return (
    <div>
      <h2>Player Intelligence</h2>
      <p className="metric-meta">Individual player metrics output using schema contract v3.</p>

      {players.map(player => (
        <div key={player.player_id} className="card">
          <h3 className="card-title">{player.player_name} (ID: #{player.player_id})</h3>
          
          <div className="metrics-grid">
            {player.metrics.map(m => (
              <div key={m.metric_id} className="metric-card">
                <div className="metric-header">
                  <span className="metric-name">{m.metric_name}</span>
                  <MetricBadge confidence={m.confidence} method={m.method} />
                </div>

                <div className="metric-value">
                  {m.value !== null ? m.value : 'N/A (Null)'}
                </div>

                <div className="metric-meta">
                  Sample Size: {m.sample_size} | Schema: {m.schema_version}
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
      ))}
    </div>
  );
}