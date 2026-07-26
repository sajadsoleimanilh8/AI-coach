import React from 'react';

export default function TabHealthProxies() {
  return (
    <div>
      <h2>Health Proxies</h2>
      <div className="card">
        <p><strong>Coming in Phase 2</strong></p>
        <p className="metric-meta">
          Biomechanical fatigue tracking and ACWR (Acute:Chronic Workload Ratio) longitudinal models are slated for Phase 2 deployment.
        </p>
        <div style={{ marginTop: '1rem', padding: '0.75rem', background: '#0f172a', borderRadius: '4px' }}>
          <code>Mock Field: acwr_historical_status = "Pending longitudinal match accumulation"</code>
        </div>
      </div>
    </div>
  );
}