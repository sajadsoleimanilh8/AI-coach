import React, { useState } from 'react';
import TabMatchAnalysis from './components/TabMatchAnalysis';
import TabPlayerIntelligence from './components/TabPlayerIntelligence';
import TabTeamIntelligence from './components/TabTeamIntelligence';
import TabHealthProxies from './components/TabHealthProxies';
import TabPsychologyProxies from './components/TabPsychologyProxies';
import TabCoachChat from './components/TabCoachChat';

const TABS = [
  { id: 'match', label: 'Match Analysis' },
  { id: 'player', label: 'Player Intelligence' },
  { id: 'team', label: 'Team Intelligence' },
  { id: 'health', label: 'Health Proxies' },
  { id: 'psych', label: 'Psychology Proxies' },
  { id: 'chat', label: 'Coach Chat' },
];

export default function App() {
  const [activeTab, setActiveTab] = useState('match');

  // FIXED (Phase 3 audit): Player/Team Intelligence tabs previously
  // fetched a hardcoded "match-999" no matter what video was actually
  // uploaded and processed. matchId now lives here, set once by
  // TabMatchAnalysis after a real upload (see its onMatchReady callback),
  // and is threaded down to every tab that needs to scope a query to a
  // specific match -- there is exactly one place in the app that decides
  // "which match are we looking at."
  const [matchId, setMatchId] = useState(null);
  const [activeJobId, setActiveJobId] = useState(null);

  return (
    <div className="app-container">
      <nav className="sidebar">
        <div className="sidebar-title">SportsStrategyCoachAI</div>
        {TABS.map((tab) => (
          <button
            key={tab.id}
            className={`nav-item ${activeTab === tab.id ? 'active' : ''}`}
            onClick={() => setActiveTab(tab.id)}
          >
            {tab.label}
          </button>
        ))}
        {matchId && (
          <div className="metric-meta" style={{ marginTop: '1.5rem', padding: '0 0.5rem', wordBreak: 'break-all' }}>
            Active match:<br />
            <code>{matchId}</code>
          </div>
        )}
      </nav>

      <main className="content-area">
        {activeTab === 'match' && (
          <TabMatchAnalysis
            onMatchReady={(newMatchId, jobId) => { setMatchId(newMatchId); setActiveJobId(jobId); }}
          />
        )}
        {activeTab === 'player' && <TabPlayerIntelligence matchId={matchId} />}
        {activeTab === 'team' && <TabTeamIntelligence matchId={matchId} />}
        {activeTab === 'health' && <TabHealthProxies />}
        {activeTab === 'psych' && <TabPsychologyProxies />}
        {activeTab === 'chat' && <TabCoachChat matchId={matchId} jobId={activeJobId} />}
      </main>
    </div>
  );
}
