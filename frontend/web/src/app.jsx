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
  { id: 'chat', label: 'Coach Chat' }
];

export default function App() {
  const [activeTab, setActiveTab] = useState('match');

  return (
    <div className="app-container">
      <nav className="sidebar">
        <div className="sidebar-title">SportsStrategyCoachAI</div>
        {TABS.map(tab => (
          <button
            key={tab.id}
            className={`nav-item ${activeTab === tab.id ? 'active' : ''}`}
            onClick={() => setActiveTab(tab.id)}
          >
            {tab.label}
          </button>
        ))}
      </nav>

      <main className="content-area">
        {activeTab === 'match' && <TabMatchAnalysis />}
        {activeTab === 'player' && <TabPlayerIntelligence />}
        {activeTab === 'team' && <TabTeamIntelligence />}
        {activeTab === 'health' && <TabHealthProxies />}
        {activeTab === 'psych' && <TabPsychologyProxies />}
        {activeTab === 'chat' && <TabCoachChat />}
      </main>
    </div>
  );
}