import React, { useState } from 'react';
import { mockApi } from '../api/mockClient';

export default function TabCoachChat({ matchId }) {
  const [messages, setMessages] = useState([
    {
      id: 1,
      sender: 'assistant',
      text: 'Hello Coach! I am your AI Strategy Consultant. Ask me about formation stability, press resistance, or player positioning.'
    }
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSend = async (e) => {
    e.preventDefault();
    if (!input.trim() || loading) return;

    const userMsg = { id: Date.now(), sender: 'user', text: input };
    setMessages((prev) => [...prev, userMsg]);
    const queryText = input;
    setInput('');
    setLoading(true);

    try {
      const response = await mockApi.sendChatMessage(queryText);
      const assistantMsg = { id: Date.now() + 1, sender: 'assistant', text: response.text };
      setMessages((prev) => [...prev, assistantMsg]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        { id: Date.now() + 1, sender: 'assistant', text: 'Error connecting to Coach Chat assistant.' }
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <h2>Coach Chat Assistant</h2>
      <p className="metric-meta">
        Interactive tactical & player intelligence consultation.
        {matchId && matchId !== 'demo' && <> Scoped to match <code>{matchId}</code>.</>}
      </p>

      <div className="chat-box">
        <div className="messages-list">
          {messages.map((m) => (
            <div key={m.id} className={`message ${m.sender}`}>
              <strong>{m.sender === 'user' ? 'Coach' : 'AI Assistant'}:</strong>
              <div>{m.text}</div>
            </div>
          ))}
          {loading && <div className="metric-meta">AI Assistant is analyzing query...</div>}
        </div>

        <form onSubmit={handleSend} className="chat-input-form">
          <input
            type="text"
            className="chat-input"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Type your tactical question here..."
          />
          <button type="submit" className="btn" disabled={loading}>
            Send
          </button>
        </form>
      </div>
    </div>
  );
}
