import React, { useState } from 'react';
import { mockApi } from '../api/mockClient';

export default function TabCoachChat() {
  const [messages, setMessages] = useState([
    { sender: 'assistant', text: 'Ask me about tactics, player performance, or match analysis.' }
  ]);
  const [input, setInput] = useState('');
  const [sending, setSending] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    const text = input.trim();
    if (!text || sending) return;

    setMessages(prev => [...prev, { sender: 'user', text }]);
    setInput('');
    setSending(true);

    try {
      const reply = await mockApi.sendChatMessage(text);
      setMessages(prev => [...prev, reply]);
    } catch (err) {
      setMessages(prev => [...prev, { sender: 'assistant', text: `Error: ${err.message}` }]);
    } finally {
      setSending(false);
    }
  };

  return (
    <div>
      <h2>Coach Chat</h2>
      <p className="metric-meta">Ask the AI coach about tactics, players, and match events.</p>

      <div className="chat-box">
        <div className="messages-list">
          {messages.map((m, i) => (
            <div key={i} className={`message ${m.sender}`}>
              {m.text}
            </div>
          ))}
          {sending && <div className="message assistant">Thinking...</div>}
        </div>

        <form className="chat-input-form" onSubmit={handleSubmit}>
          <input
            className="chat-input"
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask about a player, tactic, or match event..."
            disabled={sending}
          />
          <button className="btn" type="submit" disabled={sending || !input.trim()}>
            Send
          </button>
        </form>
      </div>
    </div>
  );
}