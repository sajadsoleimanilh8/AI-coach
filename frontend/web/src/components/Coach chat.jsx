import React, { useState } from 'react';
import { mockApi } from '../api/mockClient';

export default function TabCoachChat() {
  const [messages, setMessages] = useState([
    { sender: 'assistant', text: 'Hello Coach! Ask me anything about player performance, press resistance, or tactical metrics.' }
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;

    const userMsg = input;
    setInput('');
    setMessages(prev => [...prev, { sender: 'user', text: userMsg }]);
    setLoading(true);

    try {
      const response = await mockApi.sendChatMessage(userMsg);
      setMessages(prev => [...prev, response]);
    } catch (err) {
      setMessages(prev => [...prev, { sender: 'assistant', text: 'Error connecting to Coach AI.' }]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <h2>Coach Chat AI</h2>
      <div className="card chat-box">
        <div className="messages-list">
          {messages.map((m, idx) => (
            <div key={idx} className={`message ${m.sender}`}>
              {m.text}
            </div>
          ))}
          {loading && <div className="message assistant">Thinking...</div>}
        </div>

        <form onSubmit={handleSubmit} className="chat-input-form">
          <input
            className="chat-input"
            type="text"
            placeholder="Type tactical question..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
          />
          <button type="submit" className="btn" disabled={loading}>Send</button>
        </form>
      </div>
    </div>
  );
}