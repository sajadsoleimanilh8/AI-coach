/**
 * Real backend API client.
 *
 * FIXED (Phase 3 audit): TabPlayerIntelligence/TabTeamIntelligence
 * previously fetched a hardcoded "match-999" and silently fell back to
 * mockApi on any fetch failure -- indistinguishable from real data in the
 * UI, which meant the match_id/pipeline-wiring bugs found in this audit
 * could pass a demo dry-run unnoticed. This client:
 *   1. Never hardcodes a match_id -- callers must pass the real one,
 *      obtained from the upload response (see TabMatchAnalysis.jsx).
 *   2. Throws on failure instead of swallowing it. Callers decide whether
 *      to show an explicit "demo mode" banner (see mockClient.js's
 *      isDemoData flag) -- silence is never the fallback.
 */

const API_BASE = import.meta.env?.VITE_API_BASE || 'http://localhost:8000';

async function request(path, options = {}) {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
    ...options,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail || detail;
    } catch {
      /* response wasn't JSON, keep statusText */
    }
    const error = new Error(`${res.status} ${detail}`);
    error.status = res.status;
    throw error;
  }
  return res.json();
}

export const api = {
  async uploadVideo(file, metadata) {
    const form = new FormData();
    form.append('file', file);
    if (metadata) form.append('metadata', JSON.stringify(metadata));
    // Don't set Content-Type manually -- browser needs to set the
    // multipart boundary itself.
    const res = await fetch(`${API_BASE}/api/videos/upload`, { method: 'POST', body: form });
    if (!res.ok) throw new Error(`Upload failed: ${res.status} ${res.statusText}`);
    return res.json(); // { video_id, job_id, match_id, filename, status, message }
  },

  getProcessingStatus(jobId) {
    return request(`/api/processing/${jobId}`);
  },

  getPipelineLatency(jobId) {
    return request(`/api/pipeline/latency/${jobId}`);
  },

  getFormation(matchId, teamId) {
    return request(`/api/tactical/formation/${matchId}${teamId ? `?team_id=${teamId}` : ''}`);
  },

  getTeamShape(matchId, teamId) {
    return request(`/api/tactical/team_shape/${matchId}${teamId ? `?team_id=${teamId}` : ''}`);
  },

  getTeamIntelligence(matchId) {
    return request(`/api/team_intelligence/${matchId}`);
  },

  getPlayerIntelligence(matchId) {
    return request(`/api/player_intelligence/${matchId}`);
  },

  getPlayerMetrics(matchId, playerId) {
    return request(`/api/player_intelligence/${matchId}/${playerId}`);
  },
};
