// Real API client -- talks to the actual FastAPI backend at localhost:8000.
// Replaces mockApi for Match Analysis so results reflect the ACTUAL uploaded
// video instead of a canned/static simulation.

const API_BASE = "http://localhost:8000";

export const realApi = {
  /**
   * Upload a real video File object (from <input type="file">).
   * Matches POST /api/videos/upload (multipart/form-data).
   */
  async uploadVideoFile(file) {
    const formData = new FormData();
    formData.append("file", file);
    // metadata is optional on the backend -- omit it, or send an empty JSON object
    formData.append("metadata", JSON.stringify({}));

    const res = await fetch(`${API_BASE}/api/videos/upload`, {
      method: "POST",
      body: formData,
    });

    if (!res.ok) {
      const text = await res.text().catch(() => "");
      throw new Error(`Upload failed (${res.status}): ${text}`);
    }

    return res.json(); // { video_id, job_id, filename, status, message }
  },

  /**
   * Matches GET /api/processing/{job_id}
   */
  async getProcessingStatus(jobId) {
    const res = await fetch(`${API_BASE}/api/processing/${jobId}`);
    if (!res.ok) {
      throw new Error(`Failed to fetch job status (${res.status})`);
    }
    return res.json(); // { job_id, video_id, status, progress, message, error, ... }
  },

  /**
   * Matches GET /api/processing/{job_id}/result
   */
  async getAnalysisResult(jobId) {
    const res = await fetch(`${API_BASE}/api/processing/${jobId}/result`);
    if (!res.ok) {
      throw new Error(`Failed to fetch analysis result (${res.status})`);
    }
    return res.json(); // { job_id, video_id, status, summary, result }
  },
};
