let mockVideoUpload = null;
let mockProcessingJob = null;

const MOCK_TIMESTAMP = "2026-07-26T12:00:00Z";

export const mockApi = {
  /**
   * Upload video simulation matching backend/api/main.py POST /api/videos/upload
   */
  async uploadVideo(filename) {
    await new Promise((resolve) => setTimeout(resolve, 300));
    const videoId = "vid-uuid-" + Math.floor(Math.random() * 10000);
    const jobId = "job-uuid-" + Math.floor(Math.random() * 10000);

    mockVideoUpload = {
      video_id: videoId,
      job_id: jobId,
      filename: filename || "match_clip_first_half.mp4",
      status: "queued",
      message: "Video uploaded and queued for processing."
    };

    mockProcessingJob = {
      job_id: jobId,
      video_id: videoId,
      status: "queued",
      progress: 0,
      message: "Queued in queue pipeline",
      error: null,
      created_at: MOCK_TIMESTAMP,
      updated_at: MOCK_TIMESTAMP,
      started_at: null,
      completed_at: null
    };

    return { ...mockVideoUpload };
  },

  /**
   * Processing job status matching GET /api/processing/{job_id}
   */
  async getProcessingStatus(jobId) {
    await new Promise((resolve) => setTimeout(resolve, 200));
    if (!mockProcessingJob || mockProcessingJob.job_id !== jobId) {
      throw new Error("Job not found");
    }
    return { ...mockProcessingJob };
  },

  /**
   * Helper to advance processing state for mock demo
   */
  async advanceJobState(jobId, status, progress, message, error = null) {
    if (mockProcessingJob && mockProcessingJob.job_id === jobId) {
      mockProcessingJob.status = status;
      mockProcessingJob.progress = progress;
      mockProcessingJob.message = message;
      mockProcessingJob.error = error;
      mockProcessingJob.updated_at = new Date().toISOString();
      if (status === "processing" && !mockProcessingJob.started_at) {
        mockProcessingJob.started_at = new Date().toISOString();
      }
      if (status === "completed" || status === "failed") {
        mockProcessingJob.completed_at = new Date().toISOString();
      }
    }
    return this.getProcessingStatus(jobId);
  },

  /**
   * Returns PlayerMetrics matching PlayerMetric schema & Standard Output Contract
   */
  async getPlayerIntelligence() {
    await new Promise((resolve) => setTimeout(resolve, 400));
    return [
      {
        player_id: 10,
        player_name: "Alex Morgan",
        metrics: [
          {
            metric_id: "pm-101",
            match_id: "match-999",
            player_id: 10,
            metric_name: "first_touch_score",
            value: 86.5,
            method: "heuristic_proxy",
            confidence: "normal",
            sample_size: 14,
            sub_scores: {
              control: 76.0,
              retention: 100.0,
              direction: 96.6,
              speed: 82.4,
              pressure: 70.0
            },
            computed_at: MOCK_TIMESTAMP,
            schema_version: "v3"
          },
          {
            metric_id: "pm-102",
            match_id: "match-999",
            player_id: 10,
            metric_name: "press_resistance_score",
            value: 78.2,
            method: "heuristic_proxy",
            confidence: "normal",
            sample_size: 9,
            sub_scores: {
              retention: 80.0,
              escape: 75.0,
              pass_accuracy: 85.0,
              density: 70.0,
              decision_speed: 78.0
            },
            computed_at: MOCK_TIMESTAMP,
            schema_version: "v3"
          }
        ]
      },
      {
        player_id: 7,
        player_name: "Marcus Rashford",
        metrics: [
          {
            metric_id: "pm-201",
            match_id: "match-999",
            player_id: 7,
            metric_name: "press_resistance_score",
            value: null,
            method: "heuristic_proxy",
            confidence: "low_sample",
            sample_size: 2,
            sub_scores: {
              retention: 50.0,
              escape: 0.0
            },
            computed_at: MOCK_TIMESTAMP,
            schema_version: "v3"
          },
          {
            metric_id: "pm-202",
            match_id: "match-999",
            player_id: 7,
            metric_name: "injury_risk_score",
            value: 68.4,
            method: "heuristic_proxy",
            confidence: "normal",
            sample_size: 1,
            sub_scores: {
              distance_load: 85.0,
              sprint_load: 70.0,
              fatigue_index: 60.0,
              playing_time_load: 55.0
            },
            computed_at: MOCK_TIMESTAMP,
            schema_version: "v3"
          }
        ]
      },
      {
        player_id: 4,
        player_name: "Virgil van Dijk",
        metrics: [
          {
            metric_id: "pm-301",
            match_id: "match-999",
            player_id: 4,
            metric_name: "first_touch_score",
            value: 62.0,
            method: "heuristic_proxy",
            confidence: "low_upstream_confidence",
            sample_size: 11,
            sub_scores: {
              control: 60.0,
              retention: 70.0,
              direction: 55.0,
              speed: 60.0,
              pressure: 65.0
            },
            computed_at: MOCK_TIMESTAMP,
            schema_version: "v3"
          }
        ]
      }
    ];
  },

  /**
   * Returns TeamMetrics matching TeamMetric schema & Standard Output Contract
   */
  async getTeamIntelligence() {
    await new Promise((resolve) => setTimeout(resolve, 400));
    return [
      {
        metric_id: "tm-001",
        match_id: "match-999",
        team_id: "team-home",
        metric_name: "formation",
        value: "4-3-3",
        method: "deterministic",
        confidence: "normal",
        confidence_score: 0.92,
        sample_size: 300,
        sub_scores: {
          template_matching_distance: 1.42
        },
        computed_at: MOCK_TIMESTAMP,
        schema_version: "v3"
      },
      {
        metric_id: "tm-002",
        match_id: "match-999",
        team_id: "team-home",
        metric_name: "team_rating",
        value: 81.4,
        method: "ml_trained",
        confidence: "low_upstream_confidence",
        confidence_score: 0.54,
        sample_size: 450,
        sub_scores: {
          possession: 58.0,
          pass_accuracy: 82.5,
          xg_performance: 74.0,
          attack_creation: 88.0,
          defensive_stability: 79.0
        },
        computed_at: MOCK_TIMESTAMP,
        schema_version: "v3"
      }
    ];
  },

  /**
   * Canned LLM Chat Response Simulation
   */
  async sendChatMessage(message) {
    await new Promise((resolve) => setTimeout(resolve, 600));
    return {
      sender: "assistant",
      text: `Tactical Analysis: Alex Morgan showed exceptional press resistance under pressure (86.5 First Touch). Marcus Rashford's low sample size (${message.slice(0, 15)}...) limits current analysis reliability.`
    };
  }
};
