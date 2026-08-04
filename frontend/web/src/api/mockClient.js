let mockVideoUpload = null;
let mockProcessingJob = null;
const MOCK_TIMESTAMP = "2026-07-26T12:00:00Z";

export const mockApi = {
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
      message: "Queued in processing pipeline",
      error: null,
      created_at: MOCK_TIMESTAMP,
      updated_at: MOCK_TIMESTAMP,
      started_at: null,
      completed_at: null
    };
    return { ...mockVideoUpload };
  },

  async getProcessingStatus(jobId) {
    await new Promise((resolve) => setTimeout(resolve, 200));
    if (!mockProcessingJob || mockProcessingJob.job_id !== jobId) {
      throw new Error("Job not found");
    }
    return { ...mockProcessingJob };
  },

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
            sub_scores: { control: 76.0, retention: 100.0, direction: 96.6, speed: 82.4, pressure: 70.0 },
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
            sub_scores: { retention: 80.0, escape: 75.0, pass_accuracy: 85.0, density: 70.0, decision_speed: 78.0 },
            computed_at: MOCK_TIMESTAMP,
            schema_version: "v3"
          },
          {
            metric_id: "pm-103",
            match_id: "match-999",
            player_id: 10,
            metric_name: "off_ball_movement_score",
            value: 82.0,
            method: "heuristic_proxy",
            confidence: "normal",
            sample_size: 12,
            sub_scores: { space_creation: 85.0, attacking_presence: 80.0, efficiency: 78.0 },
            computed_at: MOCK_TIMESTAMP,
            schema_version: "v3"
          },
          {
            metric_id: "pm-104",
            match_id: "match-999",
            player_id: 10,
            metric_name: "defensive_positioning_score",
            value: 74.5,
            method: "heuristic_proxy",
            confidence: "low_upstream_confidence",
            sample_size: 8,
            sub_scores: { line_discipline: 70.0, spacing: 75.0, reaction: 80.0 },
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
            metric_name: "first_touch_score",
            value: 71.0,
            method: "heuristic_proxy",
            confidence: "normal",
            sample_size: 10,
            sub_scores: { control: 70.0, retention: 72.0, direction: 75.0, speed: 68.0, pressure: 70.0 },
            computed_at: MOCK_TIMESTAMP,
            schema_version: "v3"
          },
          {
            metric_id: "pm-202",
            match_id: "match-999",
            player_id: 7,
            metric_name: "press_resistance_score",
            value: null,
            method: "heuristic_proxy",
            confidence: "low_sample",
            sample_size: 2,
            sub_scores: { retention: 50.0, escape: 0.0 },
            computed_at: MOCK_TIMESTAMP,
            schema_version: "v3"
          },
          {
            metric_id: "pm-203",
            match_id: "match-999",
            player_id: 7,
            metric_name: "off_ball_movement_score",
            value: 88.0,
            method: "heuristic_proxy",
            confidence: "normal",
            sample_size: 15,
            sub_scores: { space_creation: 90.0, attacking_presence: 88.0, efficiency: 84.0 },
            computed_at: MOCK_TIMESTAMP,
            schema_version: "v3"
          },
          {
            metric_id: "pm-204",
            match_id: "match-999",
            player_id: 7,
            metric_name: "defensive_positioning_score",
            value: 61.0,
            method: "heuristic_proxy",
            confidence: "low_upstream_confidence",
            sample_size: 6,
            sub_scores: { line_discipline: 55.0, spacing: 65.0, reaction: 65.0 },
            computed_at: MOCK_TIMESTAMP,
            schema_version: "v3"
          }
        ]
      }
    ];
  },

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
        sub_scores: { template_matching_distance: 1.42, matching_confidence_pct: 92.0 },
        computed_at: MOCK_TIMESTAMP,
        schema_version: "v3"
      },
      {
        metric_id: "tm-002",
        match_id: "match-999",
        team_id: "team-home",
        metric_name: "compactness_score",
        value: 78.4,
        method: "deterministic",
        confidence: "normal",
        confidence_score: 0.85,
        sample_size: 300,
        sub_scores: { hull_area_m2: 480.5, raw_compactness: 78.4 },
        computed_at: MOCK_TIMESTAMP,
        schema_version: "v3"
      },
      {
        metric_id: "tm-003",
        match_id: "match-999",
        team_id: "team-home",
        metric_name: "formation_stability_score",
        value: 84.2,
        method: "deterministic",
        confidence: "normal",
        confidence_score: 0.88,
        sample_size: 300,
        sub_scores: { position_variance_m2: 3.95, raw_stability: 84.2 },
        computed_at: MOCK_TIMESTAMP,
        schema_version: "v3"
      },
      {
        metric_id: "tm-004",
        match_id: "match-999",
        team_id: "team-home",
        metric_name: "pressing_intensity_score",
        value: 68.0,
        method: "deterministic",
        confidence: "normal",
        confidence_score: 0.75,
        sample_size: 150,
        sub_scores: { mean_pressure_level: 0.68, raw_intensity: 68.0 },
        computed_at: MOCK_TIMESTAMP,
        schema_version: "v3"
      },
      {
        metric_id: "tm-005",
        match_id: "match-999",
        team_id: "team-home",
        metric_name: "weak_zone_map",
        value: 4.0,
        method: "deterministic",
        confidence: "normal",
        confidence_score: 0.80,
        sample_size: 300,
        sub_scores: {
          zone_1_1: 0.15, zone_1_2: 0.02, zone_1_3: 0.01, zone_1_4: 0.12,
          zone_2_1: 0.08, zone_2_2: 0.04, zone_2_3: 0.03, zone_2_4: 0.07,
          zone_3_1: 0.05, zone_3_2: 0.02, zone_3_3: 0.02, zone_3_4: 0.04,
          zone_4_1: 0.03, zone_4_2: 0.01, zone_4_3: 0.01, zone_4_4: 0.03,
          zone_5_1: 0.02, zone_5_2: 0.01, zone_5_3: 0.01, zone_5_4: 0.02,
          zone_6_1: 0.01, zone_6_2: 0.00, zone_6_3: 0.00, zone_6_4: 0.01
        },
        computed_at: MOCK_TIMESTAMP,
        schema_version: "v3"
      }
    ];
  },

  async sendChatMessage(message) {
    await new Promise((resolve) => setTimeout(resolve, 600));
    return {
      sender: "assistant",
      text: `Tactical Analysis: Alex Morgan showed exceptional press resistance and first touch scores (86.5 First Touch). Marcus Rashford's low sample size for press resistance limits current analysis reliability.`
    };
  }
};
