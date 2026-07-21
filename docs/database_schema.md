# Football Analysis Database Schema

## Overview

This document defines the core data structures used by the football analysis system.

It connects:
- Match data
- Video frames
- Player and ball detection
- Tracking information
- Events
- Player and team metrics

---

## Match

General match information.

Fields:
- match_id: UUID
- home_team: String
- away_team: String
- video_path: String
- duration: Float

---

## Frame

Processed video frames.

Fields:
- frame_id: Integer
- match_id: UUID
- frame_number: Integer
- timestamp: Float

---

## PlayerDetection

YOLO detection outputs.

Fields:
- detection_id: UUID
- frame_id: Integer
- player_id: Integer
- x: Float
- y: Float
- width: Float
- height: Float
- confidence: Float

---

## BallDetection

Ball detection outputs.

Fields:
- frame_id: Integer
- ball_x: Float
- ball_y: Float
- confidence: Float

---

## PlayerTracking

Player movement trajectories.

Fields:
- player_id: Integer
- frame_id: Integer
- pitch_x_meter: Float
- pitch_y_meter: Float
- speed: Float
- distance: Float

---

## Event

Detected football events.

Examples:
- pass
- shot
- touch
- possession change

Fields:
- event_id: UUID
- player_id: Integer
- event_type: String
- timestamp: Float
- metadata: JSON

---

## PlayerMetric

Analysis engine outputs.

Fields:
- player_id: Integer
- metric_name: String
- value: Float
- method: String
- confidence: String
- sub_scores: JSON

Examples:
- first_touch_score
- press_resistance
- injury_risk

---

## TeamMetric

Team-level analysis outputs.

Fields:
- team_id: Integer
- metric_name: String
- value: Float
- confidence: Float

Examples:
- formation
- possession
- team_rating

---

## Relationships

Match  
→ Frame  
→ PlayerDetection  
→ BallDetection  
→ PlayerTracking  
→ Event  
→ PlayerMetric / TeamMetric