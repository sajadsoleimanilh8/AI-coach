
---

## 6. Off-Ball Movement Score (§6)
**Reads:** `PlayerTracking` (this player's trajectory) + all opponents' trajectories for the same frames, restricted to frames where this player does NOT have the ball.
**Writes:** `PlayerMetric(metric_name="off_ball_movement_score", method="heuristic_proxy", ...)`.

### Formulas
```python
# Space creation (50%)
space_creation_score = 100 * clip(safe_ratio(avg_distance_gained_from_marker, MAX_USEFUL_SEPARATION_GAIN_M, default=0.0), 0, 1)

# Attacking-third presence (30%)
attacking_presence_score = 100 * safe_ratio(off_ball_frames_in_attacking_third, total_off_ball_frames_while_team_in_possession, default=None)

# Movement efficiency (20%)
efficiency_score = 100 * safe_ratio(net_displacement_m, total_path_length_m, default=None)

value = space_creation_score * 0.50 + attacking_presence_score * 0.30 + efficiency_score * 0.20
```

---

## 7. Defensive Positioning Score (§7)
**Reads:** `PlayerTracking` for this player + teammates with `team_id == this player's team_id` (gated on `team_assignment_confidence >= 0.5`).
**Writes:** `PlayerMetric(metric_name="defensive_positioning_score", method="heuristic_proxy", ...)`.

### Formulas
```python
# Line discipline (40%)
line_discipline_score = 100 * (1 - clip(avg_deviation_from_defensive_line_m / MAX_ACCEPTABLE_LINE_DEVIATION_M, 0, 1))

# Cover / Balance Spacing (35%)
spacing_score = 100 * gaussian_score(nearest_teammate_distance_m, ideal=IDEAL_DEFENSIVE_SPACING_M, tolerance=SPACING_TOLERANCE_M)

# Reaction to Threat (25%)
reaction_score = 100 * clip(safe_ratio(closing_speed_toward_threat_ms, REACTION_SPEED_REFERENCE_MS, default=None), 0, 1)

value = line_discipline_score * 0.40 + spacing_score * 0.35 + reaction_score * 0.25
```
