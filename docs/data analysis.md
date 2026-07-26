#  Analysis Logic Design 

## Changelog 
- Added a **standard output contract** every score returns — required so the API/DB schema and frontend don't each invent their own shape.
- Fixed **division-by-zero risks** in Press Resistance, Formation, and xG-adjacent formulas (all now guarded, with defined behavior at zero-sample).
- Homography and team assignment are no longer treated as binary "done/not done" — both now carry their own **confidence score** that propagates downstream, since a bad calibration is worse than no calibration (it produces confident-looking garbage).
- Added a concrete **calibration procedure** for `NORMALIZATION_CONSTANT` (was previously "start at 8.0, tune later" with no method).
- Added **unit test skeletons** per module — competition judges reward a demonstrable, tested pipeline over a bigger one.
- Added explicit mapping from every score to the DB schema fields it reads/writes.

---

## 0. Shared Constants & Dependencies

### 0.1 Blocking Dependencies (must exist before this doc's scores are valid)

| Dependency | Needed By | Status Check Before Coding |
|---|---|---|
| Camera homography (pixel → pitch meters) | First Touch, Press Resistance, Injury Risk, Formation | Distances in this doc are in **meters**. If homography isn't done, these formulas will silently run on pixel distances and produce garbage. Do not implement scoring until a homography transform function exists, even a rough one. |
| Team assignment (jersey clustering) | Press Resistance, Formation, Team Metrics | "Opponent" is meaningless while assignment is hardcoded to `neutral`. Gate these modules behind `team_id is not None`. |

### 0.2 NEW — Dependency Confidence Propagation

Treating homography and team assignment as binary gates is not enough: a homography transform can *exist* and still be badly calibrated (e.g. wrong pitch corners), and jersey clustering can *run* and still misassign players in similar-colored kits. Both failure modes look identical to a downstream consumer unless confidence is carried forward explicitly.

```python
# Every homography transform returns, not just coordinates:
{
    "pitch_x_m": float,
    "pitch_y_m": float,
    "homography_confidence": float,  # 0-1, from reprojection error on calibration points
}

# Every team assignment returns:
{
    "team_id": str | None,
    "team_assignment_confidence": float,  # 0-1, from cluster separation (e.g. silhouette score)
}
```

**Rule:** any score consuming these must take the **minimum** of its own confidence and the upstream confidence (see 0.3 output contract). Don't silently drop this information — a Press Resistance score built on 0.55-confidence team assignment should visibly say so, not present as equal to one built on 0.95-confidence assignment.

Practical thresholds to start with (tune after seeing real calibration data):
- `homography_confidence < 0.6` → treat pitch coordinates as unusable, skip scoring for that frame
- `team_assignment_confidence < 0.5` → treat `team_id` as `None` for that player/frame, same as unassigned

### 0.3 NEW — Standard Output Contract

Every score/module in this document returns this shape. This is what gets written to `PlayerMetric` / `TeamMetric` and what the API serves — decide this now so B's schema and D's frontend don't each guess independently.

```python
{
    "metric_name": str,              # e.g. "first_touch_score"
    "value": float,                  # 0-100
    "method": str,                   # "ml_trained" | "deterministic" | "heuristic_proxy"
    "confidence": str,               # "normal" | "low_sample" | "low_upstream_confidence"
    "sample_size": int,              # raw event count behind this score
    "sub_scores": dict[str, float],  # component breakdown, e.g. {"control": 76.0, "retention": 100.0, ...}
    "computed_at": datetime,
    "schema_version": str,           # e.g. "v3" — bump when formulas change, so old scores aren't
                                      # silently compared against new-formula scores
}
```

`confidence` resolution order (most severe wins): `low_upstream_confidence` (homography/team assignment below threshold) > `low_sample` (event count < `MIN_SAMPLE_EVENTS`) > `normal`.

### 0.4 Shared Constants (put in one `constants.py`, import everywhere — do not hardcode locally per module)

```python
PRESSURE_RADIUS_M = 5.0          # opponent within this = "applying pressure"
PRESSURE_THRESHOLD = 0.5         # pressure_level above this = "under pressure" (binary events)
RETENTION_WINDOW_S = 3.0         # min time to count possession as "retained"
TOUCH_EVAL_WINDOW_S = 2.0        # window after first touch to evaluate outcome
MAX_TOUCH_DISTANCE_M = 5.0       # ball travel beyond this after touch = poor control
DECISION_TIME_MIN_S = 0.3        # fastest realistic decision
DECISION_TIME_MAX_S = 2.0        # slowest before treated as "too slow"
MIN_SAMPLE_EVENTS = 5            # below this, flag score as "low_confidence"
HOMOGRAPHY_CONFIDENCE_MIN = 0.6  # below this, pitch coordinates are unusable
TEAM_ASSIGNMENT_CONFIDENCE_MIN = 0.5  # below this, treat team_id as unassigned
SCHEMA_VERSION = "v3"
```

### 0.5 Pressure Level (used by First Touch, Press Resistance)

```python
def pressure_level(distance_to_nearest_opponent_m: float | None) -> float | None:
    if distance_to_nearest_opponent_m is None:
        return None  # no opponent tracked nearby — not the same as zero pressure, don't conflate
    return clip(1 - distance_to_nearest_opponent_m / PRESSURE_RADIUS_M, 0, 1)
```
0 = no pressure, 1 = opponent on top of the player. **Fixed from v2:** returns `None` (not 0) when no opponent is trackable, so "confirmed no pressure" and "unknown" aren't conflated — this matters for the missing-data policy below.

### 0.6 Missing-Data Policy (applies to every score below)

- If a required tracking point is missing (occlusion, out-of-frame) for a given event, **drop that single event** from the aggregate, don't impute a fake value.
- If total valid events for a player in a match < `MIN_SAMPLE_EVENTS`, return the score with `confidence: "low_sample"` and the raw event count, rather than presenting it as equally reliable as a 30-event score.
- **NEW:** if the denominator of any ratio (see 0.7) would be zero, return the score as `null` with `confidence: "low_sample"` and `sample_size: 0` — never divide by zero, and never substitute a default value that could be misread as a real 0/100 score.

### 0.7 NEW — Zero-Denominator Guard (apply to every ratio formula in this doc)

```python
def safe_ratio(numerator: float, denominator: float, default=None) -> float | None:
    if denominator == 0:
        return default  # propagates as null, not 0 — a score of 0 means "measured and bad",
                         # not "couldn't measure"
    return numerator / denominator
```
This directly fixes silent-failure points in v2: `possessions_retained / possessions_under_pressure`, `successful_escapes / pressure_events`, `completed_passes_under_pressure / attempted_passes_under_pressure`, and `sum(success_i * density_i) / sum(density_i)` all divide by a count that can legitimately be zero (e.g. a player who was never pressured in the match). Wrap every one of these in `safe_ratio`.

---

## 1. First Touch Score

**Reads:** `Event` (first_touch type, ball positions in eval window, player position, nearest opponent position, pressure_level, touch_execution_time), all in pitch-meter coordinates.
**Writes:** `PlayerMetric(metric_name="first_touch_score", ...)` per the output contract in 0.3.

### Subscore Formulas

**Control (30%)** — how little the ball travels after the touch, relative to a bad-touch baseline:
```python
touch_distance = ball_travel_distance(from=first_touch_moment, window=TOUCH_EVAL_WINDOW_S)
control_score = 100 * (1 - min(touch_distance / MAX_TOUCH_DISTANCE_M, 1))
```

**Retention (25%)** — did the player keep the ball for the retention window without a turnover:
```python
if turnover_within(RETENTION_WINDOW_S):
    retention_score = 100 * (time_to_turnover / RETENTION_WINDOW_S)
else:
    retention_score = 100
```

**Direction (20%)** — touch direction weighted toward open space and forward progress:
```python
# FIXED from v2: guard the case where nearest opponent is at distance 0
# (unit_vector of a zero vector is undefined — this happens on tackles/50-50s)
if distance_to_nearest_opponent_m is not None and distance_to_nearest_opponent_m > 0.1:
    away_from_pressure = unit_vector(player_pos - nearest_opponent_pos)
else:
    away_from_pressure = forward  # fall back to pure forward-progress when no clear "away" direction exists

forward = unit_vector(attacking_direction)
optimal = normalize(0.5 * away_from_pressure + 0.5 * forward)
direction_score = 100 * max(0, cos_similarity(touch_direction, optimal))
```

**Speed (15%)** — execution time relative to calibrated bounds:
```python
speed_score = 100 * clip(
    (DECISION_TIME_MAX_S - touch_execution_time) / (DECISION_TIME_MAX_S - DECISION_TIME_MIN_S),
    0, 1)
```

**Pressure Handling (10%)** — only meaningful if pressure was actually present:
```python
p = pressure_level(distance_to_nearest_opponent)
if p is None:
    pressure_score = 70   # NEW: no opponent trackable — same neutral treatment as low pressure,
                           # but don't conflate with "confirmed p=0"
elif p < 0.2:
    pressure_score = 70   # neutral baseline — don't reward/penalize uncontested touches
else:
    pressure_score = 100 * control_score/100 * p
```

### Final Formula (unchanged)
```
First Touch Score = Control×0.30 + Retention×0.25 + Direction×0.20 + Speed×0.15 + Pressure×0.10
```
Confidence resolves per 0.3: if `homography_confidence < HOMOGRAPHY_CONFIDENCE_MIN` for this event, drop it from the aggregate entirely (per 0.6), don't just flag it.

### Worked Example
Player receives a pass under light pressure (opponent 4m away → p=0.2), touch_distance=1.2m, no turnover in 3s, touch angled 15° off optimal, executed in 0.6s:
```
Control    = 100*(1 - 1.2/5.0)      = 76.0
Retention  = 100                     = 100.0
Direction  = 100*cos(15°)            = 96.6
Speed      = 100*(2.0-0.6)/(2.0-0.3) = 82.4
Pressure   = 70 (p=0.2, at boundary → neutral)

Score = 76.0*.30 + 100*.25 + 96.6*.20 + 82.4*.15 + 70*.10
      = 22.8 + 25.0 + 19.3 + 12.4 + 7.0 = 86.5 → "Excellent"
```

### Unit Test Skeleton
```python
def test_first_touch_zero_distance_opponent():
    """50-50 ball, opponent at distance 0 — must not crash on unit_vector(zero)."""
    result = score_first_touch(distance_to_nearest_opponent_m=0.0, ...)
    assert result["value"] is not None

def test_first_touch_no_opponent_tracked():
    """Opponent occluded — pressure must resolve to neutral 70, not error or fake 0."""
    result = score_first_touch(distance_to_nearest_opponent_m=None, ...)
    assert result["sub_scores"]["pressure"] == 70

def test_first_touch_low_confidence_homography_dropped():
    """Event with homography_confidence below threshold must not enter the aggregate."""
    events = [make_event(homography_confidence=0.4)]
    result = score_first_touch(events)
    assert result["sample_size"] == 0
    assert result["confidence"] == "low_sample"
```

---

## 2. Press Resistance Score

**Reads:** `Event` (possession, pressure, pass outcome types) joined with `team_id` on both the possessing player and nearby opponents.
**Writes:** `PlayerMetric(metric_name="press_resistance_score", ...)`.

Requires team assignment — gate on `team_assignment_confidence >= TEAM_ASSIGNMENT_CONFIDENCE_MIN` for every player involved, not just "team_id is not None" (see 0.2). Only evaluated on possessions where `pressure_level >= PRESSURE_THRESHOLD`.

**Retention (30%)** — over all pressured possessions in the match:
```python
retention_score_raw = safe_ratio(possessions_retained, possessions_under_pressure)
retention_score = 100 * retention_score_raw if retention_score_raw is not None else None
```

**Escape Success (25%)** — ball carried/passed outside the opponent's pressure radius without a turnover:
```python
escape_score_raw = safe_ratio(successful_escapes, pressure_events)
escape_score = 100 * escape_score_raw if escape_score_raw is not None else None
```

**Pass Accuracy Under Pressure (20%)**:
```python
pass_accuracy_raw = safe_ratio(completed_passes_under_pressure, attempted_passes_under_pressure)
pass_accuracy_score = 100 * pass_accuracy_raw if pass_accuracy_raw is not None else None
```

**Density Handling (15%)** — weight each pressured event by how many opponents were within radius, so beating a double-team counts more:
```python
opponent_density = count(opponents within PRESSURE_RADIUS_M)
density_score_raw = safe_ratio(sum(success_i * density_i), sum(density_i))
density_score = 100 * density_score_raw if density_score_raw is not None else None
```

**Decision Speed (10%)**:
```python
decision_speed_score = 100 * clip(
    (DECISION_TIME_MAX_S - avg_decision_time) / (DECISION_TIME_MAX_S - DECISION_TIME_MIN_S),
    0, 1)
```

### Final Formula
```python
# FIXED from v2: if any subscore is None (zero-denominator case), redistribute its weight
# proportionally across the remaining subscores rather than treating None as 0 — a player
# with zero pressured possessions shouldn't score 0 on Press Resistance, they should have
# no score at all (see aggregate rule below).
components = {
    "retention": (retention_score, 0.30),
    "escape": (escape_score, 0.25),
    "pass_accuracy": (pass_accuracy_score, 0.20),
    "density": (density_score, 0.15),
    "decision_speed": (decision_speed_score, 0.10),
}
valid = {k: (v, w) for k, (v, w) in components.items() if v is not None}
if not valid or pressure_events < MIN_SAMPLE_EVENTS:
    final_score = None  # "not enough pressured possessions to score this player"
else:
    weight_sum = sum(w for _, w in valid.values())
    final_score = sum(v * w for v, w in valid.values()) / weight_sum
```
If `pressure_events < MIN_SAMPLE_EVENTS`, tag `confidence: "low_sample"` regardless of whether a numeric score was produced.

### Unit Test Skeleton
```python
def test_press_resistance_never_pressured():
    """Player with zero pressured possessions must return None, not 0."""
    result = score_press_resistance(possessions_under_pressure=0, ...)
    assert result["value"] is None
    assert result["confidence"] == "low_sample"

def test_press_resistance_low_team_assignment_confidence():
    """Player/opponent team assignment below confidence threshold must exclude those events."""
    ...
```

---

## 3. Formation Detection

**Reads:** `TrackingFrame` (all player positions in a rolling window), `team_id` per player.
**Writes:** `TeamMetric(metric_name="formation", value=<template_name>, ...)` plus `confidence` as a float in `sub_scores`.

### Similarity / Confidence Formula
```python
# Hungarian algorithm assigns each player to the nearest template slot (not greedy —
# greedy nearest-neighbor assignment produces wrong formations when two players are
# near two overlapping slots)
assignment = hungarian_match(avg_player_positions, template_slots)
distance_score = sum(euclidean_distance(p, slot) for p, slot in assignment)
confidence = 100 * exp(-distance_score / (n_players * NORMALIZATION_CONSTANT))
```
Run this for every candidate formation template; select highest `confidence`.

### NEW — Calibration Procedure for `NORMALIZATION_CONSTANT`
v2 said "start at 8.0, tune against real match clips" without a method. Concrete procedure:
1. Take 5-10 clips where the formation is **known from broadcast/manual labeling** (e.g. a settled 4-3-3 in open play, no transition).
2. Run the matcher against the correct template and record `distance_score` for each — this is your "should score ~95-100" reference set.
3. Run the matcher against a **deliberately wrong** template (e.g. force-match a 4-3-3 clip against a 3-5-2 template) — this is your "should score ~10-20" reference set.
4. Solve for `NORMALIZATION_CONSTANT` such that `exp(-distance_score_correct / (n*C)) ≈ 0.95-1.0` and `exp(-distance_score_wrong / (n*C)) ≈ 0.10-0.20` simultaneously. If no single constant satisfies both bounds well, the template set itself is too ambiguous (templates too geometrically similar) — revisit templates before re-tuning the constant.
5. Store the calibrated value in `constants.py` with a comment recording which clips were used, so it's reproducible when the team revisits it.

### Missing-Data Handling
- Formation should be computed over a **rolling window** (e.g., last 5 minutes of possession, not a single frame) — single-frame snapshots catch transient chaos (corners, transitions), not settled shape.
- If fewer than 8 outfield players are tracked in the window, extend the window rather than computing on incomplete data; flag `low_sample` if still incomplete after 10 minutes.
- **NEW:** if `team_assignment_confidence` is below threshold for more than 2 players in the window, don't attempt formation detection for that window at all — return `confidence: "low_upstream_confidence"` rather than a formation computed on a partially-wrong roster.

### Unit Test Skeleton
```python
def test_formation_correct_template_scores_high():
    """Known 4-3-3 clip should score >90 confidence against the 4-3-3 template."""
    ...

def test_formation_insufficient_players_extends_window():
    """<8 tracked players should extend the window before flagging low_sample."""
    ...
```

---

## 4. Injury Risk — Workload-Based Proxy (not ML)

**Important for judges/docs: label this explicitly as a heuristic sports-science proxy, not a trained predictive model.** No injury-labeled dataset exists for this competition; presenting it otherwise is the fastest way to lose credibility with technical judges.

**Reads:** `PlayerMetric` (historical distance/sprint data if available), `Event` (movement/speed within match).
**Writes:** `PlayerMetric(metric_name="injury_risk_score", method="heuristic_proxy", ...)`.

### Grounding: Acute:Chronic Workload Ratio (ACWR)
This is a real, published sports-science metric (Gabbett et al.), not invented for this project.
```python
acute_workload  = avg_distance_covered(last_7_days)      # or last N matches if days unavailable
chronic_workload = avg_distance_covered(last_28_days)
ACWR = safe_ratio(acute_workload, chronic_workload)  # FIXED: guard chronic_workload == 0
                                                       # (new player, first matches on record)

if ACWR is None:
    # fall through to MVP single-match fallback below — cannot compute ACWR without history
    use_fallback = True
elif 0.8 <= ACWR <= 1.3:
    acwr_risk = 10
elif ACWR > 1.3:
    acwr_risk = clip(10 + (ACWR - 1.3) * 150, 10, 100)
else:  # ACWR < 0.8
    acwr_risk = clip(10 + (0.8 - ACWR) * 100, 10, 100)
```
Sweet spot 0.8–1.3 (low risk). Risk rises sharply above 1.5 or below 0.8 (undertraining also raises injury risk — worth keeping since it's counter-intuitive and judges will ask about it if you only model the high end).

### MVP Fallback (competition timeline won't have 28 days of history per player)
```python
distance_load     = clip(safe_ratio(distance_covered_match, player_season_avg_distance, default=1.0), 0, 1.5) * 100
sprint_load        = clip(safe_ratio(sprint_count_match, player_season_avg_sprints, default=1.0), 0, 1.5) * 100
fatigue_index      = 100 * (1 - safe_ratio(late_match_speed, early_match_speed, default=1.0))
playing_time_load  = clip(minutes_played / 90, 0, 1) * 100
# FIXED: safe_ratio default=1.0 here (not None) because a missing season average should
# read as "assume typical load", not block the score entirely — this ratio is a fallback
# heuristic already, so failing soft is the right behavior, unlike the strict scores above.

risk_score = (distance_load*0.30 + sprint_load*0.25 + fatigue_index*0.25 + playing_time_load*0.20)
```
State clearly in documentation: **"Full ACWR model requires multi-match history; MVP uses single-match workload heuristic as a documented placeholder, upgraded to ACWR in Phase 2 once longitudinal tracking accumulates."**

Output: `Risk Score (0-100)`, Level: Low (<30) / Medium (30-60) / High (60-80) / Critical (80+).

### Unit Test Skeleton
```python
def test_injury_risk_new_player_no_history_falls_back():
    """Player with zero chronic_workload history must use MVP fallback, not crash."""
    result = score_injury_risk(chronic_workload=0, ...)
    assert result["method"] == "heuristic_proxy"
    assert "acwr" not in result["sub_scores"]  # fallback path only
```

---

## 5. Team Performance Metrics

**Reads:** `Event` (shots, passes, possessions), `TeamMetric` history.
**Writes:** `TeamMetric(metric_name="team_rating", ...)`.

### xG — Trainable in Your Timeline
Don't treat xG as a black box dependency — it's a small, well-scoped model you can train in under a day using **public open data** (e.g. StatsBomb open data has labeled shot events with outcomes).
```python
features = [
    distance_to_goal_m,
    angle_to_goal_rad,
    is_header,             # 0/1
    defenders_in_cone,     # count of opponents between shot location and goal
    is_fast_break          # 0/1, shot within 5s of a turnover
]
xg = sigmoid(w · features + b)   # trained via logistic regression on StatsBomb open data
```
This is genuinely the same class of model professional analytics providers use for baseline xG — cite this in your competition write-up as evidence of research-grounded design.

**NEW — training/serving skew guard:** the 5 features above must be computed identically at training time (from StatsBomb's coordinate system) and at inference time (from your own homography output, in your own pitch-meter coordinate system). Write one shared `compute_xg_features()` function used by both the training script and the live pipeline — do not reimplement feature extraction twice. This is a common, easy-to-miss source of silent accuracy loss.

### Team Rating (unchanged formula, now with defined xG source)
```
Team Rating = Possession×0.20 + Pass Accuracy×0.25 + xG Performance×0.25
             + Attack Creation×0.15 + Defensive Stability×0.15
```
`xG Performance = actual_goals - expected_goals` (normalized to 0-100 via a sigmoid centered at 0).

### Unit Test Skeleton
```python
def test_xg_features_match_between_training_and_inference():
    """Same shot event run through compute_xg_features() at train vs. serve time
    must produce identical feature vectors — catches coordinate-system drift."""
    ...
```

---

## Data Pipeline Connection (updated, with DB schema mapping)

| Module | Required Data | Blocking Dependency | Writes To |
|---|---|---|---|
| First Touch Score | Ball Tracking + Player Tracking + Homography | Homography | `PlayerMetric` |
| Press Resistance | Player Tracking + Team Assignment + Homography | Homography, Team Assignment | `PlayerMetric` |
| Formation Detection | Player Positions + Homography | Homography, Team Assignment | `TeamMetric` |
| Injury Risk (MVP) | Movement Tracking + Homography | Homography | `PlayerMetric` |
| Injury Risk (Phase 2, ACWR) | Multi-match historical workload | Match history accumulation | `PlayerMetric` |
| Team Metrics / xG | Event Detection + Shot Data + Public xG training set | None — trainable now | `TeamMetric` |

All writes use the **standard output contract (0.3)** — this is the row shape `PlayerMetric` and `TeamMetric` need to support: a numeric `value`, a `method` enum, a `confidence` enum, `sample_size`, a `sub_scores` JSON blob, and `schema_version`.

---

## Presentation Note for Judges
Explicitly separate, in your slides/report, which scores are:
1. **ML-trained** (xG via logistic regression on real data)
2. **Deterministic/geometric** (Formation Detection, similarity confidence)
3. **Heuristic proxy, clearly labeled as such** (First Touch, Press Resistance, Injury Risk MVP)

This matches your own transparency-with-judges principle and pre-empts the most likely technical pushback you'd otherwise get live. The `method` field in the standard output contract (0.3) makes this distinction queryable and displayable in the dashboard automatically, rather than something you have to remember to mention.
