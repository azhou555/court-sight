# Court Sight — Project Plan

## Vision

Build a computer vision system that watches tennis match footage and provides
meaningful, coaching-grade feedback on shot construction — not just what shot
was hit, but whether it was the right shot given the game state, the player's
physical situation, and the risk/reward tradeoff of the attempt.

The core question the system answers: **was this shot a good decision, and why?**

---

## What This Is Not

- Not a line-calling system
- Not a speed/spin tracker
- Not a live scoring app
- Not a pro-only tool

The focus is **shot quality feedback** for recreational and competitive amateur
players who want to understand the tactical and strategic layer of their game.

---

## System Overview

The system operates in two phases:

1. **Data Pipeline** — extract structured match data from broadcast footage
2. **Feedback Engine** — model shot quality and surface actionable recommendations

These are developed in parallel but the data pipeline is a prerequisite for
training anything meaningful.

---

## Phase 1 — Data Pipeline

### Goal

Turn raw broadcast tennis video into structured, court-coordinate-aligned match
data that can be used to train downstream models.

### Data Sources

- **Broadcast footage** — ATP/WTA matches from YouTube (wide-angle/baseline
  camera angles only)
- **Match Charting Project** (Jeff Sackmann, GitHub) — thousands of pro matches
  charted shot-by-shot with type, direction, and point outcome labels
- **THETIS dataset** — labeled tennis stroke clips for shot type classifier
  bootstrapping

### Pipeline Components

```
Broadcast frame
    │
    ├── Camera angle filter
    │       └── Keep only wide/baseline shots where full court is visible
    │
    ├── Court line detection → Homography estimation
    │       └── Maps pixel coordinates to real court coordinates (meters)
    │
    ├── Player detection + tracking
    │       └── Positions + movement vectors in court space
    │
    ├── Pose extraction (RTMPose)
    │       └── Keypoints per player, mapped to court space via homography
    │
    └── Ball tracking (TrackNet)
            └── Trajectory + bounce/landing point in court space
```

### Alignment Step

Match Charting Project labels are aligned to extracted positional data using
rally timing. This produces a unified record per shot:

```python
shot_record = {
    "shot_type":          # from Match Charting
    "direction":          # from Match Charting
    "point_outcome":      # from Match Charting
    "striker_position":   # from CV (x, y in court meters)
    "striker_velocity":   # from CV (vx, vy m/s)
    "striker_pose":       # from CV (17-keypoint array)
    "opponent_position":  # from CV
    "opponent_velocity":  # from CV
    "landing_zone":       # from ball tracking
    "rally_context":      # last N shots in the point
}
```

### Starting Point

Begin with a single clay-court match (slower rallies, cleaner tracking) that
exists in both YouTube and the Match Charting Project. Validate the pipeline
produces sensible court coordinates before scaling.

### Scale Target

30–50 processed matches → ~10,000–15,000 labeled shot events and
~3,000–5,000 labeled points — sufficient for initial model training.

---

## Phase 2 — Feedback Engine

### Goal

Model shot quality as an Expected Value (EV) problem. For any shot, compute
how good the decision was relative to the best available option given the
game state.

### Core Framework

```
EV(shot) = P(make | state) × P(win point | make, state) − P(miss | state)
```

Shot quality = how the actual shot's EV compares to the maximum available EV
from the same state. Directly analogous to centipawn loss in chess engines.

### Sub-Models

#### 2A — Shot Type Classifier

- **Input:** sequence of player pose keypoints over a stroke window
- **Output:** shot type label (forehand, backhand, serve, volley, etc.)
- **Architecture:** Temporal Convolutional Network (TCN) or small Transformer
- **Training data:** THETIS + self-collected + CV-extracted from broadcast
- **Status:** most tractable component, buildable first

#### 2B — Execution Probability Model

- **Input:** player pose at contact, movement vector, shot type, target margin
  to nearest line
- **Output:** P(successful execution)
- **Training signal:** every shot in every match — it went in or it didn't
- **Intent:** capture the risk side of the tradeoff

#### 2C — Win Probability Model

- **Input:** rally state (last N shots as structured tokens), player positions,
  movement vectors
- **Output:** P(win point from this state)
- **Architecture:** Transformer encoder over rally event sequence
- **Training:** pretrain on Match Charting Project sequences, fine-tune on
  CV-extracted positional data
- **Intent:** capture the reward side of the tradeoff

#### 2D — EV Surface

- **Derived from 2B + 2C**
- For any game state, compute EV across discretized court zones
  (deep cross, deep line, short cross, short angle, etc.)
- Produces a heatmap of shot value across the opponent's court
- Best available shot = argmax over zones
- Shot quality score = EV(actual) − EV(best available)

### State Representation

Each shot event carries:

- Last 2–3 shots in rally (type, direction, landing zone)
- Striker position + movement vector at contact
- Opponent position + movement vector at contact
- Rally length (shot index in point)
- Fatigue proxies (cumulative rally distance, recovery speed trend)
- Court opening metric (how centered/forward each player is)

### Feedback Output

Rather than binary good/bad, feedback is contextual:

> "From this position (wide, moving away from center), your down-the-line
> attempt had a 22% estimated execution probability. The highest-value zone
> from this state was deeper crosscourt with higher margin."

---

## Phase 3 — Application Layer

### Goal

Surface the feedback engine through a usable interface.

### Features (Priority Order)

1. **Automated video editing**
   - Detect point boundaries (serve trigger + dead-ball detection)
   - Strip dead time between points
   - Export trimmed match video
   - High value, relatively low complexity — ship early

2. **Shot-level feedback overlay**
   - Label each shot in video with type + quality score
   - Flag high-risk, low-EV attempts
   - Filter by shot type, game situation, quality tier

3. **Court heatmap visualization**
   - For a given player state, show EV surface across opponent's court
   - Highlight where the player actually targeted vs. optimal zones

4. **Pattern analysis**
   - Aggregate shot quality by scenario (e.g., "your forehand from the
     ad-side baseline is consistently below expected EV")
   - Trend tracking across sessions

---

## Known Constraints and Limitations

| Constraint                  | Notes                                                                                                                                |
|-----------------------------|--------------------------------------------------------------------------------------------------------------------------------------|
| Pro → recreational transfer | Win probability model trained on pro data may not transfer cleanly to club play. Treat as pre-training base, plan for fine-tuning.   |
| Broadcast camera cuts       | Rally continuity breaks on director cuts. Stitch segments using charting data timestamps.                                            |
| Pose quality at distance    | Far-baseline players are small in frame. Weight training examples by pose confidence.                                                |
| Fatigue model reliability   | Fatigue proxies are included as features but don't expect confident fatigue-conditioned recommendations from initial dataset size.    |
| Homography stability        | Re-estimate homography every N frames to handle camera drift.                                                                        |
| Skill-level calibration     | Long-term: per-player or per-tier calibration layer needed. Out of scope for V1.                                                     |

---

## Development Order

```
1.  Court homography pipeline (single match validation)
2.  Player tracking + position extraction
3.  Ball tracking + landing zone detection
4.  Point boundary detection → video editor (shippable feature)
5.  Match Charting alignment
6.  Shot type classifier
7.  Execution probability model
8.  Win probability model (Match Charting pretrain)
9.  EV surface + shot quality scoring
10. Feedback visualization layer
```

---

## Research Contribution

The combination of:

- Execution probability conditioned on biomechanical state (pose + movement)
- Strategic EV conditioned on rally context and court positions
- Applied to recreational-level tennis shot construction

…does not exist in published literature. Existing work either uses pro-only
data without positional context, or models shot outcomes without separating
execution risk from strategic value. This framing is a legitimate research
contribution at a workshop level (CVPR Sports Analytics, MIT Sloan, etc.)

---

## Out of Scope (For Now)

- Live/real-time feedback (design for offline analysis first)
- Line calling
- Serve speed / spin measurement
- Score tracking
- Android support
- Multi-player scenarios beyond 1v1
