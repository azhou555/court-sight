# EV Surface and Shot Quality Scoring (Model 2D)

## Purpose

Combine execution probability (Model 2B) and win probability (Model 2C) into
an Expected Value surface over the opponent's court. Compute a shot quality
score — analogous to centipawn loss in chess — that measures how the actual
shot compared to the optimal available option.

## EV Formula

```
EV(shot, target_zone) = P(make | state, target_zone) × P(win | make, state, target_zone)
                       − P(miss | state, target_zone)
```

Simplified (since P(miss) = 1 − P(make)):

```
EV = P_make × P_win_given_make − (1 − P_make)
   = P_make × (P_win_given_make + 1) − 1
```

For a shot that always goes in (P_make = 1): EV = P_win − 0 = P_win
For a shot that never goes in (P_make = 0): EV = 0 − 1 = −1
Maximum possible EV ≈ 0.8 (world-class shot, high-margin execution, clear winner)
Minimum EV = −1.0 (certain error)

## Court Zone Discretization

Divide the opponent's half into a 3×3 grid (9 zones) for the EV surface:

```
Zone IDs:
┌──────────┬──────────┬──────────┐
│ T_deep   │ C_deep   │ W_deep   │   deep = near baseline (high depth)
├──────────┼──────────┼──────────┤
│ T_mid    │ C_mid    │ W_mid    │   mid  = service box depth
├──────────┼──────────┼──────────┤
│ T_short  │ C_short  │ W_short  │   short = near net
└──────────┴──────────┴──────────┘
  T = toward center T   C = center   W = toward wide sideline
```

Labels are relative to the *striker* (T is always toward the middle of the
court from the striker's perspective, W is always away from center).

For each zone, run forward passes of models 2B and 2C to get EV(zone).
This produces an EV heatmap: `{zone: EV_value}` for all 9 zones.

## Shot Quality Score

```python
def shot_quality_score(actual_zone, state):
    ev_surface = compute_ev_surface(state)           # dict: zone → EV
    ev_actual = ev_surface[actual_zone]
    ev_best = max(ev_surface.values())
    ev_worst = min(ev_surface.values())

    # Raw loss (chess centipawn loss analogy)
    ev_loss = ev_best - ev_actual

    # Normalized to [0, 1] for display
    ev_range = ev_best - ev_worst
    normalized_score = 1.0 - (ev_loss / ev_range) if ev_range > 0 else 0.5

    return {
        "ev_actual": ev_actual,
        "ev_best": ev_best,
        "ev_loss": ev_loss,
        "normalized_score": normalized_score,   # 1.0 = optimal, 0.0 = worst available
        "best_zone": argmax(ev_surface),
        "ev_surface": ev_surface,
    }
```

## Feedback Text Generation

Quality tiers and associated feedback templates:

| Score     | Tier        | Feedback template                                                  |
|-----------|-------------|--------------------------------------------------------------------|
| 0.85–1.0  | Excellent   | "Strong choice — {actual_zone} was the highest-value option."      |
| 0.65–0.85 | Good        | "Reasonable choice. {best_zone} offered slightly higher EV."       |
| 0.40–0.65 | Suboptimal  | "From this position, {best_zone} had significantly higher EV."     |
| 0.00–0.40 | Poor        | "Low-percentage attempt. {actual_zone} had {p_make:.0%} execution prob from this state. {best_zone} offered {ev_best:.2f} EV vs {ev_actual:.2f}." |

Templates are filled with concrete numbers, not vague qualifiers.

## Aggregate Pattern Analysis

Across a full match (or across multiple sessions), aggregate by scenario:

```python
patterns = group_shots_by(
    position_zone="striker_court_zone",    # 3×3 grid on striker's side
    shot_type="shot_type",
    rally_depth_bucket="early|mid|late",
).aggregate(
    mean_ev_loss="ev_loss",
    mean_normalized_score="normalized_score",
    shot_count="count",
)
```

Surface patterns where `mean_normalized_score < 0.5` and `shot_count >= 5`:
these are systematic weaknesses worth flagging.

## Example Output

```json
{
  "shot_id": "rally_3_shot_7",
  "shot_type": "forehand_groundstroke",
  "actual_zone": "T_deep",
  "ev_actual": 0.31,
  "ev_best": 0.58,
  "ev_loss": 0.27,
  "best_zone": "W_deep",
  "normalized_score": 0.46,
  "tier": "suboptimal",
  "p_make_actual": 0.71,
  "p_win_actual": 0.44,
  "feedback": "From this position (wide, moving toward center), crosscourt deep offered higher value. Your backhand side was open — the W_deep zone had 0.58 EV vs 0.31 for the T attempt.",
  "ev_surface": {
    "T_deep": 0.31,  "C_deep": 0.42,  "W_deep": 0.58,
    "T_mid":  0.22,  "C_mid":  0.38,  "W_mid":  0.51,
    "T_short": 0.08, "C_short": 0.15, "W_short": 0.29
  }
}
```

## Key Dependencies

- NumPy (vectorized zone inference)
- Models 2B (execution probability) and 2C (win probability)
- Matplotlib / Seaborn (heatmap visualization)
