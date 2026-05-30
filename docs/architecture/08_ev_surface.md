# EV Surface and Shot Quality Scoring (Model 2D)

## Purpose

Combine shot margin safety (Model 2B), win probability (Model 2C), and
displacement penalty (Model 2E) into an Expected Value surface over the
opponent's court. Compute a shot quality score — analogous to centipawn loss
in chess — that measures how the actual shot compared to the optimal available
option from the same state.

---

## EV Formula

```
EV_adjusted(zone) = P(make | zone) × P(win | make, zone)
                  − P(miss | zone)
                  − displacement_penalty
```

Expanded (since P(miss) = 1 − P(make)):

```
EV_adjusted(zone) = P_make × (P_win + 1) − 1 − displacement_penalty
```

### Components

| Term | Source | Varies per zone? |
|---|---|---|
| P(make \| zone) | Model 2B — geometric margin features | Yes — margin changes with target |
| P(win \| make, zone) | Model 2C — rally context + court state | Yes — win prob depends on where you hit |
| displacement_penalty | Model 2E — recovery state at contact | **No — fixed at contact time** |

### Displacement penalty as a constant offset

The displacement penalty is computed once at contact and applied uniformly
across all 9 zones in the EV surface. Because it is constant, it does not
change which zone has the highest EV — the relative ranking of zones is
determined entirely by P(make) × P(win). The penalty affects the **absolute
EV score**, which determines how much the player's positioning compromised
the entire shot regardless of target choice.

Consequence for quality score:
```
ev_loss = EV_best - EV_actual
        = [best_base - dp] - [actual_base - dp]
        = best_base - actual_base
```

The penalty cancels in the relative score. This means:
- Shot selection quality (best zone vs. actual zone) is captured by ev_loss
- Positioning quality is captured by the displacement_penalty directly
- These are two separable dimensions of feedback, not conflated

---

## Court Zone Discretization

Divide the opponent's half into a 3×3 grid (9 zones) for the EV surface
display. Note: the underlying models (2B, 2C, 2E) operate on continuous
court coordinates — the 9-zone grid is only for the user-facing output.

```
Zone IDs (relative to striker — T is always toward center):
┌──────────┬──────────┬──────────┐
│ T_deep   │ C_deep   │ W_deep   │   deep  = near far baseline
├──────────┼──────────┼──────────┤
│ T_mid    │ C_mid    │ W_mid    │   mid   = service box depth
├──────────┼──────────┼──────────┤
│ T_short  │ C_short  │ W_short  │   short = near net
└──────────┴──────────┴──────────┘
  T = toward center T   C = center   W = toward wide sideline
```

Each zone's representative coordinate is its center point in court meters,
used when computing P(make) margin features for that zone.

---

## EV Surface Computation

For each of the 9 zones, run full inference of all three components:

```python
def compute_ev_surface(state: GameState) -> dict[str, float]:
    dp = displacement_penalty(state.recovery_state)   # computed once

    ev_surface = {}
    for zone in COURT_ZONES:
        zone_center = ZONE_CENTERS[zone]

        # 2B: geometric margin for this hypothetical target
        safety = shot_safety_model.predict(state, zone_center)
        p_make = safety["p_make"]

        # 2E: neutral position the player would target after hitting this zone
        # feeds into the opponent's likely response distribution
        next_neutral = neutral_model.predict_neutral(
            your_landing=zone_center,
            your_position=state.striker_position,
        )

        # 2C: win probability given this zone was hit and the resulting neutral
        win = win_prob_model.predict(state.rally_context + [{
            "ball_landing_zone": zone,
            "next_neutral": next_neutral,
        }])
        p_win = win["p_win_point"]

        ev_base = p_make * (p_win + 1.0) - 1.0
        ev_surface[zone] = ev_base - dp

    return ev_surface
```

**Computational cost**: 9 × (one GBT forward pass + one Transformer forward pass).
Each GBT inference is ~microseconds; Transformer forward pass is ~milliseconds.
Full EV surface per shot: well under 100ms. Not a bottleneck.

---

## Shot Quality Score

```python
def shot_quality_score(actual_zone: str, state: GameState) -> dict:
    ev_surface = compute_ev_surface(state)
    dp = displacement_penalty(state.recovery_state)

    ev_actual = ev_surface[actual_zone]
    ev_actual_base = ev_actual + dp          # base EV before positioning penalty

    best_zone = max(ev_surface, key=ev_surface.__getitem__)
    ev_best_base = ev_surface[best_zone] + dp

    ev_worst_base = min(ev_surface[z] + dp for z in COURT_ZONES)

    # Relative quality: zone choice, independent of positioning
    ev_loss = ev_best_base - ev_actual_base
    ev_range = ev_best_base - ev_worst_base
    zone_score = 1.0 - (ev_loss / ev_range) if ev_range > 0 else 0.5

    return {
        "ev_actual": round(ev_actual, 3),
        "ev_actual_base": round(ev_actual_base, 3),
        "ev_best_base": round(ev_best_base, 3),
        "ev_loss": round(ev_loss, 3),
        "best_zone": best_zone,
        "zone_score": round(zone_score, 3),       # 1.0 = optimal zone choice
        "displacement_penalty": round(dp, 3),      # separate positioning score
        "p_make_actual": state.p_make,
        "ev_surface": {z: round(v, 3) for z, v in ev_surface.items()},
    }
```

---

## Two-Dimensional Feedback

Shot quality now has two separable dimensions:

| Dimension | Score | What it measures |
|---|---|---|
| Zone selection | `zone_score` (0–1) | Did you pick the right target? |
| Positioning | `displacement_penalty` | Were you in a good position to hit? |

Feedback tier for zone selection:

| `zone_score` | Tier | Template |
|---|---|---|
| 0.85–1.0 | Excellent | "{zone} was the highest-value option from this state." |
| 0.65–0.85 | Good | "{best_zone} offered slightly higher EV (+{ev_loss:.2f})." |
| 0.40–0.65 | Suboptimal | "From this state, {best_zone} had significantly higher EV (+{ev_loss:.2f})." |
| 0.00–0.40 | Poor | "{actual_zone} had {p_make:.0%} make probability from this state. {best_zone} offered {ev_best:.2f} EV." |

Positioning feedback (displacement_penalty driven):

| `displacement_penalty` | Template |
|---|---|
| < 0.05 | "Good recovery — contacted within neutral zone." |
| 0.05–0.15 | "Slightly out of position at contact ({recovery_m:.1f}m from neutral)." |
| 0.15–0.30 | "Out of position at contact ({recovery_m:.1f}m from neutral). Attempted shot from a compromised state." |
| > 0.30 | "Severely out of position ({recovery_m:.1f}m from neutral). [body_shot / wrong_momentum / contact_deviation notes]" |

---

## Aggregate Pattern Analysis

Aggregate across a match or session, split by both dimensions:

```python
patterns = group_shots_by(
    striker_zone="striker_court_zone",
    shot_type="shot_type",
    rally_depth_bucket="early|mid|late",
).aggregate(
    mean_zone_score="zone_score",
    mean_displacement_penalty="displacement_penalty",
    shot_count="count",
)
```

Flag patterns where:
- `mean_zone_score < 0.5` and `shot_count >= 5` → systematic zone selection weakness
- `mean_displacement_penalty > 0.15` and `shot_count >= 5` → systematic recovery problem

These are different coaching recommendations: zone selection issues are about
shot construction decisions; recovery issues are about court movement habits.

---

## Example Output

```json
{
  "shot_id": "rally_3_shot_7",
  "shot_type": "forehand_groundstroke",
  "actual_zone": "T_deep",
  "ev_actual": 0.18,
  "ev_actual_base": 0.31,
  "ev_best_base": 0.58,
  "ev_loss": 0.27,
  "best_zone": "W_deep",
  "zone_score": 0.46,
  "displacement_penalty": 0.13,
  "p_make_actual": 0.71,
  "feedback_zone": "From this state, W_deep had significantly higher EV (+0.27). Opponent's backhand side was open.",
  "feedback_position": "Slightly out of position at contact (2.1m from neutral). Recovery fell 1.1m short.",
  "ev_surface": {
    "T_deep": 0.31, "C_deep": 0.42, "W_deep": 0.58,
    "T_mid":  0.22, "C_mid":  0.38, "W_mid":  0.51,
    "T_short": 0.08, "C_short": 0.15, "W_short": 0.29
  }
}
```

---

## Key Dependencies

- Model 2B: `src/models/execution_prob/model.py`
- Model 2C: `src/models/win_prob/model.py`
- Model 2E: `src/models/neutral_position/model.py`
- Displacement penalty: `src/models/neutral_position/displacement.py`
- NumPy (zone iteration, EV computation)
- Matplotlib / Seaborn (heatmap visualization)
