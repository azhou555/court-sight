# Shot Margin Safety Model (Model 2B)

## Purpose

Estimate P(make | shot geometry, shot type) — the probability a shot lands in
court given the geometric characteristics of the attempt and where it is aimed.
This is the **risk side** of the EV equation.

## Key Design Principle

This model captures **shot selection risk**: is the player choosing a margin
that is geometrically likely to succeed from their position? It does NOT model
execution quality (contact mechanics, timing errors). That component is captured
separately in the displacement penalty term applied at the EV layer.

The training signal — ball went in or didn't — is now clean for this purpose.
The features are purely geometric, so the model learns margin effects without
confounding from biomechanical noise.

## Features

All features are directly observable from ball tracking + shot classifier.
No pose features.

### Shot Geometry

| Feature | Description |
|---|---|
| `lateral_margin_m` | Distance from landing x to nearest sideline (meters) |
| `depth_margin_m` | Distance from landing y to nearest baseline (meters) |
| `net_clearance_m` | Estimated ball height at net crossing (from trajectory arc) |
| `shot_direction_deg` | Angle relative to court long axis — crosscourt bonus emerges naturally |
| `landing_x`, `landing_y` | Continuous court coordinates of landing point |

### Shot and Position Context

| Feature | Description |
|---|---|
| `shot_type` | One-hot: forehand / backhand / slice / volley / serve |
| `striker_y_m` | Striker's court depth — at net vs. baseline changes viable margins |
| `striker_x_m` | Lateral position — wide position constrains available angles |
| `incoming_depth_m` | How deep the incoming ball was — deeper = less time, tighter margins |

### Target Zone Estimation for Misses

For shots that land out:

- **Small misses** (out by < 0.5m lateral or < 1m depth): ball direction and
  trajectory arc give a reliable estimate of the intended landing zone.
  Assign the nearest in-bounds zone as a soft label with a confidence weight
  proportional to proximity.

- **Shanks** (ball direction implausible for the declared shot type, e.g.,
  forehand traveling sideways): discard from training. Expected to be rare
  in broadcast pro footage.

For makes: `landing_zone = target_zone` directly.

## Architecture: Gradient Boosted Trees

**Why GBT over MLP here:**
- Features are tabular and independently structured — no spatial relationships
  between features that MLP would exploit
- At 10k–15k training examples, GBT reaches reliable calibration faster
- Calibration is critical — the EV formula requires well-calibrated
  probabilities, not just ranked scores. GBT + isotonic regression
  consistently outperforms MLP on ECE at this scale
- Feature importance is interpretable, allowing direct validation
  (lateral margin should dominate, serve type should have low importance, etc.)

```python
model = XGBClassifier(
    n_estimators=500,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
)
# Post-fit: isotonic regression calibration on held-out split
```

## Training Data

Every shot in every processed match is a training example:
- Input: geometric features at landing + shot type + striker position
- Label: 1 (ball in court) or 0 (out) — with soft zone estimate for small misses

With 30–50 matches (~12k–15k shots post-filtering), this gives adequate
coverage across shot types and margin profiles.

## Evaluation

| Metric | Target |
|---|---|
| Brier score | < 0.18 |
| ECE | < 0.04 |
| AUC-ROC | > 0.75 |
| Lateral margin feature rank | Top 3 by SHAP importance |

Validate conditional accuracy by shot type and lateral margin bucket —
catches any systematic bias (e.g., model underestimating miss rate on slice DTL).

## Displacement Penalty

The positioning component of shot safety is computed analytically as a
separate term in the EV formula. It is NOT a feature in this model.
See `08_ev_surface.md` for the penalty definition and `10_neutral_position.md`
for the neutral position computation it depends on.

## Output Schema

```python
ShotSafetyResult = {
    "shot_id": str,
    "p_make": float,              # calibrated probability
    "p_miss": float,              # = 1 - p_make
    "lateral_margin_m": float,
    "depth_margin_m": float,
    "net_clearance_m": float,
    "target_zone_estimated": bool,  # True if target zone was imputed (miss)
}
```

## Key Dependencies

- XGBoost
- scikit-learn (isotonic calibration, evaluation)
- SHAP (feature importance validation)
- Ball tracking output (landing coordinates, trajectory arc)
- Shot type classifier output
