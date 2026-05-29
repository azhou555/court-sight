# Execution Probability Model (Model 2B)

## Purpose

Estimate P(successful execution | player state, shot type, target zone) —
the probability that a player makes the shot they're attempting given their
physical situation at contact. This is the **risk** side of the EV equation.

## Why This Matters

A player attempting a down-the-line winner while stretched wide and off-balance
has a fundamentally different execution probability than the same shot from a
neutral, centered position. The execution probability model captures this
relationship so that risky shot choices in difficult physical states are
penalized appropriately regardless of their strategic value.

## Training Signal

Unlike win probability, execution probability has a clean, abundant binary
training signal: **every shot either goes in or it doesn't**. This is a
supervised classification problem with no labeling cost beyond what the
pipeline already produces.

```
for each shot_event:
    outcome = 1 if ball landed in court else 0
    features = {player_pose, movement_vector, shot_type, target_zone}
    # →  binary cross-entropy supervision
```

## Input Features

### Biomechanical State at Contact (most important features)

| Feature                          | Description                                              |
|----------------------------------|----------------------------------------------------------|
| Pose keypoints (normalized)      | 17×2 joint positions relative to hip midpoint           |
| Joint angles                     | Elbow angle, shoulder rotation, hip tilt, knee bend      |
| Movement vector                  | (vx, vy) in m/s — direction and speed of movement       |
| Balance metric                   | Lateral displacement from center of mass baseline        |
| Recovery direction               | Moving toward center vs. away — proxy for recovery ease  |

### Shot Context

| Feature                          | Description                                              |
|----------------------------------|----------------------------------------------------------|
| Shot type (one-hot)              | From shot type classifier                                |
| Target zone (one-hot)            | 9-zone grid on opponent's court                          |
| Distance to target sideline      | Clearance margin — smaller = higher risk                 |
| Distance to net                  | Depth of target — shorter = higher net clearance needed  |
| Player court position            | Distance from center baseline, lateral displacement      |

### Rally Context

| Feature                          | Description                                              |
|----------------------------------|----------------------------------------------------------|
| Incoming ball speed (relative)   | Fast incoming ball reduces execution probability          |
| Incoming ball direction          | On-the-run vs. comfortable position                      |
| Rally length                     | Proxy for fatigue accumulation                           |

## Architecture

### Gradient Boosted Trees (Primary)

Tabular features → **XGBoost** or **LightGBM**. Reasons:
- Tabular data with mixed feature types (continuous + categorical)
- Interpretable feature importance
- Calibration: well-calibrated probabilities out of the box after isotonic
  regression post-processing
- Fast inference at serving time

```python
model = XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
)
```

### Neural Network Alternative

If the GBT underfits non-linear biomechanical interactions:

```
Input: concatenated feature vector (∼80 dims)
    │
    ├── MLP: 256 → 128 → 64
    │       Each layer: Linear → BatchNorm → ReLU → Dropout(0.3)
    │
    └── Linear(1) → sigmoid → P(make)
```

## Probability Calibration

Raw classifier outputs are calibrated using isotonic regression on a held-out
calibration set. Calibration is critical — the downstream EV computation
requires well-calibrated probabilities, not just ranked scores.

Evaluate calibration with reliability diagrams and Expected Calibration Error
(ECE). Target ECE < 0.05.

## Evaluation

Primary metrics:
- **Brier score**: proper scoring rule for probability accuracy
- **ECE**: calibration quality
- **AUC-ROC**: discrimination ability
- **Conditional accuracy**: breakdown by shot type, player position zone,
  balance state — catch any systematic biases

## Output

```python
ExecutionProbResult = {
    "shot_id": str,
    "p_make": float,                  # calibrated probability
    "p_miss": float,                  # = 1 - p_make
    "feature_importances": dict,      # top contributing features (SHAP values)
    "confidence_interval": [float, float],   # bootstrap 90% CI
}
```

## Key Dependencies

- XGBoost / LightGBM
- scikit-learn (calibration, evaluation)
- SHAP (feature importance / explainability)
- NumPy, Pandas
