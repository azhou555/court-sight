# Neutral Position / Coverage Model (Model 2E)

## Purpose

Predict where a player should be standing when their opponent makes contact
with the ball — the position that minimizes expected sprint distance to cover
the opponent's likely responses.

This serves two roles in the pipeline:

1. **Recovery position target**: after each shot, compute where the striker
   should recover to before the opponent hits
2. **Recovery displacement input**: the gap between where the player actually
   was at contact and where they should have been is a key input to the
   displacement penalty in the EV formula

---

## Why This Is Novel

Standard coaching advice gives fixed neutral positions ("recover to center
baseline"). This model makes neutral position **state-dependent** — it shifts
based on where you just hit, where the opponent is, and the likely reply
distribution. After a deep crosscourt ball that pulls the opponent wide, your
neutral is different from after a short middle ball. After a serve wide, the
opponent's return angles are geometrically constrained, pulling your neutral
T-side. The model learns these position-specific adjustments from data rather
than hard-coding them.

---

## Architecture: Two-Stage

### Stage 1 — Response Distribution Model

Predicts a **2D Gaussian** N(μ, σ²I) over the court, representing where the
opponent is likely to land their response shot.

**Why continuous 2D Gaussian instead of zone classification:**
- Gives sub-meter precision on the likely landing location
- Directly feeds into the geometric median computation without zone-boundary artifacts
- σ captures uncertainty: small σ when the opponent is stretched and constrained,
  larger σ when they have full options from a neutral position
- No information loss from coarse discretization

**Architecture: two independent GBT regressors**

```
Regressor A: features → μ_x  (lateral landing coordinate, court meters)
Regressor B: features → μ_y  (depth landing coordinate, court meters)
```

σ is estimated from calibrated residuals stratified by context bucket
(opponent displacement, incoming ball depth). It is not learned end-to-end
but updated periodically from held-out residual statistics.

**Input features:**

| Feature | Description |
|---|---|
| `your_shot_x`, `your_shot_y` | Continuous landing coordinates of your shot |
| `ball_height_at_bounce` | From trajectory arc — high bounce = more time, wider options |
| `opponent_pos_x`, `opponent_pos_y` | Where opponent was when they need to hit |
| `opponent_vel_x`, `opponent_vel_y` | Direction and speed — on-the-run vs. planted |
| `opponent_displacement_from_neutral` | How far out of position opponent was |
| `your_shot_type` | Topspin/slice/flat affects bounce characteristics |
| `rally_depth` | Shot index in point — patterns differ early vs. late |
| `prev_shot_direction` | Last 2 shot directions — encodes rally pattern context |

**Training data:**
Every pair of consecutive shots (shot N → shot N+1) in the processed match
records is a training example:
- Input: features of shot N (your shot's landing, opponent state)
- Label: (landing_x, landing_y) of shot N+1

With 30–50 matches yielding ~8k–12k consecutive shot pairs, this is adequate
for two shallow GBT regressors on tabular features.

---

### Stage 2 — Optimal Position Computation

Given the 2D Gaussian (μ, σ) from Stage 1, compute the **weighted geometric
median** — the position minimizing expected sprint distance to cover the
predicted response distribution.

```python
def optimal_neutral(mu: np.ndarray, sigma: float, n_samples: int = 200) -> np.ndarray:
    """
    mu: (2,) predicted mean landing position
    sigma: isotropic standard deviation
    Returns: (2,) optimal neutral position in court meters
    """
    rng = np.random.default_rng()
    samples = rng.multivariate_normal(mu, sigma ** 2 * np.eye(2), n_samples)

    # Clip samples to valid court bounds
    samples[:, 0] = np.clip(samples[:, 0], -4.115, 4.115)  # singles width
    samples[:, 1] = np.clip(samples[:, 1], 0.0, 23.77)      # court depth

    # Weiszfeld algorithm — geometric median of samples (equal weights)
    pos = mu.copy()
    for _ in range(100):
        dists = np.linalg.norm(samples - pos, axis=1)
        dists = np.maximum(dists, 1e-6)
        weights = 1.0 / dists
        pos_new = np.average(samples, weights=weights, axis=0)
        if np.linalg.norm(pos_new - pos) < 1e-5:
            break
        pos = pos_new
    return pos
```

For a unimodal Gaussian the geometric median converges close to μ. The value
of the Weiszfeld computation emerges when the distribution has a heavy tail
toward one side (e.g., opponent has one very likely dangerous option and several
weak ones) — the median correctly biases toward the dangerous option more than
the mean would.

---

### Stage 3 — Time Constraint (Reachable Set)

The theoretical optimal neutral is irrelevant if the player cannot reach it
before the opponent makes contact. Compute the reachable set:

```python
def reachable_neutral(
    theoretical_neutral: np.ndarray,
    current_pos: np.ndarray,
    current_vel: np.ndarray,
    t_recovery: float,               # seconds available to recover
    max_speed: float = 4.5,          # m/s (elite sprint speed)
) -> np.ndarray:
    """
    If theoretical_neutral is reachable, return it.
    Otherwise return the closest reachable point.
    """
    direction = theoretical_neutral - current_pos
    dist = np.linalg.norm(direction)
    max_dist = t_recovery * max_speed
    if dist <= max_dist:
        return theoretical_neutral
    # Project onto reachable boundary
    return current_pos + (direction / dist) * max_dist
```

`t_recovery` is estimated as:
```
t_recovery = ball_flight_time(your_shot) + opponent_approach_time
```

Where `opponent_approach_time` ≈ opponent's distance to the ball / opponent speed,
estimated from the ball trajectory and opponent tracking.

---

## Recovery Displacement

The gap between predicted neutral and actual player position at the next contact
is the **recovery displacement** — a core input to the displacement penalty:

```python
recovery_displacement_m = np.linalg.norm(
    actual_contact_position - reachable_neutral
)
```

This measures not "how far were you from the ideal neutral" but "how far were
you from the best position you could have reached given the available time."
A player who recovered well but ran out of time is scored less harshly than
one who stood still.

---

## Displacement Penalty

Combines recovery displacement with contact quality at the moment of hitting.

```python
def displacement_penalty(
    contact_pos_m: np.ndarray,
    neutral_pos_m: np.ndarray,
    movement_vector: np.ndarray,         # (vx, vy) m/s at contact
    shot_direction: np.ndarray,          # unit vector of shot direction
    ball_pos_at_contact: np.ndarray,     # ball position in court meters at contact
    shot_type: str,
    w_radial: float = 0.06,
    w_body: float = 0.10,
    w_momentum: float = 0.05,
    w_contact: float = 0.05,
) -> float:
    """
    Returns a penalty in EV units [0.0, ~0.5].
    """
    # 1. Radial distance from neutral (1m grace zone)
    dist = np.linalg.norm(contact_pos_m - neutral_pos_m)
    radial = max(0.0, dist - 1.0) * w_radial
    if dist > 3.0:
        radial += (dist - 3.0) * w_radial * 2.0  # steep beyond 3m

    # 2. Body shot (ball arriving into the torso at contact)
    torso_to_ball = ball_pos_at_contact - contact_pos_m
    body = w_body if np.linalg.norm(torso_to_ball) < 0.45 else 0.0

    # 3. Momentum misalignment with shot direction
    speed = np.linalg.norm(movement_vector)
    if speed > 0.3:
        alignment = np.dot(movement_vector / speed, shot_direction)
        momentum = max(0.0, -alignment) * w_momentum
    else:
        momentum = 0.0

    # 4. Contact point quality for shot type / wing
    ideal = IDEAL_CONTACT_OFFSET[shot_type]   # (dx, dy) relative to hip
    actual = torso_to_ball
    contact_dev = np.linalg.norm(actual - ideal) * w_contact

    return min(radial + body + momentum + contact_dev, 0.6)  # cap at 0.6
```

### Ideal contact offsets by shot type

```python
IDEAL_CONTACT_OFFSET = {
    # (x = lateral from body center, y = forward from hip)
    # positive x = toward dominant/hitting side
    "forehand_groundstroke":  np.array([1.1,  0.4]),
    "backhand_groundstroke":  np.array([-0.7, 0.5]),
    "forehand_slice":         np.array([1.0,  0.5]),
    "backhand_slice":         np.array([-0.6, 0.6]),
    "forehand_volley":        np.array([0.9,  0.7]),
    "backhand_volley":        np.array([-0.6, 0.7]),
    "overhead":               np.array([0.3,  0.0]),   # above head
    "serve":                  np.array([0.3,  0.0]),
    "lob":                    np.array([0.8,  0.3]),
    "drop_shot":              np.array([0.9,  0.5]),
}
```

These offsets are coaching priors, not learned. They should be validated against
observed contact positions on high-quality shots in the dataset.

---

## Serve Regime

After a serve, the neutral position model operates in a constrained regime:

- **Server position at contact**: fixed behind baseline, near center mark (deuce)
  or near singles sideline (ad). Range of positions is the smallest of any shot.
- **Response distribution**: strongly determined by serve placement (T/body/wide)
  and serve side (deuce/ad). The geometrically available return angles are narrow.
- **Default neutral**: center baseline T-area (~0.5m T-side of center mark).
- **Shift from prior patterns**: if the server consistently serves wide and the
  returner exploits DTL returns, neutral shifts accordingly. Use a per-match
  rolling frequency of serve placement → return direction to adjust.

The serve neutral can be a simple lookup (serve_side × serve_placement → neutral_x)
rather than running the full response distribution model.

---

## Validation

For each shot record, compare predicted neutral against actual player position
at the opponent's next contact:

```
validation_error_m = |actual_position_at_opponent_contact - predicted_neutral|
```

Stratify by:
- Shot type (different neutrals for baseline vs. net shots)
- Rally depth (patterns differ early vs. late)
- Whether the player successfully returned the next ball

Players who were closer to predicted neutral should have higher return rates.
A 1m improvement in proximity to predicted neutral should correspond to a
measurable improvement in return rate — if it doesn't, the model is learning
the wrong position.

---

## Schema Additions to ShotRecord

```python
ShotRecord additions = {
    "neutral_position_m": [float, float],         # predicted neutral at contact
    "reachable_neutral_m": [float, float],         # time-constrained neutral
    "recovery_displacement_m": float,              # |actual - reachable_neutral|
    "displacement_penalty": float,                 # computed penalty score
    "contact_point_deviation_m": float,            # deviation from ideal contact zone
    "t_recovery_available": float,                 # seconds available to recover
    "response_mu": [float, float],                 # predicted opponent landing mean
    "response_sigma": float,                       # predicted uncertainty
}
```

---

## Key Dependencies

- XGBoost (two GBT regressors for μ_x, μ_y)
- NumPy (Weiszfeld algorithm, reachable set computation)
- Player tracking output (positions, velocities)
- Ball tracking output (trajectory, contact positions)
- Shot type classifier output
