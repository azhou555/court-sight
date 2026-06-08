# Win Probability Model (Model 2C)

## Purpose

Estimate P(win point | rally state) — the probability the striker wins the
current point given the current tactical situation. This is the **reward**
side of the EV equation.

The model captures the strategic value of creating advantageous positions:
hitting to open court, forcing the opponent off-balance, reducing their
recovery time.

## Architecture

### Transformer Encoder over Rally Event Sequence

Each shot in the rally is encoded as a structured token. The Transformer
attends over the full rally history to predict win probability at each step.

```
Rally token sequence: [shot_0, shot_1, ..., shot_k] (current shot)
    │
    ├── Per-token embedding (see below)
    │       → (k+1, 128)
    │
    ├── Positional encoding (rally position index)
    │
    ├── Transformer encoder ×4
    │       4 heads, FF dim 256, dropout 0.1
    │
    ├── Extract [current shot] token → (128,)
    │
    └── Linear(128 → 1) → sigmoid → P(win point)
```

### Rally Token Embedding

Each shot token is a concatenation of:

```python
shot_token = concat([
    shot_type_embedding,          # (16,)  learned embedding
    direction_embedding,          # (8,)   learned embedding
    landing_zone_embedding,       # (8,)   learned embedding
    striker_position,             # (2,)   court meters
    striker_velocity,             # (2,)   m/s
    opponent_position,            # (2,)   court meters
    opponent_velocity,            # (2,)   m/s
    court_opening_metric,         # (1,)   opponent distance from center
    rally_depth,                  # (1,)   shot index normalized by max_rally_len
])
# Total: 42 dims → linear projection → 128
```

### Sequence Length

Maximum sequence length: 20 shots (handles 99%+ of points). Points longer
than 20 shots use a sliding window (last 20 shots).

## Pre-training Strategy

### Stage 1: Match Charting Project Pre-training

The Match Charting Project contains shot-by-shot records with type, direction,
and outcome labels — but no positional data. Pre-train the model on this data
using text-encoded shot sequences (shot type + direction tokens only, no
position features). This gives the model a strong prior on rally dynamics and
win probability patterns from thousands of matches.

Pre-training data: ~50,000 points from Match Charting Project.

### Stage 2: Fine-tuning on Pipeline-extracted Data

Fine-tune on the full feature set (including positions and velocities) from
the processed match data. The pre-trained token embeddings and attention
patterns provide a useful initialization.

Fine-tuning data: ~3,000–5,000 labeled points from Phase 1 pipeline.

## Training Details

- Loss: binary cross-entropy (point win = 1, point loss = 0)
- Optimizer: AdamW, lr=1e-4, weight_decay=1e-2
- Scheduler: cosine annealing with warm restart
- Batch size: 64 points (variable-length sequences padded with mask)
- Gradient clipping: max norm 1.0

## Evaluation

| Metric                     | Target          |
|----------------------------|-----------------|
| Brier score                | < 0.20          |
| Log-loss                   | < 0.55          |
| AUC-ROC                    | > 0.72          |
| ECE                        | < 0.05          |

Evaluate at each rally position (shot index) to check that the model improves
in accuracy as more rally context accumulates.

## Known Limitations

- **Pro → recreational transfer**: win probability patterns differ significantly
  between professional and recreational players (error rates, shot selection
  distributions). Treat the pre-trained model as a base and plan fine-tuning
  on club-level footage as a future step.
- **Sample size**: 3,000–5,000 points may be insufficient for the model to
  learn subtle positional effects. The Match Charting pre-training mitigates
  this for the tactical/sequential component but not for the positional component.
- **Serve dependency**: win probability on serve depends heavily on serve speed
  and placement, neither of which are tracked in V1. Include serve direction
  (body, T, wide) from charting data as a proxy.
- **Current-shot landing-zone optimism**: each shot token includes its own
  `ball_landing_zone`, but the bounce occurs *after* contact — at true inference
  time the landing of the shot being evaluated is not yet known. The training
  pipeline keeps this feature (it matches the token design above) but it makes
  in-sample win-probability slightly optimistic for the current shot. A strictly
  causal variant would mask the current shot's landing zone; deferred until it
  measurably affects EV ranking.

## Output Schema

```python
WinProbResult = {
    "shot_id": str,
    "p_win_point": float,           # calibrated probability
    "rally_context_used": int,      # number of prior shots in window
    "attention_weights": list,      # (optional) per-prior-shot attention
}
```

## Key Dependencies

- PyTorch + torch.nn.TransformerEncoder
- HuggingFace `transformers` (for pre-training infra, optional)
- scikit-learn (calibration, metrics)
- Pandas (data pipeline integration)
