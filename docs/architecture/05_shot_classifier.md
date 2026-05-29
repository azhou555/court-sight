# Shot Type Classifier (Model 2A)

## Purpose

Classify each stroke into a shot type label from a sequence of player pose
keypoints. This is the most tractable model component — strong training signal
(THETIS labels + Match Charting labels), clear input/output structure, and
usable before the full EV framework is built.

## Label Set

```
SHOT_TYPES = [
    "forehand_groundstroke",
    "backhand_groundstroke",
    "forehand_slice",
    "backhand_slice",
    "forehand_volley",
    "backhand_volley",
    "overhead",
    "serve",
    "lob",
    "drop_shot",
]
```

Note: serve has a distinct motion signature (ball toss, trophy position) that
makes it the easiest class to identify with high confidence.

## Architecture

### Primary: Temporal Convolutional Network (TCN)

TCN over a 30-frame pose sequence.

```
Input: (B, T, 17×2) = (batch, 30 frames, 34 pose features)
    │
    ├── Linear projection → (B, T, 128)
    │
    ├── TCN blocks ×4
    │       Each block:
    │           - Dilated causal Conv1D (kernel=3, dilation=2^i)
    │           - LayerNorm
    │           - ReLU
    │           - Residual connection
    │       Receptive field: 1 + 2*(3-1)*(1+2+4+8) = 61 frames
    │
    ├── Global average pool → (B, 128)
    │
    └── Linear → softmax → (B, 10) class probabilities
```

Why TCN over LSTM: TCNs are faster to train, have parallelizable convolutions,
and have a well-defined receptive field that can be tuned to match the
stroke duration.

### Alternative: Small Transformer Encoder

If TCN underfits (insufficient temporal context for ambiguous shots like
slice vs. topspin):

```
Input: (B, T, 34) → positional encoding → (B, T, 128)
    │
    ├── Transformer encoder ×3 (4 heads, FF dim 256)
    │
    ├── [CLS] token → (B, 128)
    │
    └── Linear → (B, 10)
```

## Input Features

Raw keypoints (17×2 court coordinates) are augmented with:
- Joint angles: elbow angle, shoulder rotation, hip-to-shoulder tilt
- Normalized coordinates: relative to player hip midpoint (removes court
  position from the signal — body-relative motion is what matters)
- Velocity features: finite differences of each joint position

Final feature vector per frame: 34 (raw) + 6 (angles) + 34 (velocity) = 74

## Training Data

| Source        | Approximate Size | Label Quality |
|---------------|-----------------|---------------|
| THETIS        | ~2,800 clips     | Gold (manually labeled) |
| Self-collected | TBD             | Gold (manually labeled) |
| Pipeline-extracted | 10k–15k shots | Silver (Match Charting aligned, some noise) |

Training strategy:
1. Pre-train on THETIS (gold labels, clean clips)
2. Fine-tune on pipeline-extracted data with confidence weighting
3. Use pose confidence scores as sample weights during fine-tuning

## Loss Function

Cross-entropy with label smoothing (ε=0.1) to prevent overconfident
predictions on ambiguous shots (e.g., heavy-topspin forehand vs. slice).

## Evaluation

Primary metric: macro-averaged F1 (class imbalance — serves and overheads
are rare compared to groundstrokes).

Expected baseline performance:
- Serve: >95% F1 (distinctive motion)
- Forehand/backhand groundstroke: >85% F1
- Volley: ~75% F1 (short motion, similar to groundstroke preparation)
- Drop shot: ~60% F1 (looks like slice until late contact)

## Output Schema

```python
ShotClassification = {
    "shot_id": str,
    "predicted_class": str,
    "class_probabilities": dict,    # {class_name: float}
    "confidence": float,            # max probability
    "model_version": str,
}
```

## Key Dependencies

- PyTorch
- torchvision (preprocessing utilities)
- scikit-learn (evaluation metrics, cross-validation)
