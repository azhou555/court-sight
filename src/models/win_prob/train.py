"""Training pipeline for the win probability model (Model 2C).

Each shot in a point becomes one training example: the sequence is the rally up
to and including that shot (left-truncated to MAX_RALLY_LEN), the label is that
shot's ``point_outcome`` (1 = that shot's striker won the point). The transformer
is trained with BCE + AdamW + cosine annealing, calibrated with isotonic
regression on a point-grouped validation split, and evaluated with
Brier / log-loss / AUC / ECE plus per-rally-position accuracy.

Usage:
    from src.models.win_prob.train import train
    train(shot_record_paths=["data/match1_shots.json", ...],
          output_dir="checkpoints/win_prob")

CLI:
    python -m src.models.win_prob.train --records data/shots/*.json \
        --output_dir checkpoints/win_prob
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from .model import RallyTokenizer, WinProbModel, pad_batch, MAX_RALLY_LEN


# ── Example construction ────────────────────────────────────────────────────────

def _point_key(record: dict) -> str:
    """Strip the trailing shot index from point_id to group shots by point."""
    pid = str(record.get("point_id", ""))
    return pid.rsplit("_", 1)[0] if "_" in pid else pid


def build_examples(records: list[dict], positions_off: bool = False) -> list:
    """One example per shot: (encoded_tuple, float_label, point_key).

    The sequence is the rally up to and including the shot (left-truncated to
    MAX_RALLY_LEN); the label is that shot's point_outcome. Shots without a
    point_outcome are skipped.
    """
    tok = RallyTokenizer()
    by_point: dict[str, list[dict]] = {}
    for r in records:
        by_point.setdefault(_point_key(r), []).append(r)

    examples = []
    for pk, shots in by_point.items():
        shots_sorted = sorted(shots, key=lambda r: r.get("shot_in_rally", 0))
        for i in range(len(shots_sorted)):
            label = shots_sorted[i].get("point_outcome")
            if label is None:
                continue
            seq = shots_sorted[: i + 1][-MAX_RALLY_LEN:]
            enc = tok.encode_rally(seq, positions_off)
            examples.append((enc, float(label), pk))

    if not examples:
        raise ValueError("No training examples extracted from records.")
    return examples


def split_by_point(examples: list, val_frac: float = 0.2, seed: int = 42):
    """Grouped split: all examples from one point land on the same side."""
    keys = sorted({e[2] for e in examples})
    rng = np.random.default_rng(seed)
    rng.shuffle(keys)
    n_val = max(1, int(len(keys) * val_frac))
    val_keys = set(keys[:n_val])
    train_ex = [e for e in examples if e[2] not in val_keys]
    val_ex = [e for e in examples if e[2] in val_keys]
    return train_ex, val_ex


# ── Metrics ─────────────────────────────────────────────────────────────────────

def expected_calibration_error(probs, labels, n_bins: int = 10) -> float:
    """Binned ECE: sum over bins of (bin weight) * |mean label - mean prob|."""
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (probs > lo) & (probs <= hi)
        if lo == 0.0:
            in_bin |= probs == 0.0
        if in_bin.sum() == 0:
            continue
        ece += in_bin.mean() * abs(labels[in_bin].mean() - probs[in_bin].mean())
    return float(ece)


def _load_records(paths: list) -> list[dict]:
    records: list[dict] = []
    for p in paths:
        data = json.loads(Path(p).read_text())
        records.extend(data if isinstance(data, list) else [data])
    return records


# ── Entry point (TEMPORARY stub — replaced in Task 3) ───────────────────────────

def train(*args, **kwargs):
    raise NotImplementedError("implemented in Task 3")
