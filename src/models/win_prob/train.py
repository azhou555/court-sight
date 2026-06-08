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


def split_by_point(examples: list, val_frac: float = 0.2, seed: int = 42) -> tuple[list, list]:
    """Grouped split: all examples from one point land on the same side."""
    keys = sorted({e[2] for e in examples})
    rng = np.random.default_rng(seed)
    rng.shuffle(keys)
    n_val = max(1, int(len(keys) * val_frac))
    val_keys = set(keys[:n_val])
    train_ex = [e for e in examples if e[2] not in val_keys]
    val_ex = [e for e in examples if e[2] in val_keys]
    if not train_ex:
        raise ValueError(
            f"split_by_point produced an empty training set "
            f"({len(keys)} unique point(s), val_frac={val_frac}). "
            "Need at least 2 distinct points."
        )
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

# ── Batching ────────────────────────────────────────────────────────────────────

def _iter_batches(examples: list, batch_size: int, shuffle: bool, seed: int = 0):
    order = np.arange(len(examples))
    if shuffle:
        np.random.default_rng(seed).shuffle(order)
    for start in range(0, len(order), batch_size):
        chunk = [examples[i] for i in order[start: start + batch_size]]
        s, d, z, f, mask = pad_batch([e[0] for e in chunk])
        y = torch.tensor([e[1] for e in chunk], dtype=torch.float32)
        yield s, d, z, f, mask, y


@torch.no_grad()
def _predict_probs(encoder, examples: list, batch_size: int):
    encoder.eval()
    probs, labels = [], []
    for s, d, z, f, mask, y in _iter_batches(examples, batch_size, shuffle=False):
        p = torch.sigmoid(encoder(s, d, z, f, mask))
        probs.extend(p.reshape(-1).tolist())
        labels.extend(y.tolist())
    return np.array(probs), np.array(labels)


def _val_logloss(encoder, examples: list, batch_size: int) -> float:
    probs, labels = _predict_probs(encoder, examples, batch_size)
    if len(set(labels.tolist())) < 2:
        # log_loss needs both classes; fall back to MSE-ish proxy
        return float(np.mean((probs - labels) ** 2))
    return float(log_loss(labels, probs, labels=[0, 1]))


# ── Entry point ─────────────────────────────────────────────────────────────────

def train(
    shot_record_paths: list,
    output_dir="checkpoints/win_prob",
    positions_off: bool = False,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-4,
    weight_decay: float = 1e-2,
    patience: int = 8,
    seed: int = 42,
) -> Path:
    """Train, calibrate, and save the win-probability model. Returns model path."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)

    records = _load_records(shot_record_paths)
    examples = build_examples(records, positions_off=positions_off)
    train_ex, val_ex = split_by_point(examples, seed=seed)

    model = WinProbModel()
    encoder = model.encoder
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.BCEWithLogitsLoss()

    best_val, best_state, bad = float("inf"), None, 0
    for epoch in range(epochs):
        encoder.train()
        for s, d, z, f, mask, y in _iter_batches(train_ex, batch_size, True, seed + epoch):
            optimizer.zero_grad()
            logits = encoder(s, d, z, f, mask)
            loss = loss_fn(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        val_loss = _val_logloss(encoder, val_ex, batch_size)
        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in encoder.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        encoder.load_state_dict(best_state)

    # Isotonic calibration on the validation split.
    val_probs, val_labels = _predict_probs(encoder, val_ex, batch_size)
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(val_probs, val_labels)
    model._calibrator = calibrator
    cal_probs = calibrator.predict(val_probs)

    both_classes = len(set(val_labels.tolist())) > 1
    metrics = {
        "brier": float(brier_score_loss(val_labels, cal_probs)),
        "log_loss": float(log_loss(val_labels, cal_probs, labels=[0, 1])),
        "auc": float(roc_auc_score(val_labels, cal_probs)) if both_classes else None,
        "ece": expected_calibration_error(cal_probs, val_labels),
        "n_train": len(train_ex),
        "n_val": len(val_ex),
        "best_val_logloss": best_val,
        "positions_off": positions_off,
    }

    model_path = output_dir / "win_prob_model.pt"
    model.save(str(model_path))
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"  brier={metrics['brier']:.3f} log_loss={metrics['log_loss']:.3f} "
          f"auc={metrics['auc']} ece={metrics['ece']:.3f}")
    print(f"Saved to {model_path}")
    return model_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train win probability model (2C)")
    parser.add_argument("--records", nargs="+", required=True, help="ShotRecord JSON file(s)")
    parser.add_argument("--output_dir", default="checkpoints/win_prob")
    parser.add_argument("--positions_off", action="store_true",
                        help="Pretrain regime: zero positional features + mask flag")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=8)
    args = parser.parse_args()
    train(
        shot_record_paths=args.records,
        output_dir=args.output_dir,
        positions_off=args.positions_off,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
    )
