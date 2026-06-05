"""Training pipeline for ShotClassifierTCN.

Two-stage training:
  1. Pre-train on THETIS data (prepare_thetis.py generates X_train/y_train.npy).
  2. Fine-tune on pipeline-extracted ShotRecord sequences (broadcast data).

Dataset format expected (produced by scripts/prepare_thetis.py):
    data_dir/
        X_train.npy   — (N, 30, 17, 2) float32 keypoint sequences
        y_train.npy   — (N,)           int64 class indices
        X_val.npy
        y_val.npy

Labels index into model.SHOT_TYPES.

Usage:
    python -m models.shot_classifier.train \\
        --data_dir data/thetis_poses \\
        --output_dir checkpoints/shot_classifier \\
        --epochs 50
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Dataset

from .model import ShotClassifierTCN, SHOT_TYPES, NUM_CLASSES, STROKE_WINDOW
from ...pipeline.pose.extractor import normalize_pose_sequence


class ShotDataset(Dataset):
    """Pose sequence dataset for shot type classification.

    Expects pre-extracted .npy files produced by scripts/prepare_thetis.py.
    Each sample is a (30, 17, 2) keypoint window → normalised to (30, 34).
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        augment: bool = True,
    ):
        assert split in ("train", "val")
        data_dir = Path(data_dir)
        X_path = data_dir / f"X_{split}.npy"
        y_path = data_dir / f"y_{split}.npy"

        if not X_path.exists() or not y_path.exists():
            raise FileNotFoundError(
                f"Missing {X_path} or {y_path}. "
                "Run scripts/prepare_thetis.py first."
            )

        self._X = np.load(str(X_path))   # (N, 30, 17, 2)
        self._y = np.load(str(y_path))   # (N,)
        self._augment = augment and split == "train"

    def __len__(self) -> int:
        return len(self._y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        keypoints = self._X[idx].copy()   # (30, 17, 2)
        label = int(self._y[idx])

        if self._augment:
            keypoints = _augment(keypoints)

        features = normalize_pose_sequence(keypoints)   # (30, 34)
        return torch.from_numpy(features), torch.tensor(label, dtype=torch.long)


# ── Augmentation ─────────────────────────────────────────────────────────────

def _augment(keypoints: np.ndarray) -> np.ndarray:
    """Light augmentation for pose sequences.

    - Random horizontal flip (inverts x-coordinate and swaps L/R joints)
    - Small Gaussian noise on keypoints
    - Random temporal shift (±2 frames, zero-padded)
    """
    # Horizontal flip (swaps left/right symmetry)
    if np.random.rand() < 0.5:
        keypoints = _hflip_pose(keypoints)

    # Gaussian noise (~1% of typical body scale)
    keypoints = keypoints + np.random.randn(*keypoints.shape).astype(np.float32) * 0.01

    # Temporal shift ±2 frames
    shift = np.random.randint(-2, 3)
    if shift != 0:
        keypoints = np.roll(keypoints, shift, axis=0)
        if shift > 0:
            keypoints[:shift] = 0.0
        else:
            keypoints[shift:] = 0.0

    return keypoints


# COCO left/right joint swap pairs for horizontal flip
_FLIP_PAIRS = [
    (1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)
]


def _hflip_pose(keypoints: np.ndarray) -> np.ndarray:
    """Flip keypoints horizontally and swap left/right joints."""
    kp = keypoints.copy()
    kp[:, :, 0] = -kp[:, :, 0]   # negate x
    for l, r in _FLIP_PAIRS:
        kp[:, l], kp[:, r] = kp[:, r].copy(), kp[:, l].copy()
    return kp


# ── Training loops ────────────────────────────────────────────────────────────

def _train_epoch(
    model: ShotClassifierTCN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: str,
    label_smoothing: float = 0.1,
) -> float:
    model.train()
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    total = 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total += loss.item()
    return total / max(len(loader), 1)


@torch.no_grad()
def _val_epoch(
    model: ShotClassifierTCN,
    loader: DataLoader,
    device: str,
) -> tuple[float, float]:
    """Returns (val_loss, val_accuracy)."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, correct, n = 0.0, 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        total_loss += criterion(logits, y).item()
        correct += (logits.argmax(dim=1) == y).sum().item()
        n += len(y)
    acc = correct / max(n, 1)
    return total_loss / max(len(loader), 1), acc


# ── Entry point ────────────────────────────────────────────────────────────────

def train(
    data_dir: str | Path,
    output_dir: str | Path = "checkpoints/shot_classifier",
    device: str = "cpu",
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 3e-4,
    num_workers: int = 4,
    resume_from: Optional[str | Path] = None,
) -> Path:
    """Train or fine-tune ShotClassifierTCN.

    Args:
        data_dir:    Directory with X_train.npy / y_train.npy / X_val.npy / y_val.npy.
        output_dir:  Where to save checkpoints and metrics.
        device:      'cpu', 'cuda', or 'mps'.
        epochs:      Training epochs.
        batch_size:  Mini-batch size.
        lr:          Peak learning rate (OneCycleLR).
        num_workers: DataLoader workers.
        resume_from: Checkpoint path to resume from.

    Returns:
        Path to the best checkpoint.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_ds = ShotDataset(data_dir, split="train", augment=True)
    val_ds   = ShotDataset(data_dir, split="val",   augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=device != "cpu",
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=device != "cpu",
    )

    if resume_from is not None:
        ckpt = torch.load(resume_from, map_location=device)
        model = ShotClassifierTCN()
        model.load_state_dict(ckpt["model_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_acc = ckpt.get("val_acc", 0.0)
        print(f"Resumed from epoch {start_epoch}, best val acc {best_val_acc:.4f}")
    else:
        model = ShotClassifierTCN()
        start_epoch = 0
        best_val_acc = 0.0

    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = OneCycleLR(
        optimizer, max_lr=lr,
        steps_per_epoch=len(train_loader),
        epochs=epochs - start_epoch,
    )

    best_path = output_dir / "best.pth"
    metrics: list[dict] = []

    for epoch in range(start_epoch, start_epoch + epochs):
        train_loss = _train_epoch(model, train_loader, optimizer, scheduler, device)
        val_loss, val_acc = _val_epoch(model, val_loader, device)

        print(
            f"epoch {epoch:3d}  "
            f"train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  "
            f"val_acc={val_acc:.4f}"
        )
        metrics.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
        })

        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "shot_types": SHOT_TYPES,
        }
        torch.save(ckpt, output_dir / f"epoch_{epoch:03d}.pth")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(ckpt, best_path)
            print(f"  → new best val_acc={best_val_acc:.4f}")

    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return best_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   required=True)
    parser.add_argument("--output_dir", default="checkpoints/shot_classifier")
    parser.add_argument("--device",     default="cpu")
    parser.add_argument("--epochs",     type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr",         type=float, default=3e-4)
    parser.add_argument("--resume_from", default=None)
    args = parser.parse_args()

    best = train(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        resume_from=args.resume_from,
    )
    print(f"Best checkpoint: {best}")
