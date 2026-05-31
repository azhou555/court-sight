"""Fine-tuning module for BallTrackerNet.

Entry point: fine_tune(). Loads pretrained weights and fine-tunes on a local
dataset in the format used by yastrebksv/TrackNet.

Dataset layout:
    data_dir/
        images/
            game1/Clip1/0000.jpg  ...
        labels_train.csv          ← columns: file,x,y,visibility
        labels_val.csv

Ground-truth heatmap: 2D Gaussian at ball (x, y), normalised 0–255 and
quantised as class labels for CrossEntropyLoss.

To obtain the dataset, download from the link in docs/architecture/04_ball_tracking.md
and extract into data_dir.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset

from .model import BallTrackerNet, INPUT_H, INPUT_W, NUM_CLASSES, load_pretrained

_HP_RADIUS = 5    # Gaussian heatmap radius in pixels at model resolution
_FPS = 30


def _draw_gaussian(shape: tuple[int, int], cx: int, cy: int, radius: int) -> np.ndarray:
    hm = np.zeros(shape, dtype=np.float32)
    h, w = shape
    x0, x1 = max(cx - radius * 3, 0), min(cx + radius * 3 + 1, w)
    y0, y1 = max(cy - radius * 3, 0), min(cy + radius * 3 + 1, h)
    xs = np.arange(x0, x1) - cx
    ys = np.arange(y0, y1) - cy
    xg, yg = np.meshgrid(xs, ys)
    g = np.exp(-(xg ** 2 + yg ** 2) / (2 * (radius / 3) ** 2))
    hm[y0:y1, x0:x1] = g
    return hm


class BallTrackDataset(Dataset):
    """Three-frame input + per-pixel GT class label for TrackNet training.

    Each sample is (frames_tensor, gt_classes) where:
        frames_tensor: (9, 360, 640) float32
        gt_classes:    (360×640,)   int64  — Gaussian heatmap quantised 0–255
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        augment: bool = True,
    ):
        assert split in ("train", "val")
        data_dir = Path(data_dir)
        csv_path = data_dir / f"labels_{split}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"{csv_path} not found. "
                "Download the TrackNet dataset and extract it into data_dir first."
            )

        self._data_dir = data_dir
        self.augment = augment and split == "train"
        self._samples: list[tuple[list[Path], int, int, int]] = []

        rows: list[dict] = []
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)

        # Build (frame_triplet, x, y, visibility) entries
        for i in range(2, len(rows)):
            vis = int(rows[i].get("visibility", 1))
            x = int(float(rows[i].get("x") or 0))
            y = int(float(rows[i].get("y") or 0))
            paths = [
                data_dir / "images" / rows[i - 2]["file"],
                data_dir / "images" / rows[i - 1]["file"],
                data_dir / "images" / rows[i]["file"],
            ]
            if all(p.exists() for p in paths):
                self._samples.append((paths, x, y, vis))

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        paths, bx, by, vis = self._samples[idx]

        frames = [cv2.imread(str(p)) for p in paths]
        orig_h, orig_w = frames[0].shape[:2]

        # Scale ball coords to model resolution
        mx = int(bx * INPUT_W / orig_w)
        my = int(by * INPUT_H / orig_h)

        channels = []
        for f in frames:
            f_r = cv2.resize(f, (INPUT_W, INPUT_H))
            f_rgb = cv2.cvtColor(f_r, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            channels.append(f_rgb)

        if self.augment:
            channels, mx, my = _augment(channels, mx, my)

        stacked = np.concatenate(channels, axis=2)          # (H, W, 9)
        inp = torch.from_numpy(stacked).permute(2, 0, 1)    # (9, H, W)

        if vis == 0:
            gt = np.zeros(INPUT_H * INPUT_W, dtype=np.int64)
        else:
            hm = _draw_gaussian((INPUT_H, INPUT_W), mx, my, _HP_RADIUS)
            gt = (hm.flatten() * 255).astype(np.int64).clip(0, 255)

        return inp, torch.from_numpy(gt)


def _augment(
    channels: list[np.ndarray], bx: int, by: int
) -> tuple[list[np.ndarray], int, int]:
    if np.random.rand() < 0.5:
        channels = [cv2.flip(f, 1) for f in channels]
        bx = INPUT_W - 1 - bx
    alpha = np.random.uniform(0.8, 1.2)
    beta = np.random.randint(-15, 15)
    channels = [np.clip(f * alpha + beta / 255.0, 0, 1) for f in channels]
    return channels, bx, by


def train_epoch(
    model: BallTrackerNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> float:
    model.train()
    criterion = nn.CrossEntropyLoss()
    total = 0.0
    for imgs, targets in loader:
        imgs, targets = imgs.to(device), targets.to(device)
        optimizer.zero_grad()
        out = model(imgs)                # (B, 256, H×W)
        loss = criterion(out, targets)   # targets: (B, H×W)
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / len(loader)


@torch.no_grad()
def validate_epoch(
    model: BallTrackerNet,
    loader: DataLoader,
    device: str,
) -> float:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total = 0.0
    for imgs, targets in loader:
        imgs, targets = imgs.to(device), targets.to(device)
        total += criterion(model(imgs), targets).item()
    return total / len(loader)


def fine_tune(
    data_dir: str | Path,
    output_dir: str | Path = "checkpoints/ball",
    weights_path: Optional[str | Path] = None,
    device: str = "cpu",
    epochs: int = 30,
    batch_size: int = 4,
    lr: float = 1e-4,
    num_workers: int = 4,
    resume_from: Optional[str | Path] = None,
) -> Path:
    """Fine-tune BallTrackerNet from pretrained weights.

    Args:
        data_dir:     Local dataset directory (images/ + labels_train.csv).
        output_dir:   Where to save checkpoints.
        weights_path: Pretrained .pth file. If None, downloads from Google Drive.
        device:       'cpu', 'cuda', or 'mps'.
        epochs:       Training epochs.
        batch_size:   Per-device batch size (keep small — inputs are large).
        lr:           Initial learning rate.
        num_workers:  DataLoader workers.
        resume_from:  Checkpoint path to resume from.

    Returns:
        Path to the best checkpoint.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if resume_from is not None:
        model = BallTrackerNet()
        ckpt = torch.load(resume_from, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val = ckpt.get("val_loss", float("inf"))
        print(f"Resuming from epoch {start_epoch}, best val loss {best_val:.6f}")
    else:
        model = load_pretrained(weights_path, device=device)
        start_epoch = 0
        best_val = float("inf")

    model = model.to(device)

    train_ds = BallTrackDataset(data_dir, split="train", augment=True)
    val_ds = BallTrackDataset(data_dir, split="val", augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=device != "cpu",
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=device != "cpu",
    )

    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.5, min_lr=1e-6)

    best_path = output_dir / "best.pth"
    for epoch in range(start_epoch, start_epoch + epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss = validate_epoch(model, val_loader, device)
        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"epoch {epoch:3d}  train={train_loss:.6f}  val={val_loss:.6f}  lr={current_lr:.2e}")

        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
        }
        torch.save(ckpt, output_dir / f"epoch_{epoch:03d}.pth")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt, best_path)
            print(f"  → new best ({best_val:.6f})")

    return best_path
