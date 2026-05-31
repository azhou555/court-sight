"""Fine-tuning module for CourtKeypointNet.

Entry point: fine_tune(). Loads pretrained weights and fine-tunes on a local
copy of the dataset in the format used by yastrebksv/TennisCourtDetector.

Dataset layout (local directory):
    data_dir/
        images/          ← PNG frames (named by annotation "id" field)
        data_train.json  ← [{"id": "<stem>", "kps": [[x,y]×14]}, ...]
        data_val.json

To obtain the dataset, run:
    python -m pipeline.homography.train download --out-dir /path/to/data_dir

Heatmap target: 2D Gaussian centered on each keypoint at 360×640, radius 55px.
Channel 14 is the court center (diagonal intersection), included for training
convergence stability and ignored at inference.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset, random_split

from .model import CourtKeypointNet, INPUT_H, INPUT_W, NUM_CHANNELS_OUT, load_pretrained

_HP_RADIUS = 55


def download_dataset(out_dir: str | Path) -> Path:
    """Download the court keypoint dataset from HuggingFace and extract it.

    The dataset is stored as a single zip in the HF repo
    (Gholamreza/tennis_court_keypoints_dataset). This function downloads it
    with huggingface_hub (handles auth, resumption, and caching) and extracts
    it to out_dir.

    Args:
        out_dir: Destination directory. Will be created if it doesn't exist.

    Returns:
        Path to the extracted data directory (contains images/ and *.json).
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise ImportError("pip install huggingface_hub") from e

    import zipfile

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Skip if already extracted
    if (out_dir / "images").exists() and (out_dir / "data_train.json").exists():
        print(f"Dataset already extracted at {out_dir}")
        return out_dir

    print("Downloading dataset from HuggingFace (7.2 GB, may take a while)...")
    zip_path = hf_hub_download(
        repo_id="Gholamreza/tennis_court_keypoints_dataset",
        filename="tennis_court_det_dataset.zip",
        repo_type="dataset",
        local_dir=str(out_dir),
    )

    print(f"Extracting {zip_path} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)

    # The zip may extract into a subdirectory — find where images/ landed
    images_dirs = list(out_dir.rglob("images"))
    if images_dirs:
        extracted_root = images_dirs[0].parent
        print(f"Extracted to {extracted_root}")
        return extracted_root

    return out_dir


def _draw_gaussian(hm: np.ndarray, center: tuple[int, int], radius: int) -> None:
    cx, cy = center
    h, w = hm.shape
    x0 = max(cx - radius * 2, 0)
    x1 = min(cx + radius * 2 + 1, w)
    y0 = max(cy - radius * 2, 0)
    y1 = min(cy + radius * 2 + 1, h)
    xs = np.arange(x0, x1) - cx
    ys = np.arange(y0, y1) - cy
    xg, yg = np.meshgrid(xs, ys)
    g = np.exp(-(xg ** 2 + yg ** 2) / (2 * (radius / 3) ** 2))
    hm[y0:y1, x0:x1] = np.maximum(hm[y0:y1, x0:x1], g)


def _line_intersection(
    x1: float, y1: float, x2: float, y2: float,
    x3: float, y3: float, x4: float, y4: float,
) -> Optional[tuple[float, float]]:
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-8:
        return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    return x1 + t * (x2 - x1), y1 + t * (y2 - y1)


class CourtKeypointDataset(Dataset):
    """Local dataset for CourtKeypointNet training.

    Expects:
        data_dir/images/<id>.png   (or .jpg)
        data_dir/data_{split}.json  [{"id": str, "kps": [[x,y]×14]}, ...]

    Images are resized to INPUT_H × INPUT_W. Heatmap targets are
    (15, INPUT_H, INPUT_W) float32.
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        augment: bool = True,
    ):
        assert split in ("train", "val")
        data_dir = Path(data_dir)
        ann_path = data_dir / f"data_{split}.json"
        if not ann_path.exists():
            raise FileNotFoundError(
                f"{ann_path} not found. "
                "Run `python -m pipeline.homography.train download --out-dir <dir>` first."
            )
        with open(ann_path) as f:
            self._data = json.load(f)
        self._img_dir = data_dir / "images"
        self.augment = augment and split == "train"

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        item = self._data[idx]
        img_id = item["id"]
        kps = np.array(item["kps"], dtype=np.float32)  # (14, 2)

        img = cv2.imread(str(self._img_dir / f"{img_id}.png"))
        if img is None:
            img = cv2.imread(str(self._img_dir / f"{img_id}.jpg"))
        if img is None:
            raise FileNotFoundError(f"Image not found: {self._img_dir}/{img_id}.(png|jpg)")

        orig_h, orig_w = img.shape[:2]
        img = cv2.resize(img, (INPUT_W, INPUT_H))
        kps[:, 0] *= INPUT_W / orig_w
        kps[:, 1] *= INPUT_H / orig_h

        if self.augment:
            img, kps = _augment(img, kps)

        inp = torch.from_numpy(
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        ).permute(2, 0, 1)

        return inp, _build_heatmaps(kps)


def _build_heatmaps(kps: np.ndarray) -> torch.Tensor:
    hm = np.zeros((NUM_CHANNELS_OUT, INPUT_H, INPUT_W), dtype=np.float32)
    for i, (x, y) in enumerate(kps):
        if 0 <= x < INPUT_W and 0 <= y < INPUT_H:
            _draw_gaussian(hm[i], (int(x), int(y)), _HP_RADIUS)
    ct = _line_intersection(
        kps[0, 0], kps[0, 1], kps[3, 0], kps[3, 1],
        kps[1, 0], kps[1, 1], kps[2, 0], kps[2, 1],
    )
    if ct is not None and 0 <= ct[0] < INPUT_W and 0 <= ct[1] < INPUT_H:
        _draw_gaussian(hm[14], (int(ct[0]), int(ct[1])), _HP_RADIUS)
    return torch.from_numpy(hm)


def _augment(img: np.ndarray, kps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if np.random.rand() < 0.5:
        img = cv2.flip(img, 1)
        kps[:, 0] = INPUT_W - 1 - kps[:, 0]
    alpha = np.random.uniform(0.8, 1.2)
    beta = np.random.randint(-20, 20)
    img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    return img, kps


def _heatmap_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return nn.functional.mse_loss(pred, target)


def train_epoch(
    model: CourtKeypointNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> float:
    model.train()
    total = 0.0
    for imgs, targets in loader:
        imgs, targets = imgs.to(device), targets.to(device)
        optimizer.zero_grad()
        loss = _heatmap_loss(model(imgs), targets)
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / len(loader)


@torch.no_grad()
def validate_epoch(
    model: CourtKeypointNet,
    loader: DataLoader,
    device: str,
) -> float:
    model.eval()
    total = 0.0
    for imgs, targets in loader:
        imgs, targets = imgs.to(device), targets.to(device)
        total += _heatmap_loss(model(imgs), targets).item()
    return total / len(loader)


def fine_tune(
    data_dir: str | Path,
    output_dir: str | Path = "checkpoints/homography",
    weights_path: Optional[str | Path] = None,
    device: str = "cpu",
    epochs: int = 30,
    batch_size: int = 8,
    lr: float = 1e-4,
    val_split: float = 0.15,
    num_workers: int = 4,
    resume_from: Optional[str | Path] = None,
) -> Path:
    """Fine-tune CourtKeypointNet from pretrained weights.

    Args:
        data_dir:     Local dataset directory (must contain images/ and data_train.json).
                      Run download_dataset() first if you don't have it.
        output_dir:   Where to save checkpoints.
        weights_path: Pretrained .pth file. If None, downloads from Google Drive.
        device:       'cpu', 'cuda', or 'mps'.
        epochs:       Training epochs.
        batch_size:   Per-device batch size.
        lr:           Initial learning rate.
        val_split:    Fraction of training data held out for validation when
                      data_val.json is absent.
        num_workers:  DataLoader workers.
        resume_from:  Checkpoint path to resume from.

    Returns:
        Path to the best checkpoint.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(data_dir)

    if resume_from is not None:
        model = CourtKeypointNet()
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

    train_ds = CourtKeypointDataset(data_dir, split="train", augment=True)

    val_json = data_dir / "data_val.json"
    if val_json.exists():
        val_ds = CourtKeypointDataset(data_dir, split="val", augment=False)
    else:
        n_val = max(1, int(len(train_ds) * val_split))
        train_ds, val_ds = random_split(train_ds, [len(train_ds) - n_val, n_val])

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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    dl = sub.add_parser("download", help="Download and extract the dataset")
    dl.add_argument("--out-dir", required=True)

    ft = sub.add_parser("train", help="Fine-tune CourtKeypointNet")
    ft.add_argument("--data-dir", required=True)
    ft.add_argument("--output-dir", default="checkpoints/homography")
    ft.add_argument("--weights", default=None)
    ft.add_argument("--device", default="cpu")
    ft.add_argument("--epochs", type=int, default=30)
    ft.add_argument("--batch-size", type=int, default=8)
    ft.add_argument("--lr", type=float, default=1e-4)
    ft.add_argument("--resume-from", default=None)

    args = parser.parse_args()

    if args.cmd == "download":
        download_dataset(args.out_dir)
    elif args.cmd == "train":
        fine_tune(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            weights_path=args.weights,
            device=args.device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            resume_from=args.resume_from,
        )
