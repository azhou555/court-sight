"""Visual verification of CourtKeypointNet channel → court coordinate mapping.

Loads pretrained weights, runs inference on one or more broadcast frames, and
plots all 15 heatmap channels overlaid on the frame. Use this to manually
confirm that channel N activates on the right court landmark after any weight
update or fine-tuning run.

Usage:
    # Single image
    python scripts/verify_keypoint_channels.py --image path/to/frame.png

    # Multiple images
    python scripts/verify_keypoint_channels.py --image a.png b.png c.png

    # Custom weights
    python scripts/verify_keypoint_channels.py --image frame.png --weights checkpoints/best.pth

To get a broadcast frame:
    ffmpeg -i match.mp4 -vframes 1 -ss 00:01:30 frame.png
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pipeline.homography.model import INPUT_H, INPUT_W, NUM_KEYPOINTS, load_pretrained
from pipeline.homography.detector import COURT_KEYPOINTS_M


def _weighted_centroid(hm: np.ndarray) -> tuple[float, float]:
    """Return (x, y) as the heatmap-weighted centroid of the top-50% activation."""
    threshold = hm.max() * 0.5
    mask = hm > threshold
    ys, xs = np.where(mask)
    if len(xs) == 0:
        flat = int(hm.argmax())
        py, px = divmod(flat, hm.shape[1])
        return float(px), float(py)
    weights = hm[mask]
    cx = float(np.average(xs, weights=weights))
    cy = float(np.average(ys, weights=weights))
    return cx, cy


_CHANNEL_LABELS = [
    "0: far baseline, doubles L",
    "1: far baseline, doubles R",
    "2: near baseline, doubles L",
    "3: near baseline, doubles R",
    "4: far baseline, singles L",
    "5: far baseline, singles R",
    "6: near baseline, singles L",
    "7: near baseline, singles R",
    "8: far service line, L T",
    "9: far service line, R T",
    "10: near service line, L T",
    "11: near service line, R T",
    "12: far center T",
    "13: near center T",
    "14: court center (training only)",
]


def load_image(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def run_inference(model, frame: np.ndarray, device: str) -> np.ndarray:
    """Return (15, H, W) heatmaps for a single frame."""
    img = cv2.resize(frame, (INPUT_W, INPUT_H))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    inp = torch.from_numpy(img_rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    with torch.no_grad():
        return model(inp.to(device))[0].cpu().numpy()  # (15, H, W)


def plot_channels(frame: np.ndarray, heatmaps: np.ndarray, out_path: Path) -> None:
    """5×3 grid: each cell = frame + one heatmap channel overlay + argmax marker."""
    img_rgb = cv2.cvtColor(cv2.resize(frame, (INPUT_W, INPUT_H)), cv2.COLOR_BGR2RGB)
    n_ch = heatmaps.shape[0]
    ncols, nrows = 5, math.ceil(n_ch / 5)

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 2.5))
    axes = axes.flatten()

    for ch in range(n_ch):
        ax = axes[ch]
        hm = heatmaps[ch]
        hm_norm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-8)

        ax.imshow(img_rgb, alpha=0.7)
        ax.imshow(hm_norm, cmap="hot", alpha=0.6, vmin=0, vmax=1)

        px, py = _weighted_centroid(hm)
        ax.plot(px, py, "c+", markersize=10, markeredgewidth=1.5)

        if ch < NUM_KEYPOINTS:
            cx_m, cy_m = COURT_KEYPOINTS_M[ch]
            coord_str = f"({cx_m:+.2f}, {cy_m:.2f})m"
        else:
            coord_str = "computed"

        peak = float(hm.max())
        ax.set_title(f"{_CHANNEL_LABELS[ch]}\npeak={peak:.3f}  {coord_str}", fontsize=6.5)
        ax.axis("off")

    for i in range(n_ch, len(axes)):
        axes[i].axis("off")

    fig.suptitle(
        "CourtKeypointNet channel verification — cyan + is argmax\n"
        "Each activation should align with its labeled court landmark",
        fontsize=9,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify keypoint channel assignments")
    parser.add_argument(
        "--image", nargs="+", required=True,
        help="Path(s) to broadcast frame(s). "
             "Extract with: ffmpeg -i match.mp4 -vframes 1 -ss 00:01:30 frame.png",
    )
    parser.add_argument("--weights", default=None, help="Path to .pth weights (default: auto-download)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-dir", default="outputs/channel_verification")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading model...")
    model = load_pretrained(weights_path=args.weights, device=args.device)
    model.eval()

    for i, img_path in enumerate(args.image):
        print(f"Processing {img_path} ({i + 1}/{len(args.image)})...")
        frame = load_image(img_path)
        heatmaps = run_inference(model, frame, args.device)
        out_path = out_dir / f"channel_map_{i:02d}.png"
        plot_channels(frame, heatmaps, out_path)

    print(
        "\nInspect the output images.\n"
        "Each subplot shows one heatmap channel overlaid on the frame.\n"
        "The cyan + should land on the labeled court landmark.\n"
        "If it doesn't, update COURT_KEYPOINTS_M in detector.py."
    )


if __name__ == "__main__":
    main()
