"""Prepare THETIS dataset for ShotClassifierTCN training.

Downloads THETIS RGB clips from GitHub, runs YOLO26-pose on each clip to
extract COCO-17 keypoint sequences, and saves train/val numpy arrays.

THETIS has 1,980 RGB clips across 12 shot classes from 55 subjects.
All 12 classes are mapped to our 10-class SHOT_TYPES label set.

Usage:
    # Download from GitHub and process (slow first run, ~2-4 hours on CPU):
    python scripts/prepare_thetis.py --output_dir data/thetis_poses --device cpu

    # Process a locally downloaded copy:
    python scripts/prepare_thetis.py \\
        --thetis_dir /data/thetis/VIDEO_RGB \\
        --output_dir data/thetis_poses \\
        --device cpu

Output:
    data/thetis_poses/
        X_train.npy   (N_train, 30, 17, 2)  float32
        y_train.npy   (N_train,)             int64
        X_val.npy     (N_val,   30, 17, 2)  float32
        y_val.npy     (N_val,)              int64
        label_map.json
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.shot_classifier.model import SHOT_TYPES

# ── Label mapping ─────────────────────────────────────────────────────────────

# Maps THETIS directory names to our SHOT_TYPES labels.
# THETIS doesn't include lob or drop_shot — those rely on fine-tuning.
THETIS_LABEL_MAP: dict[str, str] = {
    "backhand":          "backhand_groundstroke",
    "backhand2hands":    "backhand_groundstroke",
    "backhand_slice":    "backhand_slice",
    "backhand_volley":   "backhand_volley",
    "flat_service":      "serve",
    "forehand_flat":     "forehand_groundstroke",
    "forehand_openstands": "forehand_groundstroke",
    "forehand_slice":    "forehand_slice",
    "forehand_volley":   "forehand_volley",
    "kick_service":      "serve",
    "slice_service":     "serve",
    "smash":             "overhead",
}

LABEL_TO_IDX = {cls: i for i, cls in enumerate(SHOT_TYPES)}

_GITHUB_BASE = (
    "https://raw.githubusercontent.com/THETIS-dataset/dataset/master/VIDEO_RGB"
)

# ── Core processing ───────────────────────────────────────────────────────────

def extract_poses_from_clip(
    video_path: str | Path,
    model,
    device: str,
) -> np.ndarray | None:
    """Run YOLO26-pose on every frame of a clip; return (T, 17, 2) or None.

    For THETIS clips the person fills most of the frame, so we run pose
    on the full image and take the highest-confidence detection.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    all_kps = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        results = model(frame, verbose=False, device=device)
        if (
            not results
            or results[0].keypoints is None
            or len(results[0].keypoints) == 0
        ):
            all_kps.append(np.zeros((17, 2), dtype=np.float32))
            continue

        boxes_conf = (
            results[0].boxes.conf.cpu().numpy()
            if results[0].boxes is not None
            else np.array([1.0])
        )
        best = int(boxes_conf.argmax())
        kp_data = results[0].keypoints.data[best].cpu().numpy()  # (17, 3)
        kps = kp_data[:, :2].astype(np.float32)
        confs = kp_data[:, 2]
        kps[confs < 0.3] = 0.0
        all_kps.append(kps)

    cap.release()

    if len(all_kps) < 10:
        return None

    return np.array(all_kps, dtype=np.float32)   # (T, 17, 2)


def window_from_clip(poses: np.ndarray) -> np.ndarray:
    """Extract a 30-frame stroke window from a (T, 17, 2) clip.

    THETIS clips are isolated stroke demonstrations; the stroke peak is
    approximately in the centre.  We take [centre-20, centre+10).
    """
    T = len(poses)
    centre = T // 2
    start = max(0, centre - 20)
    end   = start + 30

    window = np.zeros((30, 17, 2), dtype=np.float32)
    src_len = min(30, T - start, end - start)
    window[:src_len] = poses[start:start + src_len]
    return window


# ── Download helpers ──────────────────────────────────────────────────────────

def _list_github_clips(class_name: str) -> list[str]:
    """Return list of filenames in a THETIS class directory on GitHub."""
    import urllib.request, json as _json
    api_url = (
        f"https://api.github.com/repos/THETIS-dataset/dataset"
        f"/contents/VIDEO_RGB/{class_name}"
    )
    try:
        with urllib.request.urlopen(api_url, timeout=15) as resp:
            items = _json.loads(resp.read())
        return [item["name"] for item in items if item["name"].endswith(".avi")]
    except Exception as e:
        print(f"  WARNING: GitHub API failed for {class_name}: {e}")
        return []


def _download_clip(class_name: str, filename: str, dest: Path) -> bool:
    import urllib.request
    url = f"{_GITHUB_BASE}/{class_name}/{filename}"
    try:
        urllib.request.urlretrieve(url, str(dest))
        return True
    except Exception as e:
        print(f"  WARNING: download failed {filename}: {e}")
        return False


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract THETIS pose sequences for shot classifier training"
    )
    parser.add_argument(
        "--thetis_dir", default=None,
        help="Path to local VIDEO_RGB directory (skip download if provided)"
    )
    parser.add_argument(
        "--output_dir", default="data/thetis_poses",
        help="Where to save X_train/y_train/X_val/y_val.npy"
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--val_fraction", type=float, default=0.2,
        help="Fraction of clips to hold out per class for validation"
    )
    args = parser.parse_args()

    from ultralytics import YOLO
    model = YOLO("yolo26m-pose.pt")
    print(f"YOLO26m-pose loaded on {args.device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X_train_list, y_train_list = [], []
    X_val_list,   y_val_list   = [], []

    for thetis_class, shot_type in THETIS_LABEL_MAP.items():
        label_idx = LABEL_TO_IDX.get(shot_type)
        if label_idx is None:
            continue

        print(f"\n[{thetis_class}] → {shot_type} (class {label_idx})")

        if args.thetis_dir is not None:
            # Local directory
            class_dir = Path(args.thetis_dir) / thetis_class
            clips = sorted(class_dir.glob("*.avi")) if class_dir.exists() else []
        else:
            # Download from GitHub into a temp directory
            filenames = _list_github_clips(thetis_class)
            clips = []
            tmp_dir = output_dir / "_tmp" / thetis_class
            tmp_dir.mkdir(parents=True, exist_ok=True)
            for fn in filenames:
                dest = tmp_dir / fn
                if not dest.exists():
                    if not _download_clip(thetis_class, fn, dest):
                        continue
                clips.append(dest)

        if not clips:
            print(f"  No clips found — skipping")
            continue

        # Shuffle deterministically per class for reproducible splits
        rng = np.random.default_rng(seed=42)
        order = rng.permutation(len(clips)).tolist()
        n_val = max(1, int(len(clips) * args.val_fraction))
        val_indices = set(order[:n_val])

        processed = 0
        for i, clip_path in enumerate(clips):
            poses = extract_poses_from_clip(clip_path, model, args.device)
            if poses is None:
                continue

            window = window_from_clip(poses)
            if i in val_indices:
                X_val_list.append(window)
                y_val_list.append(label_idx)
            else:
                X_train_list.append(window)
                y_train_list.append(label_idx)
            processed += 1

        print(f"  Processed {processed}/{len(clips)} clips")

    if not X_train_list:
        print("No sequences extracted. Check THETIS path or download errors.")
        return

    for split, X_list, y_list in [
        ("train", X_train_list, y_train_list),
        ("val",   X_val_list,   y_val_list),
    ]:
        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.int64)
        np.save(str(output_dir / f"X_{split}.npy"), X)
        np.save(str(output_dir / f"y_{split}.npy"), y)
        print(f"\n{split}: {len(y)} sequences — shape {X.shape}")
        for idx, name in enumerate(SHOT_TYPES):
            count = int((y == idx).sum())
            if count:
                print(f"  {name}: {count}")

    (output_dir / "label_map.json").write_text(
        json.dumps({"shot_types": SHOT_TYPES, "thetis_map": THETIS_LABEL_MAP}, indent=2)
    )
    print(f"\nSaved to {output_dir}")


if __name__ == "__main__":
    main()
