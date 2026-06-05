"""Training pipeline for ShotSafetyModel.

Builds a GBT dataset from processed ShotRecord JSON files and trains
the P(make | geometry) model.

Label assignment:
  - Shots with a detected ball bounce (ball_landing_zone is not None): label = 1
  - Shots with no bounce detected: label = 0
  This is a pragmatic proxy — TrackNet detects in-court bounces, so
  landing_zone=None correlates with out/missed shots in pro footage.
  Inter-rally shots are always label=1 (rally continued).

Missing features (net_clearance, incoming_depth) are set to 0 and can
be improved once 3D ball trajectory or per-shot context is extracted.

Usage:
    from src.models.execution_prob.train import train
    train(shot_record_paths=["data/match1_shots.json", ...],
          output_dir="checkpoints/shot_safety")

Or from CLI:
    python -m models.execution_prob.train \\
        --records data/shots/*.json \\
        --output_dir checkpoints/shot_safety
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np

from .model import ShotSafetyModel, ShotGeometryFeatures, SHOT_TYPES, zone_to_coords

# ── MCP shot code → SHOT_TYPES ────────────────────────────────────────────────

# Maps MCP shot type codes to our canonical SHOT_TYPES strings
_MCP_TO_SHOT_TYPE: dict[str, str] = {
    "serve": "serve",
    "f":     "forehand_groundstroke",
    "b":     "backhand_groundstroke",
    "r":     "forehand_slice",
    "s":     "backhand_slice",
    "v":     "forehand_volley",
    "z":     "backhand_volley",
    "l":     "lob",
    "o":     "overhead",
    "p":     "forehand_groundstroke",  # drop shot → treat as FH for geometry
    "q":     "backhand_groundstroke",  # backhand drop
    "j":     "forehand_volley",        # swing volley
    "k":     "backhand_volley",
    "y":     "lob",                    # tweener
    "u":     "forehand_volley",
    "t":     "backhand_volley",
}

# Feature names for importance reporting (must match ShotGeometryFeatures.to_vector order)
# 6 scalar features + 10 shot-type one-hot + 3 context = 19 total
FEATURE_NAMES: list[str] = [
    "lateral_margin_m",
    "depth_margin_m",
    "net_clearance_m",
    "shot_direction_deg_norm",
    "landing_x_norm",
    "landing_y_norm",
    *[f"shot_type_{s}" for s in SHOT_TYPES],   # 10
    "striker_y_norm",
    "striker_x_norm",
    "incoming_depth_norm",
]
assert len(FEATURE_NAMES) == 19, len(FEATURE_NAMES)


# ── Feature extraction ────────────────────────────────────────────────────────

def features_from_shot_record(record: dict) -> Optional[ShotGeometryFeatures]:
    """Extract ShotGeometryFeatures from a ShotRecord dict.

    Returns None if essential fields are missing.
    """
    shot_type = _MCP_TO_SHOT_TYPE.get(record.get("shot_type_mcp", ""), "")
    if shot_type not in SHOT_TYPES:
        return None

    striker_pos = record.get("striker_position_m")
    if not striker_pos or len(striker_pos) < 2:
        return None
    striker_x = float(striker_pos[0])
    striker_y = float(striker_pos[1])

    # Landing position: prefer exact contact position, fall back to zone centre
    landing_zone = record.get("ball_landing_zone")
    contact_pos = record.get("ball_contact_position_m")
    zone_estimated = False

    landing_x, landing_y = zone_to_coords(landing_zone)
    zone_estimated = True   # zone-centre is always approximate

    if contact_pos and len(contact_pos) >= 2:
        cx, cy = float(contact_pos[0]), float(contact_pos[1])
    else:
        cx, cy = striker_x, striker_y  # fallback

    # Shot direction: angle from striker to landing in degrees
    dx = landing_x - cx
    dy = landing_y - cy
    shot_direction_deg = float(np.degrees(np.arctan2(dx, max(dy, 0.1))))

    # Lateral margin: distance from landing x to nearest singles sideline
    lateral_margin = min(
        abs(landing_x - (-4.115)),
        abs(4.115 - landing_x),
    )

    # Depth margin: distance from landing y to the far baseline
    depth_margin = max(0.0, 23.77 - landing_y)

    # Opponent y as proxy for "incoming depth"
    opp_pos = record.get("opponent_position_m")
    incoming_depth = float(opp_pos[1]) if opp_pos and len(opp_pos) >= 2 else 11.885

    return ShotGeometryFeatures(
        lateral_margin_m=lateral_margin,
        depth_margin_m=depth_margin,
        net_clearance_m=0.0,          # placeholder: needs ball height at net
        shot_direction_deg=shot_direction_deg,
        landing_x=landing_x,
        landing_y=landing_y,
        shot_type=shot_type,
        striker_y_m=striker_y,
        striker_x_m=striker_x,
        incoming_depth_m=incoming_depth,
        target_zone_estimated=zone_estimated,
    )


def label_from_shot_record(record: dict, shot_idx_in_point: int, point_rally_len: int) -> int:
    """Assign binary label: 1 = shot made, 0 = miss.

    Logic:
    - If not the last shot in the point: label = 1 (rally continued)
    - If last shot AND bounce detected (ball_landing_zone not None): label = 1
    - If last shot AND no bounce: label = 0
    """
    is_last = shot_idx_in_point >= point_rally_len - 1
    if not is_last:
        return 1
    return 1 if record.get("ball_landing_zone") is not None else 0


# ── Dataset construction ──────────────────────────────────────────────────────

def build_dataset(
    shot_records: list[dict],
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a list of ShotRecord dicts to (X, y) numpy arrays.

    Groups shots by point_id to determine last-shot status.

    Returns:
        X: (N, 26) float32 feature matrix
        y: (N,)    int32 label vector
    """
    # Group by point to find rally lengths
    by_point: dict[str, list[dict]] = {}
    for r in shot_records:
        pt = r.get("point_id", "unknown")
        by_point.setdefault(pt, []).append(r)

    X_list, y_list = [], []
    for pt_id, shots in by_point.items():
        shots_sorted = sorted(shots, key=lambda r: r.get("shot_in_rally", 0))
        rally_len = len(shots_sorted)
        for idx, record in enumerate(shots_sorted):
            feat = features_from_shot_record(record)
            if feat is None:
                continue
            label = label_from_shot_record(record, idx, rally_len)
            X_list.append(feat.to_vector())
            y_list.append(label)

    if not X_list:
        raise ValueError("No valid features extracted from shot records.")

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int32)


def load_shot_records(paths: list[str | Path]) -> list[dict]:
    """Load ShotRecord dicts from one or more JSON files."""
    records = []
    for p in paths:
        data = json.loads(Path(p).read_text())
        if isinstance(data, list):
            records.extend(data)
        else:
            records.append(data)
    return records


# ── Entry point ───────────────────────────────────────────────────────────────

def train(
    shot_record_paths: list[str | Path],
    output_dir: str | Path = "checkpoints/shot_safety",
    n_estimators: int = 500,
    max_depth: int = 5,
    learning_rate: float = 0.05,
) -> Path:
    """Train and save the ShotSafetyModel.

    Args:
        shot_record_paths: JSON files containing lists of ShotRecord dicts.
        output_dir:        Where to save the model pickle.

    Returns:
        Path to the saved model file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading shot records…")
    records = load_shot_records(shot_record_paths)
    print(f"  {len(records)} total records")

    print("Building feature matrix…")
    X, y = build_dataset(records)
    print(f"  {len(y)} samples  |  make={y.sum()}  miss={(y==0).sum()}"
          f"  ({y.mean()*100:.1f}% make rate)")

    print("Training XGBClassifier + isotonic calibration…")
    model = ShotSafetyModel()
    metrics = model.train(
        X, y,
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
    )
    print(
        f"  best_iter={metrics['best_iteration']}  "
        f"cal_auc={metrics['cal_auc']:.4f}  "
        f"cal_brier={metrics['cal_brier']:.4f}  "
        f"cal_logloss={metrics['cal_logloss']:.4f}"
    )

    model_path = output_dir / "shot_safety_model.pkl"
    model.save(model_path)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Saved to {model_path}")

    # Feature importance
    importance = model.feature_importance()
    if importance:
        print("\nTop-10 features by gain:")
        for name, score in sorted(importance.items(), key=lambda x: -x[1])[:10]:
            print(f"  {name:<30} {score:.1f}")

    return model_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ShotSafetyModel")
    parser.add_argument("--records", nargs="+", required=True,
                        help="ShotRecord JSON file(s)")
    parser.add_argument("--output_dir", default="checkpoints/shot_safety")
    parser.add_argument("--n_estimators", type=int, default=500)
    parser.add_argument("--max_depth", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=0.05)
    args = parser.parse_args()

    train(
        shot_record_paths=args.records,
        output_dir=args.output_dir,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
    )
