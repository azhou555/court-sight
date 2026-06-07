"""Training pipeline for the response-distribution stage of Model 2E.

Builds a dataset of CONSECUTIVE shot pairs from processed ShotRecord JSON
files and trains the two GBT regressors that predict (mu_x, mu_y) of the
opponent's response landing distribution.

Training example construction:
  For each point, sort shots by shot_in_rally. Each adjacent pair
  (shot N, shot N+1) yields:
    - Input:  features of shot N (your landing, opponent state at your contact)
    - Label:  landing coords of shot N+1 (the opponent's response)

Label source (see docs/architecture/10_neutral_position.md):
  shot N+1's ball_contact_position_m (continuous) → zone centre fallback →
  pair skipped if neither is available.

Two ShotRecord features are not emitted by the alignment pipeline and use
placeholders here:
  - ball_height_at_bounce: 0.0 (no ball-arc extraction yet)
  - opponent_displacement_from_neutral: geometric-center proxy (distance from
    opponent to the centre of their half-court), since the true neutral is
    what this model predicts (circular).

Usage:
    from src.models.neutral_position.train import train
    train(shot_record_paths=["data/match1_shots.json", ...],
          output_dir="checkpoints/neutral_position")

Or from CLI:
    python -m src.models.neutral_position.train \\
        --records data/shots/*.json \\
        --output_dir checkpoints/neutral_position
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np

from .model import ResponseDistributionModel, ResponseDistributionFeatures
from src.models.execution_prob.model import zone_to_coords, SHOT_TYPES
from src.models.execution_prob.train import _MCP_TO_SHOT_TYPE

# ── Feature names (must match ResponseDistributionFeatures.to_vector order) ────
# 11 elements; ball_height_at_bounce is index 2, displacement is index 7.
FEATURE_NAMES: list[str] = [
    "your_landing_x_norm",
    "your_landing_y_norm",
    "ball_height_at_bounce",
    "opponent_pos_x_norm",
    "opponent_pos_y_norm",
    "opponent_vel_x_norm",
    "opponent_vel_y_norm",
    "opponent_displacement_norm",
    "rally_depth_norm",
    "prev_shot_direction_1_norm",
    "prev_shot_direction_2_norm",
    *[f"shot_type_{s}" for s in SHOT_TYPES],     # 10 dims, appended last
]
assert len(FEATURE_NAMES) == 21, len(FEATURE_NAMES)

# MCP direction code → angle in degrees (down-the-line / crosscourt geometry)
_DIRECTION_ANGLES: dict[str, float] = {"1": -45.0, "2": 0.0, "3": 45.0, "": 0.0}

# Court geometry
_NET_Y = 11.885
_FAR_HALF_CENTER = np.array([0.0, 17.83])    # centre of the far player's half
_NEAR_HALF_CENTER = np.array([0.0, 5.94])    # centre of the near player's half


# ── Helpers ───────────────────────────────────────────────────────────────────

def _direction_to_angle(direction_mcp) -> float:
    return _DIRECTION_ANGLES.get(str(direction_mcp), 0.0)


def _opponent_half_center(striker_role: str) -> np.ndarray:
    """The opponent occupies the half opposite the striker."""
    return _FAR_HALF_CENTER if striker_role == "near" else _NEAR_HALF_CENTER


def _opponent_displacement_proxy(opp_pos: list, striker_role: str) -> float:
    """Geometric-center proxy for opponent_displacement_from_neutral.

    Distance from the opponent to the centre of their half-court. Non-circular
    stand-in for the true neutral (which this model predicts). Replaced by real
    neutral-tracking at inference time.
    """
    center = _opponent_half_center(striker_role)
    d = float(np.linalg.norm(np.array([opp_pos[0], opp_pos[1]]) - center))
    return float(np.clip(d, 0.0, 10.0))


def _point_key(record: dict) -> str:
    """Strip the trailing shot index from point_id to get a point-unique key.

    ShotRecord.point_id is "{match_id}_{pt}_{shot_idx}" (shot-unique), so the
    last underscore-delimited segment must be removed to group shots by point.
    """
    pid = str(record.get("point_id", ""))
    return pid.rsplit("_", 1)[0] if "_" in pid else pid


def _label_from_next_shot(next_record: dict) -> Optional[tuple[float, float]]:
    """Landing coords (zone centre) of the opponent's response, shot N+1.

    The label is taken from shot N+1's ``ball_landing_zone`` only. Crucially we
    do NOT use ``ball_contact_position_m``: that is the *contact* point where the
    opponent struck N+1 — i.e. where YOUR shot N landed — so using it as the
    label would leak the ``your_landing`` feature and predict the wrong target.
    Only the landing zone of N+1 describes where the opponent sent their reply.
    Returns None (skip the pair) when no bounce zone was detected.
    """
    zone = next_record.get("ball_landing_zone")
    if zone is None:
        return None
    x, y = zone_to_coords(zone)
    return float(x), float(y)


# ── Feature extraction ────────────────────────────────────────────────────────

def features_from_shot_record(record: dict) -> Optional[ResponseDistributionFeatures]:
    """Extract ResponseDistributionFeatures from a single ShotRecord dict.

    Returns None if essential positional fields are missing.
    """
    striker_pos = record.get("striker_position_m")
    opp_pos = record.get("opponent_position_m")
    if not striker_pos or len(striker_pos) < 2:
        return None
    if not opp_pos or len(opp_pos) < 2:
        return None

    # Where shot N landed (your_landing). Only the bounce zone is available;
    # skip the pair if it's missing rather than injecting a center-court default.
    landing_zone = record.get("ball_landing_zone")
    if landing_zone is None:
        return None
    landing_x, landing_y = zone_to_coords(landing_zone)

    shot_type = _MCP_TO_SHOT_TYPE.get(record.get("shot_type_mcp", ""), "")
    if shot_type not in SHOT_TYPES:
        shot_type = "forehand_groundstroke"  # safe default

    opp_vel = record.get("opponent_velocity_ms") or [0.0, 0.0]
    disp = _opponent_displacement_proxy(opp_pos, record.get("striker_role", "near"))

    rc = record.get("rally_context") or []
    prev1 = _direction_to_angle(rc[-1].get("direction", "")) if len(rc) >= 1 else 0.0
    prev2 = _direction_to_angle(rc[-2].get("direction", "")) if len(rc) >= 2 else 0.0

    return ResponseDistributionFeatures(
        your_landing_x=float(landing_x),
        your_landing_y=float(landing_y),
        ball_height_at_bounce=0.0,                       # placeholder: no ball arc
        opponent_pos_x=float(opp_pos[0]),
        opponent_pos_y=float(opp_pos[1]),
        opponent_vel_x=float(opp_vel[0]),
        opponent_vel_y=float(opp_vel[1]),
        opponent_displacement_from_neutral=disp,
        your_shot_type=shot_type,
        rally_depth=int(record.get("shot_in_rally", 0)),
        prev_shot_direction_1=prev1,
        prev_shot_direction_2=prev2,
    )


# ── Dataset construction ──────────────────────────────────────────────────────

def build_dataset(
    shot_records: list[dict],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Convert ShotRecord dicts to (X, y_x, y_y, stats) for the GBT regressors.

    Groups by point (via _point_key), sorts by shot_in_rally, and emits one
    example per consecutive shot pair. Labels y_x, y_y are RAW court metres.

    Returns:
        X:     (N, 11) float32 feature matrix
        y_x:   (N,)    float32 landing-x labels (metres)
        y_y:   (N,)    float32 landing-y labels (metres)
        stats: dict with pair counts and label-quality breakdown
    """
    by_point: dict[str, list[dict]] = {}
    for r in shot_records:
        by_point.setdefault(_point_key(r), []).append(r)

    X_list, yx_list, yy_list = [], [], []
    stats = {
        "n_pairs": 0,
        "n_skipped_no_features": 0,
        "n_skipped_no_label": 0,
    }

    for shots in by_point.values():
        shots_sorted = sorted(shots, key=lambda r: r.get("shot_in_rally", 0))
        for i in range(len(shots_sorted) - 1):
            cur, nxt = shots_sorted[i], shots_sorted[i + 1]
            feat = features_from_shot_record(cur)
            if feat is None:
                stats["n_skipped_no_features"] += 1
                continue
            label = _label_from_next_shot(nxt)
            if label is None:
                stats["n_skipped_no_label"] += 1
                continue
            X_list.append(feat.to_vector())
            yx_list.append(label[0])
            yy_list.append(label[1])
            stats["n_pairs"] += 1

    if not X_list:
        raise ValueError("No valid consecutive shot pairs extracted from records.")

    return (
        np.array(X_list, dtype=np.float32),
        np.array(yx_list, dtype=np.float32),
        np.array(yy_list, dtype=np.float32),
        stats,
    )


def load_shot_records(paths: list[str | Path]) -> list[dict]:
    """Load ShotRecord dicts from one or more JSON files."""
    records: list[dict] = []
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
    output_dir: str | Path = "checkpoints/neutral_position",
    n_estimators: int = 300,
    max_depth: int = 4,
    learning_rate: float = 0.05,
    early_stopping_rounds: int = 20,
) -> Path:
    """Train and save the response-distribution model (Stage 1 of Model 2E).

    Args:
        shot_record_paths: JSON files containing lists of ShotRecord dicts.
        output_dir:        Where to save the model pickle + metrics.

    Returns:
        Path to the saved model file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading shot records…")
    records = load_shot_records(shot_record_paths)
    print(f"  {len(records)} total records")

    print("Building consecutive-pair dataset…")
    X, y_x, y_y, stats = build_dataset(records)
    print(
        f"  {stats['n_pairs']} pairs  |  "
        f"skipped(feat)={stats['n_skipped_no_features']} "
        f"skipped(label)={stats['n_skipped_no_label']}"
    )

    print("Training response-distribution regressors…")
    model = ResponseDistributionModel()
    metrics = model.train(
        X, y_x, y_y,
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        early_stopping_rounds=early_stopping_rounds,
    )
    metrics.update(stats)
    print(
        f"  mae_x={metrics['mae_x']:.3f}m  mae_y={metrics['mae_y']:.3f}m  "
        f"sigma_x={metrics['sigma_x']:.3f}m  sigma_y={metrics['sigma_y']:.3f}m"
    )

    model_path = output_dir / "response_distribution_model.pkl"
    model.save(model_path)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Saved to {model_path}")

    importance = model.feature_importance()
    if importance:
        print("\nTop features (mu_x) by gain:")
        for name, score in importance["x"][:8]:
            print(f"  {name:<28} {score:.1f}")

    return model_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train response-distribution model (2E)")
    parser.add_argument("--records", nargs="+", required=True,
                        help="ShotRecord JSON file(s)")
    parser.add_argument("--output_dir", default="checkpoints/neutral_position")
    parser.add_argument("--n_estimators", type=int, default=300)
    parser.add_argument("--max_depth", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=0.05)
    parser.add_argument("--early_stopping_rounds", type=int, default=20)
    args = parser.parse_args()

    train(
        shot_record_paths=args.records,
        output_dir=args.output_dir,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        early_stopping_rounds=args.early_stopping_rounds,
    )
