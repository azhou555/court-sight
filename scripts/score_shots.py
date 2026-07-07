"""
Score shot records with EV surface analysis.

Loads shot_records.json, runs each record through EVScorer, and writes
scored_shots.json with a "scoring" field appended to each record.

Zone resolution order:
  1. ball_landing_zone from CV (ground truth)
  2. direction_mcp + depth_mcp → derived zone (with striker-side correction)
  3. depth_mcp only → C_{depth} (depth signal preserved, lateral unknown)
  4. Neither → shot skipped (no positional info for point construction)

Usage:
    python -m scripts.score_shots <shot_records.json> <output.json>
    python -m scripts.score_shots data/shot_records.json data/scored_shots.json \\
        --safety-model   checkpoints/shot_safety/shot_safety_model.pkl \\
        --neutral-model  checkpoints/neutral_position/response_distribution_model.pkl \\
        --winprob-model  checkpoints/win_prob/win_prob_model.pt
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from src.models.ev_surface.scorer import EVScorer, RecoveryState, ZONE_CENTERS
from src.models.execution_prob.model import ShotGeometryFeatures, ShotSafetyModel
from src.models.execution_prob.train import _MCP_TO_SHOT_TYPE
from src.models.neutral_position.model import NeutralPositionModel, ResponseDistributionFeatures
from src.models.win_prob.model import WinProbModel

# MCP depth code → zone depth suffix
_DEPTH_SUFFIX = {"7": "short", "8": "mid", "9": "deep"}

# MCP direction angle for ShotGeometryFeatures
_DIR_ANGLE = {"1": -45.0, "2": 0.0, "3": 45.0}

# Baseline center — proxy neutral position until 2E is retrained on real data
# ponytail: geometric center proxy; replace with NeutralPositionModel.predict_neutral post-training
_BASELINE_CENTER = np.array([0.0, 11.89])


def _lateral_from_direction(direction: str, striker_x: float) -> str:
    """Map MCP direction code to T/C/W using striker's lateral position.

    CC ('1') goes to the opposite side from the striker; DTL ('3') stays same side.
    Striker x ≤ 0 means T-side; x > 0 means W-side.
    """
    if direction == "2":
        return "C"
    if direction == "1":  # cross-court
        return "W" if striker_x <= 0 else "T"
    if direction == "3":  # down the line
        return "T" if striker_x <= 0 else "W"
    return "C"  # unknown direction → center


def _resolve_zone(rec: dict) -> tuple[str | None, bool]:
    """Return (zone, estimated). None means skip — no positional info."""
    if zone := rec.get("ball_landing_zone"):
        return zone, False

    depth_suffix = _DEPTH_SUFFIX.get(rec.get("depth_mcp", ""))
    if not depth_suffix:
        return None, True

    direction = rec.get("direction_mcp", "")
    striker_x = float((rec.get("striker_position_m") or [0.0])[0])
    lateral = _lateral_from_direction(direction, striker_x)
    return f"{lateral}_{depth_suffix}", True


def _shot_type(rec: dict) -> str:
    return _MCP_TO_SHOT_TYPE.get(rec.get("shot_type_mcp", ""), "forehand_groundstroke")


def _prev_dir_angle(rally_context: list[dict], offset: int) -> float:
    if len(rally_context) <= offset:
        return 0.0
    return _DIR_ANGLE.get(str(rally_context[-(offset + 1)].get("direction", "")), 0.0)


def _dir_unit_vec(direction: str, striker_x: float) -> np.ndarray:
    """Shot direction as a unit vector in court coords (x=lateral, y=depth)."""
    lateral = _lateral_from_direction(direction, striker_x)
    dx = {"T": -1.0, "C": 0.0, "W": 1.0}[lateral]
    # All shots travel away from striker toward opponent baseline (positive y)
    raw = np.array([dx, 1.0])
    norm = np.linalg.norm(raw)
    return raw / norm if norm > 0 else np.array([0.0, 1.0])


def _build_features(rec: dict, zone: str, estimated: bool):
    striker_pos = np.array(rec.get("striker_position_m") or [0.0, 11.89], dtype=float)
    striker_vel = np.array(rec.get("striker_velocity_ms") or [0.0, 0.0], dtype=float)
    opponent_pos = np.array(rec.get("opponent_position_m") or [0.0, 11.89], dtype=float)
    opponent_vel = np.array(rec.get("opponent_velocity_ms") or [0.0, 0.0], dtype=float)
    ball_contact = np.array(rec.get("ball_contact_position_m") or striker_pos.tolist(), dtype=float)

    shot_type = _shot_type(rec)
    direction = rec.get("direction_mcp", "")
    striker_x = float(striker_pos[0])
    zone_center = ZONE_CENTERS[zone]
    lx, ly = float(zone_center[0]), float(zone_center[1])

    safety = ShotGeometryFeatures(
        lateral_margin_m=min(abs(lx - (-4.115)), abs(lx - 4.115)),
        depth_margin_m=min(abs(ly - 0.0), abs(ly - 23.77)),
        net_clearance_m=0.0,          # ponytail: no ball-arc extraction yet
        shot_direction_deg=_DIR_ANGLE.get(direction, 0.0),
        landing_x=lx,
        landing_y=ly,
        shot_type=shot_type,
        striker_y_m=float(striker_pos[1]),
        striker_x_m=striker_x,
        incoming_depth_m=float(opponent_pos[1]),
        target_zone_estimated=estimated,
    )

    rally_context = rec.get("rally_context") or []
    opp_displacement = float(np.linalg.norm(opponent_pos - _BASELINE_CENTER))

    response = ResponseDistributionFeatures(
        your_landing_x=lx,
        your_landing_y=ly,
        ball_height_at_bounce=0.0,    # ponytail: no ball-arc extraction yet
        opponent_pos_x=float(opponent_pos[0]),
        opponent_pos_y=float(opponent_pos[1]),
        opponent_vel_x=float(opponent_vel[0]),
        opponent_vel_y=float(opponent_vel[1]),
        opponent_displacement_from_neutral=opp_displacement,
        your_shot_type=shot_type,
        rally_depth=rec.get("shot_in_rally", 1),
        prev_shot_direction_1=_prev_dir_angle(rally_context, 0),
        prev_shot_direction_2=_prev_dir_angle(rally_context, 1),
    )

    displacement = float(np.linalg.norm(striker_pos - _BASELINE_CENTER))
    recovery = RecoveryState(
        contact_pos_m=striker_pos,
        neutral_pos_m=_BASELINE_CENTER,
        movement_vector=striker_vel,
        shot_direction=_dir_unit_vec(direction, striker_x),
        ball_pos_at_contact=ball_contact,
        shot_type=shot_type,
        recovery_displacement_m=displacement,
    )

    return safety, response, recovery, striker_pos, striker_vel


def score_shots(
    records_path: Path,
    output_path: Path,
    safety_model_path: Path,
    neutral_model_path: Path,
    winprob_model_path: Path,
) -> None:
    records = json.loads(records_path.read_text())

    scorer = EVScorer(
        safety_model=ShotSafetyModel(str(safety_model_path)),
        win_prob_model=WinProbModel(str(winprob_model_path)),
        neutral_model=NeutralPositionModel(str(neutral_model_path)),
    )

    results = []
    skipped = 0

    for rec in records:
        zone, estimated = _resolve_zone(rec)
        if zone is None:
            skipped += 1
            continue

        safety, response, recovery, striker_pos, striker_vel = _build_features(rec, zone, estimated)

        scoring = scorer.score_shot(
            actual_zone=zone,
            base_safety_features=safety,
            base_response_features=response,
            rally_context=rec.get("rally_context") or [],
            recovery_state=recovery,
            striker_pos=striker_pos,
            striker_vel=striker_vel,
            t_recovery=1.0,           # ponytail: fixed default, no inter-shot timing
            shot_id=rec.get("point_id"),
        )

        results.append({**rec, "scoring": scoring, "zone_estimated": estimated})

    output_path.write_text(json.dumps(results, indent=2))
    total = len(records)
    print(f"Scored {len(results)}/{total} shots. Skipped {skipped} (no zone info).")
    if skipped:
        print(f"  ({skipped / total:.1%} skip rate — check ball tracker miss rate)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Score shot records with EV surface analysis.")
    p.add_argument("records", type=Path, help="Path to shot_records.json")
    p.add_argument("output", type=Path, help="Path to write scored_shots.json")
    p.add_argument("--safety-model",  type=Path, default=Path("checkpoints/shot_safety/shot_safety_model.pkl"))
    p.add_argument("--neutral-model", type=Path, default=Path("checkpoints/neutral_position/response_distribution_model.pkl"))
    p.add_argument("--winprob-model", type=Path, default=Path("checkpoints/win_prob/win_prob_model.pt"))
    args = p.parse_args()

    score_shots(args.records, args.output, args.safety_model, args.neutral_model, args.winprob_model)
