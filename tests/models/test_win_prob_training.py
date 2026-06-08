"""Tests for the win-probability training pipeline (Model 2C)."""

import json

import numpy as np
import pytest

pytest.importorskip("torch")

from src.models.win_prob.train import (
    build_examples, split_by_point, _point_key, train,
    expected_calibration_error,
)

_ZONES = ["T_deep", "C_deep", "W_deep", "T_mid", "C_mid", "W_mid",
          "T_short", "C_short", "W_short"]
_SHOTS = ["f", "b", "serve"]
_DIRS = ["1", "2", "3"]


def make_synthetic_records(n_points: int, shots_per_point: int, seed: int = 7) -> list[dict]:
    rng = np.random.default_rng(seed)
    records: list[dict] = []
    for p in range(n_points):
        winner = int(rng.integers(2))   # 0 or 1; the point's eventual winner side
        for s in range(shots_per_point):
            striker_is_side0 = (s % 2 == 0)
            outcome = 1 if (striker_is_side0 == bool(winner)) else 0
            records.append({
                "point_id": f"match_0_{p}_{s}",
                "shot_in_rally": s,
                "shot_type_mcp": _SHOTS[s % len(_SHOTS)],
                "direction_mcp": _DIRS[s % len(_DIRS)],
                "ball_landing_zone": _ZONES[int(rng.integers(len(_ZONES)))],
                "point_outcome": outcome,
                "striker_position_m": [float(rng.uniform(-3, 3)), float(rng.uniform(1, 10))],
                "striker_velocity_ms": [float(rng.uniform(-2, 2)), float(rng.uniform(-2, 2))],
                "opponent_position_m": [float(rng.uniform(-3, 3)), float(rng.uniform(13, 22))],
                "opponent_velocity_ms": [float(rng.uniform(-1, 1)), float(rng.uniform(-1, 1))],
                "court_opening": float(rng.uniform(0, 4)),
            })
    return records


def test_point_key_strips_shot_index():
    assert _point_key({"point_id": "match_0_5_3"}) == "match_0_5"


def test_build_examples_one_per_shot():
    records = make_synthetic_records(n_points=3, shots_per_point=4)
    examples = build_examples(records)
    # 3 points x 4 shots = 12 examples (one per shot)
    assert len(examples) == 12
    # each example is (encoded_tuple, float_label, point_key)
    enc, label, pk = examples[0]
    assert len(enc) == 4              # (s_idx, d_idx, z_idx, floats)
    assert label in (0.0, 1.0)
    assert pk.startswith("match_0_")


def test_build_examples_label_matches_outcome():
    records = make_synthetic_records(n_points=1, shots_per_point=3)
    examples = build_examples(records)
    labels = [lbl for _, lbl, _ in examples]
    assert labels == [float(r["point_outcome"]) for r in records]


def test_build_examples_empty_raises():
    with pytest.raises(ValueError):
        build_examples([])


def test_split_by_point_is_disjoint():
    records = make_synthetic_records(n_points=10, shots_per_point=4)
    examples = build_examples(records)
    train_ex, val_ex = split_by_point(examples, val_frac=0.3, seed=1)
    train_keys = {e[2] for e in train_ex}
    val_keys = {e[2] for e in val_ex}
    assert train_keys.isdisjoint(val_keys)
    assert len(val_ex) > 0 and len(train_ex) > 0


def test_ece_perfectly_calibrated_is_low():
    # predictions equal to empirical rates → near-zero ECE
    probs = np.array([0.0, 0.0, 1.0, 1.0])
    labels = np.array([0.0, 0.0, 1.0, 1.0])
    assert expected_calibration_error(probs, labels) < 1e-6


def _write(tmp_path, records) -> str:
    path = tmp_path / "shots.json"
    path.write_text(json.dumps(records))
    return str(path)


def test_train_end_to_end(tmp_path):
    records = make_synthetic_records(n_points=40, shots_per_point=6)
    records_path = _write(tmp_path, records)
    model_path = train([records_path], output_dir=tmp_path / "ckpt",
                       epochs=3, batch_size=16, patience=3)
    assert model_path.exists()
    metrics = json.loads((tmp_path / "ckpt" / "metrics.json").read_text())
    assert 0.0 <= metrics["brier"] <= 1.0
    assert metrics["log_loss"] > 0.0
    assert metrics["n_train"] > 0 and metrics["n_val"] > 0
    # the isotonic calibrator must persist in the checkpoint (save contract)
    import torch
    ckpt = torch.load(model_path, weights_only=False)
    assert ckpt.get("calibrator") is not None


def test_train_saved_model_predicts_in_range(tmp_path):
    from src.models.win_prob.model import WinProbModel
    records = make_synthetic_records(n_points=40, shots_per_point=6)
    model_path = train([_write(tmp_path, records)], output_dir=tmp_path / "ckpt",
                       epochs=3, batch_size=16, patience=3)
    model = WinProbModel(str(model_path))
    out = model.predict(records[:3])
    assert 0.0 <= out["p_win_point"] <= 1.0


def test_train_positions_off_runs(tmp_path):
    records = make_synthetic_records(n_points=30, shots_per_point=5)
    model_path = train([_write(tmp_path, records)], output_dir=tmp_path / "ckpt_off",
                       epochs=2, batch_size=16, patience=2, positions_off=True)
    assert model_path.exists()
