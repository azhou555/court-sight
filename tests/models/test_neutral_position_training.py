"""Tests for the response-distribution training pipeline (Model 2E, Stage 1)."""

import json

import numpy as np
import pytest

from src.models.neutral_position.model import (
    ResponseDistributionModel, ResponseDistributionFeatures,
    COURT_X_MIN, COURT_X_MAX, COURT_Y_MIN, COURT_Y_MAX,
)
from src.models.neutral_position.train import (
    build_dataset, features_from_shot_record, _label_from_next_shot,
    _direction_to_angle, _point_key, train,
)

_ZONES = ["T_deep", "C_deep", "W_deep", "T_mid", "C_mid", "W_mid",
          "T_short", "C_short", "W_short"]
_SHOT_TYPES = ["f", "b", "serve"]
_DIRECTIONS = ["1", "2", "3"]


def make_synthetic_shot_records(n_points: int, shots_per_point: int,
                                seed: int = 42) -> list[dict]:
    """Generate plausible ShotRecord dicts (JSON-serialisable, no numpy)."""
    rng = np.random.default_rng(seed)
    records: list[dict] = []
    for p in range(n_points):
        context: list[dict] = []
        for s in range(shots_per_point):
            role = "near" if s % 2 == 0 else "far"
            rec = {
                "match_id": "match_0",
                "point_id": f"match_0_{p}_{s}",      # shot-unique, as in production
                "shot_in_rally": s,
                "shot_type_mcp": _SHOT_TYPES[s % len(_SHOT_TYPES)],
                "direction_mcp": _DIRECTIONS[s % len(_DIRECTIONS)],
                "striker_role": role,
                "striker_position_m": [float(rng.uniform(-3, 3)),
                                       float(rng.uniform(1, 10))],
                "striker_velocity_ms": [float(rng.uniform(-2, 2)),
                                        float(rng.uniform(-2, 2))],
                "opponent_position_m": [float(rng.uniform(-3, 3)),
                                        float(rng.uniform(13, 22))],
                "opponent_velocity_ms": [float(rng.uniform(-1, 1)),
                                         float(rng.uniform(-1, 1))],
                "ball_landing_zone": _ZONES[rng.integers(len(_ZONES))],
                "ball_contact_position_m": [float(rng.uniform(-3, 3)),
                                            float(rng.uniform(13, 22))],
                "rally_context": list(context),
            }
            records.append(rec)
            context.append({"direction": rec["direction_mcp"]})
            if len(context) > 3:
                context.pop(0)
    return records


# ── point_key / direction helpers ──────────────────────────────────────────────

def test_point_key_strips_shot_index():
    assert _point_key({"point_id": "match_0_5_3"}) == "match_0_5"
    assert _point_key({"point_id": "m_12_0"}) == "m_12"


def test_direction_angle_mapping():
    assert _direction_to_angle("1") == -45.0
    assert _direction_to_angle("2") == 0.0
    assert _direction_to_angle("3") == 45.0
    assert _direction_to_angle("") == 0.0
    assert _direction_to_angle(None) == 0.0


# ── feature extraction ──────────────────────────────────────────────────────────

def test_features_from_shot_record_valid():
    rec = make_synthetic_shot_records(1, 1)[0]
    feat = features_from_shot_record(rec)
    assert feat is not None
    assert feat.to_vector().shape == (21,)   # 11 scalars + 10 shot-type one-hot


def test_features_from_shot_record_missing_striker_returns_none():
    rec = make_synthetic_shot_records(1, 1)[0]
    rec["striker_position_m"] = None
    assert features_from_shot_record(rec) is None


def test_features_from_shot_record_missing_opponent_returns_none():
    rec = make_synthetic_shot_records(1, 1)[0]
    rec.pop("opponent_position_m")
    assert features_from_shot_record(rec) is None


def test_features_missing_landing_zone_returns_none():
    """Shot N with no detected landing zone is skipped, not center-defaulted."""
    rec = make_synthetic_shot_records(1, 1)[0]
    rec["ball_landing_zone"] = None
    assert features_from_shot_record(rec) is None


def test_features_reads_rally_context_direction_key():
    """rally_context entries use 'direction', not 'direction_mcp'."""
    rec = make_synthetic_shot_records(1, 1)[0]
    rec["rally_context"] = [{"direction": "1"}, {"direction": "3"}]
    feat = features_from_shot_record(rec)
    assert feat.prev_shot_direction_1 == 45.0    # last entry → '3'
    assert feat.prev_shot_direction_2 == -45.0   # second-to-last → '1'


# ── label extraction ────────────────────────────────────────────────────────────

def test_label_from_next_shot_uses_landing_zone():
    nxt = {"ball_landing_zone": "C_deep"}
    label = _label_from_next_shot(nxt)
    assert label is not None
    assert isinstance(label[0], float) and isinstance(label[1], float)


def test_label_ignores_contact_position():
    """Leak guard: contact_position is NOT the response landing — ignore it.

    ball_contact_position_m(N+1) is where the opponent struck the ball (≈ where
    YOUR shot N landed), so it must not become the label. With no landing zone,
    the pair must be skipped even when a contact position is present.
    """
    nxt = {"ball_contact_position_m": [1.5, 18.0], "ball_landing_zone": None}
    assert _label_from_next_shot(nxt) is None


def test_label_from_next_shot_no_zone_returns_none():
    assert _label_from_next_shot({"ball_landing_zone": None}) is None


# ── dataset construction ────────────────────────────────────────────────────────

def test_build_dataset_consecutive_pairs():
    records = make_synthetic_shot_records(n_points=5, shots_per_point=4)
    X, y_x, y_y, stats = build_dataset(records)
    # 5 points × (4 - 1) = 15 pairs
    assert stats["n_pairs"] == 15
    assert X.shape == (15, 21)
    assert y_x.shape == (15,) and y_y.shape == (15,)


def test_build_dataset_single_shot_points_yield_no_pairs():
    records = make_synthetic_shot_records(n_points=4, shots_per_point=1)
    with pytest.raises(ValueError):
        build_dataset(records)


# ── training / persistence ──────────────────────────────────────────────────────

def _write_records(tmp_path, records) -> str:
    path = tmp_path / "shots.json"
    path.write_text(json.dumps(records))
    return str(path)


def test_train_end_to_end(tmp_path):
    records = make_synthetic_shot_records(n_points=20, shots_per_point=8)
    records_path = _write_records(tmp_path, records)
    model_path = train([records_path], output_dir=tmp_path / "ckpt",
                       early_stopping_rounds=10)
    assert model_path.exists()
    assert (tmp_path / "ckpt" / "metrics.json").exists()
    metrics = json.loads((tmp_path / "ckpt" / "metrics.json").read_text())
    assert metrics["mae_x"] < 5.0      # court is ~8m wide
    assert metrics["sigma_x"] > 0.0 and metrics["sigma_y"] > 0.0


def test_model_save_load_roundtrip(tmp_path):
    records = make_synthetic_shot_records(n_points=20, shots_per_point=8)
    X, y_x, y_y, _ = build_dataset(records)
    model = ResponseDistributionModel()
    model.train(X, y_x, y_y, early_stopping_rounds=10)
    path = tmp_path / "rd.pkl"
    model.save(path)

    feat = features_from_shot_record(records[0])
    before = model.predict(feat)["mu"]

    loaded = ResponseDistributionModel(str(path))
    after = loaded.predict(feat)["mu"]
    np.testing.assert_allclose(before, after, atol=1e-4)


def test_predict_within_court_after_training(tmp_path):
    records = make_synthetic_shot_records(n_points=20, shots_per_point=8)
    X, y_x, y_y, _ = build_dataset(records)
    model = ResponseDistributionModel()
    model.train(X, y_x, y_y, early_stopping_rounds=10)

    out = model.predict(features_from_shot_record(records[1]))
    assert set(out) >= {"mu", "sigma_x", "sigma_y"}
    assert COURT_X_MIN - 2 <= out["mu"][0] <= COURT_X_MAX + 2
    assert out["sigma_x"] > 0.0
