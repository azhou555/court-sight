"""Tests for the win-probability model: tokenizer, padding, encoder, persistence."""

import pytest

from src.models.win_prob.model import (
    RallyTokenizer, RallyTransformerEncoder, WinProbModel, pad_batch,
    TOKEN_DIM, FLOAT_DIM, MAX_RALLY_LEN,
    SHOT_TYPE_VOCAB, DIRECTION_VOCAB, ZONE_VOCAB,
)

torch = pytest.importorskip("torch")


def _shot(shot_type="f", direction="1", zone="C_deep", outcome=1, idx=0,
          striker=(1.0, 5.0), opponent=(-1.0, 18.0)):
    return {
        "shot_type_mcp": shot_type,
        "direction_mcp": direction,
        "ball_landing_zone": zone,
        "point_outcome": outcome,
        "shot_in_rally": idx,
        "striker_position_m": list(striker),
        "striker_velocity_ms": [0.5, -0.5],
        "opponent_position_m": list(opponent),
        "opponent_velocity_ms": [-0.3, 0.2],
        "court_opening": 1.4,
    }


# ── tokenizer ──────────────────────────────────────────────────────────────────

def test_token_dim_is_43():
    assert TOKEN_DIM == 43
    assert FLOAT_DIM == 11


def test_tokenizer_unknown_code_maps_to_unk():
    tok = RallyTokenizer()
    s_idx, d_idx, z_idx, floats = tok.encode_shot(_shot(shot_type="ZZZ", zone="bogus"))
    assert s_idx == 1   # <unk>
    assert z_idx == 1   # <unk>


def test_tokenizer_empty_direction_has_own_slot():
    tok = RallyTokenizer()
    _, d_idx, _, _ = tok.encode_shot(_shot(direction=""))
    assert DIRECTION_VOCAB[d_idx] == ""   # not <unk>


def test_tokenizer_positions_off_zeros_and_flag():
    tok = RallyTokenizer()
    _, _, _, on = tok.encode_shot(_shot(), positions_off=False)
    _, _, _, off = tok.encode_shot(_shot(), positions_off=True)
    # floats layout: [sp_x,sp_y,sv_x,sv_y,op_x,op_y,ov_x,ov_y,court_opening,
    #                 rally_depth, positions_known]
    assert on[10] == 1.0
    assert off[10] == 0.0
    assert all(v == 0.0 for v in off[:9])   # positional dims zeroed
    assert on[:9] != [0.0] * 9               # populated when on


def test_tokenizer_rally_truncates_to_max_len():
    tok = RallyTokenizer()
    rally = [_shot(idx=i) for i in range(MAX_RALLY_LEN + 5)]
    s_idx, d_idx, z_idx, floats = tok.encode_rally(rally)
    assert s_idx.shape[0] == MAX_RALLY_LEN
    assert floats.shape == (MAX_RALLY_LEN, FLOAT_DIM)


# ── padding ──────────────────────────────────────────────────────────────────

def test_pad_batch_left_pads_and_masks():
    tok = RallyTokenizer()
    short = tok.encode_rally([_shot(idx=0)])
    long = tok.encode_rally([_shot(idx=0), _shot(idx=1), _shot(idx=2)])
    s, d, z, f, mask = pad_batch([short, long])
    assert s.shape == (2, 3)
    assert f.shape == (2, 3, FLOAT_DIM)
    # row 0 (len 1) is left-padded: first two are pad, last is real
    assert mask[0].tolist() == [True, True, False]
    assert mask[1].tolist() == [False, False, False]


# ── encoder forward ──────────────────────────────────────────────────────────

def test_forward_returns_logits_shape():
    enc = RallyTransformerEncoder()
    tok = RallyTokenizer()
    batch = [tok.encode_rally([_shot(idx=0), _shot(idx=1)]),
             tok.encode_rally([_shot(idx=0)])]
    s, d, z, f, mask = pad_batch(batch)
    out = enc(s, d, z, f, mask)
    assert out.shape == (2,)
    assert out.dtype == torch.float32


def test_input_projection_width_matches_token_dim():
    enc = RallyTransformerEncoder()
    assert enc.input_proj.in_features == TOKEN_DIM


# ── persistence + predict contract ───────────────────────────────────────────

def test_save_load_roundtrip(tmp_path):
    model = WinProbModel()
    rally = [_shot(idx=0), _shot(idx=1)]
    before = model.predict(rally)["p_win_point"]
    path = tmp_path / "wp.pt"
    model.save(str(path))
    loaded = WinProbModel(str(path))
    after = loaded.predict(rally)["p_win_point"]
    assert abs(before - after) < 1e-5


def test_predict_contract_keys():
    model = WinProbModel()
    out = model.predict([_shot(idx=0)])
    assert set(out) >= {"p_win_point", "rally_context_used"}
    assert 0.0 <= out["p_win_point"] <= 1.0
    assert out["rally_context_used"] == 1


def test_predict_ignores_point_outcome():
    """Leak guard: the label must never influence the prediction."""
    model = WinProbModel()
    win = model.predict([_shot(idx=0, outcome=1)])["p_win_point"]
    loss = model.predict([_shot(idx=0, outcome=0)])["p_win_point"]
    assert win == loss


def test_tokenizer_missing_positions_flags_unknown():
    """A shot dict with no positions → positions_known=0.0 + zeroed positions,
    even in the default (positions_off=False) path. Guards the EV-scorer's
    sparse hypothetical-zone shots."""
    tok = RallyTokenizer()
    sparse = {"ball_landing_zone": "C_deep"}            # no striker_position_m
    _, _, _, floats = tok.encode_shot(sparse)           # positions_off defaults False
    assert floats[10] == 0.0                            # positions_known flag
    assert all(v == 0.0 for v in floats[:9])            # positional dims zeroed
    # a shot WITH positions still flags known
    _, _, _, f2 = tok.encode_shot(_shot())
    assert f2[10] == 1.0
    assert f2[:9] != [0.0] * 9
