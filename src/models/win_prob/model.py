"""Win probability model — Transformer encoder over rally event sequences.

Predicts P(win point | rally state). Categorical shot attributes (type,
direction, landing zone) are embedded with learned, in-module embeddings;
positional/velocity features are concatenated as raw floats gated by a
``positions_known`` flag so the same model serves both the positions-off
pretrain regime and the full-feature fine-tune regime.

Token layout (43 dims, pre-projection):
    shot_type_emb(16) + direction_emb(8) + landing_zone_emb(8) + floats(11)
where floats = [striker_pos(2), striker_vel(2), opp_pos(2), opp_vel(2),
                court_opening(1), rally_depth(1), positions_known(1)].
"""

from __future__ import annotations

from typing import Optional

# ── Vocabularies (MCP-native codes; index 0 = <pad>, index 1 = <unk>) ──────────
PAD, UNK = "<pad>", "<unk>"
_UNK_IDX = 1

SHOT_TYPE_VOCAB = [PAD, UNK, "f", "b", "r", "s", "v", "z", "l", "o",
                   "j", "k", "y", "p", "q", "serve"]
DIRECTION_VOCAB = [PAD, UNK, "", "1", "2", "3", "4", "5", "6"]
ZONE_VOCAB = [PAD, UNK, "T_deep", "C_deep", "W_deep", "T_mid", "C_mid",
              "W_mid", "T_short", "C_short", "W_short"]

_SHOT_IDX = {t: i for i, t in enumerate(SHOT_TYPE_VOCAB)}
_DIR_IDX = {t: i for i, t in enumerate(DIRECTION_VOCAB)}
_ZONE_IDX = {t: i for i, t in enumerate(ZONE_VOCAB)}

SHOT_EMB_DIM = 16
DIR_EMB_DIM = 8
ZONE_EMB_DIM = 8
FLOAT_DIM = 11
TOKEN_DIM = SHOT_EMB_DIM + DIR_EMB_DIM + ZONE_EMB_DIM + FLOAT_DIM   # = 43
HIDDEN_DIM = 128
MAX_RALLY_LEN = 20


def _lookup(table: dict, key) -> int:
    """Vocab lookup; unseen / None keys fall back to <unk>."""
    return table.get(key, _UNK_IDX)


def _norm_direction(value) -> str:
    """Normalize a direction code to a vocab key ("" for missing)."""
    return "" if value is None else str(value)


# Torch-dependent classes live in a try block so the module imports in
# environments without torch (pure-logic tests still see the vocab constants).
try:
    import torch
    import torch.nn as nn

    class RallyTokenizer:
        """Converts shot dicts into index/float tensors. Holds no parameters."""

        def encode_shot(self, shot: dict, positions_off: bool = False):
            """Return (shot_idx, dir_idx, zone_idx, floats[list of 11])."""
            shot_idx = _lookup(_SHOT_IDX, shot.get("shot_type_mcp"))
            dir_idx = _lookup(_DIR_IDX, _norm_direction(shot.get("direction_mcp")))
            zone_idx = _lookup(_ZONE_IDX, shot.get("ball_landing_zone"))

            sp_raw = shot.get("striker_position_m")
            has_positions = sp_raw is not None and len(sp_raw) >= 2
            if positions_off or not has_positions:
                # pretrain regime (all shots) OR a shot that carries no position
                # data (e.g. a hypothetical zone appended by the EV scorer):
                # zero the positional features and flag them unknown.
                positional = [0.0] * 9
                known = 0.0
            else:
                sv = shot.get("striker_velocity_ms") or [0.0, 0.0]
                op = shot.get("opponent_position_m") or [0.0, 0.0]
                ov = shot.get("opponent_velocity_ms") or [0.0, 0.0]
                co = float(shot.get("court_opening", 0.0))
                positional = [float(sp_raw[0]), float(sp_raw[1]), float(sv[0]), float(sv[1]),
                              float(op[0]), float(op[1]), float(ov[0]), float(ov[1]), co]
                known = 1.0

            rally_depth = float(shot.get("shot_in_rally", 0)) / MAX_RALLY_LEN
            floats = [*positional, rally_depth, known]   # 11
            return shot_idx, dir_idx, zone_idx, floats

        def encode_rally(self, rally: list[dict], positions_off: bool = False):
            """Return (shot_idx, dir_idx, zone_idx, floats) tensors of length T."""
            rally = rally[-MAX_RALLY_LEN:]
            s_idx, d_idx, z_idx, f_rows = [], [], [], []
            for shot in rally:
                si, di, zi, fl = self.encode_shot(shot, positions_off)
                s_idx.append(si); d_idx.append(di); z_idx.append(zi); f_rows.append(fl)
            return (
                torch.tensor(s_idx, dtype=torch.long),
                torch.tensor(d_idx, dtype=torch.long),
                torch.tensor(z_idx, dtype=torch.long),
                torch.tensor(f_rows, dtype=torch.float32).reshape(len(f_rows), FLOAT_DIM),
            )

    def pad_batch(encoded: list):
        """Left-pad a list of (s_idx, d_idx, z_idx, floats) tuples.

        Left-padding keeps the current (most recent) shot at index T-1, so the
        encoder can read it as ``x[:, -1, :]``. Returns batched tensors plus a
        boolean padding mask where True marks a padding position.
        """
        B = len(encoded)
        T = max(e[0].shape[0] for e in encoded)
        s = torch.zeros(B, T, dtype=torch.long)        # 0 == <pad>
        d = torch.zeros(B, T, dtype=torch.long)
        z = torch.zeros(B, T, dtype=torch.long)
        f = torch.zeros(B, T, FLOAT_DIM, dtype=torch.float32)
        mask = torch.ones(B, T, dtype=torch.bool)      # True == pad
        for i, (si, di, zi, fi) in enumerate(encoded):
            L = si.shape[0]
            s[i, T - L:] = si
            d[i, T - L:] = di
            z[i, T - L:] = zi
            f[i, T - L:] = fi
            mask[i, T - L:] = False
        return s, d, z, f, mask

    class RallyTransformerEncoder(nn.Module):
        """Transformer encoder over rally tokens → win-probability logit."""

        def __init__(
            self,
            hidden_dim: int = HIDDEN_DIM,
            num_heads: int = 4,
            num_layers: int = 4,
            ff_dim: int = 256,
            dropout: float = 0.1,
            max_seq_len: int = MAX_RALLY_LEN,
        ):
            super().__init__()
            self.shot_emb = nn.Embedding(len(SHOT_TYPE_VOCAB), SHOT_EMB_DIM, padding_idx=0)
            self.dir_emb = nn.Embedding(len(DIRECTION_VOCAB), DIR_EMB_DIM, padding_idx=0)
            self.zone_emb = nn.Embedding(len(ZONE_VOCAB), ZONE_EMB_DIM, padding_idx=0)
            self.input_proj = nn.Linear(TOKEN_DIM, hidden_dim)
            self.pos_encoding = nn.Embedding(max_seq_len, hidden_dim)
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=num_heads, dim_feedforward=ff_dim,
                dropout=dropout, batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
            self.head = nn.Linear(hidden_dim, 1)

        def forward(self, shot_idx, dir_idx, zone_idx, floats, padding_mask=None):
            tokens = torch.cat([
                self.shot_emb(shot_idx),
                self.dir_emb(dir_idx),
                self.zone_emb(zone_idx),
                floats,
            ], dim=-1)                                  # (B, T, 43)
            B, T, _ = tokens.shape
            pos = torch.arange(T, device=tokens.device).unsqueeze(0).expand(B, -1)
            x = self.input_proj(tokens) + self.pos_encoding(pos)
            x = self.encoder(x, src_key_padding_mask=padding_mask)
            return self.head(x[:, -1, :]).squeeze(-1)   # logits (B,)

    class WinProbModel:
        """Inference/persistence wrapper around RallyTransformerEncoder."""

        def __init__(self, model_path: Optional[str] = None):
            self.encoder = RallyTransformerEncoder()
            self.tokenizer = RallyTokenizer()
            self._calibrator = None
            if model_path:
                self.load(model_path)

        @torch.no_grad()
        def predict(self, rally: list[dict], positions_off: bool = False) -> dict:
            self.encoder.eval()
            s, d, z, f, mask = pad_batch([self.tokenizer.encode_rally(rally, positions_off)])
            logit = self.encoder(s, d, z, f, mask)
            raw = float(torch.sigmoid(logit)[0].item())
            calibrated = (
                float(self._calibrator.predict([raw])[0])
                if self._calibrator is not None else raw
            )
            return {
                "p_win_point": calibrated,
                "rally_context_used": min(len(rally), MAX_RALLY_LEN),
            }

        def save(self, path: str) -> None:
            torch.save({
                "encoder": self.encoder.state_dict(),
                "calibrator": self._calibrator,
            }, path)

        def load(self, path: str) -> None:
            # weights_only=False: the checkpoint pickles the sklearn calibrator.
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            self.encoder.load_state_dict(ckpt["encoder"])
            self._calibrator = ckpt.get("calibrator")

except ImportError:
    # torch not available — vocab constants still import; classes raise on use.
    class RallyTokenizer:  # type: ignore[no-redef]
        def __init__(self, *a, **k):
            raise ImportError("torch is required for RallyTokenizer")

    class RallyTransformerEncoder:  # type: ignore[no-redef]
        def __init__(self, *a, **k):
            raise ImportError("torch is required for RallyTransformerEncoder")

    class WinProbModel:  # type: ignore[no-redef]
        def __init__(self, *a, **k):
            raise ImportError("torch is required for WinProbModel")

    def pad_batch(*a, **k):  # type: ignore[no-redef]
        raise ImportError("torch is required for pad_batch")
