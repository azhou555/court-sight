"""Win probability model — Transformer encoder over rally event sequences."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

SHOT_TYPES = [
    "forehand_groundstroke", "backhand_groundstroke",
    "forehand_slice", "backhand_slice",
    "forehand_volley", "backhand_volley",
    "overhead", "serve", "lob", "drop_shot",
]
DIRECTIONS = ["down_the_line", "crosscourt", "middle", "inside_in", "inside_out", "body"]
ZONES = [
    "T_deep", "C_deep", "W_deep",
    "T_mid",  "C_mid",  "W_mid",
    "T_short", "C_short", "W_short",
]

SHOT_EMB_DIM = 16
DIR_EMB_DIM = 8
ZONE_EMB_DIM = 8
POSITIONAL_DIM = 2 + 2 + 2 + 2 + 1 + 1   # striker/opp pos+vel, court_opening, rally_depth
TOKEN_DIM = SHOT_EMB_DIM + DIR_EMB_DIM + ZONE_EMB_DIM + POSITIONAL_DIM   # = 42
HIDDEN_DIM = 128
MAX_RALLY_LEN = 20

# Torch-dependent classes are defined inside a try block so the module can be
# imported in environments without torch (e.g. for pure-logic unit tests).
try:
    import torch
    import torch.nn as nn

    class RallyTransformerEncoder(nn.Module):
        """
        Transformer encoder over a sequence of rally event tokens.
        Predicts P(win point) at each shot in the rally.

        Input:  packed rally token tensors (see RallyTokenizer)
        Output: scalar win probability for the current (last) shot
        """

        def __init__(
            self,
            token_dim: int = TOKEN_DIM,
            hidden_dim: int = HIDDEN_DIM,
            num_heads: int = 4,
            num_layers: int = 4,
            ff_dim: int = 256,
            dropout: float = 0.1,
            max_seq_len: int = MAX_RALLY_LEN,
        ):
            super().__init__()
            self.input_proj = nn.Linear(token_dim, hidden_dim)
            self.pos_encoding = nn.Embedding(max_seq_len, hidden_dim)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=ff_dim,
                dropout=dropout,
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.head = nn.Linear(hidden_dim, 1)

        def forward(
            self,
            tokens: "torch.Tensor",
            padding_mask: Optional["torch.Tensor"] = None,
        ) -> "torch.Tensor":
            B, T, _ = tokens.shape
            positions = torch.arange(T, device=tokens.device).unsqueeze(0).expand(B, -1)
            x = self.input_proj(tokens) + self.pos_encoding(positions)
            x = self.encoder(x, src_key_padding_mask=padding_mask)
            last_token = x[:, -1, :]
            return torch.sigmoid(self.head(last_token)).squeeze(-1)

    class RallyTokenizer:
        """Converts a list of shot dicts into packed token tensors."""

        def __init__(self):
            self._shot_emb = nn.Embedding(len(SHOT_TYPES), SHOT_EMB_DIM)
            self._dir_emb = nn.Embedding(len(DIRECTIONS), DIR_EMB_DIM)
            self._zone_emb = nn.Embedding(len(ZONES), ZONE_EMB_DIM)

        def tokenize(self, rally: list[dict]) -> "torch.Tensor":
            tokens = []
            for shot in rally[-MAX_RALLY_LEN:]:
                shot_idx = SHOT_TYPES.index(shot.get("shot_type_mcp", SHOT_TYPES[0]))
                dir_idx = DIRECTIONS.index(shot.get("direction_mcp", DIRECTIONS[0]))
                zone_idx = ZONES.index(shot.get("ball_landing_zone", ZONES[0]))
                with torch.no_grad():
                    shot_e = self._shot_emb(torch.tensor(shot_idx))
                    dir_e = self._dir_emb(torch.tensor(dir_idx))
                    zone_e = self._zone_emb(torch.tensor(zone_idx))
                positional = torch.tensor([
                    *shot.get("striker_position_m", [0.0, 0.0]),
                    *shot.get("striker_velocity_ms", [0.0, 0.0]),
                    *shot.get("opponent_position_m", [0.0, 0.0]),
                    *shot.get("opponent_velocity_ms", [0.0, 0.0]),
                    shot.get("court_opening", 0.0),
                    shot.get("shot_in_rally", 0) / MAX_RALLY_LEN,
                ], dtype=torch.float32)
                tokens.append(torch.cat([shot_e, dir_e, zone_e, positional]))
            return torch.stack(tokens)

    class WinProbModel:
        """Wraps RallyTransformerEncoder with calibration and inference helpers."""

        def __init__(self, model_path: Optional[str] = None):
            self.encoder = RallyTransformerEncoder()
            self.tokenizer = RallyTokenizer()
            self._calibrator = None
            if model_path:
                self.load(model_path)

        @torch.no_grad()
        def predict(self, rally: list[dict]) -> dict:
            tokens = self.tokenizer.tokenize(rally).unsqueeze(0)
            raw_prob = float(self.encoder(tokens).item())
            calibrated = (
                float(self._calibrator.transform([raw_prob])[0])
                if self._calibrator else raw_prob
            )
            return {"p_win_point": calibrated, "rally_context_used": len(rally)}

        def load(self, path: str) -> None:
            checkpoint = torch.load(path, map_location="cpu")
            self.encoder.load_state_dict(checkpoint["encoder"])

        def save(self, path: str) -> None:
            torch.save({"encoder": self.encoder.state_dict()}, path)

except ImportError:
    # torch not available — classes will be defined but raise on instantiation
    class RallyTransformerEncoder:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError("torch is required for RallyTransformerEncoder")

    class RallyTokenizer:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError("torch is required for RallyTokenizer")

    class WinProbModel:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError("torch is required for WinProbModel")
