"""Shot type classifier — TCN over 30-frame pose keypoint sequences."""

from __future__ import annotations

from typing import Optional

SHOT_TYPES = [
    "forehand_groundstroke",
    "backhand_groundstroke",
    "forehand_slice",
    "backhand_slice",
    "forehand_volley",
    "backhand_volley",
    "overhead",
    "serve",
    "lob",
    "drop_shot",
]
NUM_CLASSES = len(SHOT_TYPES)
NUM_KEYPOINTS = 17
INPUT_DIM = NUM_KEYPOINTS * 2
HIDDEN_DIM = 128
STROKE_WINDOW = 30

try:
    import torch
    import torch.nn as nn

    class _TCNBlock(nn.Module):
        def __init__(self, in_channels: int, out_channels: int, dilation: int):
            super().__init__()
            padding = dilation * (3 - 1)
            self.conv = nn.Conv1d(
                in_channels, out_channels, kernel_size=3,
                dilation=dilation, padding=padding,
            )
            self.norm = nn.LayerNorm(out_channels)
            self.relu = nn.ReLU()
            self.residual = (
                nn.Conv1d(in_channels, out_channels, kernel_size=1)
                if in_channels != out_channels else nn.Identity()
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            out = self.conv(x)
            out = out[:, :, :x.size(2)]
            out = self.norm(out.transpose(1, 2)).transpose(1, 2)
            return self.relu(out + self.residual(x))

    class ShotClassifierTCN(nn.Module):
        """
        Classifies tennis shot type from a 30-frame pose keypoint sequence.

        Input:  (B, T, 17*2) — batch of stroke windows
        Output: (B, NUM_CLASSES) — class logits
        """

        def __init__(
            self,
            input_dim: int = INPUT_DIM,
            hidden_dim: int = HIDDEN_DIM,
            num_classes: int = NUM_CLASSES,
            num_blocks: int = 4,
        ):
            super().__init__()
            self.input_proj = nn.Linear(input_dim, hidden_dim)
            self.tcn_blocks = nn.ModuleList([
                _TCNBlock(hidden_dim, hidden_dim, dilation=2 ** i)
                for i in range(num_blocks)
            ])
            self.classifier = nn.Linear(hidden_dim, num_classes)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            x = self.input_proj(x)
            x = x.transpose(1, 2)
            for block in self.tcn_blocks:
                x = block(x)
            x = x.mean(dim=2)
            return self.classifier(x)

        @torch.no_grad()
        def predict(self, keypoints: "torch.Tensor") -> dict:
            """keypoints: (T, 17, 2) single stroke sequence."""
            x = keypoints.reshape(1, STROKE_WINDOW, INPUT_DIM).float()
            logits = self(x)
            probs = torch.softmax(logits, dim=-1).squeeze(0)
            idx = int(probs.argmax())
            return {
                "predicted_class": SHOT_TYPES[idx],
                "class_probabilities": {
                    cls: float(probs[i]) for i, cls in enumerate(SHOT_TYPES)
                },
                "confidence": float(probs[idx]),
            }

except ImportError:
    class ShotClassifierTCN:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError("torch is required for ShotClassifierTCN")
