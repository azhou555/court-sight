"""BallTrackerNet: TrackNet-style encoder-decoder for tennis ball detection.

Architecture mirrors yastrebksv/TrackNet. Module naming matches the pretrained
checkpoint (conv1–conv18, pool1–3, ups1–3) so state_dict loads directly.

Input:  (B, 9, 360, 640)     — 3 consecutive BGR frames, channel-stacked
Output: (B, 256, 360×640)    — per-pixel 256-class intensity scores (flattened)

Inference: argmax(dim=1) → reshape(360, 640) → HoughCircles → ball (x, y)
Training:  CrossEntropyLoss vs Gaussian GT heatmap quantized to 0–255 classes
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

_WEIGHTS_GDRIVE_ID = "1XEYZ4myUN7QT-NeBYJI0xteLsvs-ZAOl"
INPUT_H, INPUT_W = 360, 640
NUM_CLASSES = 256   # intensity quantization levels


class _ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, pad: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=pad, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class BallTrackerNet(nn.Module):
    """TrackNet encoder-decoder for ball position heatmap classification.

    Module names match the upstream checkpoint exactly for direct weight loading.
    """

    def __init__(self, out_channels: int = NUM_CLASSES):
        super().__init__()
        self.conv1 = _ConvBlock(9, 64)
        self.conv2 = _ConvBlock(64, 64)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv3 = _ConvBlock(64, 128)
        self.conv4 = _ConvBlock(128, 128)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv5 = _ConvBlock(128, 256)
        self.conv6 = _ConvBlock(256, 256)
        self.conv7 = _ConvBlock(256, 256)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv8 = _ConvBlock(256, 512)
        self.conv9 = _ConvBlock(512, 512)
        self.conv10 = _ConvBlock(512, 512)
        self.ups1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv11 = _ConvBlock(512, 256)
        self.conv12 = _ConvBlock(256, 256)
        self.conv13 = _ConvBlock(256, 256)
        self.ups2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv14 = _ConvBlock(256, 128)
        self.conv15 = _ConvBlock(128, 128)
        self.ups3 = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv16 = _ConvBlock(128, 64)
        self.conv17 = _ConvBlock(64, 64)
        self.conv18 = _ConvBlock(64, out_channels)
        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv2(self.conv1(x))
        x = self.pool1(x)
        x = self.conv4(self.conv3(x))
        x = self.pool2(x)
        x = self.conv7(self.conv6(self.conv5(x)))
        x = self.pool3(x)
        x = self.conv10(self.conv9(self.conv8(x)))
        x = self.ups1(x)
        x = self.conv13(self.conv12(self.conv11(x)))
        x = self.ups2(x)
        x = self.conv15(self.conv14(x))
        x = self.ups3(x)
        x = self.conv18(self.conv17(self.conv16(x)))
        b = x.shape[0]
        return x.reshape(b, self.conv18.block[0].out_channels, -1)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.uniform_(m.weight, -0.05, 0.05)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def load_pretrained(
    weights_path: str | Path | None = None, device: str = "cpu"
) -> BallTrackerNet:
    """Load BallTrackerNet with pretrained weights.

    If weights_path is None, downloads from Google Drive to
    ~/.cache/court_sight/ball_tracker_pretrained.pth.
    """
    model = BallTrackerNet()

    if weights_path is None:
        cache_dir = Path.home() / ".cache" / "court_sight"
        cache_dir.mkdir(parents=True, exist_ok=True)
        weights_path = cache_dir / "ball_tracker_pretrained.pth"

    weights_path = Path(weights_path)
    if not weights_path.exists():
        try:
            import gdown
        except ImportError as e:
            raise ImportError(
                "gdown is required to download pretrained weights: pip install gdown"
            ) from e
        print(f"Downloading pretrained weights to {weights_path} ...")
        gdown.download(id=_WEIGHTS_GDRIVE_ID, output=str(weights_path), quiet=False)

    state = torch.load(weights_path, map_location=device, weights_only=True)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    return model.to(device)
