"""CourtKeypointNet: TrackNet-style encoder-decoder for court keypoint heatmaps.

Architecture mirrors yastrebksv/TennisCourtDetector (BallTrackerNet).
Module naming matches the pretrained checkpoint exactly so state_dict loads
without remapping.

Input:  (B, 3, 360, 640)  — frames pre-resized to half of 720p
Output: (B, 15, 360, 640) — Gaussian heatmaps, channels 0-13 are keypoints,
                             channel 14 is court center (training only).
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


_WEIGHTS_GDRIVE_ID = "1f-Co64ehgq4uddcQm1aFBDtbnyZhQvgG"
INPUT_H, INPUT_W = 360, 640
NUM_KEYPOINTS = 14    # channels 0–13; channel 14 is center (training only)
NUM_CHANNELS_OUT = 15


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


class CourtKeypointNet(nn.Module):
    """VGG-style encoder-decoder that outputs one heatmap per court keypoint.

    Module names are kept identical to the upstream BallTrackerNet checkpoint
    (conv1–conv18, pool1–pool3, ups1–ups3) so pretrained weights load directly
    via load_state_dict with no key remapping.
    """

    def __init__(self, out_channels: int = NUM_CHANNELS_OUT):
        super().__init__()
        self.conv1 = _ConvBlock(3, 64)
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
        return x

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
) -> CourtKeypointNet:
    """Load CourtKeypointNet with pretrained weights.

    If weights_path is None, downloads from Google Drive to
    ~/.cache/court_sight/court_keypoints_pretrained.pth.
    """
    model = CourtKeypointNet()

    if weights_path is None:
        cache_dir = Path.home() / ".cache" / "court_sight"
        cache_dir.mkdir(parents=True, exist_ok=True)
        weights_path = cache_dir / "court_keypoints_pretrained.pth"

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
