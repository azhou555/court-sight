"""Ball tracking via TrackNet with velocity-based gap interpolation."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
import torch

from ..homography.detector import pixel_to_court
from .model import BallTrackerNet, INPUT_H, INPUT_W, load_pretrained


COURT_ZONES = [
    "T_deep",  "C_deep",  "W_deep",
    "T_mid",   "C_mid",   "W_mid",
    "T_short", "C_short", "W_short",
]

_FAR_SERVICE_LINE_Y = 17.37   # far service line in court meters
_NET_Y = 11.89                # net in court meters
_LATERAL_BREAKS = (-1.0, 1.0)
_MAX_GAP_FRAMES = 4              # consecutive misses before giving up on prediction
_BOUNCE_THRESHOLD = 0.1          # m/frame minimum velocity change to confirm bounce
_CONF_THRESHOLD = 0.5            # minimum normalised heatmap peak to accept detection


@dataclass
class BallTrackResult:
    frame_idx: int
    position_px: Optional[np.ndarray]    # [x, y] original frame pixels
    position_m: Optional[np.ndarray]     # [x, y] court meters
    confidence: float
    is_bounce: bool
    bounce_zone: Optional[str]
    trajectory_segment: int
    interpolated: bool = False


class BallTracker:
    """TrackNet inference + velocity extrapolation + bounce detection."""

    def __init__(
        self,
        model: Optional[BallTrackerNet] = None,
        weights_path: Optional[str] = None,
        device: str = "cpu",
        detection_threshold: float = _CONF_THRESHOLD,
        bounce_velocity_threshold: float = _BOUNCE_THRESHOLD,
    ):
        if model is not None:
            self._model = model.to(device).eval()
        else:
            self._model = load_pretrained(weights_path, device=device).eval()

        self._device = device
        self._det_thresh = detection_threshold
        self._bounce_thresh = bounce_velocity_threshold

        self._trajectory_segment = 0
        self._miss_count = 0
        # (frame_idx, position_px, position_m) for last confirmed detections
        self._recent: deque[tuple[int, np.ndarray, np.ndarray]] = deque(maxlen=5)
        # court-space positions for bounce detection
        self._position_history: list[np.ndarray] = []

    def process_frame(
        self,
        frames: tuple[np.ndarray, np.ndarray, np.ndarray],
        frame_idx: int,
        H: Optional[np.ndarray],
    ) -> BallTrackResult:
        """Process one frame triple (prev-prev, prev, curr) in BGR."""
        orig_h, orig_w = frames[2].shape[:2]
        position_px, confidence = self._infer(frames, orig_h, orig_w)

        interpolated = False
        if position_px is None or confidence < self._det_thresh:
            self._miss_count += 1
            if self._miss_count <= _MAX_GAP_FRAMES and len(self._recent) >= 2:
                position_px = self._extrapolate(frame_idx)
                confidence = 0.0
                interpolated = True
            else:
                return BallTrackResult(
                    frame_idx=frame_idx,
                    position_px=None,
                    position_m=None,
                    confidence=0.0,
                    is_bounce=False,
                    bounce_zone=None,
                    trajectory_segment=self._trajectory_segment,
                    interpolated=False,
                )
        else:
            self._miss_count = 0

        if H is None:
            return BallTrackResult(
                frame_idx=frame_idx,
                position_px=position_px,
                position_m=None,
                confidence=confidence,
                is_bounce=False,
                bounce_zone=None,
                trajectory_segment=self._trajectory_segment,
                interpolated=interpolated,
            )

        position_m = pixel_to_court(position_px, H)

        if not interpolated:
            self._recent.append((frame_idx, position_px.copy(), position_m.copy()))

        self._position_history.append(position_m)
        if len(self._position_history) > 10:
            self._position_history.pop(0)

        is_bounce, bounce_zone = self._detect_bounce()
        if is_bounce:
            self._trajectory_segment += 1

        return BallTrackResult(
            frame_idx=frame_idx,
            position_px=position_px,
            position_m=position_m,
            confidence=confidence,
            is_bounce=is_bounce,
            bounce_zone=bounce_zone,
            trajectory_segment=self._trajectory_segment,
            interpolated=interpolated,
        )

    def _infer(
        self,
        frames: tuple[np.ndarray, np.ndarray, np.ndarray],
        orig_h: int,
        orig_w: int,
    ) -> tuple[Optional[np.ndarray], float]:
        inp = _preprocess(frames)
        with torch.no_grad():
            out = self._model(inp.to(self._device))   # (1, 256, H×W)

        intensity = out.argmax(dim=1).cpu().numpy()[0]  # (H×W,)
        intensity = intensity.reshape(INPUT_H, INPUT_W).astype(np.float32)

        # Normalise and extract ball via HoughCircles (same as reference impl)
        heatmap = (intensity * (255.0 / 255.0)).astype(np.uint8)
        _, binary = cv2.threshold(heatmap, 127, 255, cv2.THRESH_BINARY)
        circles = cv2.HoughCircles(
            binary,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=1,
            param1=50,
            param2=2,
            minRadius=2,
            maxRadius=7,
        )
        if circles is None or len(circles[0]) != 1:
            return None, 0.0

        cx, cy = circles[0][0][:2]
        # Confidence: normalised peak intensity at detected location
        confidence = float(intensity[int(cy), int(cx)]) / 255.0

        # Scale back to original frame resolution
        sx = orig_w / INPUT_W
        sy = orig_h / INPUT_H
        return np.array([cx * sx, cy * sy], dtype=np.float64), confidence

    def _extrapolate(self, frame_idx: int) -> np.ndarray:
        """Constant-velocity prediction from last 2 confirmed detections."""
        (f0, p0, _), (f1, p1, _) = self._recent[-2], self._recent[-1]
        dt_known = f1 - f0
        if dt_known == 0:
            return p1.copy()
        velocity = (p1 - p0) / dt_known
        dt_pred = frame_idx - f1
        return p1 + velocity * dt_pred

    def _detect_bounce(self) -> tuple[bool, Optional[str]]:
        if len(self._position_history) < 3:
            return False, None
        recent = self._position_history[-3:]
        dy = np.diff([p[1] for p in recent])
        if dy[0] < -self._bounce_thresh and dy[1] > self._bounce_thresh:
            return True, _classify_zone(recent[1])
        return False, None

    def reset(self) -> None:
        self._trajectory_segment = 0
        self._miss_count = 0
        self._recent.clear()
        self._position_history.clear()


def _preprocess(
    frames: tuple[np.ndarray, np.ndarray, np.ndarray]
) -> torch.Tensor:
    """Stack 3 BGR frames into a (1, 9, 360, 640) float tensor."""
    channels = []
    for f in frames:
        f_resized = cv2.resize(f, (INPUT_W, INPUT_H))
        f_rgb = cv2.cvtColor(f_resized, cv2.COLOR_BGR2RGB)
        channels.append(f_rgb.astype(np.float32) / 255.0)
    stacked = np.concatenate(channels, axis=2)          # (360, 640, 9)
    tensor = torch.from_numpy(stacked).permute(2, 0, 1) # (9, 360, 640)
    return tensor.unsqueeze(0)                           # (1, 9, 360, 640)


def _classify_zone(position_m: np.ndarray) -> str:
    """Classify a court-meter bounce position into one of 9 landing zones.

    Depth is measured in absolute y (0 = near baseline, 23.77 = far baseline):
      deep:  y > 17.37  (past far service line, near far baseline)
      mid:   11.89 < y ≤ 17.37  (far service box)
      short: y ≤ 11.89  (near net / drop shot territory)

    Lateral is relative to court center (positive = deuce/right side):
      T: x < −1.0 m   (toward center T)
      W: x >  1.0 m   (toward wide/sideline)
      C: otherwise    (center)
    """
    x, y = position_m

    if y > _FAR_SERVICE_LINE_Y:
        depth = "deep"
    elif y > _NET_Y:
        depth = "mid"
    else:
        depth = "short"

    if x < _LATERAL_BREAKS[0]:
        lateral = "T"
    elif x > _LATERAL_BREAKS[1]:
        lateral = "W"
    else:
        lateral = "C"

    return f"{lateral}_{depth}"
