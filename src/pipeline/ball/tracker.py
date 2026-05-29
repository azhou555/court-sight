"""Ball tracking via TrackNetV2 with Kalman filtering and bounce detection."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional

from src.pipeline.homography.detector import pixel_to_court


COURT_ZONES = [
    "T_deep",  "C_deep",  "W_deep",
    "T_mid",   "C_mid",   "W_mid",
    "T_short", "C_short", "W_short",
]

# Zone boundaries relative to opponent's baseline (y=0 for their side)
# depth: short < 6.4m, mid 6.4–11.89m, deep > 11.89m (past service line)
# lateral: T < -1.0m, W > 1.0m, C in between (relative to center)
_DEPTH_BREAKS = (6.4, 11.89)     # service line depth, center of court
_LATERAL_BREAKS = (-1.0, 1.0)


@dataclass
class BallTrackResult:
    frame_idx: int
    position_px: Optional[np.ndarray]
    position_m: Optional[np.ndarray]
    confidence: float
    is_bounce: bool
    bounce_zone: Optional[str]
    trajectory_segment: int


class BallTracker:
    """
    Wraps TrackNetV2 heatmap regression with a Kalman filter for
    continuity across missed detections.
    """

    def __init__(
        self,
        tracknet_weights: str = "tracknet_v2.pth",
        detection_threshold: float = 0.5,
        bounce_velocity_threshold: float = 0.1,
    ):
        self.detection_threshold = detection_threshold
        self.bounce_velocity_threshold = bounce_velocity_threshold
        self._trajectory_segment = 0
        self._position_history: list[tuple[int, np.ndarray]] = []
        # TODO: load TrackNetV2 model
        # self._model = TrackNetV2.load(tracknet_weights)

    def process_frame(
        self,
        frames: tuple[np.ndarray, np.ndarray, np.ndarray],
        frame_idx: int,
        H: Optional[np.ndarray],
    ) -> BallTrackResult:
        """frames: (prev, curr, next) BGR frames."""
        position_px, confidence = self._infer(frames)

        if confidence < self.detection_threshold or H is None:
            return BallTrackResult(
                frame_idx=frame_idx,
                position_px=position_px,
                position_m=None,
                confidence=confidence,
                is_bounce=False,
                bounce_zone=None,
                trajectory_segment=self._trajectory_segment,
            )

        position_m = pixel_to_court(position_px, H)
        self._position_history.append((frame_idx, position_m))
        is_bounce, bounce_zone = self._detect_bounce(position_m)

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
        )

    def _infer(
        self, frames: tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> tuple[Optional[np.ndarray], float]:
        # TODO: stack frames, run TrackNetV2 inference, return centroid + confidence
        raise NotImplementedError

    def _detect_bounce(self, position_m: np.ndarray) -> tuple[bool, Optional[str]]:
        if len(self._position_history) < 3:
            return False, None
        recent = [p for _, p in self._position_history[-3:]]
        dy = np.diff([p[1] for p in recent])
        if dy[0] < -self.bounce_velocity_threshold and dy[1] > self.bounce_velocity_threshold:
            zone = _classify_zone(position_m)
            return True, zone
        return False, None

    def reset_segment(self) -> None:
        self._trajectory_segment = 0
        self._position_history.clear()


def _classify_zone(position_m: np.ndarray) -> str:
    """Classify a court-meter position into one of the 9 landing zones."""
    x, y = position_m

    # Depth classification (y = 0 at near baseline, 23.77 at far baseline)
    # For the far side (opponent), depth is measured from far baseline (y=23.77)
    depth_from_baseline = 23.77 - y
    if depth_from_baseline < _DEPTH_BREAKS[0]:
        depth = "short"
    elif depth_from_baseline < _DEPTH_BREAKS[1]:
        depth = "mid"
    else:
        depth = "deep"

    # Lateral classification
    if x < _LATERAL_BREAKS[0]:
        lateral = "T"
    elif x > _LATERAL_BREAKS[1]:
        lateral = "W"
    else:
        lateral = "C"

    return f"{lateral}_{depth}"
