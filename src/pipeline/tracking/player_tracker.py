"""Player detection and tracking via YOLOv8 + ByteTrack."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from src.pipeline.homography.detector import pixel_to_court


@dataclass
class PlayerState:
    track_id: int
    bbox_px: np.ndarray           # [x1, y1, x2, y2]
    position_m: np.ndarray        # [x, y] court meters (feet contact point)
    velocity_ms: np.ndarray       # [vx, vy] m/s
    confidence: float


@dataclass
class PlayerTrackResult:
    frame_idx: int
    near_player: Optional[PlayerState]
    far_player: Optional[PlayerState]


class PlayerTracker:
    """
    Wraps YOLOv8 person detector and ByteTrack multi-object tracker.
    Projects detections to court coordinates via the current homography.
    """

    NEAR_SIDE_MAX_Y = 11.885   # meters — center of court (net)

    def __init__(
        self,
        yolo_model_path: str = "yolov8m.pt",
        detection_threshold: float = 0.5,
        iou_threshold: float = 0.4,
        velocity_smoothing_window: int = 7,
    ):
        self.detection_threshold = detection_threshold
        self.iou_threshold = iou_threshold
        self.velocity_smoothing_window = velocity_smoothing_window
        self._position_history: dict[int, list] = {}
        # TODO: initialize YOLO and ByteTrack here
        # self._detector = YOLO(yolo_model_path)
        # self._tracker = ByteTrack(...)

    def process_frame(
        self, frame: np.ndarray, frame_idx: int, H: Optional[np.ndarray]
    ) -> PlayerTrackResult:
        if H is None:
            return PlayerTrackResult(frame_idx=frame_idx, near_player=None, far_player=None)

        raw_detections = self._detect(frame)
        tracks = self._update_tracks(raw_detections, frame)
        player_states = self._assign_roles(tracks, H, frame_idx)
        return PlayerTrackResult(frame_idx=frame_idx, **player_states)

    def _detect(self, frame: np.ndarray) -> list[dict]:
        # TODO: run YOLOv8 inference, filter to person class, apply court mask
        raise NotImplementedError

    def _update_tracks(self, detections: list[dict], frame: np.ndarray) -> list[dict]:
        # TODO: run ByteTrack update step
        raise NotImplementedError

    def _assign_roles(
        self, tracks: list[dict], H: np.ndarray, frame_idx: int
    ) -> dict:
        near, far = None, None
        for track in tracks:
            foot_px = np.array([
                (track["bbox"][0] + track["bbox"][2]) / 2,
                track["bbox"][3],
            ])
            pos_m = pixel_to_court(foot_px, H)
            vel_ms = self._compute_velocity(track["track_id"], pos_m, frame_idx)
            state = PlayerState(
                track_id=track["track_id"],
                bbox_px=np.array(track["bbox"]),
                position_m=pos_m,
                velocity_ms=vel_ms,
                confidence=track["confidence"],
            )
            if pos_m[1] <= self.NEAR_SIDE_MAX_Y:
                near = state
            else:
                far = state
        return {"near_player": near, "far_player": far}

    def _compute_velocity(
        self, track_id: int, pos_m: np.ndarray, frame_idx: int
    ) -> np.ndarray:
        history = self._position_history.setdefault(track_id, [])
        history.append((frame_idx, pos_m.copy()))
        if len(history) < 2:
            return np.zeros(2)
        # Finite difference over last 3 frames, assume 30fps
        window = history[-3:]
        dt = (window[-1][0] - window[0][0]) / 30.0
        if dt == 0:
            return np.zeros(2)
        return (window[-1][1] - window[0][1]) / dt
