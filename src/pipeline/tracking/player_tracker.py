"""Player detection and tracking via YOLO26m + BoT-SORT."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.signal import savgol_filter

from ..homography.detector import pixel_to_court


# Court boundary constants (meters) used for ball-boy suppression mask
_COURT_X_LIMIT = 5.485   # doubles sideline
_COURT_Y_MIN = 0.0
_COURT_Y_MAX = 23.77
_MASK_MARGIN = 3.0        # meters of tolerance beyond court boundary

_SG_WINDOW = 7
_SG_POLY = 2
_FPS_DEFAULT = 30.0
_MAX_HISTORY = _SG_WINDOW + 10


@dataclass
class PlayerState:
    track_id: int
    bbox_px: np.ndarray        # [x1, y1, x2, y2] original frame pixels
    position_m: np.ndarray     # [x, y] court meters, foot contact point
    velocity_ms: np.ndarray    # [vx, vy] m/s
    confidence: float


@dataclass
class PlayerTrackResult:
    frame_idx: int
    near_player: Optional[PlayerState]
    far_player: Optional[PlayerState]


class PlayerTracker:
    """YOLO26m detector + BoT-SORT tracker with court-coordinate projection."""

    NEAR_SIDE_MAX_Y = 11.885   # net center in court meters

    def __init__(
        self,
        model_path: str = "yolo26m.pt",
        detection_threshold: float = 0.5,
        iou_threshold: float = 0.4,
        fps: float = _FPS_DEFAULT,
        device: str = "cpu",
    ):
        self._fps = fps
        self._detection_threshold = detection_threshold
        self._iou_threshold = iou_threshold
        self._position_history: dict[int, list[tuple[int, np.ndarray]]] = defaultdict(list)

        from ultralytics import YOLO
        self._detector = YOLO(model_path)

        from boxmot.trackers import BotSort
        self._tracker = BotSort(
            with_reid=False,
            cmc_method="ecc",
            frame_rate=int(fps),
            track_high_thresh=detection_threshold,
            new_track_thresh=detection_threshold + 0.1,
            track_buffer=30,
        )

    def process_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        H: Optional[np.ndarray],
        camera_cut: bool = False,
    ) -> PlayerTrackResult:
        if H is None:
            return PlayerTrackResult(frame_idx=frame_idx, near_player=None, far_player=None)

        if camera_cut:
            self._tracker.reset()
            self._position_history.clear()

        dets = self._detect(frame, H)
        tracks = self._update_tracks(dets, frame)
        return self._assign_roles(tracks, H, frame_idx)

    def _detect(self, frame: np.ndarray, H: np.ndarray) -> np.ndarray:
        """Run YOLO26m, filter to persons within the court mask.

        Returns:
            (N, 6) float32 array: [x1, y1, x2, y2, conf, cls]
        """
        results = self._detector(
            frame,
            conf=self._detection_threshold,
            iou=self._iou_threshold,
            classes=[0],   # person only
            verbose=False,
        )
        rows = []
        for r in results:
            boxes = r.boxes
            if boxes is None or len(boxes) == 0:
                continue
            xyxy = boxes.xyxy.cpu().numpy()    # (N, 4)
            confs = boxes.conf.cpu().numpy()   # (N,)
            for (x1, y1, x2, y2), conf in zip(xyxy, confs):
                foot_px = np.array([(x1 + x2) / 2.0, y2])
                if not self._in_court_mask(foot_px, H):
                    continue
                rows.append([x1, y1, x2, y2, conf, 0.0])

        if not rows:
            return np.zeros((0, 6), dtype=np.float32)
        return np.array(rows, dtype=np.float32)

    def _in_court_mask(self, foot_px: np.ndarray, H: np.ndarray) -> bool:
        pos = pixel_to_court(foot_px, H)
        x, y = pos
        return (
            -(_COURT_X_LIMIT + _MASK_MARGIN) <= x <= (_COURT_X_LIMIT + _MASK_MARGIN)
            and (_COURT_Y_MIN - _MASK_MARGIN) <= y <= (_COURT_Y_MAX + _MASK_MARGIN)
        )

    def _update_tracks(self, dets: np.ndarray, frame: np.ndarray) -> list[dict]:
        """Run BoT-SORT update. Returns list of active track dicts."""
        tracks = self._tracker.update(dets, frame)
        # Output columns: [x1, y1, x2, y2, track_id, conf, cls, det_idx]
        result = []
        for t in tracks:
            result.append({
                "bbox":      t[:4].tolist(),
                "track_id":  int(t[4]),
                "confidence": float(t[5]),
            })
        return result

    def _assign_roles(
        self,
        tracks: list[dict],
        H: np.ndarray,
        frame_idx: int,
    ) -> PlayerTrackResult:
        near: Optional[PlayerState] = None
        far: Optional[PlayerState] = None

        for track in tracks:
            x1, y1, x2, y2 = track["bbox"]
            foot_px = np.array([(x1 + x2) / 2.0, y2])
            pos_m = pixel_to_court(foot_px, H)
            vel_ms = self._compute_velocity(track["track_id"], pos_m, frame_idx)

            state = PlayerState(
                track_id=track["track_id"],
                bbox_px=np.array(track["bbox"]),
                position_m=pos_m,
                velocity_ms=vel_ms,
                confidence=track["confidence"],
            )

            if pos_m[1] < self.NEAR_SIDE_MAX_Y:
                near = state
            else:
                far = state

        return PlayerTrackResult(
            frame_idx=frame_idx,
            near_player=near,
            far_player=far,
        )

    def _compute_velocity(
        self, track_id: int, pos_m: np.ndarray, frame_idx: int
    ) -> np.ndarray:
        history = self._position_history[track_id]
        history.append((frame_idx, pos_m.copy()))
        if len(history) > _MAX_HISTORY:
            del history[0]

        n = len(history)
        if n < 2:
            return np.zeros(2)

        positions = np.array([h[1] for h in history])   # (n, 2)
        frame_idxs = np.array([h[0] for h in history])  # (n,)

        if n >= _SG_WINDOW:
            mean_dt = float(np.mean(np.diff(frame_idxs))) / self._fps
            deriv = savgol_filter(
                positions, _SG_WINDOW, _SG_POLY, deriv=1, delta=mean_dt, axis=0
            )
            return deriv[-1].astype(np.float64)

        # Finite difference over available window (< 7 frames)
        window = history[-min(3, n):]
        dt = (window[-1][0] - window[0][0]) / self._fps
        if dt == 0.0:
            return np.zeros(2)
        return (window[-1][1] - window[0][1]) / dt
