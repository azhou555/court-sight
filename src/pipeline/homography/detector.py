"""Court line detection and homography estimation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
import torch

from .model import CourtKeypointNet, INPUT_H, INPUT_W, NUM_KEYPOINTS, load_pretrained


# Real-world court keypoints in meters (origin = center of near baseline,
# x positive toward deuce/right, y positive toward far baseline).
# Ordering matches CourtReference.key_points from yastrebksv/TennisCourtDetector,
# which is the channel ordering used by CourtKeypointNet.
# Run scripts/verify_keypoint_channels.py to visually confirm after any weight update.
COURT_KEYPOINTS_M = np.array([
    [-5.485, 23.77],   # 0: far baseline, doubles left corner
    [ 5.485, 23.77],   # 1: far baseline, doubles right corner
    [-5.485,  0.00],   # 2: near baseline, doubles left corner
    [ 5.485,  0.00],   # 3: near baseline, doubles right corner
    [-4.115, 23.77],   # 4: far baseline, singles left corner
    [-4.115,  0.00],   # 5: near baseline, singles left corner  [verified from pixel positions]
    [ 4.115, 23.77],   # 6: far baseline, singles right corner  [verified from pixel positions]
    [ 4.115,  0.00],   # 7: near baseline, singles right corner
    [-4.115, 17.37],   # 8: far service line, left T
    [ 4.115, 17.37],   # 9: far service line, right T
    [-4.115,  6.40],   # 10: near service line, left T
    [ 4.115,  6.40],   # 11: near service line, right T
    [ 0.000, 17.37],   # 12: far center T
    [ 0.000,  6.40],   # 13: near center T
], dtype=np.float64)

# Channels whose pixel coords must appear in H estimation (near baseline anchors).
_NEAR_BASELINE_CHANNELS = {2, 3, 6, 7}

_CONF_THRESHOLD = 0.3
_MIN_INLIERS = 4
_MAX_REPROJ_ERROR_M = 0.50   # pretrained model accuracy; tighten to 0.15m after fine-tuning

# Metric tolerances for self-consistency check (meters).
# Pretrained model has ~0.37m reprojection error so tolerances are kept loose.
# Tighten _METRIC_TOL to 0.25m after fine-tuning achieves <0.15m error.
_SINGLES_WIDTH = 8.23
_COURT_DEPTH = 23.77
_SERVICE_DEPTH = 6.40
_METRIC_TOL = 1.0


@dataclass
class HomographyResult:
    frame_idx: int
    H: Optional[np.ndarray]
    reprojection_error: float
    inlier_count: int
    confidence: float
    camera_angle_valid: bool
    keypoints_px: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    keypoint_confidences: np.ndarray = field(default_factory=lambda: np.zeros(0))


class CourtHomographyEstimator:
    """Estimates H mapping image pixels → court meters for each frame."""

    def __init__(
        self,
        model: Optional[CourtKeypointNet] = None,
        weights_path: Optional[str] = None,
        device: str = "cpu",
        reestimate_interval: int = 30,
    ):
        if model is not None:
            self._model = model.to(device).eval()
        else:
            self._model = load_pretrained(weights_path, device=device).eval()

        self._device = device
        self.reestimate_interval = reestimate_interval
        self._last_H: Optional[np.ndarray] = None
        self._last_frame: int = -reestimate_interval

    def process_frame(self, frame: np.ndarray, frame_idx: int) -> HomographyResult:
        should_reestimate = (
            self._last_H is None
            or (frame_idx - self._last_frame) >= self.reestimate_interval
        )
        if not should_reestimate:
            return HomographyResult(
                frame_idx=frame_idx,
                H=self._last_H,
                reprojection_error=0.0,
                inlier_count=0,
                confidence=0.5,
                camera_angle_valid=True,
            )

        result = self._estimate(frame, frame_idx)
        if result.H is not None:
            self._last_H = result.H
            self._last_frame = frame_idx
        elif self._last_H is not None:
            return HomographyResult(
                frame_idx=frame_idx,
                H=self._last_H,
                reprojection_error=float("inf"),
                inlier_count=0,
                confidence=0.1,
                camera_angle_valid=False,
            )
        return result

    def _estimate(self, frame: np.ndarray, frame_idx: int) -> HomographyResult:
        src_pts, dst_pts, confs, channel_ids = self._detect_court_keypoints(frame)

        near_baseline_found = len(_NEAR_BASELINE_CHANNELS & set(channel_ids)) >= 2
        if len(src_pts) < _MIN_INLIERS or not near_baseline_found:
            return HomographyResult(
                frame_idx=frame_idx,
                H=None,
                reprojection_error=float("inf"),
                inlier_count=0,
                confidence=0.0,
                camera_angle_valid=False,
                keypoints_px=np.array(src_pts),
                keypoint_confidences=np.array(confs),
            )

        src = np.array(src_pts, dtype=np.float64)
        dst = np.array(dst_pts, dtype=np.float64)

        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if H is None:
            return HomographyResult(
                frame_idx=frame_idx,
                H=None,
                reprojection_error=float("inf"),
                inlier_count=0,
                confidence=0.0,
                camera_angle_valid=True,
                keypoints_px=src,
                keypoint_confidences=np.array(confs),
            )

        inliers = mask.ravel().astype(bool)
        err = _reprojection_error(H, src[inliers], dst[inliers])

        if not self._self_consistency_check(H, channel_ids, src):
            return HomographyResult(
                frame_idx=frame_idx,
                H=None,
                reprojection_error=err,
                inlier_count=int(inliers.sum()),
                confidence=0.0,
                camera_angle_valid=False,
                keypoints_px=src,
                keypoint_confidences=np.array(confs),
            )

        inlier_ratio = inliers.sum() / max(len(inliers), 1)
        confidence = float(inlier_ratio * max(0.0, 1.0 - err / _MAX_REPROJ_ERROR_M))

        return HomographyResult(
            frame_idx=frame_idx,
            H=H,
            reprojection_error=float(err),
            inlier_count=int(inliers.sum()),
            confidence=confidence,
            camera_angle_valid=err < _MAX_REPROJ_ERROR_M,
            keypoints_px=src,
            keypoint_confidences=np.array(confs),
        )

    def _detect_court_keypoints(
        self, frame: np.ndarray
    ) -> tuple[list, list, list, list]:
        """Run heatmap inference and return confident keypoints.

        Returns:
            src_pts:     list of (x, y) pixel coords in original frame space
            dst_pts:     list of (x, y) court meter coords (from COURT_KEYPOINTS_M)
            confidences: list of float heatmap peak values
            channel_ids: list of int channel indices (0-13)
        """
        orig_h, orig_w = frame.shape[:2]
        scale_x = orig_w / INPUT_W
        scale_y = orig_h / INPUT_H

        inp = _preprocess(frame)
        with torch.no_grad():
            heatmaps = self._model(inp.to(self._device))[0, :NUM_KEYPOINTS]  # (14, H, W)

        src_pts, dst_pts, confs, channel_ids = [], [], [], []
        for ch in range(NUM_KEYPOINTS):
            hm = heatmaps[ch].cpu().numpy()
            peak_val = float(hm.max())
            if peak_val < _CONF_THRESHOLD:
                continue
            px, py = _weighted_centroid(hm)
            # Scale from inference resolution back to original frame resolution
            src_pts.append([px * scale_x, py * scale_y])
            dst_pts.append(COURT_KEYPOINTS_M[ch].tolist())
            confs.append(peak_val)
            channel_ids.append(ch)

        return src_pts, dst_pts, confs, channel_ids

    def _self_consistency_check(
        self,
        H: np.ndarray,
        channel_ids: list[int],
        src_pts: np.ndarray,
    ) -> bool:
        ch_to_px = {ch: src_pts[i] for i, ch in enumerate(channel_ids)}

        def court_pt(ch: int) -> Optional[np.ndarray]:
            if ch not in ch_to_px:
                return None
            return pixel_to_court(ch_to_px[ch], H)

        checks_run = 0
        near_left = court_pt(5)   # ch 5 = near-baseline singles left (verified)
        near_right = court_pt(7)
        if near_left is not None and near_right is not None:
            width = float(np.linalg.norm(near_right - near_left))
            if abs(width - _SINGLES_WIDTH) > _METRIC_TOL:
                return False
            checks_run += 1

        far_left = court_pt(4)
        if near_left is not None and far_left is not None:
            depth = float(np.linalg.norm(far_left - near_left))
            if abs(depth - _COURT_DEPTH) > _METRIC_TOL * 2:
                return False
            checks_run += 1

        near_svc_left = court_pt(10)
        if near_left is not None and near_svc_left is not None:
            svc_depth = float(np.linalg.norm(near_svc_left - near_left))
            if abs(svc_depth - _SERVICE_DEPTH) > _METRIC_TOL:
                return False
            checks_run += 1

        # Require at least one geometric check to pass
        return checks_run >= 1


def _weighted_centroid(hm: np.ndarray) -> tuple[float, float]:
    """Return (x, y) as the heatmap-weighted centroid of activations above 50% of peak."""
    threshold = hm.max() * 0.5
    mask = hm > threshold
    ys, xs = np.where(mask)
    if len(xs) == 0:
        flat = int(hm.argmax())
        py, px = divmod(flat, hm.shape[1])
        return float(px), float(py)
    weights = hm[mask]
    return float(np.average(xs, weights=weights)), float(np.average(ys, weights=weights))


def _preprocess(frame: np.ndarray) -> torch.Tensor:
    img = cv2.resize(frame, (INPUT_W, INPUT_H))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(img).float() / 255.0
    return tensor.permute(2, 0, 1).unsqueeze(0)


def _reprojection_error(H: np.ndarray, src: np.ndarray, dst: np.ndarray) -> float:
    ones = np.ones((len(src), 1))
    src_h = np.hstack([src, ones])
    proj_h = (H @ src_h.T).T
    proj = proj_h[:, :2] / proj_h[:, 2:3]
    return float(np.mean(np.linalg.norm(proj - dst, axis=1)))


def pixel_to_court(pixel_pt: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Project a pixel coordinate to court-meter coordinates via homography H."""
    pt = np.array([*pixel_pt, 1.0], dtype=np.float64)
    result = H @ pt
    return result[:2] / result[2]
