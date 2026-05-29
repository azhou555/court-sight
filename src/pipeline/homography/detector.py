"""Court line detection and homography estimation."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# Real-world court keypoints in meters (origin = center of near baseline).
# Singles court: 23.77m deep, 8.23m wide (half = 4.115m each side).
COURT_KEYPOINTS_M = np.array([
    [-4.115,  0.0],    # near baseline left corner
    [ 4.115,  0.0],    # near baseline right corner
    [-4.115, 23.77],   # far baseline left corner
    [ 4.115, 23.77],   # far baseline right corner
    [-3.05,  6.40],    # near service T left
    [ 3.05,  6.40],    # near service T right
    [-3.05, 17.37],    # far service T left
    [ 3.05, 17.37],    # far service T right
    [ 0.0,   6.40],    # near service T center
    [ 0.0,  17.37],    # far service T center
], dtype=np.float64)


@dataclass
class HomographyResult:
    frame_idx: int
    H: Optional[np.ndarray]
    reprojection_error: float
    inlier_count: int
    confidence: float
    camera_angle_valid: bool
    keypoints_px: np.ndarray = field(default_factory=lambda: np.array([]))


class CourtHomographyEstimator:
    """Estimates H mapping image pixels → court meters for each frame."""

    def __init__(
        self,
        reestimate_interval: int = 30,
        max_reprojection_error: float = 0.15,
        min_inliers: int = 4,
    ):
        self.reestimate_interval = reestimate_interval
        self.max_reprojection_error = max_reprojection_error
        self.min_inliers = min_inliers
        self._last_H: Optional[np.ndarray] = None
        self._last_frame: int = -reestimate_interval

    def process_frame(self, frame: np.ndarray, frame_idx: int) -> HomographyResult:
        should_reestimate = (
            self._last_H is None
            or (frame_idx - self._last_frame) >= self.reestimate_interval
        )
        if should_reestimate:
            result = self._estimate(frame, frame_idx)
            if result.H is not None:
                self._last_H = result.H
                self._last_frame = frame_idx
            return result
        return HomographyResult(
            frame_idx=frame_idx,
            H=self._last_H,
            reprojection_error=0.0,
            inlier_count=0,
            confidence=0.5,
            camera_angle_valid=True,
        )

    def _estimate(self, frame: np.ndarray, frame_idx: int) -> HomographyResult:
        keypoints_px = self._detect_court_keypoints(frame)
        if keypoints_px is None or len(keypoints_px) < 4:
            return HomographyResult(
                frame_idx=frame_idx,
                H=None,
                reprojection_error=float("inf"),
                inlier_count=0,
                confidence=0.0,
                camera_angle_valid=False,
            )

        src_pts = keypoints_px.astype(np.float64)
        dst_pts = COURT_KEYPOINTS_M[: len(src_pts)]

        import cv2
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if H is None:
            return HomographyResult(
                frame_idx=frame_idx,
                H=None,
                reprojection_error=float("inf"),
                inlier_count=0,
                confidence=0.0,
                camera_angle_valid=True,
            )

        inliers = mask.ravel().astype(bool)
        err = self._reprojection_error(H, src_pts[inliers], dst_pts[inliers])
        inlier_ratio = inliers.sum() / max(len(inliers), 1)
        confidence = inlier_ratio * max(0.0, 1.0 - err / self.max_reprojection_error)

        return HomographyResult(
            frame_idx=frame_idx,
            H=H,
            reprojection_error=err,
            inlier_count=int(inliers.sum()),
            confidence=float(confidence),
            camera_angle_valid=err < self.max_reprojection_error,
            keypoints_px=keypoints_px,
        )

    def _detect_court_keypoints(self, frame: np.ndarray) -> Optional[np.ndarray]:
        # TODO: implement court line detection via Hough transform + RANSAC
        # Returns (N, 2) array of pixel coordinates matching COURT_KEYPOINTS_M order.
        raise NotImplementedError

    @staticmethod
    def _reprojection_error(
        H: np.ndarray, src: np.ndarray, dst: np.ndarray
    ) -> float:
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
