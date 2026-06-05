"""Pose keypoint extraction via YOLO26-pose and projection to court coordinates.

Uses yolo26m-pose.pt (Ultralytics) instead of the originally planned RTMPose,
to avoid the mmpose/mmcv install stack.  COCO-17 keypoints, same schema.
Upgrade to RTMPose-m if per-joint accuracy becomes a bottleneck.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..homography.detector import pixel_to_court


COCO_KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
NUM_KEYPOINTS = 17
STROKE_WINDOW_FRAMES = 30
STROKE_PRE_CONTACT = 20     # frames before contact in the window
STROKE_POST_CONTACT = 10    # frames after contact

POSE_CONFIDENCE_THRESHOLD = 0.3   # per-keypoint minimum
POSE_QUALITY_THRESHOLD = 0.5      # mean of key joints to flag a frame

# Joints used for pose quality scoring (shoulders, elbows, wrists, left hip)
_QUALITY_JOINTS = [5, 6, 7, 8, 9, 10, 11]

# Indices for normalization
_LEFT_HIP  = 11
_RIGHT_HIP = 12
_LEFT_SHOULDER  = 5
_RIGHT_SHOULDER = 6


@dataclass
class PoseResult:
    frame_idx: int
    player_role: str
    keypoints_px: np.ndarray    # (17, 2) in original frame pixels
    keypoints_m: np.ndarray     # (17, 2) in court meters
    confidences: np.ndarray     # (17,)
    pose_quality: float         # mean confidence of key joints


@dataclass
class StrokePoseSequence:
    """30-frame pose window centred on a ball-contact frame."""

    shot_id: str
    contact_frame: int
    player_role: str
    keypoints: np.ndarray       # (30, 17, 2) court-meter coords
    confidences: np.ndarray     # (30, 17)
    pose_quality: float         # mean quality across window


class PoseExtractor:
    """Extract COCO-17 body pose using YOLO26-pose.

    process_player() takes a tracked player bounding box and a BGR frame,
    crops the player region, runs YOLO26-pose inference, and returns keypoints
    projected to court-meter space via the current homography.
    """

    def __init__(
        self,
        model_path: str = "yolo26m-pose.pt",
        device: str = "cpu",
    ):
        from ultralytics import YOLO
        self._model = YOLO(model_path)
        self._device = device

    def process_player(
        self,
        frame: np.ndarray,
        bbox_px: np.ndarray,
        player_role: str,
        frame_idx: int,
        H: Optional[np.ndarray],
    ) -> Optional[PoseResult]:
        """Run pose on a player crop; return PoseResult or None."""
        if H is None:
            return None

        keypoints_px, confidences = self._infer(frame, bbox_px)
        if keypoints_px is None:
            return None

        keypoints_m = _project_to_court(keypoints_px, H)
        quality = float(np.mean(confidences[_QUALITY_JOINTS]))

        return PoseResult(
            frame_idx=frame_idx,
            player_role=player_role,
            keypoints_px=keypoints_px,
            keypoints_m=keypoints_m,
            confidences=confidences,
            pose_quality=quality,
        )

    def _infer(
        self,
        frame: np.ndarray,
        bbox_px: np.ndarray,
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Run YOLO26-pose on a padded player crop.

        Returns:
            keypoints_px: (17, 2) full-frame pixel coords, zeros where conf < threshold
            confidences:  (17,) per-keypoint confidence scores
        """
        x1, y1, x2, y2 = bbox_px.astype(int)
        h, w = frame.shape[:2]

        # 10% padding so limbs near box edges are not cropped
        pad = int((y2 - y1) * 0.10)
        x1c = max(0, x1 - pad)
        y1c = max(0, y1 - pad)
        x2c = min(w, x2 + pad)
        y2c = min(h, y2 + pad)

        crop = frame[y1c:y2c, x1c:x2c]
        if crop.size == 0:
            return None, None

        results = self._model(crop, verbose=False, device=self._device)
        if (
            not results
            or results[0].keypoints is None
            or len(results[0].keypoints) == 0
        ):
            return None, None

        # When multiple people appear in the crop, take the highest-conf detection
        boxes_conf = (
            results[0].boxes.conf.cpu().numpy()
            if results[0].boxes is not None
            else np.array([1.0])
        )
        best = int(boxes_conf.argmax())
        kp_data = results[0].keypoints.data[best].cpu().numpy()  # (17, 3)

        keypoints_px = kp_data[:, :2].copy()
        confidences = kp_data[:, 2].copy()

        # Re-map crop-local coords to full-frame coords
        keypoints_px[:, 0] += x1c
        keypoints_px[:, 1] += y1c

        # Zero out low-confidence keypoints so they don't corrupt normalization
        keypoints_px[confidences < POSE_CONFIDENCE_THRESHOLD] = 0.0

        return keypoints_px, confidences

    def build_stroke_sequence(
        self,
        pose_buffer: list[PoseResult],
        contact_frame: int,
        shot_id: str,
        player_role: str,
    ) -> Optional[StrokePoseSequence]:
        """Slice a 30-frame window from a running pose buffer.

        Args:
            pose_buffer:   All PoseResult objects accumulated so far.
            contact_frame: Ball-contact frame index (centre of window).
            shot_id:       Identifier for the shot (e.g. "{match}_{pt}_{shot}").
            player_role:   "near" or "far".

        Returns:
            StrokePoseSequence, or None if fewer than 20 frames are available.
        """
        start = contact_frame - STROKE_PRE_CONTACT
        end = contact_frame + STROKE_POST_CONTACT

        by_frame = {
            p.frame_idx: p for p in pose_buffer if p.player_role == player_role
        }
        window = [by_frame.get(i) for i in range(start, end)]

        if sum(p is not None for p in window) < 20:
            return None

        keypoints = np.zeros((STROKE_WINDOW_FRAMES, NUM_KEYPOINTS, 2), dtype=np.float32)
        confidences = np.zeros((STROKE_WINDOW_FRAMES, NUM_KEYPOINTS), dtype=np.float32)
        for t, pose in enumerate(window):
            if pose is not None:
                keypoints[t] = pose.keypoints_m
                confidences[t] = pose.confidences

        valid = confidences[:, _QUALITY_JOINTS]
        quality = float(np.mean(valid[valid > 0])) if (valid > 0).any() else 0.0

        return StrokePoseSequence(
            shot_id=shot_id,
            contact_frame=contact_frame,
            player_role=player_role,
            keypoints=keypoints,
            confidences=confidences,
            pose_quality=quality,
        )


# ── Utilities ────────────────────────────────────────────────────────────────

def _project_to_court(keypoints_px: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Project (17, 2) pixel keypoints to court-meter coordinates."""
    ones = np.ones((NUM_KEYPOINTS, 1))
    kp_h = np.hstack([keypoints_px, ones])
    court_h = (H @ kp_h.T).T
    return (court_h[:, :2] / court_h[:, 2:3]).astype(np.float32)


def normalize_pose_sequence(keypoints: np.ndarray) -> np.ndarray:
    """Normalize (T, 17, 2) keypoint array to body-relative coordinates.

    1. Translate each frame so the hip midpoint is the origin.
    2. Scale by the mean shoulder-to-hip distance across the window.

    Returns:
        (T, 34) float32 array of normalised flattened keypoints.
    """
    T = keypoints.shape[0]

    hip_mid = (keypoints[:, _LEFT_HIP] + keypoints[:, _RIGHT_HIP]) / 2.0  # (T, 2)
    centred = keypoints - hip_mid[:, np.newaxis, :]

    shoulder_mid = (keypoints[:, _LEFT_SHOULDER] + keypoints[:, _RIGHT_SHOULDER]) / 2.0
    scale = float(np.linalg.norm(shoulder_mid - hip_mid, axis=1).mean()) + 1e-6
    normalised = centred / scale

    return normalised.reshape(T, -1).astype(np.float32)
