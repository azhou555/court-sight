"""Pose keypoint extraction via RTMPose and projection to court coordinates."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional

from src.pipeline.homography.detector import pixel_to_court


COCO_KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
NUM_KEYPOINTS = 17
STROKE_WINDOW_FRAMES = 30       # 1 second at 30fps
STROKE_PRE_CONTACT = 20         # frames before contact
STROKE_POST_CONTACT = 10        # frames after contact
POSE_CONFIDENCE_THRESHOLD = 0.3
POSE_QUALITY_THRESHOLD = 0.5    # mean confidence of 7 key joints


@dataclass
class PoseResult:
    frame_idx: int
    player_role: str                     # "near" or "far"
    keypoints_px: np.ndarray             # (17, 2) pixel coordinates
    keypoints_m: np.ndarray              # (17, 2) court-meter coordinates
    confidences: np.ndarray              # (17,) per-keypoint confidence
    pose_quality: float                  # mean confidence of key joints


@dataclass
class StrokePoseSequence:
    shot_id: str
    contact_frame: int
    player_role: str
    keypoints: np.ndarray                # (30, 17, 2) court-meter coords
    confidences: np.ndarray              # (30, 17)
    pose_quality: float                  # mean quality across window


class PoseExtractor:
    """
    Runs RTMPose on cropped player bounding boxes and projects keypoints
    to court-meter coordinates via the current homography.
    """

    # Key joints for pose quality scoring
    _QUALITY_JOINTS = [5, 6, 7, 8, 9, 10, 11]   # shoulders, elbows, wrists, hips

    def __init__(self, model_config: str = "rtmpose-m_8xb256-420e_coco-256x192"):
        # TODO: initialize MMPose RTMPose model
        # self._model = init_model(model_config, ...)
        self._model_config = model_config

    def process_player(
        self,
        frame: np.ndarray,
        bbox_px: np.ndarray,
        player_role: str,
        frame_idx: int,
        H: Optional[np.ndarray],
    ) -> Optional[PoseResult]:
        if H is None:
            return None

        keypoints_px, confidences = self._infer(frame, bbox_px)
        if keypoints_px is None:
            return None

        keypoints_m = self._project_to_court(keypoints_px, H)
        quality = float(np.mean(confidences[self._QUALITY_JOINTS]))

        return PoseResult(
            frame_idx=frame_idx,
            player_role=player_role,
            keypoints_px=keypoints_px,
            keypoints_m=keypoints_m,
            confidences=confidences,
            pose_quality=quality,
        )

    def _infer(
        self, frame: np.ndarray, bbox_px: np.ndarray
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        # TODO: crop bbox, run RTMPose inference, return (17,2) keypoints + (17,) confidences
        raise NotImplementedError

    @staticmethod
    def _project_to_court(
        keypoints_px: np.ndarray, H: np.ndarray
    ) -> np.ndarray:
        """Project (17,2) pixel keypoints to court-meter coordinates."""
        ones = np.ones((NUM_KEYPOINTS, 1))
        kp_h = np.hstack([keypoints_px, ones])
        court_h = (H @ kp_h.T).T
        return court_h[:, :2] / court_h[:, 2:3]

    def build_stroke_sequence(
        self,
        pose_buffer: list[PoseResult],
        contact_frame: int,
        shot_id: str,
        player_role: str,
    ) -> Optional[StrokePoseSequence]:
        """Extract STROKE_WINDOW_FRAMES poses centered around contact_frame."""
        start = contact_frame - STROKE_PRE_CONTACT
        end = contact_frame + STROKE_POST_CONTACT

        frames_in_buffer = {p.frame_idx: p for p in pose_buffer if p.player_role == player_role}
        window_poses = [frames_in_buffer.get(i) for i in range(start, end)]

        if sum(p is not None for p in window_poses) < 20:
            return None

        keypoints = np.zeros((STROKE_WINDOW_FRAMES, NUM_KEYPOINTS, 2))
        confidences = np.zeros((STROKE_WINDOW_FRAMES, NUM_KEYPOINTS))
        for t, pose in enumerate(window_poses):
            if pose is not None:
                keypoints[t] = pose.keypoints_m
                confidences[t] = pose.confidences

        quality = float(np.mean(confidences[:, self._QUALITY_JOINTS][confidences[:, self._QUALITY_JOINTS] > 0]))

        return StrokePoseSequence(
            shot_id=shot_id,
            contact_frame=contact_frame,
            player_role=player_role,
            keypoints=keypoints,
            confidences=confidences,
            pose_quality=quality,
        )
