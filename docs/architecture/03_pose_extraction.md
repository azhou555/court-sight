# Pose Extraction

## Purpose

Extract 17-keypoint body pose for each player at each frame and project
those keypoints into court-meter coordinates. Pose is the primary biomechanical
input to both the shot type classifier and the execution probability model.

## Model: YOLO26-pose (V1)

**Implementation uses `yolo26m-pose.pt`** (Ultralytics) instead of the
originally planned RTMPose, to avoid the mmpose/mmcv dependency stack.
Same COCO-17 keypoint schema; ~3-4 AP lower than RTMPose-m but sufficient
for shot classification. Upgrade to RTMPose-m if joint accuracy becomes a
bottleneck.

| Model         | COCO AP | Notes                              |
|---------------|---------|------------------------------------|
| YOLO26m-pose  | ~72     | Already in stack, no new deps ✓    |
| RTMPose-m     | 75.8    | Best tradeoff — future upgrade     |
| ViTPose-B     | 77.1    | More accurate, heavier             |

## Keypoint Schema (COCO 17-point)

```
0:  nose
1:  left_eye       2:  right_eye
3:  left_ear       4:  right_ear
5:  left_shoulder  6:  right_shoulder
7:  left_elbow     8:  right_elbow
9:  left_wrist    10:  right_wrist
11: left_hip      12:  right_hip
13: left_knee     14:  right_knee
15: left_ankle    16:  right_ankle
```

## Processing Pipeline

```
Detected player bounding box
    │
    ├── Crop + resize to 256×192 (standard RTMPose input)
    │
    ├── RTMPose inference → 17 keypoints in crop coordinates
    │
    └── Unproject to full-frame pixels
            │
            └── Apply homography H → court-meter coordinates
```

### Homography Projection for Keypoints

```python
def pose_to_court(keypoints_px, H):
    """
    keypoints_px: (17, 2) array of pixel coordinates
    Returns: (17, 2) array of court-meter coordinates
    """
    ones = np.ones((17, 1))
    kp_h = np.hstack([keypoints_px, ones])          # (17, 3)
    court_h = (H @ kp_h.T).T                        # (17, 3)
    return court_h[:, :2] / court_h[:, 2:3]          # perspective divide
```

Note: the homography is calibrated to map **ground-plane** points. Upper-body
keypoints (head, shoulders, wrists) are NOT on the ground plane, so their
projected court coordinates are approximate and useful only for relative limb
angles, not absolute court positions. Foot keypoints (ankles) are treated as
the player's ground-truth court position.

## Confidence Filtering

Each RTMPose keypoint carries a confidence score (0–1).

- **Per-keypoint threshold**: 0.3 — below this, treat as missing
- **Pose-level confidence**: mean confidence of the 7 most important joints
  (shoulders, elbows, wrists, hips). Below 0.5 → flag frame as low-quality.
- Low-confidence poses are included in the dataset but are weighted down
  during model training.

## Stroke Window Extraction

For the shot type classifier, we need a temporal window of poses centered
on ball contact.

- Window size: 30 frames (1 second at 30fps)
- Contact frame: estimated from ball tracking (minimum ball velocity
  after predicted bounce near player)
- Window: [contact − 20, contact + 10] frames

```python
StrokePoseSequence = {
    "shot_id": str,
    "contact_frame": int,
    "player_role": "near" | "far",
    "keypoints": np.ndarray,      # (30, 17, 2) court-meter coords
    "confidences": np.ndarray,    # (30, 17) per-keypoint scores
    "pose_quality": float,         # mean confidence across window
}
```

## Key Dependencies

- MMPose (`mmpretrain`, `mmpose`) — RTMPose weights available via model zoo
- OpenCV — crop/resize preprocessing
- NumPy
