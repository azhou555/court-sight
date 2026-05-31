# Court Homography Pipeline

## Purpose

Map pixel coordinates in a broadcast frame to real-world court coordinates
(meters, origin at center of near baseline). This is the foundation for every
downstream spatial measurement — player positions, ball landing zones, and
court openings all depend on an accurate homography.

## Court Coordinate System

```
        -5.485m  -4.115m     0m      +4.115m  +5.485m
           │        │        │          │         │
  0.00m ───┼────────┼─────────┼──────────┼─────────┼─── near baseline
           │        │        │          │         │
  6.40m ───┼────────┼─────────┼──────────┼─────────┼─── near service line
           │        │        │          │         │
 11.89m ───┼────────┼─────────┼──────────┼─────────┼─── net
           │        │        │          │         │
 17.37m ───┼────────┼─────────┼──────────┼─────────┼─── far service line
           │        │        │          │         │
 23.77m ───┼────────┼─────────┼──────────┼─────────┼─── far baseline
        doubles  singles              singles  doubles
```

- Origin: center of the near baseline
- X axis: cross-court (positive toward deuce/right side when facing net)
- Y axis: depth (positive toward far baseline)
- Units: meters

## Model Architecture

The keypoint detector is a **TrackNet-style encoder-decoder** (not ResNet-based).
The design mirrors the architecture from `yastrebksv/TennisCourtDetector`.

```
Input: (B, 3, 360, 640)   ← frames resized to half of 720p

Encoder
  conv1–2  →  (B, 64,  360, 640)
  pool1    →  (B, 64,  180, 320)
  conv3–4  →  (B, 128, 180, 320)
  pool2    →  (B, 128,  90, 160)
  conv5–7  →  (B, 256,  90, 160)
  pool3    →  (B, 256,  45,  80)
  conv8–10 →  (B, 512,  45,  80)

Decoder
  ups1×2   →  (B, 512,  90, 160)
  conv11–13→  (B, 256,  90, 160)
  ups2×2   →  (B, 256, 180, 320)
  conv14–15→  (B, 128, 180, 320)
  ups3×2   →  (B, 128, 360, 640)
  conv16–17→  (B,  64, 360, 640)
  conv18   →  (B,  15, 360, 640)   ← 14 keypoints + 1 center court

Output: (B, 15, 360, 640)
```

- Each of channels 0–13 is a Gaussian heatmap for one court keypoint.
- Channel 14 is the court center (diagonal intersection), used only during
  training for convergence stability and ignored at inference.
- No sigmoid/softmax on output; raw logits, trained with MSE against
  Gaussian target heatmaps.

## Keypoint Channel → Court Coordinate Mapping

Channel ordering follows `CourtReference.key_points` from the reference repo.
The visualization script (`scripts/verify_keypoint_channels.py`) must be run
after any weight update to confirm this mapping holds.

| Channel | Semantic label | Court coords (x, y) m |
|---------|---------------|----------------------|
| 0 | far baseline, doubles left corner | (−5.485, 23.77) |
| 1 | far baseline, doubles right corner | (+5.485, 23.77) |
| 2 | near baseline, doubles left corner | (−5.485,  0.00) |
| 3 | near baseline, doubles right corner | (+5.485,  0.00) |
| 4 | far baseline, singles left corner | (−4.115, 23.77) |
| 5 | far baseline, singles right corner | (+4.115, 23.77) |
| 6 | near baseline, singles left corner | (−4.115,  0.00) |
| 7 | near baseline, singles right corner | (+4.115,  0.00) |
| 8 | far service line, left T | (−4.115, 17.37) |
| 9 | far service line, right T | (+4.115, 17.37) |
| 10 | near service line, left T | (−4.115,  6.40) |
| 11 | near service line, right T | (+4.115,  6.40) |
| 12 | far center T | ( 0.000, 17.37) |
| 13 | near center T | ( 0.000,  6.40) |
| 14 | court center (training only) | — |

## Algorithm

### Step 1 — Preprocessing

Resize frame to 640×360, normalize to [0, 1], convert BGR→RGB.

### Step 2 — Heatmap Inference

Run the CourtKeypointNet forward pass to get `(1, 15, 360, 640)` heatmaps.
Channel 14 (center) is discarded. For each of channels 0–13, the keypoint
pixel coordinate is the argmax of the heatmap channel, and the per-keypoint
confidence is the peak heatmap value (normalized 0–1).

### Step 3 — Confidence Filtering

Per-keypoint confidence threshold: **0.3**.
Keypoints below threshold are dropped before calling `findHomography`.

Minimum viable keypoints: **4**, with at least 2 from the near baseline
(channels 2, 3, 6, 7). If fewer than 4 confident keypoints are available,
the frame is rejected and the previous H is reused.

### Step 4 — Homography Estimation

```python
H, mask = cv2.findHomography(
    src_pts,   # confident keypoint pixel coords (N×2, N≥4)
    dst_pts,   # corresponding court meter coords (N×2)
    cv2.RANSAC,
    ransacReprojThreshold=5.0,
)
```

With up to 14 keypoints, RANSAC has strong redundancy against 1–3 detection
errors.

### Step 5 — Service T Verification

Project the near/far center T court coordinates (0.0, 6.40) and (0.0, 17.37)
through H into pixel space and check whether a detected keypoint exists within
15px. Match rate across both T's contributes to the confidence score but does
not block H acceptance.

### Step 6 — Self-Consistency Validation

Verify H by projecting known court dimensions and checking metric distances:

```python
# Baseline width check
near_left  = pixel_to_court(kp_px[6], H)   # channel 6: near baseline singles left
near_right = pixel_to_court(kp_px[7], H)   # channel 7: near baseline singles right
assert abs(np.linalg.norm(near_right - near_left) - 8.23) < 0.20  # ±20cm

# Court depth check
far_left   = pixel_to_court(kp_px[4], H)   # channel 4: far baseline singles left
assert abs(np.linalg.norm(far_left - near_left) - 23.77) < 0.30
```

Checks:
- Singles baseline width ≈ 8.23m
- Court depth ≈ 23.77m
- Service box depth ≈ 6.40m (near service line y)

Frames failing any check are flagged invalid and do not update `_last_H`.

### Step 7 — Temporal Stabilization

Re-estimate H every 30 frames (1 second at 30fps). Between keyframes, reuse
the last valid H. On a detected camera cut (frame-difference spike above
threshold), force re-estimation on the next frame.

```python
reprojection_error = mean(norm(project(H, src_pts[inliers]) - dst_pts[inliers]))
# Threshold: 0.15m (15cm)
```

## Camera Angle Filter

Only wide-angle baseline camera views are usable. Reject a frame if:
- Fewer than 4 keypoints are detected with confidence > 0.3
- Near baseline corners (channels 2, 3, 6, 7) are all missing
- The projected court quadrilateral has aspect ratio outside expected range

## Dataset and Weights

| Item | Detail |
|------|--------|
| Dataset | `Gholamreza/tennis_court_keypoints_dataset` (HuggingFace) |
| Size | 8,841 images, all court surfaces, 1280×720 |
| Labels | 14 keypoints per image |
| Reference impl | `yastrebksv/TennisCourtDetector` (GitHub) |
| Pretrained weights | Google Drive `1f-Co64ehgq4uddcQm1aFBDtbnyZhQvgG` (download via gdown) |
| Fine-tune strategy | Load pretrained weights as starting point; fine-tune on target match footage whenever accuracy degrades on a new surface or camera angle |

## Implementation Notes

- The model's VGG-style encoder-decoder generalizes across court surfaces
  better than Hough-based detection, which requires per-surface color tuning.
- TV graphics (score bugs, name strips) suppress confidence in the lower third
  — expected, not a bug.
- Doubles sideline corners (channels 0–3) are included in H estimation when
  visible; they are dropped by confidence filtering when the broadcast crops
  them out.
- Segmentation → Hough is the documented fallback if heatmap accuracy proves
  insufficient for a specific footage type.

## Known Limitations

- Model pretrained on broadcast wide-angle footage; accuracy on recreational
  camera angles is untested and likely lower.
- Player occlusion of baseline corners is handled by the minimum-4-keypoints
  fallback but reduces H accuracy.
- At 360×640 inference resolution, keypoint localization precision is ≈2px,
  which projects to ≈5–10cm court error at typical broadcast framing.

## Key Dependencies

- PyTorch (heatmap inference and fine-tuning)
- OpenCV (`cv2.findHomography`)
- `huggingface_hub` + `datasets` (dataset access, dev only)
- `gdown` (pretrained weight download, dev only)

## Output Schema

```python
@dataclass
class HomographyResult:
    frame_idx: int
    H: Optional[np.ndarray]          # 3×3 homography matrix
    reprojection_error: float
    inlier_count: int
    confidence: float                 # 0–1
    camera_angle_valid: bool
    keypoints_px: np.ndarray          # (N, 2) pixel coords used for H
    keypoint_confidences: np.ndarray  # (N,) heatmap peak values
```
