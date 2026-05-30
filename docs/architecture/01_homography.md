# Court Homography Pipeline

## Purpose

Map pixel coordinates in a broadcast frame to real-world court coordinates
(meters, origin at center of near baseline). This is the foundation for every
downstream spatial measurement — player positions, ball landing zones, and
court openings all depend on an accurate homography.

## Court Coordinate System

```
        -4.115m     0m      +4.115m  (singles sidelines)
           │        │          │
  0.00m ───┼────────┼──────────┼─── near baseline (server side)
           │        │          │
  6.40m ───┼────────┼──────────┼─── near service line
           │        │          │
 11.89m ───┼────────┼──────────┼─── net
           │        │          │
 17.37m ───┼────────┼──────────┼─── far service line
           │        │          │
 23.77m ───┼────────┼──────────┼─── far baseline (returner side)
```

- Origin: center of the near baseline
- X axis: cross-court (positive toward deuce/right side)
- Y axis: depth (positive toward far baseline)
- Units: meters

## Algorithm

### Step 1 — Keypoint Detection via Heatmap Regression

Run a ResNet50-based heatmap regression model that outputs 14 Gaussian heatmaps,
one per court keypoint. The model is pretrained on 8,841 broadcast tennis images
from the `Gholamreza/tennis_court_keypoints_dataset` (HuggingFace), following
the TennisCourtDetector architecture (yastrebksv/TennisCourtDetector).

```
Input:  (1, 3, H, W) frame
Output: (1, 14, H/4, W/4) heatmaps  — one channel per keypoint
```

Keypoint pixel coordinates are extracted as the argmax of each heatmap channel,
with per-keypoint confidence derived from the peak heatmap value.

### Step 2 — Correspondence: Heatmap Channel → Court Coordinate

Each of the 14 output channels maps to a known court coordinate in meters.
The mapping is established from the dataset's keypoint ordering (verified against
the TennisCourtDetector label schema). Assignment uses geometric labeling —
detected points are matched to semantic roles (e.g. near-left baseline corner)
by their relative spatial position in the image, providing a fallback if the
channel ordering is ambiguous.

Primary keypoints used for H estimation (8 points):
- Near baseline: left corner, right corner
- Far baseline: left corner, right corner
- Near service line: left T, right T
- Far service line: left T, right T

Center T's (2 points) are excluded from H estimation and used only as
post-hoc verification. Net posts are ignored (frequently occluded).

### Step 3 — Confidence Filtering

Per-keypoint confidence threshold: **0.3** (peak heatmap value, normalized 0–1).
Keypoints below threshold are dropped before calling `findHomography`.

Minimum viable keypoints: **4**, with at least 2 from the near baseline.
If fewer than 4 confident keypoints are available, the frame is rejected
and the previous H is reused (or the frame is flagged as low-confidence).

### Step 4 — Homography Estimation

```python
H, mask = cv2.findHomography(
    src_pts,   # confident keypoint pixel coords (N×2, N≥4)
    dst_pts,   # corresponding court meter coords (N×2)
    cv2.RANSAC,
    ransacReprojThreshold=5.0,
)
```

With 8 keypoints RANSAC has enough redundancy to handle 1–2 detection errors.

### Step 5 — Service T Verification

Project the center T court coordinates through H into pixel space and check
whether a detected keypoint (or high heatmap activation) exists within 15px of
the predicted location. Match rate across both center T's is used as an
additional confidence signal — it doesn't block H acceptance but lowers the
confidence score.

### Step 6 — Self-Consistency Validation

Verify H by projecting known court dimensions through it and checking metric
distances:

```python
near_left  = pixel_to_court(keypoints_px[0], H)
near_right = pixel_to_court(keypoints_px[1], H)
baseline_width = np.linalg.norm(near_right - near_left)
assert abs(baseline_width - 8.23) < 0.20   # ±20cm tolerance
```

Checks performed:
- Baseline width ≈ 8.23m
- Court depth ≈ 23.77m
- Service box depth ≈ 6.40m

Frames failing any check are flagged as invalid and do not update `_last_H`.

### Step 7 — Temporal Stabilization

Re-estimate H every 30 frames (1 second at 30fps). Between keyframes, reuse
the last valid H. On a detected camera cut (frame-difference spike above
threshold), force re-estimation on the next frame regardless of interval.

```python
reprojection_error = mean(norm(project(H, src_pts[inliers]) - dst_pts[inliers]))
# Threshold: 0.15m (15cm)
```

## Camera Angle Filter

Only wide-angle baseline camera views are usable. Reject a frame if:
- Fewer than 4 keypoints are detected with confidence > 0.3
- The projected court quadrilateral has aspect ratio outside expected range
- Both near baseline corners are missing (cannot establish court depth)

## Dataset and Pretrained Weights

| Item | Detail |
|------|--------|
| Dataset | `Gholamreza/tennis_court_keypoints_dataset` (HuggingFace) |
| Size | 8,841 images, all court surfaces |
| Labels | 14 keypoints per image |
| Reference impl | `yastrebksv/TennisCourtDetector` (GitHub) |
| Pretrained weights | Available from TennisCourtDetector repo |
| Strategy | Use pretrained weights for Phase 1 broadcast footage; fine-tune if accuracy degrades on target matches |

## Implementation Notes

- Heatmap regression generalizes better across court surfaces than classical
  Hough-based detection, which requires per-surface color tuning
- TV graphics overlays (score bugs, player name strips) can suppress keypoint
  confidence in the lower third of the frame — expected, not a bug
- Segmentation → Hough is the documented fallback if heatmap accuracy proves
  insufficient for a specific footage type (see limitations section)

## Known Limitations

- Model was trained on broadcast wide-angle footage; accuracy on non-broadcast
  camera angles (recreational footage) is untested and likely lower
- Occlusion of baseline corners by players is handled by the minimum-4-keypoints
  fallback but reduces H accuracy
- Doubles sideline keypoints (if present in the 14-point schema) are not used
  for H estimation and are ignored

## Key Dependencies

- PyTorch (heatmap inference)
- OpenCV (`cv2.findHomography`)
- TennisCourtDetector weights (`yastrebksv/TennisCourtDetector`)

## Output Schema

```python
HomographyResult = {
    "frame_idx": int,
    "H": np.ndarray,               # 3×3 homography matrix
    "reprojection_error": float,
    "inlier_count": int,
    "confidence": float,           # 0–1, from reprojection error + inlier ratio + center T verification
    "camera_angle_valid": bool,
    "keypoints_px": np.ndarray,    # (N, 2) detected pixel keypoints used for H
    "keypoint_confidences": np.ndarray,  # (N,) per-keypoint heatmap peak values
}
```
