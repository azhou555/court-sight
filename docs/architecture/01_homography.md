# Court Homography Pipeline

## Purpose

Map pixel coordinates in a broadcast frame to real-world court coordinates
(meters, origin at center of baseline). This is the foundation for every
downstream spatial measurement — player positions, ball landing zones, and
court openings all depend on an accurate homography.

## Court Coordinate System

```
        0m          11.89m (singles sideline)
        │                │
  0m ───┼────────────────┼─── baseline (server side)
        │                │
 11.89m ─── service line │
        │                │
 23.77m ───┼────────────────┼─── baseline (returner side)
```

- Origin: center of the near baseline
- X axis: cross-court (positive toward deuce side)
- Y axis: depth (positive toward far baseline)
- Units: meters

## Algorithm

### Step 1 — Court Line Detection

Use a combination of:
- **Hough line transform** on a court-color-masked binary edge image
- **Semantic segmentation** (fine-tuned DeepLabV3 or lightweight equivalent)
  to classify court surface vs. lines vs. background

Court mask: filter to court-surface HSV range, then detect white/yellow line
pixels within that mask.

### Step 2 — Keypoint Extraction

From detected lines, extract the 4–6 most reliable court keypoints:
- Baseline corners (2 pts)
- Service T intersections (2 pts)
- Net posts (2 pts, optional — often occluded)

Robustness note: use RANSAC to reject spurious line detections before
computing intersections.

### Step 3 — Homography Estimation

```python
# H maps image points → court points
H, mask = cv2.findHomography(
    src_pts,   # detected pixel keypoints (N×2)
    dst_pts,   # corresponding court meter coordinates (N×2)
    cv2.RANSAC,
    ransacReprojThreshold=5.0,
)
```

Minimum 4 point correspondences required. Using 6+ gives robustness via
RANSAC inlier selection.

### Step 4 — Temporal Stabilization

Re-estimate H every 30 frames (1 second at 30fps). Between keyframes,
interpolate using affine blend. Flag frames where reprojection error exceeds
threshold — these likely have a camera cut and need re-initialization.

```python
reprojection_error = mean(
    norm(project(H, src_pts[inliers]) - dst_pts[inliers])
)
# Threshold: 0.15m (15cm) — tolerable court localization error
```

## Camera Angle Filter

Only wide-angle baseline camera views are usable. Filter criteria:
- Both baseline corners visible
- At least one full service box visible
- Estimated court coverage > 60% of frame width
- Aspect ratio check on projected court (catches severe off-angle shots)

Reject zoomed-in rally shots, player close-ups, and overhead views.

## Implementation Notes

- Baseline keypoints are more reliably detected than net posts (net occludes
  the court surface beneath it)
- Clay courts have better contrast for line detection than hard courts under
  certain lighting conditions
- TV graphics overlays (score bugs, player names) can interfere — mask the
  standard broadcast overlay region before line detection

## Key Dependencies

- OpenCV (`cv2.findHomography`, `cv2.HoughLinesP`)
- Optional: MMSegmentation for semantic court segmentation

## Output Schema

```python
HomographyResult = {
    "frame_idx": int,
    "H": np.ndarray,          # 3×3 homography matrix
    "reprojection_error": float,
    "inlier_count": int,
    "confidence": float,       # 0–1, derived from reprojection error + inlier ratio
    "camera_angle_valid": bool,
}
```
