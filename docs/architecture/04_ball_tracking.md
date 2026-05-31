# Ball Tracking

## Purpose

Track the tennis ball across frames to determine trajectory, detect bounce
events, and compute landing zone coordinates in court-meter space.

## Model: TrackNet

**TrackNet** (yastrebksv PyTorch implementation, based on Huang et al. 2019)
is the standard approach for tennis ball tracking from broadcast footage.
It treats ball tracking as a per-pixel classification problem over stacked
consecutive frames, which handles motion blur better than single-frame
detection.

Note: TrackNetV4 (ICASSP 2025) improves on this architecture with motion
attention maps but has no public weights yet. Update to V4 when weights become
available.

### Architecture

Same VGG-style encoder-decoder as CourtKeypointNet (conv1–conv18, pool1–3,
ups1–3) with two differences:

```
Input:  (B, 9, 360, 640)   — 3 consecutive frames × 3 channels, stacked
Output: (B, 256, H×W)      — 256 intensity classes per spatial pixel,
                              flattened: (B, 256, 360×640)
```

Training uses **CrossEntropyLoss**: each spatial pixel is a 256-class
classification where the ground-truth class is the pixel's Gaussian heatmap
value (0–255). `argmax(dim=1)` during inference gives a predicted intensity map
that is thresholded and processed with HoughCircles to locate the ball.

### Ball Localization

```
output: (B, 256, H×W)
  → argmax(dim=1)        → (B, H×W)        predicted intensity index
  → reshape(360, 640)    → (360, 640)       intensity map
  → scale × 255 → uint8 → threshold(127)   binary mask
  → HoughCircles(minR=2, maxR=7)           → (x, y) at model resolution
  → × (orig_w/640, orig_h/360)             → original frame pixel coords
```

### Gap Handling

When HoughCircles finds no ball (missed detection):
- **≤ 4 consecutive misses**: extrapolate using constant velocity from last
  2 known positions (online velocity-based prediction)
- **> 4 consecutive misses**: return None; treat next detection as a new
  trajectory segment

### Pretrained Weights

| Item | Detail |
|------|--------|
| Repo | `yastrebksv/TrackNet` (GitHub) |
| Weights | Google Drive `1XEYZ4myUN7QT-NeBYJI0xteLsvs-ZAOl` (download via gdown) |
| Dataset | 10 broadcast match clips, 19,835 labeled frames, 1280×720 30fps |
| Fine-tune strategy | Fine-tune on target match footage when tracking confidence degrades |

## Processing Pipeline

```
Frame buffer [t-2, t-1, t]
    │
    ├── Stack → (9, 360, 640) input tensor
    │
    ├── TrackNet inference → (256, H×W) class scores
    │
    ├── Argmax → intensity map → HoughCircles → pixel centroid
    │
    ├── Gap interpolation (velocity extrapolation, ≤4 frames)
    │
    ├── Bounce detection (y-velocity sign change in court space)
    │
    └── Homography projection → court-meter coordinates
```

## Bounce Detection

A bounce occurs when the court-space Y velocity changes from negative
(descending) to positive (ascending) across 3 consecutive positions:

```python
dy = np.diff([p[1] for p in recent_3_positions])
is_bounce = dy[0] < -threshold and dy[1] > threshold   # threshold = 0.1 m/frame
```

## Landing Zone Classification

9 zones on the opponent's court, relative to the *striker's* perspective
(T side = toward center service line, W = toward singles sideline):

```
┌────────────┬────────────┬────────────┐
│  T_deep    │  C_deep    │  W_deep    │  y > 17.37m from near baseline
├────────────┼────────────┼────────────┤
│  T_mid     │  C_mid     │  W_mid     │  6.40m < y ≤ 17.37m
├────────────┼────────────┼────────────┤
│  T_short   │  C_short   │  W_short   │  y ≤ 6.40m (net area)
└────────────┴────────────┴────────────┘
   x < −1.0m   |x| ≤ 1.0m   x > 1.0m
```

Depth measured from the far baseline (y = 23.77m) into the opponent's court.

## Output Schema

```python
@dataclass
class BallTrackResult:
    frame_idx: int
    position_px: Optional[np.ndarray]   # [x, y] original frame pixels
    position_m: Optional[np.ndarray]    # [x, y] court meters
    confidence: float                    # 0–1, HoughCircles peak intensity
    is_bounce: bool
    bounce_zone: Optional[str]           # e.g. "T_deep", "W_mid"
    trajectory_segment: int              # increments on each bounce
    interpolated: bool                   # True if position is extrapolated
```

## Known Challenges

- **Ball occlusion during contact**: ball hidden for 1–3 frames. Gap
  interpolation handles these automatically.
- **Broadcast overlays**: score bugs suppress detections in lower-third.
  Not masked explicitly — HoughCircles will simply miss the ball in those frames.
- **Slow-motion replays**: TrackNet expects real-time frame rate. Detect
  replay segments (frame duplication pattern) and skip them.
- **Serve trajectory**: high-arc toss + strike creates unusual motion profile.
  Bounce detection handles the first bounce normally; the toss itself is not
  tracked (ball exits/enters frame).

## Key Dependencies

- PyTorch (TrackNet inference and fine-tuning)
- OpenCV (`cv2.HoughCircles`)
- `gdown` (pretrained weight download, dev only)
