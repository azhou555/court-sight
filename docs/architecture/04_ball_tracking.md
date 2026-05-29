# Ball Tracking

## Purpose

Track the tennis ball across frames to determine trajectory, detect bounce
events, and compute landing zone coordinates in court-meter space.

## Model: TrackNet

**TrackNet** (Huang et al., 2019 / TrackNetV2) is the standard approach for
tennis ball tracking from broadcast footage. It treats ball tracking as a
heatmap regression problem over stacked input frames, which handles motion
blur better than single-frame detection.

### Architecture

- Input: 3 consecutive frames stacked → (H, W, 9) tensor
- Encoder: VGG16-style feature extractor
- Decoder: upsampling path → (H, W, 1) Gaussian heatmap
- Output: predicted ball centroid via heatmap argmax

Pre-trained weights available for tennis specifically (trained on TTNet
dataset + broadcast clips).

### Alternative: MonoTrack / WASB-SBD

If TrackNet confidence is poor on a given match, fall back to:
- **WASB-SBD** (Weak Appearance-based Sport Ball Detector) — works on
  low-resolution ball regions
- **Physics-based trajectory fitting** — fit a parabola to the last N
  detected positions and extrapolate

## Processing Pipeline

```
Frame stack [t-1, t, t+1]
    │
    ├── TrackNet inference → ball heatmap
    │
    ├── Heatmap → pixel centroid (argmax + sub-pixel refinement)
    │
    ├── Kalman filter smoothing (handles missed detections)
    │
    ├── Bounce detection (velocity sign change in Y)
    │
    └── Homography projection → court-meter coordinates
```

## Bounce Detection

A bounce occurs when the vertical velocity component (in 2D projected space)
changes from negative (descending) to positive (ascending). Detection criteria:

```python
def detect_bounce(positions_m, frame_indices):
    """
    positions_m: (N, 2) ball positions in court meters
    Returns: list of (frame_idx, court_position) for detected bounces
    """
    dy = np.diff(positions_m[:, 1])   # depth change per frame
    # Sign change from negative to positive (ball going toward far baseline)
    bounce_candidates = np.where((dy[:-1] < -0.1) & (dy[1:] > 0.1))[0] + 1
    ...
```

Minimum depth velocity threshold (0.1m/frame) avoids false positives from
tracking noise.

## Landing Zone Classification

Divide the opponent's court into 9 zones:

```
┌───────────┬───────────┬───────────┐
│  T  deep  │ mid deep  │ wide deep │  ← baseline area
├───────────┼───────────┼───────────┤
│  T  mid   │ mid mid   │ wide mid  │  ← service box depth
├───────────┼───────────┼───────────┤
│  T  short │ mid short │ wide short│  ← net area
└───────────┴───────────┴───────────┘
     T side      center    wide side
```

Where T side = toward the center service line and wide side = toward the
singles sideline. Relative to the *striker* (so "T" is always the safer,
higher-percentage target).

## Serve Detection

Serves are a special case:
- Ball appears from below the top of the frame (toss)
- Travels in a high arc to the service box
- Detected via trajectory shape (upward then downward arc from near side)

Serve vs. return disambiguation uses player position: server is stationary
behind baseline at start of point.

## Output Schema

```python
BallTrackResult = {
    "frame_idx": int,
    "position_px": [x, y],
    "position_m": [x, y],          # court meters
    "confidence": float,
    "is_bounce": bool,
    "bounce_zone": str | None,      # e.g. "T_deep", "wide_mid"
    "trajectory_segment": int,      # which rally segment (0 = serve)
}
```

## Known Challenges

- **Ball occlusion by players**: during contact frames, ball is hidden.
  Interpolate across 2–3 frames around contact.
- **Broadcast overlays**: score bugs at screen edges — mask before inference.
- **Slow-motion replays**: TrackNet expects real-time frame rate. Detect
  replay segments (frame duplication pattern) and skip them.

## Key Dependencies

- TrackNetV2 (PyTorch implementation, available on GitHub)
- NumPy, SciPy (Kalman filter, signal processing)
- OpenCV (frame preprocessing)
