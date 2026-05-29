# Player Detection and Tracking

## Purpose

Locate both players in every frame, assign consistent identities across the
match, and compute position and velocity in court-meter coordinates.

## Detection

Use a fine-tuned **YOLOv8** person detector. The tennis-specific context
means:
- Exactly 2 players in frame (or 1 if only one side of court is shown)
- Players are distinguishable by court half (near vs. far baseline)
- Ball boys/girls are present — suppress detections outside the court area
  using the homography mask from the homography pipeline

Detection confidence threshold: 0.5. Non-maximum suppression IoU threshold: 0.4.

## Tracking

Use **ByteTrack** for multi-object tracking. ByteTrack handles low-confidence
detections (partially occluded players) better than SORT-family trackers.

Track assignment to "near player" / "far player" roles:
- Project player centroid through H to court coordinates
- Assign role based on Y coordinate (< 11.88m = near side, > 11.88m = far side)
- Re-assign roles each frame to handle camera flip edits

## Position and Velocity

Player position = court-coordinate centroid of bounding box bottom edge
(feet contact point is more stable than box center, which shifts with
arm/racket position).

```python
def pixel_to_court(pixel_pt, H):
    pt = np.array([*pixel_pt, 1.0], dtype=np.float64)
    court_pt = H @ pt
    return court_pt[:2] / court_pt[2]  # perspective divide
```

Velocity = finite difference of position over 3-frame window, smoothed with
a Savitzky-Golay filter (window=7, poly=2) to reduce jitter.

## Player Identity

Within a match, maintain two persistent track identities. On camera cuts:
- Detect cut using frame-difference spike or homography re-initialization flag
- Re-match tracks to roles using court position after cut

## Output Schema

```python
PlayerTrackResult = {
    "frame_idx": int,
    "near_player": {
        "track_id": int,
        "bbox_px": [x1, y1, x2, y2],
        "position_m": [x, y],           # court meters
        "velocity_ms": [vx, vy],         # m/s
        "confidence": float,
    },
    "far_player": { ... },               # same structure
}
```

## Known Challenges

- **Occlusion by net**: near-baseline rallies occasionally have players
  partially behind the net. ByteTrack's re-ID handles short occlusions.
- **Ball boy interference**: suppress detections in fixed boundary zones
  (corners behind baselines).
- **Doubles matches**: out of scope for V1 — skip doubles footage in dataset
  collection.

## Key Dependencies

- Ultralytics YOLOv8 (`ultralytics` package)
- ByteTrack (standalone or via BoxMOT)
- NumPy + SciPy (Savitzky-Golay smoothing)
