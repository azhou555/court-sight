# Player Detection and Tracking

## Purpose

Locate both players in every frame, assign consistent identities across the
match, and compute position and velocity in court-meter coordinates.

## Detection

Use **YOLO26m** for person detection. YOLO26 is the current default Ultralytics
model (as of v8.4), featuring NMS-free end-to-end inference and the MuSGD
optimizer — comparable accuracy to YOLO11m (~51.5% mAP COCO) with faster CPU
inference.

- Class filter: person only (class 0)
- Confidence threshold: 0.5
- Court mask: suppress detections where the foot point (bbox bottom-center)
  projects outside the court boundary + 3m margin (catches ball boys behind
  the baseline and line judges at the sides)
- Pretrained COCO weights used initially; fine-tune on broadcast tennis footage
  if false-positive rate on ball boys/line judges is unacceptable

## Tracking

Use **BoT-SORT** (via BoxMOT) in motion-only mode (`with_reid=False`).

BoT-SORT extends ByteTrack with **global motion compensation (GMC)** using ECC
(Enhanced Correlation Coefficient) optical flow, which explicitly corrects for
broadcast camera pan and tilt during rallies. This is the key improvement over
ByteTrack for broadcast footage.

ReID features are omitted — with exactly 2 players distinguishable by court
half, the added accuracy does not justify the overhead of downloading and
running a ReID model.

Configuration:
```python
BotSort(
    with_reid=False,
    cmc_method='ecc',      # camera motion compensation via optical flow
    frame_rate=30,
    track_high_thresh=0.5,
    new_track_thresh=0.6,
    track_buffer=30,       # frames to keep a lost track alive
)
```

Track assignment to "near player" / "far player" roles:
- Project player foot point through H to court coordinates
- Assign role based on Y coordinate (< 11.885m = near side, ≥ 11.885m = far side)
- Re-assign roles each frame to handle camera-flip edits

## Position and Velocity

**Position**: court-coordinate projection of the bbox bottom-center (foot
contact point). More stable than box center, which shifts with arm/racket pose.

**Velocity**: Savitzky-Golay filter (window=7, poly=2) over the rolling position
history, differentiated at the last point. Falls back to 3-frame finite
difference until 7 frames of history are available.

```python
from scipy.signal import savgol_filter

deriv = savgol_filter(positions, window=7, polyorder=2, deriv=1, delta=dt, axis=0)
velocity_ms = deriv[-1]   # m/s at most recent frame
```

## Player Identity

Two persistent track IDs per match. On camera cuts (detected via homography
re-initialization flag from the homography pipeline):
- Clear position history for both tracks
- Re-match track IDs to near/far roles using court position on the next frame

## Output Schema

```python
@dataclass
class PlayerState:
    track_id: int
    bbox_px: np.ndarray        # [x1, y1, x2, y2] in original frame pixels
    position_m: np.ndarray     # [x, y] court meters (foot contact point)
    velocity_ms: np.ndarray    # [vx, vy] m/s
    confidence: float

@dataclass
class PlayerTrackResult:
    frame_idx: int
    near_player: Optional[PlayerState]
    far_player: Optional[PlayerState]
```

## Known Challenges

- **Occlusion by net**: ByteTrack / BoT-SORT re-ID handles short occlusions
  via the track buffer (30 frames).
- **Ball boy suppression**: Court mask handles most cases; edge cases (ball boy
  running onto court) will produce a spurious detection but role assignment
  limits it to whichever side has fewer than 2 players.
- **Doubles footage**: Out of scope for V1 — skip doubles footage during
  dataset collection.
- **Frame rate variation**: Some broadcast clips run at 25fps. Pass the actual
  frame rate to `BotSort(frame_rate=fps)` and the velocity `delta` parameter.

## Key Dependencies

- `ultralytics>=8.4` (YOLO26m)
- `boxmot>=10.0` (BotSort)
- `scipy` (Savitzky-Golay filter)
