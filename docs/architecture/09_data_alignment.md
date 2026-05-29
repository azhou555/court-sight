# Match Charting Project Data Alignment

## Purpose

The Match Charting Project (MCP) provides shot-level labels (type, direction,
outcome) for thousands of professional matches. The pipeline aligns these
labels to the CV-extracted positional data (player positions, pose, ball
trajectory) to produce fully annotated shot records.

## Match Charting Project Format

MCP stores data as CSV files with one row per shot:

```
match_id, set, game, point, shot_in_rally, server, shot_type, direction,
depth, outcome, ...
```

Shot type codes (subset):
- `f` = forehand, `b` = backhand, `s` = slice, `v` = volley
- `r` = serve (first), `q` = serve (second)
- `o` = overhead

Direction codes:
- `1` = down the line, `2` = crosscourt, `3` = middle, `4` = inside-in, etc.

## Alignment Strategy

The core challenge: MCP timestamps are at the point level (set/game/point
counts), while video is at the frame level. We need to map each shot row in
MCP to a specific frame range in the video.

### Step 1 — Point Boundary Detection

Detect point start and end frames from the video:
- **Point start**: serve detected (ball toss from behind baseline, player
  stationary)
- **Point end**: dead-ball state (players stationary, ball out of play, visible
  score update, or crowd reaction pattern)

This produces a list: `[(start_frame, end_frame, point_score)]`

### Step 2 — Score Alignment

The video's detected score sequence is aligned to the MCP score sequence using
dynamic time warping (DTW) or simple edit-distance alignment. Score changes are
the alignment anchors.

This handles:
- Broadcast replays (which duplicate point footage)
- Commentary segments between games
- Warm-up footage at the start of recordings

### Step 3 — Shot-level Alignment within Points

Within a matched point, align individual shots using:
1. MCP rally length (shot count) vs. detected shot count from ball tracking
2. Serve shot is always shot 1 — align the serve detection as shot 1
3. Subsequent shots: assign each ball-tracking-detected contact event to the
   next MCP shot row in sequence

Edge cases:
- **Extra shots detected**: ball-tracking false positives. If CV detects more
  shots than MCP records, drop the lowest-confidence detections.
- **Missing shots**: a shot was missed by ball tracking. Interpolate using
  player contact frame (minimum distance between player position and ball
  trajectory).

### Step 4 — Quality Filtering

An aligned shot is marked high-confidence if:
- The point was matched to MCP with DTW distance < threshold
- The shot-level timing is consistent (inter-shot intervals plausible for
  rally pace)
- Pose confidence > 0.5 for the striker at contact frame
- Ball tracking confidence > 0.4 at contact frame

Low-confidence shots are retained but down-weighted during training.

## Output: Unified Shot Record

```python
ShotRecord = {
    # From Match Charting Project
    "match_id": str,
    "point_id": str,               # set-game-point-shot
    "shot_in_rally": int,
    "shot_type_mcp": str,
    "direction_mcp": str,
    "depth_mcp": str,
    "point_outcome": int,          # 1 = point won by striker, 0 = lost

    # From CV pipeline
    "contact_frame": int,
    "striker_role": str,           # "near" or "far"
    "striker_position_m": [float, float],
    "striker_velocity_ms": [float, float],
    "striker_pose": list,          # (17, 2) keypoints in court meters
    "striker_pose_confidence": float,
    "opponent_position_m": [float, float],
    "opponent_velocity_ms": [float, float],
    "ball_landing_zone": str,
    "ball_contact_position_m": [float, float],

    # Derived
    "rally_context": list,         # last 3 ShotRecord summaries
    "alignment_confidence": float,
    "pose_quality": float,
}
```

## Key Dependencies

- Pandas (MCP CSV loading, join operations)
- SciPy (DTW via `scipy.spatial.distance.cdist` + custom DTW)
- NumPy
- MCP dataset: https://github.com/JeffSackmann/tennis_MatchChartingProject
