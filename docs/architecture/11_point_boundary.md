# Point Boundary Detection and Video Editor

## Purpose

Detect the start and end of each tennis point in a broadcast match video, then
export a trimmed version with between-point dead time removed. This is the first
shippable feature: useful standalone even before the full feedback engine is
trained.

## Approach: Rule-Based State Machine

Academic temporal action localization models (E2E-Spot, T-DEED) are overkill
here — we already emit ball tracking confidence, player velocities, and
homography validity per frame. A deterministic state machine over those signals
is sufficient and interpretable.

## State Machine

```
           ball appears         ball appears
           after dead period    within cooldown
DEAD ──────────────────► LIVE ◄──────────────── COOLDOWN
 ▲                          │                       │
 │    extended dead          │ ball lost             │ ball lost
 └───────────────────────────┴───────────────────────┘
       (cooldown expires + both players slow)
```

**States:**
- `DEAD` — between points; no ball, players walking/stationary
- `LIVE` — point in progress; ball tracked, rally active
- `COOLDOWN` — ball just lost; brief window before declaring dead ball

**Transition rules:**

| From | To | Condition |
|------|-----|-----------|
| `DEAD` | `LIVE` | ball visible after ≥ `dead_min_frames` of no ball |
| `LIVE` | `COOLDOWN` | ball tracking lost |
| `COOLDOWN` | `LIVE` | ball reappears within `cooldown_frames` |
| `COOLDOWN` | `DEAD` | cooldown expires AND `both_players_slow` |

**Signal definitions (per frame):**
- `ball_visible`: `ball.confidence > threshold AND ball.position_px is not None`
- `both_players_slow`: max velocity across both players < 1.0 m/s
- `dead_min_frames`: 60 frames (2 sec) — minimum dead period before next serve
- `cooldown_frames`: 15 frames (0.5 sec) — occlusion tolerance during rallies

## Point Segment Output

```python
@dataclass
class PointSegment:
    point_id: int
    start_frame: int    # first live frame (serve toss area)
    end_frame: int      # last live frame (ball lost / out)
    start_sec: float
    end_sec: float
```

Each segment is padded with a configurable pre-roll (default 60 frames) and
post-roll (default 45 frames) to include serve setup and reaction.

## Video Editor

Takes a video path + list of `PointSegment` objects and writes a trimmed output
video using **ffmpeg stream copy** (no re-encode — fast, lossless quality).

```
Input:  match.mp4 + [PointSegment, ...]
Step 1: For each segment, extract with -ss/-to -c copy → temp_0000.mp4 ...
Step 2: ffmpeg concat demuxer → trimmed_match.mp4
Output: trimmed_match.mp4 (only rally time, dead time stripped)
```

Stream copy preserves exact quality and runs ~50× faster than re-encode. Minor
GOP alignment artifacts at cut points are acceptable for a data pipeline; if
exact frame accuracy is required for UI display, add `-force_key_frames` on
encode.

## Saved Metadata

The boundary detector writes `match_boundaries.json` alongside the trimmed
video:

```json
{
  "source_video": "match.mp4",
  "fps": 30.0,
  "total_points": 142,
  "points": [
    {"point_id": 0, "start_frame": 420, "end_frame": 1050,
     "start_sec": 14.0, "end_sec": 35.0},
    ...
  ]
}
```

This file can be used by downstream pipeline stages (MCP alignment, shot
annotation) without re-running the boundary detector.

## Known Limitations

- **Slow-motion replays**: frame duplication inflates frame counts and fools
  ball tracker. Detect via frame difference drop below threshold and skip replay
  segments. Not implemented in V1 — flag as known gap.
- **Tight service lets / net cords**: ball may briefly disappear on net contact,
  triggering cooldown correctly, but if the let is replayed immediately the state
  machine may emit a spurious LIVE transition. Post-hoc merging of very short
  segments (< 10 frames) handles this.
- **Crowd/player obscuring serve toss**: first serve sometimes not detected until
  ball is airborne. The pre-roll buffer compensates.

## Key Dependencies

- `ffmpeg-python>=0.2` (video trimming and concatenation)
- Existing pipeline outputs: `BallTrackResult`, `PlayerTrackResult`
