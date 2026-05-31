# Match Charting Project Data Alignment

## Purpose

The Match Charting Project (MCP) provides shot-level labels (type, direction,
outcome) for thousands of professional matches. The pipeline aligns these
labels to the CV-extracted positional data (player positions, ball trajectory)
to produce fully annotated shot records for downstream model training.

## MCP CSV Format

MCP stores data as one CSV row per **point** (not per shot). The key columns:

```
match_id, Pt, Set1, Set2, Gm1, Gm2, Pts, Svr, 1st, 2nd, Notes, PtWinner
```

- `1st` — shot string for first-serve rally (empty if first serve faulted)
- `2nd` — shot string for second-serve rally (empty if first serve was in)
- `Svr` — 1 or 2 (which player is serving)
- `PtWinner` — 1 or 2

### Shot String Encoding

Each shot string is a compact sequence encoding the entire rally:

```
4b38f1r2f-1l1o1*
```

| Token     | Meaning                                              |
|-----------|------------------------------------------------------|
| `4`       | Serve: 4=wide, 5=body, 6=T                           |
| `f` / `b` | Shot type: f=forehand, b=backhand, r=FH slice, etc.  |
| `-` / `+` | Position modifier: `-`=at-net, `+`=approach          |
| `1-3`     | Direction: 1=DTL, 2=CC/middle, 3=opposite-DTL        |
| `7-9`     | Depth (optional): 7=shallow, 8=mid, 9=deep           |
| `*`/`@`/`#`/`nwdx` | Outcome (terminal): winner / UE / FE / miss type |

The full shot type alphabet: `f b r s v z l o j k y p q`
(forehand, backhand, fh-slice, bh-slice, volley, bh-volley, lob, overhead,
fh-swing-volley, bh-swing-volley, tweener, fh-drop, bh-drop)

Semicolons `;` are contributor-inserted separators and are stripped.

### Rally Length Convention

`rally_len` = number of shots **including serve**, **excluding the error shot**.
- Ace: `4*` → rally_len = 1
- Serve + return + error: `4f27f3d@` → rally_len = 2 (serve + return; f3 error excluded)

## Alignment Strategy

### Step 1 — Contact Detection

From ball tracking (`BallTrackResult.trajectory_segment`): each segment
increment marks a court bounce, after which the opponent hits the next shot.
The first frame of each trajectory segment ≈ the corresponding contact frame.

Contact count per video point ≈ rally length (within ±1 for error shots).

### Step 2 — Point-Level Alignment (DTW over Rally Lengths)

Match the sequence of video points to MCP rows using DTW over rally-length
sequences. This handles:

- **Broadcast cuts** — MCP rows without a corresponding video point
- **Replays** — video points that duplicate MCP entries
- **Misdetections** — single-point rally count errors (±1 tolerance)

Score-based alignment (scoreboard OCR) is deferred. Rally-length DTW is
sufficient for unedited or lightly-edited broadcast footage.

**DTW cost:** `|cv_contact_count − mcp_rally_len|`

**Alignment confidence:** `1.0 − cost / max(rally_len, 1)`

A manual alignment override file can be supplied to fix anchor points when
DTW fails on heavily-edited footage (future feature).

### Step 3 — Shot-Level Assignment Within Points

Within each matched point:
1. Contacts are zipped with MCP shot tokens in order (serve = index 0).
2. Server role is inferred from ball y-position at the first contact:
   `y < net_y → near player serving, y ≥ net_y → far player serving`.
3. Shots alternate between striker/receiver starting from the server.
4. Excess contacts (more CV than MCP) → extras dropped.
5. Missing contacts (fewer CV than MCP) → unmatched MCP shots skipped.

### Step 4 — Quality Filtering

`alignment_confidence` encodes per-point quality:
- 1.0 — exact rally count match
- 0.5 — off by half the rally length
- 0.0 — completely mismatched

Low-confidence records are retained in the output but should be downweighted
during training.

## Output: ShotRecord

```python
@dataclass
class ShotRecord:
    # From MCP
    match_id: str
    point_id: str           # "{match_id}_{pt}_{shot_idx}"
    shot_in_rally: int
    shot_type_mcp: str      # e.g. 'f', 'b', 'serve'
    direction_mcp: str      # '1', '2', '3' or ''
    depth_mcp: str          # '7', '8', '9' or ''
    point_outcome: int      # 1 = striker won, 0 = lost

    # From CV
    contact_frame: int
    striker_role: str       # "near" or "far"
    striker_position_m: list
    striker_velocity_ms: list
    opponent_position_m: list
    opponent_velocity_ms: list
    ball_landing_zone: Optional[str]
    ball_contact_position_m: Optional[list]

    # Pose (populated when RTMPose step is implemented)
    striker_pose: Optional[list] = None   # (17, 2) keypoints in court meters
    striker_pose_confidence: float = 0.0

    # Derived
    rally_context: list = field(default_factory=list)
    alignment_confidence: float = 0.0
    pose_quality: float = 0.0
```

## Known Limitations

- **Score-based alignment not implemented**: Rally-length DTW is a proxy.
  Heavily-edited broadcasts (match highlights, truncated recordings) may
  produce poor alignment. Score OCR is the planned fix.
- **Pose fields are empty in V1**: `striker_pose` is `None` until RTMPose
  is integrated.
- **Server role is heuristic**: Ball y-position at serve contact infers
  who is serving; rare camera angles or occlusions can flip this.
- **Contact timing is approximate**: Trajectory-segment boundaries overcount
  by 1 for error shots (no ±0 correction applied per shot).

## Key Dependencies

- `pandas>=2.2` — MCP CSV loading
- `numpy>=1.26` — DTW DP matrix
- Existing pipeline outputs: `BallTrackResult`, `PlayerTrackResult`,
  `PointSegment`
