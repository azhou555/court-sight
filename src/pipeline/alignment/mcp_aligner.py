"""Align Match Charting Project labels to CV-extracted positional data.

Alignment strategy:
  1. For each detected video point, count contacts from ball trajectory segments.
  2. Match the full sequence of video points to MCP rows via DTW over rally
     lengths — this handles broadcast cuts and replays without score OCR.
  3. Within each matched point, zip contact frames to MCP shot tokens.
  4. Assemble one ShotRecord per shot with CV player/ball state.

Score-based alignment (via scoreboard OCR) is intentionally deferred to a
future iteration.  Rally-length DTW is sufficient for most broadcast footage.

See docs/architecture/09_data_alignment.md for the full design.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np

from .contact_detector import detect_contacts
from .mcp_parser import MCPPoint, ShotToken, load_mcp_csv
from ..ball.tracker import BallTrackResult
from ..events.boundary_detector import PointSegment
from ..tracking.player_tracker import PlayerState, PlayerTrackResult

_NET_Y = 11.885   # court-meter y of net center


@dataclass
class ShotRecord:
    """One fully annotated shot event."""

    # From Match Charting Project
    match_id: str
    point_id: str               # "{match_id}_{pt}_{shot_idx}"
    shot_in_rally: int
    shot_type_mcp: str          # e.g. 'f', 'b', 'serve'
    direction_mcp: str          # '1', '2', '3' or ''
    depth_mcp: str              # '7', '8', '9' or ''
    point_outcome: int          # 1 = striker won point, 0 = lost

    # From CV pipeline
    contact_frame: int
    striker_role: str           # "near" or "far"
    striker_position_m: list    # [x, y] court meters
    striker_velocity_ms: list   # [vx, vy] m/s
    opponent_position_m: list   # [x, y] court meters
    opponent_velocity_ms: list  # [vx, vy] m/s
    ball_landing_zone: Optional[str]
    ball_contact_position_m: Optional[list]

    # Pose — populated when RTMPose is implemented (pipeline step 3b)
    striker_pose: Optional[list] = None   # (17, 2) keypoints in court meters
    striker_pose_confidence: float = 0.0

    # Derived
    rally_context: list = field(default_factory=list)  # last ≤3 ShotRecord summaries
    alignment_confidence: float = 0.0
    pose_quality: float = 0.0


class MCPAligner:
    """Aligns MCP CSV data to CV-extracted frame-level match data.

    Usage:
        aligner = MCPAligner(fps=30.0)
        records = aligner.align(
            mcp_path="charting-m-points-2020s.csv",
            point_boundaries=boundary_detector.completed_points,
            ball_tracks=ball_track_results,
            player_tracks=player_track_results,
        )
        MCPAligner.save(records, "shot_records.json")
    """

    def __init__(self, fps: float = 30.0):
        self.fps = fps

    def align(
        self,
        mcp_path: str | Path,
        point_boundaries: list[PointSegment],
        ball_tracks: list[BallTrackResult],
        player_tracks: list[PlayerTrackResult],
    ) -> list[ShotRecord]:
        """Align MCP labels to CV data for a full match.

        Args:
            mcp_path:         Path to MCP points CSV filtered to this match.
            point_boundaries: PointSegments from BoundaryDetector.
            ball_tracks:      All per-frame BallTrackResult objects.
            player_tracks:    All per-frame PlayerTrackResult objects.

        Returns:
            List of annotated ShotRecord objects in rally order.
        """
        mcp_points = load_mcp_csv(mcp_path)

        ball_by_frame: dict[int, BallTrackResult] = {
            r.frame_idx: r for r in ball_tracks
        }
        player_by_frame: dict[int, PlayerTrackResult] = {
            r.frame_idx: r for r in player_tracks
        }

        video_contacts: list[list[int]] = [
            detect_contacts(ball_tracks, seg.start_frame, seg.end_frame)
            for seg in point_boundaries
        ]

        alignment = _dtw_align_points(video_contacts, mcp_points)

        records: list[ShotRecord] = []
        for video_idx, mcp_idx, conf in alignment:
            seg = point_boundaries[video_idx]
            mcp_pt = mcp_points[mcp_idx]
            contacts = video_contacts[video_idx]

            shot_records = _assign_shots(
                mcp_pt=mcp_pt,
                contacts=contacts,
                alignment_conf=conf,
                ball_by_frame=ball_by_frame,
                player_by_frame=player_by_frame,
                recent_records=records[-3:],
            )
            records.extend(shot_records)

        return records

    @staticmethod
    def save(records: list[ShotRecord], path: str | Path) -> None:
        """Serialize ShotRecords to JSON."""
        Path(path).write_text(json.dumps([asdict(r) for r in records], indent=2))

    @staticmethod
    def load(path: str | Path) -> list[ShotRecord]:
        """Deserialize ShotRecords from JSON."""
        data = json.loads(Path(path).read_text())
        return [ShotRecord(**d) for d in data]


# ─────────────────────────────── point alignment ────────────────────────────

def _dtw_align_points(
    video_contacts: list[list[int]],
    mcp_points: list[MCPPoint],
) -> list[tuple[int, int, float]]:
    """DTW alignment of video points to MCP rows via rally-length sequences.

    Returns sorted (video_idx, mcp_idx, confidence) triples representing
    1:1 point matches.  Confidence is 1.0 when rally lengths agree exactly,
    decreasing linearly with the absolute count delta.

    Broadcast cuts manifest as MCP rows with no matching video point;
    replays manifest as video points with no matching MCP row.  Both are
    silently dropped from the output — they produce no ShotRecords.
    """
    v_counts = [len(c) for c in video_contacts]
    m_counts = [p.rally_len for p in mcp_points]

    if not v_counts or not m_counts:
        return []

    nv, nm = len(v_counts), len(m_counts)
    INF = float("inf")
    dp = [[INF] * (nm + 1) for _ in range(nv + 1)]
    dp[0][0] = 0.0

    for i in range(1, nv + 1):
        for j in range(1, nm + 1):
            cost = abs(v_counts[i - 1] - m_counts[j - 1])
            dp[i][j] = cost + min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1])

    # Standard DTW traceback
    path: list[tuple[int, int]] = []
    i, j = nv, nm
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        diag = dp[i - 1][j - 1]
        up = dp[i - 1][j]
        left = dp[i][j - 1]
        best = min(diag, up, left)
        if diag == best:
            i -= 1
            j -= 1
        elif up == best:
            i -= 1
        else:
            j -= 1
    path.reverse()

    # Deduplicate to 1:1 pairings (first occurrence wins)
    result: list[tuple[int, int, float]] = []
    seen_v: set[int] = set()
    seen_m: set[int] = set()
    for vi, mi in path:
        if vi in seen_v or mi in seen_m:
            continue
        seen_v.add(vi)
        seen_m.add(mi)
        delta = abs(v_counts[vi] - m_counts[mi])
        conf = max(0.0, 1.0 - delta / max(m_counts[mi], 1))
        result.append((vi, mi, conf))

    return result


# ─────────────────────────────── shot assignment ────────────────────────────

def _assign_shots(
    mcp_pt: MCPPoint,
    contacts: list[int],
    alignment_conf: float,
    ball_by_frame: dict[int, BallTrackResult],
    player_by_frame: dict[int, PlayerTrackResult],
    recent_records: list[ShotRecord],
) -> list[ShotRecord]:
    """Zip contact frames with MCP shot tokens and assemble ShotRecords."""
    shots = mcp_pt.shots
    if not shots or not contacts:
        return []

    server_role = _detect_server_role(contacts[0], ball_by_frame)
    server_wins = mcp_pt.svr == mcp_pt.pt_winner

    records: list[ShotRecord] = []
    context = [_shot_summary(r) for r in recent_records]

    # Truncate to the shorter list — extras on either side are unmatched
    n = min(len(shots), len(contacts))
    for i in range(n):
        shot = shots[i]
        contact_frame = contacts[i]

        ball = ball_by_frame.get(contact_frame)
        player = _nearest_player(player_by_frame, contact_frame)
        if player is None:
            continue

        is_server_shot = i % 2 == 0
        striker, striker_role, opponent = _resolve_striker(
            player, is_server_shot, server_role
        )
        if striker is None:
            continue

        point_outcome = 1 if (is_server_shot == server_wins) else 0
        landing_zone = _find_landing_zone(contact_frame, ball_by_frame)

        record = ShotRecord(
            match_id=mcp_pt.match_id,
            point_id=f"{mcp_pt.match_id}_{mcp_pt.pt}_{i}",
            shot_in_rally=i,
            shot_type_mcp=shot.shot_type,
            direction_mcp=str(shot.direction) if shot.direction is not None else "",
            depth_mcp=str(shot.depth) if shot.depth is not None else "",
            point_outcome=point_outcome,
            contact_frame=contact_frame,
            striker_role=striker_role,
            striker_position_m=striker.position_m.tolist(),
            striker_velocity_ms=striker.velocity_ms.tolist(),
            opponent_position_m=opponent.position_m.tolist() if opponent else [0.0, 0.0],
            opponent_velocity_ms=opponent.velocity_ms.tolist() if opponent else [0.0, 0.0],
            ball_landing_zone=landing_zone,
            ball_contact_position_m=(
                ball.position_m.tolist()
                if ball and ball.position_m is not None
                else None
            ),
            rally_context=list(context),
            alignment_confidence=alignment_conf,
        )
        records.append(record)

        context.append(_shot_summary(record))
        if len(context) > 3:
            context.pop(0)

    return records


# ─────────────────────────────── utilities ──────────────────────────────────

def _detect_server_role(
    first_contact: int,
    ball_by_frame: dict[int, BallTrackResult],
) -> str:
    """Infer which side is serving from ball y-position at serve contact.

    At the serve, the ball is near the server's baseline.  If y < net_y the
    near player is serving; if y > net_y the far player is serving.
    """
    ball = ball_by_frame.get(first_contact)
    if ball and ball.position_m is not None:
        return "near" if float(ball.position_m[1]) < _NET_Y else "far"
    return "near"


def _resolve_striker(
    player: PlayerTrackResult,
    is_server_shot: bool,
    server_role: str,
) -> tuple[Optional[PlayerState], str, Optional[PlayerState]]:
    """Return (striker, striker_role, opponent) for this shot."""
    near = player.near_player
    far = player.far_player

    if near is None and far is None:
        return None, "near", None

    striker_is_near = (is_server_shot and server_role == "near") or (
        not is_server_shot and server_role == "far"
    )

    if striker_is_near:
        if near is not None:
            return near, "near", far
        return far, "far", None
    else:
        if far is not None:
            return far, "far", near
        return near, "near", None


def _nearest_player(
    player_by_frame: dict[int, PlayerTrackResult],
    frame_idx: int,
    window: int = 5,
) -> Optional[PlayerTrackResult]:
    """Return the PlayerTrackResult closest to frame_idx within window."""
    candidates = sorted(
        (abs(f - frame_idx), f)
        for f in player_by_frame
        if abs(f - frame_idx) <= window
    )
    return player_by_frame[candidates[0][1]] if candidates else None


def _find_landing_zone(
    contact_frame: int,
    ball_by_frame: dict[int, BallTrackResult],
    lookahead: int = 25,
) -> Optional[str]:
    """Find the first bounce zone within lookahead frames after contact."""
    for f in range(contact_frame, contact_frame + lookahead):
        b = ball_by_frame.get(f)
        if b and b.is_bounce and b.bounce_zone:
            return b.bounce_zone
    return None


def _shot_summary(record: ShotRecord) -> dict:
    """Compact summary of a ShotRecord for rally context."""
    return {
        "shot_type": record.shot_type_mcp,
        "direction": record.direction_mcp,
        "zone": record.ball_landing_zone,
        "role": record.striker_role,
    }
