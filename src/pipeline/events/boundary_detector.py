"""Point boundary detection via rule-based state machine.

Consumes per-frame pipeline outputs (BallTrackResult, PlayerTrackResult) and
emits PointSegment objects marking the start and end of each tennis point.

State machine: DEAD ↔ LIVE with a COOLDOWN buffer to tolerate brief occlusions.
See docs/architecture/11_point_boundary.md for the full design.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from pathlib import Path
from typing import Optional

from ..ball.tracker import BallTrackResult
from ..tracking.player_tracker import PlayerTrackResult


class _State(Enum):
    DEAD = auto()
    LIVE = auto()
    COOLDOWN = auto()


@dataclass
class PointSegment:
    point_id: int
    start_frame: int
    end_frame: int
    start_sec: float
    end_sec: float


@dataclass
class BoundaryDetector:
    """Stateful per-frame point boundary detector.

    Args:
        fps:               Video frame rate.
        dead_min_frames:   Minimum consecutive frames without ball before the
                           state is considered truly dead (default: 2 sec).
        cooldown_frames:   Frames to wait after ball loss before declaring dead
                           (occlusion tolerance, default: 0.5 sec).
        player_slow_ms:    Max player speed (m/s) to count as "stationary".
        pre_roll_frames:   Frames prepended before detected point start.
        post_roll_frames:  Frames appended after detected point end.
        min_point_frames:  Minimum point length; shorter segments are discarded
                           (guards against replays and false triggers).
    """

    fps: float = 30.0
    dead_min_frames: int = 60
    cooldown_frames: int = 15
    player_slow_ms: float = 1.0
    pre_roll_frames: int = 60
    post_roll_frames: int = 45
    min_point_frames: int = 10

    # Internal state — not part of the public interface
    _state: _State = field(default=_State.DEAD, init=False, repr=False)
    _dead_counter: int = field(default=0, init=False, repr=False)
    _cooldown_counter: int = field(default=0, init=False, repr=False)
    _live_start_frame: int = field(default=0, init=False, repr=False)
    _last_live_frame: int = field(default=0, init=False, repr=False)
    _completed: list[PointSegment] = field(default_factory=list, init=False, repr=False)
    _point_counter: int = field(default=0, init=False, repr=False)

    def process_frame(
        self,
        frame_idx: int,
        ball: Optional[BallTrackResult],
        players: Optional[PlayerTrackResult],
    ) -> Optional[PointSegment]:
        """Update state machine for one frame.

        Returns a completed PointSegment when a point ends, otherwise None.
        """
        ball_visible = (
            ball is not None
            and ball.position_px is not None
            and ball.confidence > 0.0
        )
        both_slow = _both_players_slow(players, self.player_slow_ms)

        completed: Optional[PointSegment] = None

        if self._state is _State.DEAD:
            if ball_visible:
                self._dead_counter += 1
                if self._dead_counter >= self.dead_min_frames:
                    # Enough dead time has passed — this ball appearance is a serve
                    self._state = _State.LIVE
                    self._live_start_frame = frame_idx
                    self._last_live_frame = frame_idx
                    self._dead_counter = 0
            else:
                self._dead_counter = 0

        elif self._state is _State.LIVE:
            if ball_visible:
                self._last_live_frame = frame_idx
                self._cooldown_counter = 0
            else:
                self._state = _State.COOLDOWN
                self._cooldown_counter = 1

        elif self._state is _State.COOLDOWN:
            if ball_visible:
                self._last_live_frame = frame_idx
                self._cooldown_counter = 0
                self._state = _State.LIVE
            else:
                self._cooldown_counter += 1
                if self._cooldown_counter > self.cooldown_frames and both_slow:
                    completed = self._close_point()
                    self._state = _State.DEAD
                    self._dead_counter = 0

        return completed

    def flush(self) -> Optional[PointSegment]:
        """Close any open point at end of video."""
        if self._state in (_State.LIVE, _State.COOLDOWN):
            return self._close_point()
        return None

    def _close_point(self) -> Optional[PointSegment]:
        duration = self._last_live_frame - self._live_start_frame
        if duration < self.min_point_frames:
            return None

        start = max(0, self._live_start_frame - self.pre_roll_frames)
        end = self._last_live_frame + self.post_roll_frames

        seg = PointSegment(
            point_id=self._point_counter,
            start_frame=start,
            end_frame=end,
            start_sec=start / self.fps,
            end_sec=end / self.fps,
        )
        self._completed.append(seg)
        self._point_counter += 1
        return seg

    @property
    def completed_points(self) -> list[PointSegment]:
        return list(self._completed)

    def save(self, path: str | Path, source_video: str = "") -> None:
        """Write completed point boundaries to JSON."""
        data = {
            "source_video": source_video,
            "fps": self.fps,
            "total_points": len(self._completed),
            "points": [asdict(s) for s in self._completed],
        }
        Path(path).write_text(json.dumps(data, indent=2))

    @staticmethod
    def load(path: str | Path) -> tuple[list[PointSegment], dict]:
        """Load previously saved boundaries. Returns (segments, metadata)."""
        raw = json.loads(Path(path).read_text())
        segments = [PointSegment(**p) for p in raw["points"]]
        meta = {k: v for k, v in raw.items() if k != "points"}
        return segments, meta


def _both_players_slow(
    players: Optional[PlayerTrackResult], threshold: float
) -> bool:
    """True when both detected players are moving below threshold m/s."""
    if players is None:
        return True
    speeds = []
    for p in (players.near_player, players.far_player):
        if p is not None:
            speeds.append(float(p.velocity_ms.__abs__().max()))
    if not speeds:
        return True
    return max(speeds) < threshold
