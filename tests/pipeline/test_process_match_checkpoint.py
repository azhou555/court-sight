"""Checkpoint/resume round-trip for process_match.py.

Guards against regressing to the old scheme where ball/player tracks were
re-serialized in full on every checkpoint (O(n^2) over a match): checkpoint.json
must stay small (no track data) and the JSONL track logs must survive a
simulated crash (extra trailing lines past what the checkpoint vouches for).
"""

import json

import numpy as np
import pytest

from scripts.process_match import (
    _load_checkpoint,
    _tracks_paths,
    _write_checkpoint,
    _ser_ball,
    _ser_player,
)
from src.pipeline.ball.tracker import BallTrackResult
from src.pipeline.tracking.player_tracker import PlayerTrackResult
from src.pipeline.events.boundary_detector import PointSegment


class _FakeHomography:
    _last_frame = 3
    _last_H = np.eye(3)


def _ball(i):
    return BallTrackResult(
        frame_idx=i, position_px=np.array([1.0, 2.0]), position_m=np.array([0.1, 0.2]),
        confidence=0.9, is_bounce=False, bounce_zone=None, trajectory_segment=0,
    )


def _player(i):
    return PlayerTrackResult(frame_idx=i, near_player=None, far_player=None)


def test_checkpoint_omits_track_data(tmp_path):
    ckpt_path = tmp_path / "checkpoint.json"
    _write_checkpoint(
        ckpt_path, next_frame=4, skipped=0, gate_keepalive=0, point_counter=1,
        point_segments=[], homography=_FakeHomography(),
    )
    raw = json.loads(ckpt_path.read_text())
    assert "ball_tracks" not in raw and "player_tracks" not in raw


def test_resume_truncates_partial_trailing_line(tmp_path):
    ball_path, player_path = _tracks_paths(tmp_path)
    with open(ball_path, "w") as f, open(player_path, "w") as f2:
        for i in range(4):
            f.write(json.dumps(_ser_ball(_ball(i))) + "\n")
            f2.write(json.dumps(_ser_player(_player(i))) + "\n")
        # simulate a crash mid-append past what checkpoint.json will vouch for
        f.write('{"frame_idx": 4, "garbage')

    ckpt_path = tmp_path / "checkpoint.json"
    _write_checkpoint(
        ckpt_path, next_frame=4, skipped=0, gate_keepalive=0, point_counter=1,
        point_segments=[], homography=_FakeHomography(),
    )

    loaded = _load_checkpoint(ckpt_path, tmp_path)
    assert len(loaded["ball_tracks"]) == 4
    assert len(loaded["player_tracks"]) == 4
    assert loaded["ball_tracks"][-1].frame_idx == 3
