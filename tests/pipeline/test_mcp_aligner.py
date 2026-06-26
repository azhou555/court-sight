"""Tests for MCP alignment logic: DTW point alignment and shot assignment."""

import numpy as np
import pytest

from src.pipeline.alignment.mcp_aligner import (
    ShotRecord,
    _assign_shots,
    _detect_server_role,
    _dtw_align_points,
    _find_landing_zone,
    _nearest_player,
    _resolve_striker,
)
from src.pipeline.alignment.mcp_parser import MCPPoint, ShotToken
from src.pipeline.ball.tracker import BallTrackResult
from src.pipeline.tracking.player_tracker import PlayerState, PlayerTrackResult


# ─────────────────────────────── helpers ────────────────────────────────────

def _mcp_point(rally_len: int, pt: int = 1, svr: int = 1, pt_winner: int = 1) -> MCPPoint:
    shots = [
        ShotToken(is_serve=True, shot_type="serve", direction=None, depth=None, serve_placement=4),
    ] + [
        ShotToken(is_serve=False, shot_type="f", direction=2, depth=None)
        for _ in range(rally_len - 1)
    ]
    return MCPPoint(
        match_id="TEST", pt=pt, set1=0, set2=0, gm1=0, gm2=0, pts="0-0",
        svr=svr, pt_winner=pt_winner, shots=shots, rally_len=rally_len,
    )


def _ball(frame_idx: int, y_m: float = 5.0, is_bounce: bool = False,
          bounce_zone: str | None = None) -> BallTrackResult:
    return BallTrackResult(
        frame_idx=frame_idx,
        position_px=np.array([320.0, 240.0]),
        position_m=np.array([0.0, y_m]),
        confidence=0.9,
        is_bounce=is_bounce,
        bounce_zone=bounce_zone,
        trajectory_segment=0,
    )


def _player(frame_idx: int, near_y: float = 2.0, far_y: float = 21.0) -> PlayerTrackResult:
    def _state(y: float) -> PlayerState:
        return PlayerState(
            track_id=1,
            bbox_px=np.array([0.0, 0.0, 50.0, 100.0]),
            position_m=np.array([0.0, y]),
            velocity_ms=np.array([0.0, 0.0]),
            confidence=0.9,
        )
    return PlayerTrackResult(
        frame_idx=frame_idx,
        near_player=_state(near_y),
        far_player=_state(far_y),
    )


# ─────────────────────────────── _dtw_align_points ──────────────────────────

def test_dtw_perfect_match():
    video_contacts = [[1, 2, 3], [4, 5]]        # rally_len 3, 2
    mcp = [_mcp_point(3, pt=1), _mcp_point(2, pt=2)]
    result = _dtw_align_points(video_contacts, mcp)
    assert len(result) == 2
    vis, mis, confs = zip(*result)
    assert list(vis) == [0, 1]
    assert list(mis) == [0, 1]
    assert all(c == 1.0 for c in confs)


def test_dtw_confidence_decreases_with_mismatch():
    video_contacts = [[1, 2]]        # rally_len 2
    mcp = [_mcp_point(4, pt=1)]      # rally_len 4 — delta = 2
    result = _dtw_align_points(video_contacts, mcp)
    assert len(result) == 1
    _, _, conf = result[0]
    assert conf < 1.0


def test_dtw_broadcast_cut_skipped():
    # 2 video points, 3 MCP points — one MCP row has no video match
    video_contacts = [[1], [2]]
    mcp = [_mcp_point(1, pt=1), _mcp_point(5, pt=2), _mcp_point(1, pt=3)]
    result = _dtw_align_points(video_contacts, mcp)
    # Each video point gets at most one MCP match
    vis = [r[0] for r in result]
    assert len(vis) == len(set(vis))            # no duplicate video indices


def test_dtw_empty_inputs():
    assert _dtw_align_points([], [_mcp_point(2)]) == []
    assert _dtw_align_points([[1, 2]], []) == []
    assert _dtw_align_points([], []) == []


def test_dtw_one_to_one_deduplication():
    # All rally lengths equal — no duplicates in output
    video_contacts = [[i] for i in range(5)]
    mcp = [_mcp_point(1, pt=i) for i in range(5)]
    result = _dtw_align_points(video_contacts, mcp)
    vis = [r[0] for r in result]
    mis = [r[1] for r in result]
    assert len(vis) == len(set(vis))
    assert len(mis) == len(set(mis))


# ─────────────────────────────── _detect_server_role ────────────────────────

def test_detect_server_role_near():
    # Ball y < net (11.885) → near player serving
    ball_by_frame = {10: _ball(10, y_m=3.0)}
    assert _detect_server_role(10, ball_by_frame) == "near"


def test_detect_server_role_far():
    ball_by_frame = {10: _ball(10, y_m=20.0)}
    assert _detect_server_role(10, ball_by_frame) == "far"


def test_detect_server_role_no_ball_defaults_near():
    assert _detect_server_role(10, {}) == "near"


# ─────────────────────────────── _nearest_player ────────────────────────────

def test_nearest_player_exact_frame():
    player_by_frame = {10: _player(10), 20: _player(20)}
    result = _nearest_player(player_by_frame, 10)
    assert result is not None
    assert result.frame_idx == 10


def test_nearest_player_within_window():
    player_by_frame = {12: _player(12)}
    result = _nearest_player(player_by_frame, 10, window=5)
    assert result is not None


def test_nearest_player_outside_window():
    player_by_frame = {20: _player(20)}
    result = _nearest_player(player_by_frame, 10, window=5)
    assert result is None


# ─────────────────────────────── _find_landing_zone ─────────────────────────

def test_find_landing_zone_found():
    ball_by_frame = {
        10: _ball(10),
        15: _ball(15, is_bounce=True, bounce_zone="C_deep"),
    }
    zone = _find_landing_zone(10, ball_by_frame, lookahead=25)
    assert zone == "C_deep"


def test_find_landing_zone_not_found():
    ball_by_frame = {10: _ball(10)}
    assert _find_landing_zone(10, ball_by_frame, lookahead=5) is None


def test_find_landing_zone_ignores_pre_contact_bounce():
    # Bounce at frame 5 (before contact at 10) should not be returned
    ball_by_frame = {
        5:  _ball(5, is_bounce=True, bounce_zone="W_short"),
        15: _ball(15, is_bounce=True, bounce_zone="T_deep"),
    }
    zone = _find_landing_zone(10, ball_by_frame, lookahead=25)
    assert zone == "T_deep"


# ─────────────────────────────── _resolve_striker ───────────────────────────

def test_resolve_striker_server_near_first_shot():
    player = _player(10)
    striker, role, opponent = _resolve_striker(player, is_server_shot=True, server_role="near")
    assert role == "near"
    assert striker is not None


def test_resolve_striker_receiver_far_first_shot():
    player = _player(10)
    striker, role, opponent = _resolve_striker(player, is_server_shot=False, server_role="near")
    assert role == "far"


def test_resolve_striker_no_players_returns_none():
    player = PlayerTrackResult(frame_idx=10, near_player=None, far_player=None)
    striker, role, opponent = _resolve_striker(player, is_server_shot=True, server_role="near")
    assert striker is None


# ─────────────────────────────── _assign_shots ──────────────────────────────

def test_assign_shots_basic():
    mcp_pt = _mcp_point(rally_len=2, svr=1, pt_winner=1)
    contacts = [10, 30]
    ball_by_frame = {
        10: _ball(10, y_m=3.0),
        20: _ball(20, is_bounce=True, bounce_zone="C_deep"),
        30: _ball(30, y_m=20.0),
    }
    player_by_frame = {10: _player(10), 30: _player(30)}

    records = _assign_shots(
        mcp_pt=mcp_pt,
        contacts=contacts,
        alignment_conf=0.8,
        ball_by_frame=ball_by_frame,
        player_by_frame=player_by_frame,
        recent_records=[],
    )
    assert len(records) == 2
    assert records[0].shot_type_mcp == "serve"
    assert records[1].shot_type_mcp == "f"
    assert all(r.alignment_confidence == 0.8 for r in records)


def test_assign_shots_truncates_to_shorter_list():
    # 3 MCP shots but only 2 contacts → 2 records max
    mcp_pt = _mcp_point(rally_len=3, svr=1, pt_winner=1)
    contacts = [10, 30]
    ball_by_frame = {10: _ball(10, y_m=3.0), 30: _ball(30)}
    player_by_frame = {10: _player(10), 30: _player(30)}

    records = _assign_shots(mcp_pt, contacts, 1.0, ball_by_frame, player_by_frame, [])
    assert len(records) <= 2


def test_assign_shots_empty_contacts():
    mcp_pt = _mcp_point(rally_len=2)
    records = _assign_shots(mcp_pt, [], 1.0, {}, {}, [])
    assert records == []


def test_assign_shots_point_outcome():
    # svr=1, pt_winner=1 → server wins; shot 0 (server) → outcome=1
    mcp_pt = _mcp_point(rally_len=2, svr=1, pt_winner=1)
    contacts = [10, 30]
    ball_by_frame = {10: _ball(10, y_m=3.0), 30: _ball(30)}
    player_by_frame = {10: _player(10), 30: _player(30)}

    records = _assign_shots(mcp_pt, contacts, 1.0, ball_by_frame, player_by_frame, [])
    assert records[0].point_outcome == 1   # server's shot → server wins
    assert records[1].point_outcome == 0   # receiver's shot → server still wins → receiver lost
