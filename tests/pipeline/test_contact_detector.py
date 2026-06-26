"""Tests for ball-racket contact frame detection."""

import pytest

from src.pipeline.alignment.contact_detector import detect_contacts
from src.pipeline.ball.tracker import BallTrackResult

import numpy as np


def _ball(frame_idx: int, segment: int, interpolated: bool = False) -> BallTrackResult:
    return BallTrackResult(
        frame_idx=frame_idx,
        position_px=np.array([100.0, 100.0]),
        position_m=np.array([0.0, 5.0]),
        confidence=0.9,
        is_bounce=False,
        bounce_zone=None,
        trajectory_segment=segment,
        interpolated=interpolated,
    )


def _no_ball(frame_idx: int) -> BallTrackResult:
    return BallTrackResult(
        frame_idx=frame_idx,
        position_px=None,
        position_m=None,
        confidence=0.0,
        is_bounce=False,
        bounce_zone=None,
        trajectory_segment=0,
    )


# ─────────────────────────────── basic cases ────────────────────────────────

def test_single_serve_contact():
    # Ball appears at frame 10 (segment 0) — one contact
    tracks = [_ball(10, segment=0), _ball(11, segment=0), _ball(12, segment=0)]
    contacts = detect_contacts(tracks, start_frame=0, end_frame=20)
    assert contacts == [10]


def test_two_contacts_on_segment_change():
    # Segment increments at frame 15 → second contact
    tracks = [
        _ball(10, segment=0),
        _ball(11, segment=0),
        _ball(15, segment=1),
        _ball(16, segment=1),
    ]
    contacts = detect_contacts(tracks, start_frame=0, end_frame=20)
    assert contacts == [10, 15]


def test_three_contacts():
    tracks = [
        _ball(5,  segment=0),
        _ball(10, segment=1),
        _ball(20, segment=2),
    ]
    contacts = detect_contacts(tracks, start_frame=0, end_frame=30)
    assert contacts == [5, 10, 20]


def test_no_ball_returns_empty():
    tracks = [_no_ball(i) for i in range(10)]
    assert detect_contacts(tracks, start_frame=0, end_frame=9) == []


def test_empty_tracks():
    assert detect_contacts([], start_frame=0, end_frame=10) == []


# ─────────────────────────────── window filtering ───────────────────────────

def test_frames_outside_window_excluded():
    tracks = [
        _ball(5,  segment=0),   # before window
        _ball(15, segment=1),   # inside
        _ball(25, segment=2),   # after window
    ]
    contacts = detect_contacts(tracks, start_frame=10, end_frame=20)
    assert contacts == [15]


def test_window_boundaries_inclusive():
    tracks = [_ball(10, segment=0), _ball(20, segment=1)]
    contacts = detect_contacts(tracks, start_frame=10, end_frame=20)
    assert contacts == [10, 20]


# ─────────────────────────────── interpolated frames ────────────────────────

def test_interpolated_frames_excluded():
    # Interpolated frame at segment boundary should NOT count as a contact
    tracks = [
        _ball(10, segment=0),
        _ball(15, segment=1, interpolated=True),   # gap-fill, skip
        _ball(16, segment=1, interpolated=False),
    ]
    contacts = detect_contacts(tracks, start_frame=0, end_frame=20)
    # Contact at 10 (seg 0 start), then 16 (first real seg 1 frame)
    assert 15 not in contacts
    assert 16 in contacts
