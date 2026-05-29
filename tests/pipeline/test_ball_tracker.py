"""Tests for ball zone classification logic."""

import numpy as np
import pytest

from src.pipeline.ball.tracker import _classify_zone, COURT_ZONES


def test_classify_zone_T_deep():
    # Near center T, well behind service line on far side
    zone = _classify_zone(np.array([-2.0, 2.0]))    # deep on far side
    assert zone == "T_deep"


def test_classify_zone_wide_short():
    zone = _classify_zone(np.array([3.5, 21.0]))    # wide, short
    assert zone == "W_short"


def test_classify_zone_center_mid():
    zone = _classify_zone(np.array([0.0, 14.0]))
    assert zone == "C_mid"


def test_all_zones_reachable():
    """Verify all 9 zones can be produced by _classify_zone."""
    test_points = [
        ([-2.0, 2.0],  "T_deep"),
        ([0.0,  2.0],  "C_deep"),
        ([3.0,  2.0],  "W_deep"),
        ([-2.0, 14.0], "T_mid"),
        ([0.0,  14.0], "C_mid"),
        ([3.0,  14.0], "W_mid"),
        ([-2.0, 21.0], "T_short"),
        ([0.0,  21.0], "C_short"),
        ([3.0,  21.0], "W_short"),
    ]
    for pt, expected in test_points:
        result = _classify_zone(np.array(pt))
        assert result == expected, f"Point {pt}: expected {expected}, got {result}"
