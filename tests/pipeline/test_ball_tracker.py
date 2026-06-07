"""Tests for ball zone classification logic."""

import numpy as np
import pytest

from src.pipeline.ball.tracker import _classify_zone, COURT_ZONES


def test_classify_zone_T_deep():
    # Deep = near the far baseline (high y); center-T lateral.
    zone = _classify_zone(np.array([-2.0, 21.0]))
    assert zone == "T_deep"


def test_classify_zone_wide_short():
    # Short = near the net (low y); wide lateral.
    zone = _classify_zone(np.array([3.5, 2.0]))
    assert zone == "W_short"


def test_classify_zone_center_mid():
    zone = _classify_zone(np.array([0.0, 14.0]))
    assert zone == "C_mid"


def test_all_zones_reachable():
    """Verify all 9 zones can be produced by _classify_zone.

    Depth convention (matches _classify_zone and zone_to_coords): low y is
    short (near net), high y is deep (near far baseline).
    """
    test_points = [
        ([-2.0, 21.0], "T_deep"),
        ([0.0,  21.0], "C_deep"),
        ([3.0,  21.0], "W_deep"),
        ([-2.0, 14.0], "T_mid"),
        ([0.0,  14.0], "C_mid"),
        ([3.0,  14.0], "W_mid"),
        ([-2.0, 2.0],  "T_short"),
        ([0.0,  2.0],  "C_short"),
        ([3.0,  2.0],  "W_short"),
    ]
    for pt, expected in test_points:
        result = _classify_zone(np.array(pt))
        assert result == expected, f"Point {pt}: expected {expected}, got {result}"
