"""Tests for EV surface computation and shot quality scoring."""

import numpy as np
import pytest

from src.models.ev_surface.scorer import compute_ev, _get_tier, COURT_ZONES


def test_compute_ev_certain_make_high_win():
    ev = compute_ev(p_make=1.0, p_win_given_make=0.9)
    assert abs(ev - (1.0 * (0.9 + 1.0) - 1.0)) < 1e-6


def test_compute_ev_certain_miss():
    ev = compute_ev(p_make=0.0, p_win_given_make=0.9)
    assert abs(ev - (-1.0)) < 1e-6


def test_compute_ev_range():
    for p_make in np.linspace(0, 1, 11):
        for p_win in np.linspace(0, 1, 11):
            ev = compute_ev(p_make, p_win)
            assert -1.0 <= ev <= 1.0 + 1e-9


def test_get_tier_boundaries():
    assert _get_tier(0.90) == "excellent"
    assert _get_tier(0.75) == "good"
    assert _get_tier(0.50) == "suboptimal"
    assert _get_tier(0.20) == "poor"
    assert _get_tier(0.00) == "poor"


def test_court_zones_count():
    assert len(COURT_ZONES) == 9
