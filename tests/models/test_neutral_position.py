"""Tests for neutral position model components."""

import numpy as np
import pytest

from src.models.neutral_position.model import (
    optimal_neutral, reachable_neutral, serve_neutral,
    COURT_X_MIN, COURT_X_MAX, COURT_Y_MIN, COURT_Y_MAX,
)


def test_optimal_neutral_near_mean():
    """For tight Gaussian, geometric median should be close to mean."""
    mu = np.array([1.0, 8.0])
    result = optimal_neutral(mu, sigma_x=0.3, sigma_y=0.3)
    np.testing.assert_allclose(result, mu, atol=0.3)


def test_optimal_neutral_within_court():
    """Result must always be inside court bounds."""
    rng = np.random.default_rng(42)
    for _ in range(20):
        mu = rng.uniform([-4.0, 1.0], [4.0, 22.0])
        result = optimal_neutral(mu, sigma_x=1.0, sigma_y=2.0, rng=rng)
        assert COURT_X_MIN <= result[0] <= COURT_X_MAX
        assert COURT_Y_MIN <= result[1] <= COURT_Y_MAX


def test_reachable_neutral_reachable():
    """If neutral is within sprint range, return it exactly."""
    current = np.array([0.0, 5.0])
    theoretical = np.array([2.0, 5.0])   # 2m away
    result = reachable_neutral(theoretical, current, np.zeros(2), t_recovery=1.0)
    np.testing.assert_allclose(result, theoretical)


def test_reachable_neutral_constrained():
    """If neutral is too far, project onto reachable boundary."""
    current = np.array([0.0, 5.0])
    theoretical = np.array([10.0, 5.0])  # 10m — unreachable in 0.5s
    result = reachable_neutral(theoretical, current, np.zeros(2), t_recovery=0.5)
    dist = float(np.linalg.norm(result - current))
    assert dist <= 0.5 * 4.5 + 1e-6   # max_speed * t_recovery


def test_reachable_neutral_direction_preserved():
    """Constrained neutral should be in the right direction."""
    current = np.array([0.0, 0.0])
    theoretical = np.array([3.0, 4.0])  # 5m away, unreachable in 0.5s
    result = reachable_neutral(theoretical, current, np.zeros(2), t_recovery=0.5)
    # Should be along the (3, 4) direction
    ratio = result[0] / result[1] if result[1] != 0 else float("inf")
    np.testing.assert_allclose(ratio, 3.0 / 4.0, atol=0.01)


def test_serve_neutral_defaults_present():
    for side in ("deuce", "ad"):
        for placement in ("T", "body", "wide"):
            pos = serve_neutral(side, placement)
            assert pos.shape == (2,)
            assert COURT_X_MIN <= pos[0] <= COURT_X_MAX


def test_serve_neutral_pattern_bias():
    """Pattern bias should shift neutral laterally."""
    base = serve_neutral("deuce", "T", pattern_bias=0.0)
    biased = serve_neutral("deuce", "T", pattern_bias=0.5)
    assert biased[0] == pytest.approx(base[0] + 0.5)
    assert biased[1] == pytest.approx(base[1])
