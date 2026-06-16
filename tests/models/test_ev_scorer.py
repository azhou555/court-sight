"""Tests for EV surface computation and shot quality scoring."""

import numpy as np
import pytest

from src.models.ev_surface.scorer import compute_ev, _get_tier, COURT_ZONES


def test_compute_ev_certain_make_high_win():
    ev = compute_ev(p_make=1.0, p_win=0.9)
    assert abs(ev - (1.0 * (0.9 + 1.0) - 1.0)) < 1e-6


def test_compute_ev_certain_miss():
    ev = compute_ev(p_make=0.0, p_win=0.9)
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


from src.models.ev_surface.scorer import EVScorer, RecoveryState
from src.models.execution_prob.model import ShotGeometryFeatures
from src.models.neutral_position.model import ResponseDistributionFeatures


class _FakeSafety:
    def __init__(self, p_make=0.8):
        self.p = p_make
    def predict(self, features):
        return {"p_make": self.p, "p_miss": 1.0 - self.p}


class _FakeWinProb:
    """p_win keyed off the last (hypothetical) shot's landing zone."""
    def __init__(self, zone_pwin):
        self.m = zone_pwin
    def predict(self, rally):
        zone = rally[-1].get("ball_landing_zone")
        return {"p_win_point": self.m.get(zone, 0.5), "rally_context_used": len(rally)}


class _FakeNeutral:
    """reachable neutral biased by your_landing_x so zones differ."""
    def predict_neutral(self, features, current_pos, current_vel, t_recovery, rng=None):
        return {"reachable_neutral_m": np.array([features.your_landing_x * 0.5, 2.0])}


def _safety_features():
    return ShotGeometryFeatures(
        lateral_margin_m=1.0, depth_margin_m=2.0, net_clearance_m=0.3,
        shot_direction_deg=0.0, landing_x=0.0, landing_y=10.0,
        shot_type="forehand_groundstroke", striker_y_m=2.0, striker_x_m=0.0,
        incoming_depth_m=18.0,
    )


def _response_features():
    return ResponseDistributionFeatures(
        your_landing_x=0.0, your_landing_y=10.0, ball_height_at_bounce=0.0,
        opponent_pos_x=0.0, opponent_pos_y=18.0, opponent_vel_x=0.0, opponent_vel_y=0.0,
        opponent_displacement_from_neutral=1.0, your_shot_type="forehand_groundstroke",
        rally_depth=4, prev_shot_direction_1=0.0, prev_shot_direction_2=0.0,
    )


def _uniform_pwin(value=0.5):
    return {z: value for z in COURT_ZONES}


def test_compute_ev_surface_returns_nine_finite_evs():
    scorer = EVScorer(_FakeSafety(0.8), _FakeWinProb(_uniform_pwin(0.5)), _FakeNeutral())
    surface = scorer.compute_ev_surface(_safety_features(), [{"ball_landing_zone": "C_deep"}], dp=0.0)
    assert len(surface) == 9
    assert all(np.isfinite(v) for v in surface.values())


def test_compute_ev_surface_dp_is_constant_offset():
    """dp shifts every zone equally and never changes the argmax."""
    zone_pwin = _uniform_pwin(0.3); zone_pwin["W_deep"] = 0.95
    scorer = EVScorer(_FakeSafety(0.8), _FakeWinProb(zone_pwin), _FakeNeutral())
    feats, rally = _safety_features(), [{"ball_landing_zone": "C_deep"}]
    s0 = scorer.compute_ev_surface(feats, rally, dp=0.0)
    s2 = scorer.compute_ev_surface(feats, rally, dp=0.2)
    for z in COURT_ZONES:
        assert abs((s0[z] - 0.2) - s2[z]) < 1e-6
    assert max(s0, key=s0.get) == max(s2, key=s2.get) == "W_deep"


def test_predict_zone_neutrals_one_per_zone():
    scorer = EVScorer(_FakeSafety(), _FakeWinProb(_uniform_pwin()), _FakeNeutral())
    neutrals = scorer.predict_zone_neutrals(
        _response_features(), striker_pos=np.array([0.0, 2.0]),
        striker_vel=np.array([0.0, 0.0]), t_recovery=0.8,
    )
    assert set(neutrals) == set(COURT_ZONES)
    from src.models.ev_surface.scorer import ZONE_CENTERS
    for z in COURT_ZONES:
        assert abs(neutrals[z][0] - ZONE_CENTERS[z][0] * 0.5) < 1e-6
