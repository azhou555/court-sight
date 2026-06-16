"""EV surface computation and shot quality scoring."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional
import numpy as np

from src.models.execution_prob.model import ShotSafetyModel, ShotGeometryFeatures
from src.models.execution_prob.displacement import displacement_penalty
from src.models.win_prob.model import WinProbModel
from src.models.neutral_position.model import NeutralPositionModel, ResponseDistributionFeatures

COURT_ZONES = [
    "T_deep",  "C_deep",  "W_deep",
    "T_mid",   "C_mid",   "W_mid",
    "T_short", "C_short", "W_short",
]

# Representative center coordinates (x, y) for each zone in court meters.
# These are used to compute geometric margin features for each hypothetical target.
ZONE_CENTERS: dict[str, np.ndarray] = {
    "T_deep":   np.array([-2.5, 3.0]),
    "C_deep":   np.array([ 0.0, 3.0]),
    "W_deep":   np.array([ 2.5, 3.0]),
    "T_mid":    np.array([-2.5, 10.0]),
    "C_mid":    np.array([ 0.0, 10.0]),
    "W_mid":    np.array([ 2.5, 10.0]),
    "T_short":  np.array([-2.5, 18.0]),
    "C_short":  np.array([ 0.0, 18.0]),
    "W_short":  np.array([ 2.5, 18.0]),
}

QUALITY_TIERS = {
    (0.85, 1.01): "excellent",
    (0.65, 0.85): "good",
    (0.40, 0.65): "suboptimal",
    (0.00, 0.40): "poor",
}

ZONE_FEEDBACK_TEMPLATES = {
    "excellent": "{zone} was the highest-value option from this state.",
    "good":      "{best_zone} offered slightly higher EV (+{ev_loss:.2f}).",
    "suboptimal":"From this state, {best_zone} had significantly higher EV (+{ev_loss:.2f}).",
    "poor":      "{actual_zone} had {p_make:.0%} make probability from this state. "
                 "{best_zone} offered {ev_best:.2f} EV vs {ev_actual:.2f}.",
}

POSITION_FEEDBACK_TEMPLATES = [
    (0.05,  "Good recovery — contacted within neutral zone."),
    (0.15,  "Slightly out of position at contact ({displacement_m:.1f}m from neutral)."),
    (0.30,  "Out of position at contact ({displacement_m:.1f}m from neutral). "
            "Attempted shot from a compromised state."),
    (float("inf"),
            "Severely out of position ({displacement_m:.1f}m from neutral)."),
]


@dataclass
class RecoveryState:
    """Positioning state at contact, required for displacement penalty."""
    contact_pos_m: np.ndarray
    neutral_pos_m: np.ndarray
    movement_vector: np.ndarray
    shot_direction: np.ndarray
    ball_pos_at_contact: np.ndarray
    shot_type: str
    recovery_displacement_m: float     # pre-computed |contact - neutral|


def compute_ev(p_make: float, p_win: float) -> float:
    return p_make * (p_win + 1.0) - 1.0


class EVScorer:
    """
    Computes per-zone EV surface and shot quality score.

    Shot quality is two-dimensional:
    - zone_score: relative quality of target choice (0–1)
    - displacement_penalty: absolute positioning penalty (EV units)
    """

    def __init__(
        self,
        safety_model: ShotSafetyModel,
        win_prob_model: WinProbModel,
        neutral_model: NeutralPositionModel,
    ):
        self.safety_model = safety_model
        self.win_prob_model = win_prob_model
        self.neutral_model = neutral_model

    def compute_ev_surface(
        self,
        base_safety_features: ShotGeometryFeatures,
        rally_context: list[dict],
        dp: float = 0.0,
    ) -> dict[str, float]:
        """EV (adjusted) per zone. dp is a constant offset applied uniformly —
        it does not change which zone is best."""
        ev_surface = {}
        for zone in COURT_ZONES:
            zone_center = ZONE_CENTERS[zone]

            # 2B: geometric margin for this hypothetical target zone
            zone_features = _build_zone_features(base_safety_features, zone_center)
            p_make = self.safety_model.predict(zone_features)["p_make"]

            # 2C: win probability given this zone was the landing target
            rally_with_zone = rally_context + [{"ball_landing_zone": zone}]
            p_win = self.win_prob_model.predict(rally_with_zone)["p_win_point"]

            ev_surface[zone] = round(compute_ev(p_make, p_win) - dp, 4)
        return ev_surface

    def predict_zone_neutrals(
        self,
        base_response_features: ResponseDistributionFeatures,
        striker_pos: np.ndarray,
        striker_vel: np.ndarray,
        t_recovery: float,
    ) -> dict[str, np.ndarray]:
        """Per-zone reachable neutral (2E). Output-only — does NOT enter EV math.
        For each zone, the player's landing is set to that zone's center."""
        neutrals = {}
        for zone in COURT_ZONES:
            zone_center = ZONE_CENTERS[zone]
            feats = replace(
                base_response_features,
                your_landing_x=float(zone_center[0]),
                your_landing_y=float(zone_center[1]),
            )
            result = self.neutral_model.predict_neutral(
                feats, striker_pos, striker_vel, t_recovery,
            )
            neutrals[zone] = result["reachable_neutral_m"]
        return neutrals

    def score_shot(
        self,
        actual_zone: str,
        base_safety_features: ShotGeometryFeatures,
        rally_context: list[dict],
        recovery_state: RecoveryState,
        striker_pos: np.ndarray,
        striker_vel: np.ndarray,
        t_recovery: float,
        shot_id: Optional[str] = None,
    ) -> dict:
        dp = displacement_penalty(
            contact_pos_m=recovery_state.contact_pos_m,
            neutral_pos_m=recovery_state.neutral_pos_m,
            movement_vector=recovery_state.movement_vector,
            shot_direction=recovery_state.shot_direction,
            ball_pos_at_contact=recovery_state.ball_pos_at_contact,
            shot_type=recovery_state.shot_type,
        )

        ev_surface = self.compute_ev_surface(
            base_safety_features, rally_context, recovery_state,
            striker_pos, striker_vel, t_recovery,
        )

        # Base EV values (before displacement penalty)
        ev_actual_base = ev_surface[actual_zone] + dp
        ev_best_zone = max(COURT_ZONES, key=lambda z: ev_surface[z] + dp)
        ev_best_base = ev_surface[ev_best_zone] + dp
        ev_worst_base = min(ev_surface[z] + dp for z in COURT_ZONES)

        ev_loss = ev_best_base - ev_actual_base
        ev_range = ev_best_base - ev_worst_base
        zone_score = 1.0 - (ev_loss / ev_range) if ev_range > 0 else 0.5

        tier = _get_tier(zone_score)
        p_make = self.safety_model.predict(base_safety_features)["p_make"]

        return {
            "shot_id": shot_id,
            "actual_zone": actual_zone,
            "ev_actual": round(ev_surface[actual_zone], 3),
            "ev_actual_base": round(ev_actual_base, 3),
            "ev_best_base": round(ev_best_base, 3),
            "ev_loss": round(ev_loss, 3),
            "best_zone": ev_best_zone,
            "zone_score": round(zone_score, 3),
            "displacement_penalty": round(dp, 3),
            "p_make_actual": round(p_make, 3),
            "feedback_zone": _zone_feedback(
                tier, actual_zone, ev_best_zone, ev_loss,
                ev_best_base, ev_surface[actual_zone], p_make,
            ),
            "feedback_position": _position_feedback(
                dp, recovery_state.recovery_displacement_m
            ),
            "ev_surface": {z: round(v, 3) for z, v in ev_surface.items()},
        }


def _build_zone_features(
    base: ShotGeometryFeatures, zone_center: np.ndarray
) -> ShotGeometryFeatures:
    """Rebuild margin features for a hypothetical target zone center."""
    lateral_margin = min(
        abs(zone_center[0] - (-4.115)),
        abs(zone_center[0] - 4.115),
    )
    depth_margin = min(
        abs(zone_center[1] - 0.0),
        abs(zone_center[1] - 23.77),
    )
    return replace(
        base,
        landing_x=float(zone_center[0]),
        landing_y=float(zone_center[1]),
        lateral_margin_m=lateral_margin,
        depth_margin_m=depth_margin,
        # net_clearance estimated from zone depth — zones near net = lower clearance
        net_clearance_m=max(0.1, (zone_center[1] - 11.89) * 0.05),
    )


def _get_tier(score: float) -> str:
    for (low, high), tier in QUALITY_TIERS.items():
        if low <= score < high:
            return tier
    return "poor"


def _zone_feedback(
    tier: str, actual_zone: str, best_zone: str, ev_loss: float,
    ev_best: float, ev_actual: float, p_make: float,
) -> str:
    return ZONE_FEEDBACK_TEMPLATES[tier].format(
        zone=actual_zone, best_zone=best_zone, ev_loss=ev_loss,
        ev_best=ev_best, ev_actual=ev_actual, p_make=p_make,
        actual_zone=actual_zone,
    )


def _position_feedback(dp: float, displacement_m: float) -> str:
    for threshold, template in POSITION_FEEDBACK_TEMPLATES:
        if dp < threshold:
            return template.format(displacement_m=displacement_m)
    return POSITION_FEEDBACK_TEMPLATES[-1][1].format(displacement_m=displacement_m)
