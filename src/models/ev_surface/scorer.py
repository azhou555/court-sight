"""EV surface computation and shot quality scoring."""

from __future__ import annotations

from typing import Optional
import numpy as np

from src.models.execution_prob.model import ExecutionProbModel, ExecutionProbFeatures
from src.models.win_prob.model import WinProbModel

COURT_ZONES = [
    "T_deep", "C_deep", "W_deep",
    "T_mid",  "C_mid",  "W_mid",
    "T_short","C_short","W_short",
]

QUALITY_TIERS = {
    (0.85, 1.01): "excellent",
    (0.65, 0.85): "good",
    (0.40, 0.65): "suboptimal",
    (0.00, 0.40): "poor",
}

FEEDBACK_TEMPLATES = {
    "excellent": "Strong choice — {actual_zone} was the highest-value option.",
    "good": "Reasonable choice. {best_zone} offered slightly higher EV ({ev_delta:.2f}).",
    "suboptimal": "From this position, {best_zone} had significantly higher EV (+{ev_delta:.2f}).",
    "poor": (
        "Low-percentage attempt. {actual_zone} had {p_make:.0%} execution probability from "
        "this state. {best_zone} offered {ev_best:.2f} EV vs {ev_actual:.2f}."
    ),
}


def compute_ev(p_make: float, p_win_given_make: float) -> float:
    return p_make * (p_win_given_make + 1.0) - 1.0


class EVScorer:
    """
    Combines execution probability and win probability models to compute
    an EV surface and shot quality score for any game state.
    """

    def __init__(
        self,
        execution_model: ExecutionProbModel,
        win_prob_model: WinProbModel,
    ):
        self.execution_model = execution_model
        self.win_prob_model = win_prob_model

    def compute_ev_surface(
        self,
        base_features: ExecutionProbFeatures,
        rally_context: list[dict],
    ) -> dict[str, float]:
        ev_surface = {}
        for zone in COURT_ZONES:
            features = ExecutionProbFeatures(
                **{**base_features.__dict__, "target_zone": zone}
            )
            exec_result = self.execution_model.predict(features)
            p_make = exec_result["p_make"]

            rally_with_zone = rally_context + [{"ball_landing_zone": zone}]
            win_result = self.win_prob_model.predict(rally_with_zone)
            p_win = win_result["p_win_point"]

            ev_surface[zone] = compute_ev(p_make, p_win)
        return ev_surface

    def score_shot(
        self,
        actual_zone: str,
        base_features: ExecutionProbFeatures,
        rally_context: list[dict],
        shot_id: Optional[str] = None,
    ) -> dict:
        ev_surface = self.compute_ev_surface(base_features, rally_context)

        ev_actual = ev_surface[actual_zone]
        best_zone = max(ev_surface, key=ev_surface.__getitem__)
        ev_best = ev_surface[best_zone]
        ev_worst = min(ev_surface.values())

        ev_loss = ev_best - ev_actual
        ev_range = ev_best - ev_worst
        normalized_score = 1.0 - (ev_loss / ev_range) if ev_range > 0 else 0.5

        tier = _get_tier(normalized_score)
        exec_result = self.execution_model.predict(base_features)
        feedback = FEEDBACK_TEMPLATES[tier].format(
            actual_zone=actual_zone,
            best_zone=best_zone,
            ev_delta=ev_loss,
            ev_best=ev_best,
            ev_actual=ev_actual,
            p_make=exec_result["p_make"],
        )

        return {
            "shot_id": shot_id,
            "actual_zone": actual_zone,
            "ev_actual": round(ev_actual, 3),
            "ev_best": round(ev_best, 3),
            "ev_loss": round(ev_loss, 3),
            "best_zone": best_zone,
            "normalized_score": round(normalized_score, 3),
            "tier": tier,
            "p_make_actual": round(exec_result["p_make"], 3),
            "feedback": feedback,
            "ev_surface": {z: round(v, 3) for z, v in ev_surface.items()},
        }


def _get_tier(score: float) -> str:
    for (low, high), tier in QUALITY_TIERS.items():
        if low <= score < high:
            return tier
    return "poor"
