"""Execution probability model — XGBoost over biomechanical + context features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np

COURT_ZONES = [
    "T_deep", "C_deep", "W_deep",
    "T_mid",  "C_mid",  "W_mid",
    "T_short","C_short","W_short",
]
SHOT_TYPES = [
    "forehand_groundstroke", "backhand_groundstroke",
    "forehand_slice", "backhand_slice",
    "forehand_volley", "backhand_volley",
    "overhead", "serve", "lob", "drop_shot",
]


@dataclass
class ExecutionProbFeatures:
    """All features required by the execution probability model."""
    # Biomechanical state
    keypoints_normalized: np.ndarray    # (17, 2) hip-centered
    joint_angles: np.ndarray            # (6,) elbow, shoulder, hip angles
    movement_vector: np.ndarray         # (2,) m/s
    balance_metric: float               # lateral displacement from CoM baseline

    # Shot context
    shot_type: str
    target_zone: str
    distance_to_sideline: float         # meters
    distance_to_net: float              # meters (depth of target)
    player_x_position: float            # meters from center
    player_y_position: float            # meters from baseline

    # Rally context
    incoming_ball_speed_rel: float      # relative to player average
    rally_length: int                   # shot index in point

    def to_vector(self) -> np.ndarray:
        shot_onehot = np.eye(len(SHOT_TYPES))[SHOT_TYPES.index(self.shot_type)]
        zone_onehot = np.eye(len(COURT_ZONES))[COURT_ZONES.index(self.target_zone)]
        return np.concatenate([
            self.keypoints_normalized.ravel(),          # 34
            self.joint_angles,                          # 6
            self.movement_vector,                       # 2
            [self.balance_metric],                      # 1
            shot_onehot,                                # 10
            zone_onehot,                                # 9
            [self.distance_to_sideline],                # 1
            [self.distance_to_net],                     # 1
            [self.player_x_position],                   # 1
            [self.player_y_position],                   # 1
            [self.incoming_ball_speed_rel],             # 1
            [self.rally_length / 30.0],                 # 1 (normalized)
        ])                                              # total: 68


class ExecutionProbModel:
    """
    Gradient-boosted trees model estimating P(make | biomechanical state, shot context).
    Wraps XGBoost with isotonic regression calibration.
    """

    def __init__(self, model_path: Optional[str] = None):
        self._model = None
        self._calibrator = None
        if model_path:
            self.load(model_path)

    def predict(self, features: ExecutionProbFeatures) -> dict:
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() or train() first.")
        x = features.to_vector().reshape(1, -1)
        raw_prob = float(self._model.predict_proba(x)[0, 1])
        calibrated = (
            float(self._calibrator.transform([raw_prob])[0])
            if self._calibrator else raw_prob
        )
        return {
            "p_make": calibrated,
            "p_miss": 1.0 - calibrated,
        }

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        # TODO: fit XGBoost + isotonic calibration on held-out split
        raise NotImplementedError

    def load(self, path: str) -> None:
        # TODO: load XGBoost model + calibrator from disk
        raise NotImplementedError

    def save(self, path: str) -> None:
        # TODO: persist model + calibrator
        raise NotImplementedError
