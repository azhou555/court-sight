"""Shot margin safety model — GBT on geometric features predicting P(make)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np

SHOT_TYPES = [
    "forehand_groundstroke", "backhand_groundstroke",
    "forehand_slice", "backhand_slice",
    "forehand_volley", "backhand_volley",
    "overhead", "serve", "lob", "drop_shot",
]


@dataclass
class ShotGeometryFeatures:
    """
    All geometric features required by the margin safety model.
    No pose features — purely about the shot's geometric characteristics.
    """
    # Shot margin geometry (from ball tracking)
    lateral_margin_m: float        # distance from landing x to nearest sideline
    depth_margin_m: float          # distance from landing y to nearest baseline
    net_clearance_m: float         # estimated ball height above net at crossing
    shot_direction_deg: float      # angle relative to court long axis (0=straight, 90=cross)
    landing_x: float               # continuous court x coordinate
    landing_y: float               # continuous court y coordinate

    # Shot and position context
    shot_type: str
    striker_y_m: float             # striker's court depth
    striker_x_m: float             # striker's lateral position
    incoming_depth_m: float        # how deep the incoming ball was

    # Miss estimation flag
    target_zone_estimated: bool = False   # True if target zone was imputed

    def to_vector(self) -> np.ndarray:
        type_onehot = np.eye(len(SHOT_TYPES))[SHOT_TYPES.index(self.shot_type)]
        return np.concatenate([
            [self.lateral_margin_m],
            [self.depth_margin_m],
            [self.net_clearance_m],
            [self.shot_direction_deg / 90.0],    # normalized
            [self.landing_x / 4.115],            # normalized to [-1, 1]
            [self.landing_y / 23.77],            # normalized to [0, 1]
            type_onehot,                          # 10
            [self.striker_y_m / 23.77],
            [self.striker_x_m / 4.115],
            [self.incoming_depth_m / 23.77],
        ])                                        # total: ~20 features


class ShotSafetyModel:
    """
    GBT model estimating P(make | geometric shot features).
    Captures shot selection risk — geometric margin and placement difficulty.
    Positioning risk is computed separately in the displacement penalty.
    """

    def __init__(self, model_path: Optional[str] = None):
        self._model = None
        self._calibrator = None
        if model_path:
            self.load(model_path)

    def predict(self, features: ShotGeometryFeatures) -> dict:
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
            "lateral_margin_m": features.lateral_margin_m,
            "depth_margin_m": features.depth_margin_m,
            "net_clearance_m": features.net_clearance_m,
            "target_zone_estimated": features.target_zone_estimated,
        }

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        # TODO: fit XGBoost + isotonic calibration on held-out split
        raise NotImplementedError

    def load(self, path: str) -> None:
        # TODO: load XGBoost model + calibrator
        raise NotImplementedError

    def save(self, path: str) -> None:
        raise NotImplementedError
