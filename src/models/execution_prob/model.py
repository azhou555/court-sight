"""Shot margin safety model — GBT on geometric features predicting P(make).

Estimates P(make | shot geometry, shot type): the probability a shot lands
in court given the geometric characteristics of the attempt.  This is the
risk side of the EV equation — shot selection risk — not execution quality.

See docs/architecture/06_execution_probability.md for the full design.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

SHOT_TYPES = [
    "forehand_groundstroke", "backhand_groundstroke",
    "forehand_slice", "backhand_slice",
    "forehand_volley", "backhand_volley",
    "overhead", "serve", "lob", "drop_shot",
]

# Zone centre coordinates (court meters) used when exact landing coords unavailable
_ZONE_CENTERS: dict[str, tuple[float, float]] = {
    "T_deep":  (-2.6, 20.6), "C_deep":  (0.0, 20.6), "W_deep":  (2.6, 20.6),
    "T_mid":   (-2.6, 14.6), "C_mid":   (0.0, 14.6), "W_mid":   (2.6, 14.6),
    "T_short": (-2.6,  5.9), "C_short": (0.0,  5.9), "W_short": (2.6,  5.9),
}

_SINGLES_X_LIMIT = 4.115   # singles sideline
_FAR_BASELINE_Y  = 23.77


@dataclass
class ShotGeometryFeatures:
    """All geometric features for the margin safety model.

    No pose features — purely about shot geometry and court context.
    """

    # Shot margin geometry (from ball tracking + zone)
    lateral_margin_m: float     # distance from landing x to nearest sideline
    depth_margin_m: float       # distance from landing y to far baseline
    net_clearance_m: float      # estimated ball height at net (0 if unavailable)
    shot_direction_deg: float   # angle relative to court long axis
    landing_x: float            # continuous court x (from zone centre if approx)
    landing_y: float            # continuous court y

    # Shot and position context
    shot_type: str              # must be in SHOT_TYPES
    striker_y_m: float
    striker_x_m: float
    incoming_depth_m: float     # proxy: opponent y position at contact

    # Quality flag
    target_zone_estimated: bool = False   # True if landing coords approximated from zone

    def to_vector(self) -> np.ndarray:
        """Flat numpy feature vector for GBT input."""
        if self.shot_type not in SHOT_TYPES:
            shot_type = "forehand_groundstroke"  # safe fallback
        else:
            shot_type = self.shot_type
        type_onehot = np.eye(len(SHOT_TYPES))[SHOT_TYPES.index(shot_type)]
        return np.concatenate([
            [self.lateral_margin_m],
            [self.depth_margin_m],
            [self.net_clearance_m],
            [self.shot_direction_deg / 90.0],
            [self.landing_x / _SINGLES_X_LIMIT],
            [self.landing_y / _FAR_BASELINE_Y],
            type_onehot,                             # 10 dims
            [self.striker_y_m / _FAR_BASELINE_Y],
            [self.striker_x_m / _SINGLES_X_LIMIT],
            [self.incoming_depth_m / _FAR_BASELINE_Y],
        ]).astype(np.float32)                        # total: 6 + 10 + 3 = 19 dims


def zone_to_coords(zone: Optional[str]) -> tuple[float, float]:
    """Return approximate (x, y) court metres for a landing zone label."""
    if zone is None:
        return 0.0, 20.0
    return _ZONE_CENTERS.get(zone, (0.0, 20.0))


class ShotSafetyModel:
    """P(make | shot geometry, shot type) via XGBoost + isotonic calibration.

    Usage:
        model = ShotSafetyModel()
        model.train(X, y)          # numpy arrays
        model.save("checkpoints/shot_safety/model.pkl")
        result = model.predict(features)  # ShotGeometryFeatures instance
    """

    def __init__(self, model_path: Optional[str] = None):
        self._xgb = None
        self._calibrator = None
        if model_path:
            self.load(model_path)

    # ── Inference ────────────────────────────────────────────────────────────

    def predict(self, features: ShotGeometryFeatures) -> dict:
        """Return calibrated P(make) and diagnostics."""
        if self._xgb is None:
            raise RuntimeError("Model not loaded. Call load() or train() first.")
        x = features.to_vector().reshape(1, -1)
        raw = float(self._xgb.predict_proba(x)[0, 1])
        p_make = (
            float(self._calibrator.transform([raw])[0])
            if self._calibrator is not None
            else raw
        )
        p_make = float(np.clip(p_make, 0.0, 1.0))
        return {
            "p_make":               p_make,
            "p_miss":               1.0 - p_make,
            "lateral_margin_m":     features.lateral_margin_m,
            "depth_margin_m":       features.depth_margin_m,
            "net_clearance_m":      features.net_clearance_m,
            "target_zone_estimated": features.target_zone_estimated,
        }

    # ── Training ─────────────────────────────────────────────────────────────

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cal_fraction: float = 0.2,
        n_estimators: int = 500,
        max_depth: int = 5,
        learning_rate: float = 0.05,
        early_stopping_rounds: int = 30,
        random_state: int = 42,
    ) -> dict:
        """Fit XGBClassifier + isotonic calibration.

        Args:
            X:  (N, F) feature matrix from ShotGeometryFeatures.to_vector().
            y:  (N,)   binary labels — 1 = shot made, 0 = miss.
            cal_fraction: Fraction held out for isotonic calibration.

        Returns:
            dict with train/val metrics.
        """
        from xgboost import XGBClassifier
        from sklearn.isotonic import IsotonicRegression
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss

        # Split: train for XGB, calibration for isotonic regression
        X_tr, X_cal, y_tr, y_cal = train_test_split(
            X, y, test_size=cal_fraction, random_state=random_state, stratify=y
        )
        # Further split train into train/eval for early stopping
        X_fit, X_eval, y_fit, y_eval = train_test_split(
            X_tr, y_tr, test_size=0.15, random_state=random_state, stratify=y_tr
        )

        self._xgb = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            early_stopping_rounds=early_stopping_rounds,
            random_state=random_state,
            verbosity=0,
        )
        self._xgb.fit(
            X_fit, y_fit,
            eval_set=[(X_eval, y_eval)],
            verbose=False,
        )

        # Isotonic calibration on the held-out calibration set
        raw_cal = self._xgb.predict_proba(X_cal)[:, 1]
        self._calibrator = IsotonicRegression(out_of_bounds="clip")
        self._calibrator.fit(raw_cal, y_cal)

        # Evaluate on calibration set after calibration
        cal_probs = np.clip(self._calibrator.transform(raw_cal), 0.0, 1.0)
        metrics = {
            "n_train": len(y_tr),
            "n_cal":   len(y_cal),
            "best_iteration": self._xgb.best_iteration,
            "cal_auc":    float(roc_auc_score(y_cal, cal_probs)),
            "cal_logloss": float(log_loss(y_cal, cal_probs)),
            "cal_brier":  float(brier_score_loss(y_cal, cal_probs)),
        }
        return metrics

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Save XGB model + calibrator to a pickle file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"xgb": self._xgb, "calibrator": self._calibrator}, f)

    def load(self, path: str | Path) -> None:
        """Load XGB model + calibrator from a pickle file."""
        with open(path, "rb") as f:
            bundle = pickle.load(f)
        self._xgb = bundle["xgb"]
        self._calibrator = bundle["calibrator"]

    def feature_importance(self) -> dict[str, float]:
        """Return feature importance by SHAP-style gain, keyed by feature name."""
        if self._xgb is None:
            return {}
        from src.models.execution_prob.train import FEATURE_NAMES
        scores = self._xgb.get_booster().get_score(importance_type="gain")
        # Booster uses 'f0', 'f1', ... keys
        return {
            FEATURE_NAMES[int(k[1:])]: float(v)
            for k, v in scores.items()
            if int(k[1:]) < len(FEATURE_NAMES)
        }
