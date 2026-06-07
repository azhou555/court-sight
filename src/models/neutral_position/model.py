"""Neutral position / coverage model — two-stage: response distribution + geometric median."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np

from src.models.execution_prob.model import SHOT_TYPES

# Court-coordinate normalisers (must match ResponseDistributionFeatures.to_vector)
_NORM_X = 4.115
_NORM_Y = 23.77

# Court bounds in meters
COURT_X_MIN, COURT_X_MAX = -4.115, 4.115
COURT_Y_MIN, COURT_Y_MAX = 0.0, 23.77

MAX_SPRINT_SPEED_MS = 4.5   # elite baseline sprint speed
WEISZFELD_ITERS = 100
WEISZFELD_EPS = 1e-5
N_GAUSSIAN_SAMPLES = 200


@dataclass
class ResponseDistributionFeatures:
    """Features for predicting where the opponent will land their response."""
    your_landing_x: float
    your_landing_y: float
    ball_height_at_bounce: float
    opponent_pos_x: float
    opponent_pos_y: float
    opponent_vel_x: float
    opponent_vel_y: float
    opponent_displacement_from_neutral: float
    your_shot_type: str
    rally_depth: int
    prev_shot_direction_1: float    # last shot direction angle (degrees)
    prev_shot_direction_2: float    # shot before that

    def to_vector(self) -> np.ndarray:
        # Shot-type one-hot is appended LAST so the leading scalar indices
        # (esp. col 1 = landing_y, col 7 = displacement) stay stable for the
        # stratified-sigma bucketing in ResponseDistributionModel.
        st = self.your_shot_type if self.your_shot_type in SHOT_TYPES \
            else "forehand_groundstroke"
        type_onehot = np.eye(len(SHOT_TYPES))[SHOT_TYPES.index(st)]
        return np.concatenate([
            [self.your_landing_x / 4.115],
            [self.your_landing_y / 23.77],
            [self.ball_height_at_bounce],
            [self.opponent_pos_x / 4.115],
            [self.opponent_pos_y / 23.77],
            [self.opponent_vel_x / 5.0],
            [self.opponent_vel_y / 5.0],
            [self.opponent_displacement_from_neutral / 5.0],
            [self.rally_depth / 20.0],
            [self.prev_shot_direction_1 / 90.0],
            [self.prev_shot_direction_2 / 90.0],
            type_onehot,                              # 10 dims (SHOT_TYPES)
        ]).astype(np.float32)                          # total: 11 + 10 = 21


class ResponseDistributionModel:
    """
    Two GBT regressors predicting mu_x, mu_y of a 2D Gaussian over
    where the opponent will land their response shot.

    Outputs a 2D Gaussian N(mu, sigma^2 * I) — continuous coordinates,
    not zone labels. Sigma is estimated from calibrated residuals.
    """

    def __init__(self, model_path: Optional[str] = None):
        self._model_x = None    # GBT regressor for mu_x
        self._model_y = None    # GBT regressor for mu_y
        self._sigma_x = 1.5     # default sigma (meters) — updated after training
        self._sigma_y = 2.0
        self._sigma_buckets: Optional[dict] = None   # stratified sigma lookup
        if model_path:
            self.load(model_path)

    def predict(self, features: ResponseDistributionFeatures) -> dict:
        if self._model_x is None or self._model_y is None:
            raise RuntimeError("Model not loaded.")
        x = features.to_vector().reshape(1, -1)
        mu_x = float(self._model_x.predict(x)[0]) * 4.115
        mu_y = float(self._model_y.predict(x)[0]) * 23.77
        return {
            "mu": np.array([mu_x, mu_y]),
            "sigma_x": self._sigma_x,
            "sigma_y": self._sigma_y,
        }

    def train(
        self,
        X: np.ndarray,
        y_x: np.ndarray,
        y_y: np.ndarray,
        eval_fraction: float = 0.2,
        n_estimators: int = 300,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        early_stopping_rounds: int = 20,
        random_state: int = 42,
    ) -> dict:
        """Fit two XGBRegressors predicting normalised (mu_x, mu_y).

        Labels y_x, y_y are RAW court metres; they are normalised internally
        (÷ _NORM_X, ÷ _NORM_Y) so the regressors learn the same scale that
        predict() rescales from. Sigma is estimated from held-out residuals
        and stored in metres.
        """
        from xgboost import XGBRegressor
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_absolute_error, mean_squared_error

        yx = np.asarray(y_x, dtype=np.float32) / _NORM_X
        yy = np.asarray(y_y, dtype=np.float32) / _NORM_Y

        # Hold out an eval set for residual-based sigma estimation
        X_tr, X_ev, yx_tr, yx_ev, yy_tr, yy_ev = train_test_split(
            X, yx, yy, test_size=eval_fraction, random_state=random_state
        )
        # Inner split for early stopping
        X_fit, X_es, yx_fit, yx_es, yy_fit, yy_es = train_test_split(
            X_tr, yx_tr, yy_tr, test_size=0.15, random_state=random_state
        )

        def _make() -> "XGBRegressor":
            return XGBRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="reg:squarederror",
                eval_metric="rmse",
                early_stopping_rounds=early_stopping_rounds,
                random_state=random_state,
                verbosity=0,
            )

        self._model_x = _make()
        self._model_x.fit(X_fit, yx_fit, eval_set=[(X_es, yx_es)], verbose=False)
        self._model_y = _make()
        self._model_y.fit(X_fit, yy_fit, eval_set=[(X_es, yy_es)], verbose=False)

        # Residuals on the held-out eval set (normalised space)
        res_x = yx_ev - self._model_x.predict(X_ev)
        res_y = yy_ev - self._model_y.predict(X_ev)
        self._sigma_x = max(float(np.std(res_x)) * _NORM_X, 0.1)
        self._sigma_y = max(float(np.std(res_y)) * _NORM_Y, 0.1)
        self._sigma_buckets = self._compute_sigma_buckets(X_ev, res_x, res_y)

        # Metrics in metres
        pred_x_m = self._model_x.predict(X_ev) * _NORM_X
        pred_y_m = self._model_y.predict(X_ev) * _NORM_Y
        true_x_m = yx_ev * _NORM_X
        true_y_m = yy_ev * _NORM_Y
        return {
            "n_train": int(len(X_tr)),
            "n_eval": int(len(X_ev)),
            "best_iter_x": int(getattr(self._model_x, "best_iteration", 0) or 0),
            "best_iter_y": int(getattr(self._model_y, "best_iteration", 0) or 0),
            "mae_x": float(mean_absolute_error(true_x_m, pred_x_m)),
            "mae_y": float(mean_absolute_error(true_y_m, pred_y_m)),
            "rmse_x": float(mean_squared_error(true_x_m, pred_x_m) ** 0.5),
            "rmse_y": float(mean_squared_error(true_y_m, pred_y_m) ** 0.5),
            "sigma_x": self._sigma_x,
            "sigma_y": self._sigma_y,
        }

    @staticmethod
    def _compute_sigma_buckets(X_ev: np.ndarray, res_x: np.ndarray,
                               res_y: np.ndarray) -> dict:
        """Stratified residual std by (depth, displacement) context bucket.

        Stored for optional inference-time use; the global sigma stays the
        default. Keys are "{depth}|{disp}" with depth in {short,mid,deep} and
        disp in {low,high}. Buckets with <5 samples are skipped. Sigma is in
        metres; X columns are de-normalised (col 1 = landing_y, col 7 = disp).
        """
        buckets: dict[str, tuple[float, float]] = {}
        depth = X_ev[:, 1] * _NORM_Y
        disp = X_ev[:, 7] * 5.0
        depth_label = np.where(depth < 8.0, "short",
                               np.where(depth < 16.0, "mid", "deep"))
        disp_label = np.where(disp < 2.0, "low", "high")
        for d in ("short", "mid", "deep"):
            for p in ("low", "high"):
                mask = (depth_label == d) & (disp_label == p)
                if mask.sum() >= 5:
                    buckets[f"{d}|{p}"] = (
                        float(np.std(res_x[mask])) * _NORM_X,
                        float(np.std(res_y[mask])) * _NORM_Y,
                    )
        return buckets

    def save(self, path: str | Path) -> None:
        """Save both regressors + sigma to a pickle bundle."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "model_x": self._model_x,
                "model_y": self._model_y,
                "sigma_x": self._sigma_x,
                "sigma_y": self._sigma_y,
                "sigma_buckets": self._sigma_buckets,
            }, f)

    def load(self, path: str | Path) -> None:
        """Load both regressors + sigma from a pickle bundle."""
        with open(path, "rb") as f:
            bundle = pickle.load(f)
        self._model_x = bundle["model_x"]
        self._model_y = bundle["model_y"]
        self._sigma_x = bundle.get("sigma_x", 1.5)
        self._sigma_y = bundle.get("sigma_y", 2.0)
        self._sigma_buckets = bundle.get("sigma_buckets")

    def feature_importance(self) -> dict[str, list]:
        """Gain-based importance per regressor, keyed 'x' and 'y'."""
        if self._model_x is None or self._model_y is None:
            return {}
        from src.models.neutral_position.train import FEATURE_NAMES

        def _imp(model) -> list:
            scores = model.get_booster().get_score(importance_type="gain")
            return sorted(
                ((FEATURE_NAMES[int(k[1:])], float(v))
                 for k, v in scores.items() if int(k[1:]) < len(FEATURE_NAMES)),
                key=lambda t: -t[1],
            )
        return {"x": _imp(self._model_x), "y": _imp(self._model_y)}


def optimal_neutral(
    mu: np.ndarray,
    sigma_x: float,
    sigma_y: float,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Compute the weighted geometric median of a 2D Gaussian response distribution.

    For a unimodal distribution this converges near mu. The Weiszfeld iteration
    adds value when the distribution has heavy tails toward dangerous zones —
    the median biases toward them more than the mean would.
    """
    if rng is None:
        rng = np.random.default_rng()

    cov = np.diag([sigma_x ** 2, sigma_y ** 2])
    samples = rng.multivariate_normal(mu, cov, N_GAUSSIAN_SAMPLES)

    # Clip to valid court bounds
    samples[:, 0] = np.clip(samples[:, 0], COURT_X_MIN, COURT_X_MAX)
    samples[:, 1] = np.clip(samples[:, 1], COURT_Y_MIN, COURT_Y_MAX)

    pos = mu.copy()
    for _ in range(WEISZFELD_ITERS):
        dists = np.linalg.norm(samples - pos, axis=1)
        dists = np.maximum(dists, 1e-6)
        weights = 1.0 / dists
        pos_new = np.average(samples, weights=weights, axis=0)
        if np.linalg.norm(pos_new - pos) < WEISZFELD_EPS:
            break
        pos = pos_new

    return np.clip(pos, [COURT_X_MIN, COURT_Y_MIN], [COURT_X_MAX, COURT_Y_MAX])


def reachable_neutral(
    theoretical: np.ndarray,
    current_pos: np.ndarray,
    current_vel: np.ndarray,
    t_recovery: float,
    max_speed: float = MAX_SPRINT_SPEED_MS,
) -> np.ndarray:
    """
    If theoretical neutral is reachable in t_recovery seconds, return it.
    Otherwise return the closest point on the reachable boundary.
    """
    direction = theoretical - current_pos
    dist = float(np.linalg.norm(direction))
    max_dist = t_recovery * max_speed

    if dist == 0 or dist <= max_dist:
        return theoretical

    return current_pos + (direction / dist) * max_dist


class NeutralPositionModel:
    """
    Full neutral position pipeline: response distribution → geometric median
    → time-constrained reachable neutral.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.response_model = ResponseDistributionModel(model_path)

    def predict_neutral(
        self,
        features: ResponseDistributionFeatures,
        current_pos: np.ndarray,
        current_vel: np.ndarray,
        t_recovery: float,
        rng: Optional[np.random.Generator] = None,
    ) -> dict:
        dist = self.response_model.predict(features)
        mu = dist["mu"]
        sigma_x = dist["sigma_x"]
        sigma_y = dist["sigma_y"]

        theoretical = optimal_neutral(mu, sigma_x, sigma_y, rng)
        reachable = reachable_neutral(theoretical, current_pos, current_vel, t_recovery)

        return {
            "theoretical_neutral_m": theoretical,
            "reachable_neutral_m": reachable,
            "response_mu": mu,
            "response_sigma_x": sigma_x,
            "response_sigma_y": sigma_y,
            "t_recovery": t_recovery,
        }


# Serve regime — simplified lookup for serve + 1 scenarios
SERVE_NEUTRAL_DEFAULTS: dict[tuple[str, str], np.ndarray] = {
    # (side, serve_placement) → default neutral (x, y) in court meters
    ("deuce", "T"):    np.array([-0.5, 1.5]),
    ("deuce", "body"): np.array([0.0,  1.5]),
    ("deuce", "wide"): np.array([1.0,  1.5]),
    ("ad",    "T"):    np.array([0.5,  1.5]),
    ("ad",    "body"): np.array([0.0,  1.5]),
    ("ad",    "wide"): np.array([-1.0, 1.5]),
}


def serve_neutral(side: str, placement: str, pattern_bias: float = 0.0) -> np.ndarray:
    """
    pattern_bias: positive = shift toward deuce side, negative = ad side.
    Populated from per-match serve direction frequency statistics.
    """
    base = SERVE_NEUTRAL_DEFAULTS.get((side, placement), np.array([0.0, 1.5]))
    return base + np.array([pattern_bias, 0.0])
