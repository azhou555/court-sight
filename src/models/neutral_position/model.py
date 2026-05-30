"""Neutral position / coverage model — two-stage: response distribution + geometric median."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np

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
        return np.array([
            self.your_landing_x / 4.115,
            self.your_landing_y / 23.77,
            self.ball_height_at_bounce,
            self.opponent_pos_x / 4.115,
            self.opponent_pos_y / 23.77,
            self.opponent_vel_x / 5.0,
            self.opponent_vel_y / 5.0,
            self.opponent_displacement_from_neutral / 5.0,
            self.rally_depth / 20.0,
            self.prev_shot_direction_1 / 90.0,
            self.prev_shot_direction_2 / 90.0,
        ])


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
    ) -> None:
        # TODO: fit two XGBRegressors, estimate sigma from residuals
        raise NotImplementedError

    def load(self, path: str) -> None:
        raise NotImplementedError

    def save(self, path: str) -> None:
        raise NotImplementedError


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
