"""Displacement penalty — analytically computed positioning quality score."""

from __future__ import annotations

import numpy as np

# Ideal contact offset (dx, dy) relative to player hip midpoint in court meters.
# dx: positive = toward dominant/hitting side
# dy: positive = forward from hip
IDEAL_CONTACT_OFFSET: dict[str, np.ndarray] = {
    "forehand_groundstroke": np.array([1.1,  0.4]),
    "backhand_groundstroke": np.array([-0.7, 0.5]),
    "forehand_slice":        np.array([1.0,  0.5]),
    "backhand_slice":        np.array([-0.6, 0.6]),
    "forehand_volley":       np.array([0.9,  0.7]),
    "backhand_volley":       np.array([-0.6, 0.7]),
    "overhead":              np.array([0.3,  0.0]),
    "serve":                 np.array([0.3,  0.0]),
    "lob":                   np.array([0.8,  0.3]),
    "drop_shot":             np.array([0.9,  0.5]),
}

BODY_SHOT_RADIUS_M = 0.45      # ball within this distance of torso = body shot
NEUTRAL_GRACE_ZONE_M = 1.0     # no penalty within 1m of neutral
STEEP_PENALTY_THRESHOLD_M = 3.0
MAX_PENALTY = 0.60


def displacement_penalty(
    contact_pos_m: np.ndarray,
    neutral_pos_m: np.ndarray,
    movement_vector: np.ndarray,
    shot_direction: np.ndarray,
    ball_pos_at_contact: np.ndarray,
    shot_type: str,
    w_radial: float = 0.06,
    w_body: float = 0.10,
    w_momentum: float = 0.05,
    w_contact: float = 0.05,
) -> float:
    """
    Compute the displacement penalty for a shot in EV units [0.0, MAX_PENALTY].

    Parameters
    ----------
    contact_pos_m : (2,) player position at contact in court meters
    neutral_pos_m : (2,) predicted neutral position at contact time
    movement_vector : (2,) player velocity in m/s at contact
    shot_direction : (2,) unit vector in direction of shot
    ball_pos_at_contact : (2,) ball position at contact frame in court meters
    shot_type : declared shot type string
    """
    # 1. Radial distance from neutral (grace zone within 1m)
    dist = float(np.linalg.norm(contact_pos_m - neutral_pos_m))
    radial = max(0.0, dist - NEUTRAL_GRACE_ZONE_M) * w_radial
    if dist > STEEP_PENALTY_THRESHOLD_M:
        radial += (dist - STEEP_PENALTY_THRESHOLD_M) * w_radial * 2.0

    # 2. Body shot — ball arriving into the torso at contact
    torso_to_ball = ball_pos_at_contact - contact_pos_m
    body = w_body if float(np.linalg.norm(torso_to_ball)) < BODY_SHOT_RADIUS_M else 0.0

    # 3. Momentum misalignment with shot direction
    speed = float(np.linalg.norm(movement_vector))
    if speed > 0.3:
        shot_dir_norm = shot_direction / (np.linalg.norm(shot_direction) + 1e-9)
        alignment = float(np.dot(movement_vector / speed, shot_dir_norm))
        momentum = max(0.0, -alignment) * w_momentum
    else:
        momentum = 0.0

    # 4. Contact point deviation from ideal for this wing/shot type
    ideal = IDEAL_CONTACT_OFFSET.get(shot_type, np.array([1.0, 0.4]))
    contact_dev = float(np.linalg.norm(torso_to_ball - ideal)) * w_contact

    return min(radial + body + momentum + contact_dev, MAX_PENALTY)
