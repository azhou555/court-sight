"""Tests for displacement penalty computation."""

import numpy as np
import pytest

from src.models.execution_prob.displacement import (
    displacement_penalty,
    NEUTRAL_GRACE_ZONE_M,
    MAX_PENALTY,
    IDEAL_CONTACT_OFFSET,
)


def _penalty(
    contact=None, neutral=None, movement=None,
    shot_dir=None, ball_pos=None, shot_type="forehand_groundstroke"
):
    if contact is None:  contact = np.array([0.0, 5.0])
    if neutral is None:  neutral = np.array([0.0, 5.0])
    if movement is None: movement = np.array([0.0, 0.0])
    if shot_dir is None: shot_dir = np.array([0.0, 1.0])
    if ball_pos is None: ball_pos = np.array([1.1, 5.4])
    return displacement_penalty(contact, neutral, movement, shot_dir, ball_pos, shot_type)


def test_at_neutral_no_body_no_misalignment():
    """Player at neutral, stationary, ball at ideal contact point → minimal penalty."""
    ideal = IDEAL_CONTACT_OFFSET["forehand_groundstroke"]
    contact = np.array([0.0, 5.0])
    ball_pos = contact + ideal
    p = _penalty(contact=contact, neutral=contact, ball_pos=ball_pos)
    assert p < 0.05


def test_within_grace_zone():
    """Less than 1m from neutral → no radial penalty."""
    p_inside = _penalty(contact=np.array([0.5, 5.0]), neutral=np.array([0.0, 5.0]))
    p_outside = _penalty(contact=np.array([2.0, 5.0]), neutral=np.array([0.0, 5.0]))
    assert p_inside < p_outside


def test_body_shot_increases_penalty():
    """Ball arriving at torso should add body penalty."""
    normal_ball = np.array([1.1, 5.4])   # arm's length
    body_ball = np.array([0.1, 5.05])    # in torso
    p_normal = _penalty(ball_pos=normal_ball)
    p_body = _penalty(ball_pos=body_ball)
    assert p_body > p_normal


def test_wrong_momentum_increases_penalty():
    """Moving away from shot direction should increase penalty."""
    forward = np.array([0.0, 2.0])
    backward = np.array([0.0, -2.0])
    shot_dir = np.array([0.0, 1.0])
    p_forward = _penalty(movement=forward, shot_dir=shot_dir)
    p_backward = _penalty(movement=backward, shot_dir=shot_dir)
    assert p_backward > p_forward


def test_penalty_capped_at_max():
    """Extreme displacement should not exceed MAX_PENALTY."""
    far_contact = np.array([8.0, 0.0])
    p = _penalty(contact=far_contact, neutral=np.array([0.0, 5.0]))
    assert p <= MAX_PENALTY


def test_penalty_non_negative():
    for shot_type in IDEAL_CONTACT_OFFSET:
        p = _penalty(shot_type=shot_type)
        assert p >= 0.0, f"Negative penalty for {shot_type}"


def test_all_shot_types_have_offsets():
    shot_types = [
        "forehand_groundstroke", "backhand_groundstroke",
        "forehand_slice", "backhand_slice",
        "forehand_volley", "backhand_volley",
        "overhead", "serve", "lob", "drop_shot",
    ]
    for st in shot_types:
        assert st in IDEAL_CONTACT_OFFSET
