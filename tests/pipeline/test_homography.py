"""Tests for court homography utilities."""

import numpy as np
import pytest

from src.pipeline.homography.detector import pixel_to_court, COURT_KEYPOINTS_M


def test_pixel_to_court_identity():
    """With identity homography, pixel_to_court should return input coordinates."""
    H = np.eye(3, dtype=np.float64)
    pt = np.array([100.0, 200.0])
    result = pixel_to_court(pt, H)
    np.testing.assert_allclose(result, pt, atol=1e-6)


def test_pixel_to_court_scale():
    """Homography scaling x by 2 should double x coordinate."""
    H = np.diag([2.0, 1.0, 1.0])
    pt = np.array([5.0, 3.0])
    result = pixel_to_court(pt, H)
    np.testing.assert_allclose(result, [10.0, 3.0], atol=1e-6)


def test_court_keypoints_shape():
    assert COURT_KEYPOINTS_M.shape == (14, 2)


def test_court_keypoints_symmetry():
    """Baseline corners should be symmetric about x=0."""
    left_x = COURT_KEYPOINTS_M[0, 0]
    right_x = COURT_KEYPOINTS_M[1, 0]
    assert abs(left_x + right_x) < 1e-6
