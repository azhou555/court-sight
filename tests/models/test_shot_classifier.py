"""Tests for the shot type classifier model."""

import torch
import pytest

from src.models.shot_classifier.model import ShotClassifierTCN, SHOT_TYPES, STROKE_WINDOW, INPUT_DIM, NUM_CLASSES


def make_random_input(batch_size: int = 2):
    return torch.randn(batch_size, STROKE_WINDOW, INPUT_DIM)


def test_output_shape():
    model = ShotClassifierTCN()
    x = make_random_input(batch_size=4)
    out = model(x)
    assert out.shape == (4, NUM_CLASSES)


def test_output_is_logits():
    model = ShotClassifierTCN()
    x = make_random_input()
    out = model(x)
    # Logits — not bounded to [0,1]
    assert out.dtype == torch.float32


def test_predict_returns_valid_class():
    model = ShotClassifierTCN()
    model.eval()
    keypoints = torch.randn(STROKE_WINDOW, 17, 2)
    result = model.predict(keypoints)
    assert result["predicted_class"] in SHOT_TYPES
    assert 0.0 <= result["confidence"] <= 1.0
    assert abs(sum(result["class_probabilities"].values()) - 1.0) < 1e-4


def test_gradient_flow():
    model = ShotClassifierTCN()
    x = make_random_input()
    loss = model(x).sum()
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
