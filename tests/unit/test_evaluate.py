"""Ride-level evaluation and metrics."""

import numpy as np
import pytest

from ride_classifier.evaluate import compute_metrics, ride_level


def test_ride_level_averages_window_probabilities():
    proba = np.array(
        [
            [0.9, 0.1, 0, 0, 0],  # ride A, window 1
            [0.2, 0.8, 0, 0, 0],  # ride A, window 2  -> average favours class 0
            [0.0, 0.0, 0, 1, 0],  # ride B
        ]
    )
    y_true, y_pred = ride_level(proba, np.array(["A", "A", "B"]), np.array([0, 0, 3]))
    assert y_true.tolist() == [0, 3]
    assert y_pred.tolist() == [0, 3]


def test_perfect_predictions_score_one():
    y = np.array([0, 1, 2, 3, 4])
    metrics = compute_metrics(y, y)
    assert metrics == pytest.approx({"accuracy": 1, "balanced_accuracy": 1, "macro_f1": 1, "f1_reckless": 1})


def test_majority_class_has_high_accuracy_but_low_macro_f1():
    y_true = np.array([0] * 8 + [1, 4])
    y_pred = np.zeros_like(y_true)
    metrics = compute_metrics(y_true, y_pred)
    assert metrics["accuracy"] == pytest.approx(0.8)
    assert metrics["macro_f1"] < 0.2
    assert metrics["f1_reckless"] == 0
