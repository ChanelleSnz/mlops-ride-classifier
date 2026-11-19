"""Ride-level evaluation shared by the baseline and the deep learning model."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, f1_score

from ride_classifier.data import CLASSES


def ride_level(proba: np.ndarray, ride_ids: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Turn window predictions into one prediction per ride.

    The probabilities of all windows of a ride are averaged,
    and the ride gets the class with the highest average.
    """
    df = pd.DataFrame(proba).assign(ride=ride_ids, y=y)
    grouped = df.groupby("ride")
    y_true = grouped["y"].first().to_numpy()
    y_pred = grouped[list(range(proba.shape[1]))].mean().to_numpy().argmax(axis=1)
    return y_true, y_pred


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Metrics that stay meaningful when classes are imbalanced."""
    labels = list(range(len(CLASSES)))
    f1_per_class = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "f1_reckless": f1_per_class[CLASSES.index("Reckless")],
    }


def report(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    """Detailed per-class report, readable in the terminal."""
    return classification_report(
        y_true, y_pred, labels=list(range(len(CLASSES))), target_names=CLASSES, zero_division=0
    )
