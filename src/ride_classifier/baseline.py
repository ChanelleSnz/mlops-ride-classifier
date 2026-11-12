"""Naive baselines: the score a model must beat to be better than chance."""

from __future__ import annotations

import numpy as np
from sklearn.dummy import DummyClassifier

from ride_classifier.data import PROCESSED_DIR, SEED
from ride_classifier.evaluate import compute_metrics, ride_level

STRATEGIES = {
    "most_frequent": "always predicts the majority class (Safe)",
    "stratified": "predicts at random, following the class proportions",
}


def run_baselines() -> dict[str, dict[str, float]]:
    data = np.load(PROCESSED_DIR / "dataset.npz")
    X_train = np.zeros((len(data["y_train"]), 1))  # a dummy model ignores the input
    X_test = np.zeros((len(data["y_test"]), 1))

    results = {}
    for strategy, description in STRATEGIES.items():
        model = DummyClassifier(strategy=strategy, random_state=SEED)
        model.fit(X_train, data["y_train"])
        proba = model.predict_proba(X_test)
        y_true, y_pred = ride_level(proba, data["ride_test"], data["y_test"])
        results[strategy] = compute_metrics(y_true, y_pred)
        print(f"\n{strategy}: {description}")
        for name, value in results[strategy].items():
            print(f"  {name:<18} {value:.3f}")
    return results


if __name__ == "__main__":
    run_baselines()
