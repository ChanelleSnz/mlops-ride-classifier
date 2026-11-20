"""Ride-level model trained on the feature store's training set.

A classic model (random forest) on the 33 ride features served by Feast, evaluated on exactly the same
test rides as the CNN (data.split_rides). train.py logs its scores in the same MLflow run, so the CNN is
compared with a feature-based model, not only with dummy baselines.

Usage: uv run python -m ride_classifier.feature_model
Needs data/processed/ride_features_training.parquet (built by feature_repo/build_training_set.py).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from ride_classifier.data import CLASSES, PROCESSED_DIR, RAW_DIR, SEED, split_rides
from ride_classifier.evaluate import compute_metrics
from ride_classifier.features import FEATURE_NAMES

TRAINING_SET_PATH = PROCESSED_DIR / "ride_features_training.parquet"
PARAMS = {"n_estimators": 300, "min_samples_leaf": 2, "class_weight": "balanced"}


def split_training_set(
    training: pd.DataFrame, labels: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Join the labels to the store features and split with the same rides as the CNN."""
    missing = set(FEATURE_NAMES) - set(training.columns)
    if missing:
        raise ValueError(f"Training set is missing features: {sorted(missing)}")

    label_of = dict(zip(labels["RIDE_ID"], labels["LABEL"], strict=True))
    features = training.set_index("ride_id")[FEATURE_NAMES]
    train_ids, test_ids = split_rides(labels)

    def xy(ride_ids) -> tuple[np.ndarray, np.ndarray]:
        X = features.loc[list(ride_ids)].to_numpy(dtype=np.float32)
        y = np.array([CLASSES.index(label_of[ride_id]) for ride_id in ride_ids])
        return X, y

    return (*xy(train_ids), *xy(test_ids))


def train_feature_model(
    training_set_path: Path = TRAINING_SET_PATH, raw_dir: Path = RAW_DIR
) -> tuple[RandomForestClassifier, dict[str, float]] | None:
    """Train and evaluate the feature model. Returns None when the feature store has not been built."""
    if not training_set_path.exists():
        print(f"No feature store training set ({training_set_path}): feature model skipped")
        return None

    training = pd.read_parquet(training_set_path)
    labels = pd.read_csv(raw_dir / "GBG500_labels.csv")
    X_train, y_train, X_test, y_test = split_training_set(training, labels)

    model = RandomForestClassifier(**PARAMS, random_state=SEED, n_jobs=-1)
    model.fit(X_train, y_train)
    return model, compute_metrics(y_test, model.predict(X_test))


if __name__ == "__main__":
    result = train_feature_model()
    if result:
        for name, value in result[1].items():
            print(f"  {name:<18} {value:.3f}")
