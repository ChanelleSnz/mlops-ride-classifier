"""Ride-level features (feature store offline source) and the model trained on them."""

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from ride_classifier.data import CLASSES, SENSORS
from ride_classifier.feature_model import split_training_set, train_feature_model
from ride_classifier.features import FEATURE_NAMES, build_feature_table, ride_features


def test_feature_names_are_unique_and_match_the_feast_definitions():
    # feature_repo/definitions.py rebuilds the same 33 names (Feast runs in its own environment)
    assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES)) == 33


def test_ride_features_on_a_known_signal():
    ride = pd.DataFrame({"time_index": [0, 100, 200, 300], **{s: [1.0, 2.0, 3.0, np.nan] for s in SENSORS}})
    features = ride_features(ride)
    assert set(features) == set(FEATURE_NAMES)
    assert features["duration_s"] == pytest.approx(0.4)  # 4 measurements at 10 per second
    assert features["missing_ratio"] == pytest.approx(0.25)
    assert features["ax_max"] == pytest.approx(3.0)  # the missing last value is filled with the previous one
    assert features["acc_mag_mean"] == pytest.approx(np.sqrt(3) * 9 / 4)  # norm of (v, v, v) = v·√3


def test_ride_features_are_insensitive_to_row_order(synthetic_ride):
    shuffled = synthetic_ride.sample(frac=1, random_state=0)
    assert ride_features(shuffled) == pytest.approx(ride_features(synthetic_ride))


def test_feature_table_is_in_the_feast_format(synthetic_ride):
    rides = pd.concat([synthetic_ride.assign(ride_id="b"), synthetic_ride.head(700).assign(ride_id="a")])
    computed_at = datetime(2026, 11, 20, tzinfo=UTC)
    table = build_feature_table(rides, computed_at)
    assert table.columns.tolist() == ["ride_id", *FEATURE_NAMES, "event_timestamp"]
    assert table["ride_id"].tolist() == ["a", "b"]  # one row per ride
    assert (table[FEATURE_NAMES].dtypes == "float32").all()
    assert (table["event_timestamp"] == pd.Timestamp(computed_at)).all()


def fake_store_training_set(n_per_class: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Labels + feature store training set where the class is easy to read in every feature."""
    rng = np.random.default_rng(0)
    ride_ids = [f"ride_{i}" for i in range(n_per_class * len(CLASSES))]
    labels = pd.DataFrame({"RIDE_ID": ride_ids, "LABEL": [c for c in CLASSES for _ in range(n_per_class)]})
    class_index = np.array([CLASSES.index(label) for label in labels["LABEL"]])
    values = class_index[:, None] * 10 + rng.normal(size=(len(ride_ids), len(FEATURE_NAMES)))
    training = pd.DataFrame(values, columns=FEATURE_NAMES).assign(ride_id=ride_ids)
    return training, labels


def test_split_training_set_uses_the_same_rides_as_the_cnn():
    training, labels = fake_store_training_set()
    X_train, y_train, X_test, y_test = split_training_set(training, labels)
    assert X_train.shape == (35, len(FEATURE_NAMES)) and X_test.shape == (15, len(FEATURE_NAMES))
    assert sorted(np.bincount(y_test)) == [3, 3, 3, 3, 3]  # stratified 30 % test split


def test_split_training_set_rejects_missing_features():
    training, labels = fake_store_training_set()
    with pytest.raises(ValueError, match="missing features"):
        split_training_set(training.drop(columns="duration_s"), labels)


def test_feature_model_learns_an_easy_signal(tmp_path):
    training, labels = fake_store_training_set()
    training.to_parquet(tmp_path / "training.parquet")
    labels.to_csv(tmp_path / "GBG500_labels.csv", index=False)
    _, metrics = train_feature_model(tmp_path / "training.parquet", raw_dir=tmp_path)
    assert metrics["macro_f1"] == pytest.approx(1.0)


def test_feature_model_is_skipped_when_the_store_was_not_built(tmp_path):
    assert train_feature_model(tmp_path / "missing.parquet", raw_dir=tmp_path) is None
