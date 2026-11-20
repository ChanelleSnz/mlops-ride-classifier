"""Feast definitions and pipeline, on a temporary store filled with fake features.

Run: uv run --project feature_repo pytest feature_repo
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from feast import FeatureStore

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from build_training_set import build_training_set  # noqa: E402
from definitions import FEATURE_NAMES, FEATURE_SERVICE, definitions  # noqa: E402

RIDES = ["ride_a", "ride_b", "ride_c"]


@pytest.fixture
def store(tmp_path: Path) -> FeatureStore:
    """A local Feast store in a temporary folder, with 3 fake rides in its offline parquet file."""
    rng = np.random.default_rng(0)
    features = pd.DataFrame(rng.normal(size=(len(RIDES), len(FEATURE_NAMES))).astype("float32"), columns=FEATURE_NAMES)
    features.insert(0, "ride_id", RIDES)
    features["event_timestamp"] = pd.Timestamp("2026-11-01", tz="UTC")
    features_path = tmp_path / "ride_features.parquet"
    features.to_parquet(features_path, index=False)

    (tmp_path / "feature_store.yaml").write_text(
        "project: ride_classifier_test\n"
        "provider: local\n"
        "registry: registry.db\n"
        "online_store: {type: sqlite, path: online_store.db}\n"
        "offline_store: {type: file}\n"
        "entity_key_serialization_version: 3\n"
    )
    feature_store = FeatureStore(repo_path=str(tmp_path))
    feature_store.apply(definitions(features_path))
    feature_store.features = features  # keep the expected values for the assertions
    return feature_store


def test_feature_names_are_unique_and_complete():
    assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES)) == 33


def test_feature_service_serves_the_ride_stats_view(store):
    service = store.get_feature_service(FEATURE_SERVICE)
    assert [projection.name for projection in service.feature_view_projections] == ["ride_stats"]


def test_training_set_has_one_row_per_ride_with_the_stored_values(store):
    training = build_training_set(store, pd.Series(RIDES[::-1]))
    assert training["ride_id"].tolist() == RIDES
    expected = store.features.set_index("ride_id").loc[RIDES, FEATURE_NAMES]
    np.testing.assert_allclose(training[FEATURE_NAMES].to_numpy(), expected.to_numpy(), rtol=1e-6)


def test_training_set_fails_for_a_ride_without_features(store):
    with pytest.raises(ValueError, match="no features"):
        build_training_set(store, pd.Series(["ride_a", "unknown_ride"]))


def test_online_store_returns_the_latest_features(store):
    store.materialize_incremental(end_date=datetime.now(UTC))
    online = store.get_online_features(
        features=store.get_feature_service(FEATURE_SERVICE), entity_rows=[{"ride_id": "ride_b"}]
    ).to_dict()
    expected = store.features.set_index("ride_id").loc["ride_b", "acc_mag_mean"]
    assert online["acc_mag_mean"][0] == pytest.approx(expected)
