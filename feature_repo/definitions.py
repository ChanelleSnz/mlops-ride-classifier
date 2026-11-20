"""Feast definitions: the ride entity, the ride_stats feature view and the feature service used by training.

The feature names must match src/ride_classifier/features.py (FEATURE_NAMES). They are rebuilt here with the
same rule because Feast runs in its own environment and cannot import the ride_classifier package;
tests/test_feature_repo.py checks both lists against the parquet file.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

from feast import Entity, FeatureService, FeatureView, Field, FileSource
from feast.types import Float32
from feast.value_type import ValueType

REPO_DIR = Path(__file__).resolve().parent
FEATURES_PATH = REPO_DIR / "data" / "ride_features.parquet"

SENSORS = ["ax", "ay", "az", "rx", "ry", "rz"]
FEATURE_NAMES = [
    "duration_s",
    "missing_ratio",
    *[f"{sensor}_{stat}" for sensor in SENSORS for stat in ["mean", "std", "min", "max"]],
    *[f"{name}_mag_{stat}" for name in ["acc", "gyro"] for stat in ["mean", "std", "p95"]],
    "acc_jerk_mean",
]
FEATURE_SERVICE = "ride_classifier_v1"


def definitions(features_path: Path = FEATURES_PATH) -> list:
    """All Feast objects, built for a given offline parquet file (tests use a temporary one)."""
    ride = Entity(
        name="ride",
        join_keys=["ride_id"],
        value_type=ValueType.STRING,
        description="One micromobility ride (GBG500 ride_id)",
    )
    source = FileSource(
        name="ride_stats_source",
        path=str(features_path),
        timestamp_field="event_timestamp",
        description="Written by `python -m ride_classifier.features`",
    )
    ride_stats = FeatureView(
        name="ride_stats",
        entities=[ride],
        ttl=timedelta(0),  # 0 = features never expire (a finished ride does not change)
        schema=[Field(name=name, dtype=Float32) for name in FEATURE_NAMES],
        source=source,
        online=True,
        description="Summary statistics of the IMU sensors over the whole ride",
        tags={"owner": "data-and-model", "team": "ride-classifier"},
    )
    service = FeatureService(
        name=FEATURE_SERVICE,
        features=[ride_stats],
        description="Features used to train the ride-level model logged in MLflow",
    )
    return [ride, source, ride_stats, service]
