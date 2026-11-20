"""Ride-level features for the feature store.

One row per ride: summary statistics of each sensor and of the acceleration / rotation
magnitudes. They are written as a parquet file, which is the offline source of the Feast
feature view defined in feature_repo/ (Feast runs in its own environment, see feature_repo/README.md).

Usage: uv run python -m ride_classifier.features [--raw-dir data/raw]
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ride_classifier.data import RAW_DIR, ROOT, SENSORS, clean_ride

FEATURE_REPO = ROOT / "feature_repo"
FEATURES_PATH = FEATURE_REPO / "data" / "ride_features.parquet"
SAMPLING_HZ = 10  # 10 measurements per second

STATS = ["mean", "std", "min", "max"]
MAGNITUDES = {"acc": ["ax", "ay", "az"], "gyro": ["rx", "ry", "rz"]}
MAGNITUDE_STATS = ["mean", "std", "p95"]

FEATURE_NAMES = [
    "duration_s",
    "missing_ratio",
    *[f"{sensor}_{stat}" for sensor in SENSORS for stat in STATS],
    *[f"{name}_mag_{stat}" for name in MAGNITUDES for stat in MAGNITUDE_STATS],
    "acc_jerk_mean",
]


def ride_features(ride: pd.DataFrame) -> dict[str, float]:
    """Summarise ONE raw ride (time_index, ax, ay, az, rx, ry, rz) into a fixed set of features.

    The ride is cleaned exactly like the CNN input (data.clean_ride), so both models see the same signal.
    """
    missing_ratio = float(ride[SENSORS].isna().to_numpy().mean())
    signal = clean_ride(ride)  # (time steps, 6 sensors), sorted, deduplicated, interpolated

    features: dict[str, float] = {"duration_s": len(signal) / SAMPLING_HZ, "missing_ratio": missing_ratio}
    for i, sensor in enumerate(SENSORS):
        column = signal[:, i]
        features.update(
            {
                f"{sensor}_mean": float(column.mean()),
                f"{sensor}_std": float(column.std()),
                f"{sensor}_min": float(column.min()),
                f"{sensor}_max": float(column.max()),
            }
        )
    for name, axes in MAGNITUDES.items():
        magnitude = np.linalg.norm(signal[:, [SENSORS.index(a) for a in axes]], axis=1)
        features.update(
            {
                f"{name}_mag_mean": float(magnitude.mean()),
                f"{name}_mag_std": float(magnitude.std()),
                f"{name}_mag_p95": float(np.percentile(magnitude, 95)),
            }
        )

    # Jerk: how abruptly the acceleration changes between two measurements (m/s³)
    acc = np.linalg.norm(signal[:, :3], axis=1)
    features["acc_jerk_mean"] = float(np.abs(np.diff(acc)).mean() * SAMPLING_HZ) if len(acc) > 1 else 0.0
    return features


def build_feature_table(rides: pd.DataFrame, computed_at: datetime | None = None) -> pd.DataFrame:
    """One row per ride, in the format Feast expects: entity key, event timestamp, features.

    The rides only have a relative time index (no date), so the event timestamp is the
    time the features were computed. Feast uses it for point-in-time joins.
    """
    computed_at = computed_at or datetime.now(UTC)
    rows = [{"ride_id": ride_id, **ride_features(ride)} for ride_id, ride in rides.groupby("ride_id", sort=True)]
    table = pd.DataFrame(rows, columns=["ride_id", *FEATURE_NAMES])
    table[FEATURE_NAMES] = table[FEATURE_NAMES].astype("float32")
    table["event_timestamp"] = pd.Timestamp(computed_at)
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute ride-level features for the feature store")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR, help="folder containing GBG500.parquet")
    parser.add_argument("--out", type=Path, default=FEATURES_PATH)
    args = parser.parse_args()

    rides = pd.read_parquet(args.raw_dir / "GBG500.parquet")
    table = build_feature_table(rides)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(args.out, index=False)
    print(f"{len(table)} rides x {len(FEATURE_NAMES)} features -> {args.out}")


if __name__ == "__main__":
    main()
