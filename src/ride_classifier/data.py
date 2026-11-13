"""Data loading and preprocessing for ride classification.

Every function used to prepare training data is also used by the API,
so a ride is transformed exactly the same way at training and prediction time.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SENSORS = ["ax", "ay", "az", "rx", "ry", "rz"]
CLASSES = ["Safe", "Safe overall", "Bad weather", "Tandem", "Reckless"]
WINDOW_SIZE = 600  # 60 seconds at 10 measurements per second
STRIDE = 300  # windows overlap by 50 %
TEST_SIZE = 0.3
SEED = 42

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"


def load_raw(raw_dir: Path = RAW_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the sensor measurements and the ride labels."""
    rides = pd.read_parquet(raw_dir / "GBG500.parquet")
    labels = pd.read_csv(raw_dir / "GBG500_labels.csv")
    return rides, labels


def clean_ride(ride: pd.DataFrame) -> np.ndarray:
    """Sort one ride by time, drop duplicate timestamps and fill missing values.

    Returns an array of shape (time steps, 6 sensors).
    """
    ride = ride.sort_values("time_index").drop_duplicates("time_index")
    values = ride[SENSORS].interpolate(limit_direction="both")
    return values.to_numpy(dtype=np.float32)


def make_windows(signal: np.ndarray, size: int = WINDOW_SIZE, stride: int = STRIDE) -> np.ndarray:
    """Cut a (time, sensors) signal into overlapping windows.

    Returns an array of shape (number of windows, size, sensors).
    A ride shorter than one window is padded by repeating its last value.
    """
    if len(signal) < size:
        signal = np.pad(signal, ((0, size - len(signal)), (0, 0)), mode="edge")
    starts = range(0, len(signal) - size + 1, stride)
    return np.stack([signal[s : s + size] for s in starts])


def fit_normalizer(windows: np.ndarray) -> dict[str, list[float]]:
    """Compute the mean and standard deviation of each sensor (training data only)."""
    flat = windows.reshape(-1, windows.shape[-1])
    return {"mean": flat.mean(axis=0).tolist(), "std": (flat.std(axis=0) + 1e-8).tolist()}


def normalize(windows: np.ndarray, stats: dict[str, list[float]]) -> np.ndarray:
    """Put every sensor on the same scale (mean 0, standard deviation 1)."""
    mean = np.asarray(stats["mean"], dtype=np.float32)
    std = np.asarray(stats["std"], dtype=np.float32)
    return ((windows - mean) / std).astype(np.float32)


def _windows_for(ride_ids, rides_by_id, label_of):
    X, y, owner = [], [], []
    for ride_id in ride_ids:
        windows = make_windows(clean_ride(rides_by_id[ride_id]))
        X.append(windows)
        y += [CLASSES.index(label_of[ride_id])] * len(windows)
        owner += [ride_id] * len(windows)
    return np.concatenate(X), np.asarray(y), np.asarray(owner)


def build_dataset(raw_dir: Path = RAW_DIR, out_dir: Path = PROCESSED_DIR) -> None:
    """Split rides into train/test, cut them into windows, normalise and save."""
    rides, labels = load_raw(raw_dir)
    label_of = dict(zip(labels["RIDE_ID"], labels["LABEL"], strict=True))

    # Split by ride, not by window: windows of the same ride must never be
    # in both train and test, otherwise the test score is artificially high.
    train_ids, test_ids = train_test_split(
        labels["RIDE_ID"],
        test_size=TEST_SIZE,
        stratify=labels["LABEL"],
        random_state=SEED,
    )

    rides_by_id = dict(tuple(rides.groupby("ride_id")))
    X_train, y_train, ride_train = _windows_for(train_ids, rides_by_id, label_of)
    X_test, y_test, ride_test = _windows_for(test_ids, rides_by_id, label_of)

    stats = fit_normalizer(X_train)  # computed on train only, never on test
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "dataset.npz",
        X_train=normalize(X_train, stats),
        y_train=y_train,
        ride_train=ride_train,
        X_test=normalize(X_test, stats),
        y_test=y_test,
        ride_test=ride_test,
    )
    (out_dir / "normalizer.json").write_text(json.dumps(stats, indent=2))

    print(f"Train: {len(train_ids)} rides -> {X_train.shape}")
    print(f"Test:  {len(test_ids)} rides -> {X_test.shape}")
    for name, y in [("train", y_train), ("test", y_test)]:
        counts = {c: int((y == i).sum()) for i, c in enumerate(CLASSES)}
        print(f"Windows per class ({name}): {counts}")
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    build_dataset()
