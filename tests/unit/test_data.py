"""Preprocessing: cleaning, windowing and normalisation."""

import numpy as np
import pandas as pd

from ride_classifier.data import (
    CLASSES,
    SENSORS,
    WINDOW_SIZE,
    clean_ride,
    fit_normalizer,
    make_windows,
    normalize,
)


def test_classes_are_the_five_expected_labels():
    assert CLASSES == ["Safe", "Safe overall", "Bad weather", "Tandem", "Reckless"]


def test_clean_ride_sorts_by_time_and_drops_duplicates():
    ride = pd.DataFrame({"time_index": [200, 0, 100, 100], **{s: [3.0, 1.0, 2.0, 2.0] for s in SENSORS}})
    cleaned = clean_ride(ride)
    assert cleaned.shape == (3, len(SENSORS))
    assert cleaned[:, 0].tolist() == [1.0, 2.0, 3.0]


def test_clean_ride_fills_missing_values():
    ride = pd.DataFrame({"time_index": [0, 100, 200], **{s: [1.0, np.nan, 3.0] for s in SENSORS}})
    cleaned = clean_ride(ride)
    assert not np.isnan(cleaned).any()
    assert cleaned[1, 0] == 2.0  # linear interpolation between 1 and 3


def test_make_windows_shape_and_overlap():
    signal = np.arange(1200 * 6, dtype=np.float32).reshape(1200, 6)
    windows = make_windows(signal, size=600, stride=300)
    assert windows.shape == (3, 600, 6)  # starts at 0, 300, 600
    np.testing.assert_array_equal(windows[1][0], signal[300])


def test_make_windows_pads_short_rides():
    signal = np.ones((100, 6), dtype=np.float32)
    windows = make_windows(signal)
    assert windows.shape == (1, WINDOW_SIZE, 6)
    np.testing.assert_array_equal(windows[0][-1], signal[-1])  # padded with the last value


def test_normalize_gives_zero_mean_and_unit_std():
    rng = np.random.default_rng(0)
    windows = rng.normal(loc=50, scale=10, size=(20, 600, 6)).astype(np.float32)
    normalized = normalize(windows, fit_normalizer(windows))
    flat = normalized.reshape(-1, 6)
    np.testing.assert_allclose(flat.mean(axis=0), 0, atol=1e-3)
    np.testing.assert_allclose(flat.std(axis=0), 1, atol=1e-3)
    assert normalized.dtype == np.float32
