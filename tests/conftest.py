"""Shared test fixtures."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ride_classifier.data import CLASSES, SENSORS

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"


@pytest.fixture
def example_csv() -> bytes:
    """A real ride (2050 measurements) shipped in examples/."""
    return (EXAMPLES_DIR / "ride_safe.csv").read_bytes()


@pytest.fixture
def synthetic_ride() -> pd.DataFrame:
    """A small fake ride: 1000 measurements of random sensor values."""
    rng = np.random.default_rng(0)
    ride = pd.DataFrame(rng.normal(size=(1000, len(SENSORS))), columns=SENSORS)
    ride.insert(0, "time_index", np.arange(1000) * 100)
    return ride


class FakeModel:
    """Stands in for the MLflow model: always predicts 'Bad weather'."""

    def predict(self, ride: pd.DataFrame) -> pd.DataFrame:
        proba = {f"proba_{c}": 0.0 for c in CLASSES}
        proba["proba_Bad weather"] = 1.0
        return pd.DataFrame([{"label": "Bad weather", **proba}])


@pytest.fixture
def fake_model() -> FakeModel:
    return FakeModel()
