"""End-to-end: the real stack (MLflow + MinIO + API with the champion model) must be running.

Run:  docker compose up -d
      uv run pytest -m e2e
"""

import os
from pathlib import Path

import httpx
import pytest

API_URL = os.environ.get("E2E_API_URL", "http://localhost:8000")
EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "ride_bad_weather.csv"

pytestmark = pytest.mark.e2e


@pytest.fixture(scope="module")
def api_url() -> str:
    try:
        httpx.get(f"{API_URL}/health", timeout=5).raise_for_status()
    except httpx.HTTPError as error:
        pytest.skip(f"API not ready at {API_URL}: {error}")
    return API_URL


def test_api_serves_the_champion_model(api_url):
    model = httpx.get(f"{api_url}/model", timeout=5).json()
    assert model["alias"] == "champion"
    assert model["version"] is not None


def test_real_prediction_from_csv_to_label(api_url):
    response = httpx.post(
        f"{api_url}/predict",
        files={"file": (EXAMPLE.name, EXAMPLE.read_bytes(), "text/csv")},
        timeout=60,
    )
    assert response.status_code == 200
    body = response.json()
    assert body["label"] in {"Safe", "Safe overall", "Bad weather", "Tandem", "Reckless"}
    assert sum(body["probabilities"].values()) == pytest.approx(1, abs=1e-3)
