"""API routes, tested with a fake model (no MLflow server needed)."""

import pytest
from fastapi.testclient import TestClient

from ride_classifier import api


@pytest.fixture
def client(fake_model, monkeypatch):
    """API client with the fake model loaded. TestClient without `with` skips startup (no MLflow)."""
    monkeypatch.setattr(api.store, "model", fake_model)
    monkeypatch.setattr(api.store, "info", api.ModelInfo(name="ride-classifier", alias="champion", version="7"))
    return TestClient(api.app)


@pytest.fixture
def client_without_model(monkeypatch):
    monkeypatch.setattr(api.store, "model", None)
    return TestClient(api.app)


def upload(content: bytes) -> dict:
    return {"file": ("ride.csv", content, "text/csv")}


def test_health_ok_when_model_loaded(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model"]["version"] == "7"


def test_health_503_without_model(client_without_model):
    assert client_without_model.get("/health").status_code == 503


def test_model_info(client):
    assert client.get("/model").json()["alias"] == "champion"


def test_predict_returns_label_probabilities_and_model(client, example_csv):
    response = client.post("/predict", files=upload(example_csv))
    assert response.status_code == 200
    body = response.json()
    assert body["label"] == "Bad weather"
    assert set(body["probabilities"]) == {"Safe", "Safe overall", "Bad weather", "Tandem", "Reckless"}
    assert body["n_measurements"] == 2050
    assert body["model"]["version"] == "7"


def test_predict_rejects_missing_columns(client):
    response = client.post("/predict", files=upload(b"time_index,ax\n0,1.0\n"))
    assert response.status_code == 422
    assert "Missing columns" in response.json()["detail"]


def test_predict_rejects_empty_file(client):
    response = client.post("/predict", files=upload(b"time_index,ax,ay,az,rx,ry,rz\n"))
    assert response.status_code == 422


def test_predict_rejects_too_many_rows(client, monkeypatch, example_csv):
    monkeypatch.setattr(api, "MAX_ROWS", 100)
    assert client.post("/predict", files=upload(example_csv)).status_code == 413


def test_predict_503_without_model(client_without_model, example_csv):
    assert client_without_model.post("/predict", files=upload(example_csv)).status_code == 503


def test_swagger_documents_all_routes(client):
    paths = client.get("/openapi.json").json()["paths"]
    assert {"/predict", "/health", "/model", "/model/reload"} <= set(paths)


def test_metrics_count_requests_and_predictions(client, example_csv):
    client.post("/predict", files=upload(example_csv))
    metrics = client.get("/metrics/").text
    assert 'model_predictions_total{label="Bad weather"}' in metrics
    assert 'api_http_requests_total{method="POST",route="/predict",status="200"}' in metrics
    assert "model_prediction_confidence_bucket" in metrics
    assert "input_ride_measurements_count" in metrics
