"""REST API that classifies a ride from its raw sensor measurements.

Run locally:  uv run uvicorn ride_classifier.api:app --reload --port 8000
Swagger UI:   http://localhost:8000/docs
"""

from __future__ import annotations

import io
import os
from contextlib import asynccontextmanager

import mlflow
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from mlflow.tracking import MlflowClient
from pydantic import BaseModel, Field

from ride_classifier.data import CLASSES, SENSORS

TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5001")
MODEL_NAME = os.environ.get("MODEL_NAME", "ride-classifier")
MODEL_ALIAS = os.environ.get("MODEL_ALIAS", "champion")
REQUIRED_COLUMNS = ["time_index", *SENSORS]
MAX_ROWS = 20_000  # about 33 minutes of measurements at 10 per second


class ModelInfo(BaseModel):
    name: str = Field(examples=["ride-classifier"])
    alias: str = Field(examples=["champion"])
    version: str | None = Field(default=None, examples=["1"])
    run_id: str | None = Field(default=None, description="MLflow run that produced the model")


class Health(BaseModel):
    status: str = Field(examples=["ok"])
    model_loaded: bool
    model: ModelInfo


class Prediction(BaseModel):
    label: str = Field(description="Predicted class", examples=["Safe"])
    probabilities: dict[str, float] = Field(
        description="Probability of each class (sums to 1)",
        examples=[{"Safe": 0.81, "Safe overall": 0.09, "Bad weather": 0.06, "Tandem": 0.03, "Reckless": 0.01}],
    )
    n_measurements: int = Field(description="Number of rows read from the file", examples=[3059])
    model: ModelInfo


class ModelStore:
    """Holds the model currently served. Loaded at startup and on /model/reload."""

    def __init__(self) -> None:
        self.model = None
        self.info = ModelInfo(name=MODEL_NAME, alias=MODEL_ALIAS)

    def load(self) -> None:
        mlflow.set_tracking_uri(TRACKING_URI)
        version = MlflowClient().get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
        self.model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@{MODEL_ALIAS}")
        self.info = ModelInfo(name=MODEL_NAME, alias=MODEL_ALIAS, version=version.version, run_id=version.run_id)


store = ModelStore()


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        store.load()
        print(f"Loaded {MODEL_NAME}@{MODEL_ALIAS} (version {store.info.version})")
    except Exception as error:  # the API still starts; /health reports the problem
        print(f"Model not loaded: {error}")
    yield


app = FastAPI(
    title="Ride Classifier API",
    description=(
        "Classifies a micromobility ride into **Safe, Safe overall, Bad weather, Tandem or Reckless** "
        "from its IMU measurements (accelerometer ax/ay/az, gyroscope rx/ry/rz, 10 measurements per second).\n\n"
        "The model is loaded from the MLflow Model Registry (`ride-classifier@champion`)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


def read_ride(content: bytes) -> pd.DataFrame:
    """Parse and validate the uploaded CSV. Raises a 422 error with a clear message."""
    try:
        ride = pd.read_csv(io.BytesIO(content))
    except Exception as error:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, f"Not a readable CSV file: {error}")

    missing = [c for c in REQUIRED_COLUMNS if c not in ride.columns]
    if missing:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, f"Missing columns: {missing}")
    if ride.empty:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "The file contains no measurements")
    if len(ride) > MAX_ROWS:
        raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, f"Too many rows (max {MAX_ROWS})")

    ride = ride[REQUIRED_COLUMNS]
    if ride[SENSORS].isna().all().any():
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "At least one sensor column is completely empty")
    return ride


@app.get("/health", response_model=Health, tags=["monitoring"], summary="Is the API ready to predict?")
def health():
    """Returns 200 when a model is loaded, 503 otherwise (used by Kubernetes readiness checks)."""
    body = Health(status="ok" if store.model else "no model", model_loaded=store.model is not None, model=store.info)
    if store.model is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, body.model_dump())
    return body


@app.get("/model", response_model=ModelInfo, tags=["model"], summary="Which model version is served?")
def model_info():
    return store.info


@app.post("/model/reload", response_model=ModelInfo, tags=["model"], summary="Load the current champion")
def reload_model():
    """Reloads the model the `champion` alias points to (after a new training, without restarting the API)."""
    try:
        store.load()
    except Exception as error:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, f"Could not load the model: {error}")
    return store.info


@app.post("/predict", response_model=Prediction, tags=["prediction"], summary="Classify one ride")
async def predict(
    file: UploadFile = File(
        description="CSV of ONE ride with columns time_index, ax, ay, az, rx, ry, rz (one row per measurement)"
    ),
):
    if store.model is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "No model loaded, try POST /model/reload")

    ride = read_ride(await file.read())
    result = store.model.predict(ride).iloc[0]
    return Prediction(
        label=result["label"],
        probabilities={c: round(float(result[f"proba_{c}"]), 4) for c in CLASSES},
        n_measurements=len(ride),
        model=store.info,
    )
