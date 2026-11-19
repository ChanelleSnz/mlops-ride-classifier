"""Train the CNN, track the run in MLflow and register the model.

Usage: uv run python -m ride_classifier.train
The MLflow server must be running (docker compose up -d).
"""

from __future__ import annotations

import os
import tempfile
from importlib.metadata import version
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no window: figures are only saved

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.metrics import ConfusionMatrixDisplay
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ride_classifier import data as data_module
from ride_classifier.baseline import run_baselines
from ride_classifier.data import CLASSES, PROCESSED_DIR, SEED
from ride_classifier.evaluate import compute_metrics, report, ride_level
from ride_classifier.model import RideCNN
from ride_classifier.pyfunc_model import RideClassifier

TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5001")
EXPERIMENT = "ride-classifier"
MODEL_NAME = "ride-classifier"
CHAMPION = "champion"  # alias of the model version served by the API
PARAMS = {"epochs": 20, "batch_size": 64, "learning_rate": 1e-3, "dropout": 0.3}


def class_weights(y: np.ndarray) -> torch.Tensor:
    """Rare classes weigh more in the loss, so the model cannot ignore them."""
    counts = np.bincount(y, minlength=len(CLASSES))
    return torch.tensor(len(y) / (len(CLASSES) * np.maximum(counts, 1)), dtype=torch.float32)


def predict_proba(model: nn.Module, X: np.ndarray, batch_size: int = 256) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        batches = [model(torch.from_numpy(X[i : i + batch_size])) for i in range(0, len(X), batch_size)]
    return torch.softmax(torch.cat(batches), dim=1).numpy()


def train(params: dict = PARAMS):
    """Train on the windows of data.py, return the model and the ride-level test results."""
    torch.manual_seed(SEED)
    data = np.load(PROCESSED_DIR / "dataset.npz")
    X_train, y_train = data["X_train"], data["y_train"]

    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=params["batch_size"],
        shuffle=True,
    )
    model = RideCNN(dropout=params["dropout"])
    loss_fn = nn.CrossEntropyLoss(weight=class_weights(y_train))
    optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])

    for epoch in range(1, params["epochs"] + 1):
        model.train()
        total = 0.0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(y_batch)
        epoch_loss = total / len(y_train)
        mlflow.log_metric("train_loss", epoch_loss, step=epoch)
        print(f"epoch {epoch:>2}/{params['epochs']}  loss {epoch_loss:.4f}")

    proba = predict_proba(model, data["X_test"])
    y_true, y_pred = ride_level(proba, data["ride_test"], data["y_test"])
    return model, y_true, y_pred


def confusion_matrix_figure(y_true: np.ndarray, y_pred: np.ndarray):
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, labels=list(range(len(CLASSES))), display_labels=CLASSES,
        xticks_rotation=45, cmap="Blues", ax=ax, colorbar=False,
    )
    ax.set_title("Test set, one prediction per ride")
    fig.tight_layout()
    return fig


def promote_if_better(version_number: str, macro_f1: float) -> None:
    """Give the 'champion' alias to the new version only if it beats the current champion."""
    client = MlflowClient()
    try:
        champion = client.get_model_version_by_alias(MODEL_NAME, CHAMPION)
        champion_f1 = client.get_run(champion.run_id).data.metrics["macro_f1"]
    except MlflowException:
        champion, champion_f1 = None, -1.0

    if macro_f1 > champion_f1:
        client.set_registered_model_alias(MODEL_NAME, CHAMPION, version_number)
        print(f"Version {version_number} is the new champion (macro F1 {macro_f1:.3f} > {champion_f1:.3f})")
    else:
        print(f"Champion stays version {champion.version} (macro F1 {champion_f1:.3f} >= {macro_f1:.3f})")


def log_model(model: nn.Module) -> str:
    """Log the model with its preprocessing as an MLflow pyfunc and register a new version."""
    rides, labels = data_module.load_raw()
    example = rides[rides["ride_id"] == labels["RIDE_ID"].iloc[0]].drop(columns="ride_id").head(700)

    with tempfile.TemporaryDirectory() as tmp:
        weights = Path(tmp) / "ride_cnn.pt"
        torch.save(model.state_dict(), weights)
        artifacts = {"weights": str(weights), "normalizer": str(PROCESSED_DIR / "normalizer.json")}

        wrapper = RideClassifier()
        wrapper.load_context(type("Context", (), {"artifacts": artifacts})())
        signature = infer_signature(example, wrapper.predict(None, example))

        info = mlflow.pyfunc.log_model(
            name="model",
            python_model=wrapper,
            artifacts=artifacts,
            code_paths=[str(Path(__file__).parent)],
            signature=signature,
            input_example=example,
            registered_model_name=MODEL_NAME,
            pip_requirements=[
                f"{package}=={version(package)}"
                for package in ["torch", "pandas", "numpy", "scikit-learn", "pyarrow"]
            ],
        )
    return info.registered_model_version


def main() -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)

    with mlflow.start_run():
        mlflow.log_params(
            {
                **PARAMS,
                "window_size": data_module.WINDOW_SIZE,
                "stride": data_module.STRIDE,
                "test_size": data_module.TEST_SIZE,
                "seed": SEED,
            }
        )
        baseline_f1 = run_baselines()["most_frequent"]["macro_f1"]
        mlflow.log_metric("baseline_macro_f1", baseline_f1)

        model, y_true, y_pred = train(PARAMS)
        metrics = compute_metrics(y_true, y_pred)
        mlflow.log_metrics(metrics)
        mlflow.log_text(report(y_true, y_pred), "classification_report.txt")
        mlflow.log_figure(confusion_matrix_figure(y_true, y_pred), "confusion_matrix.png")

        print("\n" + report(y_true, y_pred))
        for name, value in metrics.items():
            print(f"  {name:<18} {value:.3f}")

        version_number = log_model(model)
        promote_if_better(version_number, metrics["macro_f1"])


if __name__ == "__main__":
    main()
