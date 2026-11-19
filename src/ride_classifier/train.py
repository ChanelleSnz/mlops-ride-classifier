"""Train the CNN on the windows produced by data.py and evaluate it per ride."""

from __future__ import annotations

import json

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ride_classifier.data import CLASSES, PROCESSED_DIR, ROOT, SEED
from ride_classifier.evaluate import compute_metrics, report, ride_level
from ride_classifier.model import RideCNN

MODELS_DIR = ROOT / "models"
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


def train(params: dict = PARAMS) -> tuple[nn.Module, dict[str, float]]:
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
        print(f"epoch {epoch:>2}/{params['epochs']}  loss {total / len(y_train):.4f}")

    proba = predict_proba(model, data["X_test"])
    y_true, y_pred = ride_level(proba, data["ride_test"], data["y_test"])
    metrics = compute_metrics(y_true, y_pred)
    print("\n" + report(y_true, y_pred))
    for name, value in metrics.items():
        print(f"  {name:<18} {value:.3f}")
    return model, metrics


if __name__ == "__main__":
    model, metrics = train()
    MODELS_DIR.mkdir(exist_ok=True)
    torch.save(model.state_dict(), MODELS_DIR / "ride_cnn.pt")
    (MODELS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nModel saved to {MODELS_DIR / 'ride_cnn.pt'}")
