"""MLflow model that takes a raw ride and returns its class.

The preprocessing (cleaning, windowing, normalisation) is packaged together
with the network weights, so the API only has to send the raw sensor values
of one ride.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlflow.pyfunc
import pandas as pd
import torch

from ride_classifier.data import CLASSES, clean_ride, make_windows, normalize
from ride_classifier.model import RideCNN


class RideClassifier(mlflow.pyfunc.PythonModel):
    def load_context(self, context) -> None:
        self.stats = json.loads(Path(context.artifacts["normalizer"]).read_text())
        self.model = RideCNN()
        self.model.load_state_dict(torch.load(context.artifacts["weights"], map_location="cpu"))
        self.model.eval()

    def predict(self, context, model_input: pd.DataFrame, params=None) -> pd.DataFrame:
        """Input: the raw measurements of ONE ride (time_index, ax, ay, az, rx, ry, rz).

        Output: one row with the predicted label and the probability of each class.
        """
        windows = normalize(make_windows(clean_ride(model_input)), self.stats)
        with torch.no_grad():
            proba = torch.softmax(self.model(torch.from_numpy(windows)), dim=1).mean(dim=0)
        return pd.DataFrame(
            [
                {
                    "label": CLASSES[int(proba.argmax())],
                    **{f"proba_{c}": float(p) for c, p in zip(CLASSES, proba)},
                }
            ]
        )
