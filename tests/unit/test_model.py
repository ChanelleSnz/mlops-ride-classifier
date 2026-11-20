"""CNN architecture, class weights and the MLflow pyfunc wrapper."""

import json

import numpy as np
import pytest
import torch

from ride_classifier.data import CLASSES, SENSORS
from ride_classifier.model import RideCNN
from ride_classifier.pyfunc_model import RideClassifier
from ride_classifier.train import class_weights


@pytest.mark.parametrize("length", [300, 600, 1000])
def test_cnn_outputs_one_score_per_class(length):
    model = RideCNN().eval()
    scores = model(torch.zeros(4, length, len(SENSORS)))
    assert scores.shape == (4, len(CLASSES))


def test_rare_classes_get_bigger_weights():
    y = np.array([0] * 90 + [1] * 9 + [4])
    weights = class_weights(y)
    assert weights[4] > weights[1] > weights[0]


def test_pyfunc_returns_one_label_and_probabilities(tmp_path, synthetic_ride):
    weights_path = tmp_path / "weights.pt"
    normalizer_path = tmp_path / "normalizer.json"
    torch.save(RideCNN().state_dict(), weights_path)
    normalizer_path.write_text(json.dumps({"mean": [0.0] * 6, "std": [1.0] * 6}))
    context = type("Context", (), {"artifacts": {"weights": str(weights_path), "normalizer": str(normalizer_path)}})

    model = RideClassifier()
    model.load_context(context)
    result = model.predict(None, synthetic_ride)

    assert len(result) == 1
    assert result["label"].iloc[0] in CLASSES
    probabilities = result[[f"proba_{c}" for c in CLASSES]].iloc[0]
    assert probabilities.sum() == pytest.approx(1, abs=1e-5)
