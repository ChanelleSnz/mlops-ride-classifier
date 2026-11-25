"""Export the champion CNN to ONNX and check that ONNX Runtime gives the same results as PyTorch.

Only the network is exported: the preprocessing (clean -> windows -> normalise) stays the same numpy/pandas code.
The batch dimension is dynamic, so one file serves 1 ride (≈ 9 windows) as well as the whole test set.

Usage (from the project root):
    MLFLOW_TRACKING_URI=... uv run --group ai-infra python ai-infrastructure/optimization/export_onnx.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "benchmarks"))

from benchmark import test_rides  # noqa: E402
from device_benchmark import load_wrapper  # noqa: E402

from ride_classifier.data import clean_ride, make_windows, normalize  # noqa: E402

MODEL_DIR = HERE / "model"
ONNX_PATH = MODEL_DIR / "ride_cnn.onnx"
OPSET = 18


def export(net: torch.nn.Module, path: Path) -> None:
    example = torch.randn(9, 600, 6)
    batch = torch.export.Dim("batch", min=1, max=8192)
    torch.onnx.export(
        net,
        (example,),
        str(path),
        input_names=["windows"],
        output_names=["logits"],
        dynamic_shapes=({0: batch},),
        opset_version=OPSET,
        dynamo=True,  # torch.export-based exporter (the default and recommended one in PyTorch 2.x)
        external_data=False,  # one self-contained file (weights inside), easier to ship and to measure
    )
    onnx.checker.check_model(onnx.load(str(path)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-uri", default="models:/ride-classifier@champion")
    parser.add_argument("--out", type=Path, default=ONNX_PATH)
    args = parser.parse_args()

    wrapper = load_wrapper(args.model_uri)
    net = wrapper.model.eval()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    export(net, args.out)

    # Correctness on the real workload: every window of the 150 test rides, and the ride-level decisions
    session = ort.InferenceSession(str(args.out), providers=["CPUExecutionProvider"])
    rides = test_rides()
    prepared = [normalize(make_windows(clean_ride(ride)), wrapper.stats) for _, ride, _ in rides]
    windows = np.concatenate(prepared)
    with torch.no_grad():
        reference = net(torch.from_numpy(windows)).numpy()
    onnx_logits = session.run(None, {"windows": windows})[0]

    def ride_labels(logits_of) -> list[int]:
        labels = []
        for w in prepared:
            logits = logits_of(w)
            proba = np.exp(logits - logits.max(axis=1, keepdims=True))
            labels.append(int((proba / proba.sum(axis=1, keepdims=True)).mean(axis=0).argmax()))
        return labels

    with torch.no_grad():
        torch_labels = ride_labels(lambda w: net(torch.from_numpy(w)).numpy())
    onnx_labels = ride_labels(lambda w: session.run(None, {"windows": w})[0])

    report = {
        "onnx_file": args.out.name,
        "onnx_size_bytes": args.out.stat().st_size,
        "opset": OPSET,
        "torch": torch.__version__,
        "onnx": onnx.__version__,
        "onnxruntime": ort.__version__,
        "windows_checked": int(len(windows)),
        "max_abs_logit_diff": float(np.abs(onnx_logits - reference).max()),
        "same_window_argmax": float((onnx_logits.argmax(1) == reference.argmax(1)).mean()),
        "same_ride_predictions": f"{sum(a == b for a, b in zip(torch_labels, onnx_labels, strict=True))}/{len(rides)}",
    }
    (MODEL_DIR / "export-check.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
