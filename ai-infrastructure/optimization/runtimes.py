"""The two runtimes compared, with no heavy import at module level.

Imported by the cold-start measurement in a fresh process: the ONNX Runtime path must never import PyTorch,
otherwise its start-up time and memory would include PyTorch's.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

ONNX_PATH = Path(__file__).resolve().parent / "model" / "ride_cnn.onnx"
RUNTIMES = ["pytorch", "onnxruntime"]


def make_forward(runtime: str, threads: int, state_dict=None, onnx_path: Path = ONNX_PATH):
    """A function windows (n, 600, 6) float32 -> logits (n, 5) numpy, for the given runtime."""
    if runtime == "pytorch":
        import torch

        from ride_classifier.model import RideCNN

        torch.set_num_threads(threads)
        net = RideCNN().eval()
        net.load_state_dict(state_dict)

        def forward(windows: np.ndarray) -> np.ndarray:
            with torch.no_grad():  # what the MLflow pyfunc does today
                return net(torch.from_numpy(windows)).numpy()

        return forward

    if runtime == "onnxruntime":
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(str(onnx_path), options, providers=["CPUExecutionProvider"])
        return lambda windows: session.run(None, {"windows": windows})[0]

    raise ValueError(f"unknown runtime {runtime!r}")


def ride_label(logits: np.ndarray) -> int:
    """Same decision as the pyfunc: mean of the window probabilities, then argmax."""
    proba = np.exp(logits - logits.max(axis=1, keepdims=True))
    return int((proba / proba.sum(axis=1, keepdims=True)).mean(axis=0).argmax())


def load_weights(path: str):
    """PyTorch weights for the pytorch runtime (imports torch: only call it for that runtime)."""
    import torch

    return torch.load(path, map_location="cpu")
