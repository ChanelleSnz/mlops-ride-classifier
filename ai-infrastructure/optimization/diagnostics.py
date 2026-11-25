"""Diagnostics behind the interpretation of results.csv (not part of the fair comparison).

1. Is oneDNN (PyTorch's optimised CPU kernels) available in this PyTorch build?
2. Does a contiguous input (instead of the transposed view used by RideCNN.forward) change PyTorch's speed?
3. How do both runtimes react to the number of threads?

Usage: MLFLOW_TRACKING_URI=... uv run --group ai-infra python ai-infrastructure/optimization/diagnostics.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "benchmarks"))
sys.path.insert(0, str(HERE))

from device_benchmark import load_wrapper  # noqa: E402
from runtimes import make_forward  # noqa: E402


def mean_ms(function, x: np.ndarray, repetitions: int) -> float:
    for _ in range(3):
        function(x)
    start = time.perf_counter()
    for _ in range(repetitions):
        function(x)
    return (time.perf_counter() - start) / repetitions * 1000


def main() -> None:
    net = load_wrapper("models:/ride-classifier@champion").model.eval()
    rng = np.random.default_rng(0)
    small = rng.normal(size=(9, 600, 6)).astype(np.float32)
    large = rng.normal(size=(1323, 600, 6)).astype(np.float32)

    def torch_as_is(x):
        with torch.no_grad():
            return net(torch.from_numpy(x))

    def torch_contiguous(x):
        with torch.no_grad():
            return net.classifier(net.features(torch.from_numpy(x).transpose(1, 2).contiguous()))

    print(f"torch {torch.__version__}, oneDNN (mkldnn) available: {torch.backends.mkldnn.is_available()}")
    torch.set_num_threads(4)
    same = torch.allclose(torch_as_is(large), torch_contiguous(large), atol=1e-5)
    print(f"same outputs with a contiguous input: {same}")
    for name, function in [("pytorch, as served", torch_as_is), ("pytorch, contiguous input", torch_contiguous)]:
        small_ms, large_ms = mean_ms(function, small, 200), mean_ms(function, large, 5)
        print(f"{name:<30} 4 threads: 9 windows {small_ms:6.3f} ms | 1323 windows {large_ms:7.1f} ms")
    for threads in (1, 2, 4):
        torch.set_num_threads(threads)
        for runtime in ("pytorch", "onnxruntime"):
            forward = make_forward(runtime, threads, net.state_dict())
            small_ms, large_ms = mean_ms(forward, small, 200), mean_ms(forward, large, 5)
            print(f"{runtime:<12} {threads} thread(s): 9 windows {small_ms:6.3f} ms | 1323 windows {large_ms:7.1f} ms")


if __name__ == "__main__":
    main()
