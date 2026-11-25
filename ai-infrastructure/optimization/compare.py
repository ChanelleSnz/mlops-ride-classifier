"""Baseline (PyTorch) vs optimized (ONNX Runtime) inference of the same CNN, on the same CPU.

Fair comparison: same weights, same preprocessing (numpy/pandas), same post-processing (numpy softmax + mean),
same test rides, same number of CPU threads, same warm-up and repetitions, 3 runs. Only the runtime that executes
the network changes:
- baseline:  PyTorch eager, `torch.no_grad()` (what the MLflow pyfunc does today);
- optimized: ONNX Runtime CPU, all graph optimizations (Conv+BatchNorm folding, Conv+ReLU fusion, ...).

Measured: latency of one ride (end to end and network only), batch-size sweep, cold start in a new process
(import + load + first prediction), process memory, macro F1.

Usage (from the project root, after export_onnx.py):
    MLFLOW_TRACKING_URI=... uv run --group ai-infra python ai-infrastructure/optimization/compare.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "benchmarks"))

from benchmark import Context, latency_stats, test_rides, time_calls  # noqa: E402
from device_benchmark import write_rows  # noqa: E402
from runtimes import RUNTIMES, make_forward, ride_label  # noqa: E402

RESULTS_CSV = HERE / "results.csv"
BATCH_SIZES = [9, 64, 256, 1024]  # windows; the whole test set (1,323) is added at run time


COLD_START = """
import json, resource, sys, time
start = time.perf_counter()
sys.path.insert(0, sys.argv[3])
import numpy as np
from runtimes import load_weights, make_forward
runtime, threads = sys.argv[1], int(sys.argv[2])
state_dict = load_weights(sys.argv[4]) if runtime == "pytorch" else None
imported = time.perf_counter()
forward = make_forward(runtime, threads, state_dict)
loaded = time.perf_counter()
forward(np.zeros((9, 600, 6), dtype=np.float32))
first = time.perf_counter()
forward(np.zeros((1323, 600, 6), dtype=np.float32))
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
peak_mb = peak / 2**20 if sys.platform == "darwin" else peak / 2**10
print(json.dumps({"import": imported - start, "load": loaded - imported, "first": first - loaded, "rss": peak_mb}))
"""


def cold_start(ctx: Context, runtime: str, threads: int, weights: Path, repeats: int) -> None:
    """New Python process: import the runtime, load the model, first prediction; memory without the dataset."""
    samples = []
    for _ in range(repeats):
        command = [sys.executable, "-c", COLD_START, runtime, str(threads), str(HERE), str(weights)]
        output = subprocess.run(command, capture_output=True, text=True, check=True).stdout.strip().splitlines()[-1]
        samples.append(json.loads(output))
    notes = f"new process, median of {repeats}"
    for key, metric in [("import", "cold_import_time"), ("load", "cold_load_time"), ("first", "first_prediction")]:
        ctx.add(metric, np.median([s[key] for s in samples]) * 1000, "ms", runtime=runtime, notes=notes)
    ctx.add(
        "process_peak_rss",
        np.median([s["rss"] for s in samples]),
        "MB",
        runtime=runtime,
        notes=f"model + 1 batch of 1,323 windows, no dataset; {notes}",
    )


def benchmark_runtime(ctx: Context, runtime: str, forward, prepared, frames, labels, warmup: int, repetitions: int):
    from ride_classifier.data import clean_ride, make_windows, normalize

    stats = ctx.stats

    def predict(frame) -> int:
        return ride_label(forward(normalize(make_windows(clean_ride(frame)), stats)))

    latencies = time_calls(predict, frames, warmup, repetitions)
    for name, value in latency_stats(latencies).items():
        ctx.add(
            f"ride_latency_{name}",
            value,
            "ms",
            runtime=runtime,
            repetitions=repetitions,
            notes="1 ride: preprocessing + network + post-processing",
        )
    network = time_calls(forward, prepared, warmup, repetitions)
    for name, value in latency_stats(network).items():
        ctx.add(
            f"network_latency_{name}",
            value,
            "ms",
            runtime=runtime,
            repetitions=repetitions,
            notes="network only, ≈ 9 windows",
        )

    all_windows = np.concatenate(prepared)
    for size in [*BATCH_SIZES, len(all_windows)]:
        batch = all_windows[:size]
        reps = max(5, min(repetitions, 20_000 // size))
        timings = time_calls(forward, [batch], warmup, reps)
        notes = f"network only, {size} windows, {reps} repetitions"
        ctx.add(
            "batch_latency_mean", timings.mean(), "ms", runtime=runtime, concurrency=size, repetitions=reps, notes=notes
        )
        ctx.add(
            "batch_throughput",
            size / (timings.mean() / 1000),
            "windows/s",
            runtime=runtime,
            concurrency=size,
            notes=notes,
        )

    from ride_classifier.evaluate import compute_metrics

    y_pred = np.array([predict(frame) for frame in frames])
    ctx.add(
        "macro_f1",
        compute_metrics(labels, y_pred)["macro_f1"],
        "",
        runtime=runtime,
        repetitions=len(frames),
        notes="150 test rides",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-uri", default="models:/ride-classifier@champion")
    parser.add_argument("--threads", type=int, default=4, help="CPU threads for both runtimes")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=200)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--cold-repeats", type=int, default=5)
    parser.add_argument("--out", type=Path, default=RESULTS_CSV)
    args = parser.parse_args()

    import tempfile

    import onnxruntime as ort
    import torch
    from device_benchmark import load_wrapper

    from ride_classifier.data import clean_ride, make_windows, normalize

    wrapper = load_wrapper(args.model_uri)
    rides = test_rides()
    frames = [ride for _, ride, _ in rides]
    labels = np.array([label for _, _, label in rides])
    prepared = [normalize(make_windows(clean_ride(frame)), wrapper.stats) for frame in frames]

    versions = {"pytorch": f"PyTorch {torch.__version__} eager", "onnxruntime": f"ONNX Runtime {ort.__version__}"}
    ctx = Context(model_version="ride-classifier v1 (@champion)", runtime="")
    ctx.stats = wrapper.stats
    print(f"{ctx.hardware} | {args.threads} threads for both runtimes", flush=True)

    with tempfile.TemporaryDirectory() as tmp:
        weights = Path(tmp) / "ride_cnn.pt"
        torch.save(wrapper.model.state_dict(), weights)
        for runtime in RUNTIMES:
            print(f"Cold start: {runtime}", flush=True)
            cold_start(ctx, runtime, args.threads, weights, args.cold_repeats)

    forwards = {runtime: make_forward(runtime, args.threads, wrapper.model.state_dict()) for runtime in RUNTIMES}
    for run in range(1, args.runs + 1):
        ctx.run = run
        for runtime in RUNTIMES if run % 2 else RUNTIMES[::-1]:  # alternate the order between runs
            print(f"Run {run}/{args.runs}: {runtime}", flush=True)
            benchmark_runtime(ctx, runtime, forwards[runtime], prepared, frames, labels, args.warmup, args.repetitions)

    for row in ctx.rows:
        row["runtime"] = f"{versions[row['runtime']]}, CPU, {args.threads} threads"
    write_rows(ctx, args.out, print_csv=False)


if __name__ == "__main__":
    main()
