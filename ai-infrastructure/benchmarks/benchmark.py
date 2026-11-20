"""Inference benchmark of the ride classifier: latency, throughput, memory and quality.

Two levels:
- model: the MLflow pyfunc (preprocessing + CNN) called in-process, one ride per call;
- api:   POST /predict on a running API (HTTP + CSV parsing + model), with several concurrency levels.

Usage (from the project root, an MLflow server or local registry with the model must be reachable):
    MLFLOW_TRACKING_URI=... uv run python ai-infrastructure/benchmarks/benchmark.py --api-url http://localhost:8000

Raw per-request latencies go to benchmarks/raw/, the summary to benchmarks/baseline-results.csv.
"""

from __future__ import annotations

import argparse
import csv
import io
import platform
import resource
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
from mlflow.tracking import MlflowClient

from ride_classifier.data import CLASSES, load_raw, make_windows, normalize, split_rides
from ride_classifier.data import clean_ride as clean
from ride_classifier.evaluate import compute_metrics

HERE = Path(__file__).resolve().parent
RESULTS_CSV = HERE / "baseline-results.csv"
RAW_DIR = HERE / "raw"
COLUMNS = [
    "run",
    "metric",
    "value",
    "unit",
    "model_version",
    "runtime",
    "batch_rides",
    "concurrency",
    "repetitions",
    "hardware",
    "date",
    "notes",
]


@dataclass
class Context:
    """Fixed conditions recorded with every result row."""

    model_version: str
    runtime: str
    hardware: str = field(default_factory=lambda: hardware_description())
    date: str = field(default_factory=lambda: datetime.now(UTC).strftime("%Y-%m-%d"))
    rows: list[dict] = field(default_factory=list)
    run: int = 1

    def add(self, metric: str, value: float, unit: str, *, concurrency=1, repetitions="", notes="", runtime=None):
        self.rows.append(
            {
                "run": self.run,
                "metric": metric,
                "value": round(float(value), 3),
                "unit": unit,
                "model_version": self.model_version,
                "runtime": runtime or self.runtime,
                "batch_rides": 1,
                "concurrency": concurrency,
                "repetitions": repetitions,
                "hardware": self.hardware,
                "date": self.date,
                "notes": notes,
            }
        )


def hardware_description() -> str:
    if sys.platform == "darwin":
        cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout
        memory = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout)
    else:
        cpu, memory = platform.processor(), 0
    gigabytes = f", {memory / 2**30:.0f} GB RAM" if memory else ""
    threads = f"{torch.get_num_threads()} torch threads"
    return f"{cpu.strip()} ({threads}{gigabytes}), {platform.system()} {platform.release()}"


def peak_rss_mb() -> float:
    """Peak resident memory of this process (ru_maxrss is in bytes on macOS, in KB on Linux)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 2**20 if sys.platform == "darwin" else peak / 2**10


def latency_stats(latencies_ms: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(latencies_ms.mean()),
        "p50": float(np.percentile(latencies_ms, 50)),
        "p95": float(np.percentile(latencies_ms, 95)),
        "max": float(latencies_ms.max()),
    }


def test_rides() -> list[tuple[str, pd.DataFrame, int]]:
    """The 150 held-out rides (same split as training), raw, as the API receives them."""
    rides, labels = load_raw()
    _, test_ids = split_rides(labels)
    label_of = dict(zip(labels["RIDE_ID"], labels["LABEL"], strict=True))
    by_id = dict(tuple(rides.groupby("ride_id")))
    return [
        (ride_id, by_id[ride_id].drop(columns="ride_id").reset_index(drop=True), CLASSES.index(label_of[ride_id]))
        for ride_id in test_ids
    ]


def time_calls(function, inputs: list, warmup: int, repetitions: int) -> np.ndarray:
    """Warm up, then call function on the inputs in a loop and return each call's duration in ms."""
    for i in range(warmup):
        function(inputs[i % len(inputs)])
    durations = []
    for i in range(repetitions):
        start = time.perf_counter()
        function(inputs[i % len(inputs)])
        durations.append((time.perf_counter() - start) * 1000)
    return np.asarray(durations)


def save_raw(ctx: Context, name: str, latencies_ms: np.ndarray) -> None:
    RAW_DIR.mkdir(exist_ok=True)
    path = RAW_DIR / f"run{ctx.run}_{name}.csv"
    pd.DataFrame({"latency_ms": latencies_ms}).to_csv(path, index_label="request")


def load_worker(api_url: str, bodies: list[bytes], offset: int, step: int, deadline: float):
    """One client of the closed-loop load test: send rides back to back until the deadline."""
    import httpx

    ok, failed, timings, i = 0, 0, [], offset
    with httpx.Client(base_url=api_url, timeout=60) as client:
        while time.perf_counter() < deadline:
            start = time.perf_counter()
            files = {"file": ("ride.csv", bodies[i % len(bodies)], "text/csv")}
            response = client.post("/predict", files=files)
            timings.append((time.perf_counter() - start) * 1000)
            ok, failed = (ok + 1, failed) if response.status_code == 200 else (ok, failed + 1)
            i += step
    return ok, failed, timings


def benchmark_model(ctx: Context, model, rides, warmup: int, repetitions: int) -> None:
    frames = [ride for _, ride, _ in rides]

    latencies = time_calls(model.predict, frames, warmup, repetitions)
    save_raw(ctx, "model_latency", latencies)
    stats = latency_stats(latencies)
    for name, value in stats.items():
        ctx.add(f"model_latency_{name}", value, "ms", repetitions=repetitions, notes="pyfunc.predict, 1 ride per call")
    ctx.add("model_throughput", 1000 / stats["mean"], "rides/s", repetitions=repetitions, notes="sequential, 1 / mean")

    # Where does the time go? Preprocessing (numpy/pandas) vs network forward pass (PyTorch)
    wrapper = model.unwrap_python_model()
    prepared = [normalize(make_windows(clean(frame)), wrapper.stats) for frame in frames]
    preprocess = time_calls(lambda f: normalize(make_windows(clean(f)), wrapper.stats), frames, warmup, repetitions)

    def forward(windows):
        with torch.no_grad():
            return torch.softmax(wrapper.model(torch.from_numpy(windows)), dim=1).mean(dim=0)

    network = time_calls(forward, prepared, warmup, repetitions)
    ctx.add("preprocessing_latency_mean", preprocess.mean(), "ms", repetitions=repetitions, notes="clean + windows")
    ctx.add("network_latency_mean", network.mean(), "ms", repetitions=repetitions, notes="CNN forward + softmax")
    ctx.add("windows_per_ride_mean", np.mean([len(w) for w in prepared]), "windows", notes="600 x 6 each")


def benchmark_quality(ctx: Context, model, rides) -> None:
    y_true = np.array([label for _, _, label in rides])
    y_pred = np.array([CLASSES.index(model.predict(ride).iloc[0]["label"]) for _, ride, _ in rides])
    metrics = compute_metrics(y_true, y_pred)
    ctx.add("macro_f1", metrics["macro_f1"], "", repetitions=len(rides), notes="150 held-out rides")
    ctx.add("accuracy", metrics["accuracy"], "", repetitions=len(rides), notes="150 held-out rides")


def benchmark_api(ctx: Context, api_url: str, rides, warmup: int, repetitions: int, levels, duration: float) -> None:
    import httpx

    bodies = []
    for _, ride, _ in rides:
        buffer = io.StringIO()
        ride.to_csv(buffer, index=False)
        bodies.append(buffer.getvalue().encode())

    with httpx.Client(base_url=api_url, timeout=60) as client:

        def post(body: bytes) -> None:
            client.post("/predict", files={"file": ("ride.csv", body, "text/csv")}).raise_for_status()

        latencies = time_calls(post, bodies, warmup, repetitions)
    save_raw(ctx, "api_latency", latencies)
    for name, value in latency_stats(latencies).items():
        ctx.add(f"api_latency_{name}", value, "ms", repetitions=repetitions, notes="POST /predict, CSV of 1 ride")

    for concurrency in levels:
        done, errors, per_request = 0, 0, []
        start = time.perf_counter()
        deadline = start + duration
        with ThreadPoolExecutor(concurrency) as pool:
            futures = [
                pool.submit(load_worker, api_url, bodies, offset, concurrency, deadline)
                for offset in range(concurrency)
            ]
            for future in futures:
                ok, failed, timings = future.result()
                done, errors, per_request = done + ok, errors + failed, per_request + timings
        elapsed = time.perf_counter() - start
        timings = np.asarray(per_request)
        save_raw(ctx, f"api_load_c{concurrency}", timings)
        notes = f"{duration:.0f} s closed loop, {errors} errors"
        ctx.add("api_throughput", done / elapsed, "requests/s", concurrency=concurrency, repetitions=done, notes=notes)
        ctx.add("api_latency_p95_under_load", np.percentile(timings, 95), "ms", concurrency=concurrency, notes=notes)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-uri", default="models:/ride-classifier@champion")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=200)
    parser.add_argument("--runs", type=int, default=3, help="repeat the latency/throughput measurements")
    parser.add_argument("--api-url", help="also benchmark a running API, e.g. http://localhost:8000")
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--duration", type=float, default=20, help="seconds per concurrency level")
    parser.add_argument("--api-pid", type=int, help="PID of the API server process, to record its memory (RSS)")
    parser.add_argument("--out", type=Path, default=RESULTS_CSV)
    args = parser.parse_args()

    name, alias = args.model_uri.removeprefix("models:/").split("@")
    model_version = MlflowClient().get_model_version_by_alias(name, alias).version
    runtime = f"PyTorch {torch.__version__} CPU, MLflow {version('mlflow')} pyfunc"
    ctx = Context(model_version=f"{name} v{model_version} (@{alias})", runtime=runtime)
    print(f"Model {ctx.model_version} | {ctx.hardware}")

    rss_before = peak_rss_mb()
    rides = test_rides()
    rss_data = peak_rss_mb()
    local_model = Path(mlflow.artifacts.download_artifacts(args.model_uri))
    model = mlflow.pyfunc.load_model(str(local_model))
    ctx.add("process_peak_rss_after_load", peak_rss_mb(), "MB", notes=f"data loaded: {rss_data - rss_before:.0f} MB")

    weights = next(local_model.rglob("ride_cnn.pt"))
    artifact = sum(f.stat().st_size for f in local_model.rglob("*") if f.is_file())
    ctx.add("model_weights_size", weights.stat().st_size / 1e6, "MB", notes="ride_cnn.pt state_dict")
    ctx.add("mlflow_model_artifact_size", artifact / 1e6, "MB", notes="weights + normalizer + code + env files")

    benchmark_quality(ctx, model, rides)  # deterministic: measured once

    for run in range(1, args.runs + 1):
        ctx.run = run
        print(f"Run {run}/{args.runs}")
        benchmark_model(ctx, model, rides, args.warmup, args.repetitions)
        ctx.add("process_peak_rss_after_run", peak_rss_mb(), "MB", notes="benchmark process, model in-process")

        if args.api_url:
            before = len(ctx.rows)
            benchmark_api(ctx, args.api_url, rides, args.warmup, args.repetitions, args.concurrency, args.duration)
            if args.api_pid:
                rss_kb = int(subprocess.run(["ps", "-o", "rss=", "-p", str(args.api_pid)], capture_output=True).stdout)
                ctx.add("api_process_rss_after_load_test", rss_kb / 1024, "MB", notes="ps rss of the uvicorn process")
            for row in ctx.rows[before:]:
                row["runtime"] = f"FastAPI + uvicorn (1 worker), {runtime}"

    with args.out.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(ctx.rows)
    for row in ctx.rows:
        print(f"  run {row['run']}  {row['metric']:<32} {row['value']:>10} {row['unit']:<11} c={row['concurrency']}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
