"""CPU vs GPU inference benchmark: the same model (same weights, same preprocessing) on each device.

The served MLflow pyfunc always runs on CPU, so this script reuses its weights and normaliser and runs the
same computation (clean -> windows -> normalise -> CNN -> mean of window probabilities) on the chosen device.

Measured per device:
- first prediction (moving the network to the device + first forward: includes CUDA initialisation);
- latency of one ride (≈ 9 windows), split into preprocessing (CPU), host->device transfer and network;
- batch-size sweep up to all test windows: throughput in windows/s;
- memory: peak process RSS (CPU) or peak allocated VRAM (GPU);
- quality: macro F1 on the 150 test rides (must be the same on every device).

Usage (from the project root):
    uv run python ai-infrastructure/benchmarks/device_benchmark.py --model-uri models:/ride-classifier@champion \\
        --devices cpu cuda --out ai-infrastructure/benchmarks/results-<machine>.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import mlflow
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark import COLUMNS, Context, latency_stats, peak_rss_mb, test_rides, time_calls  # noqa: E402

from ride_classifier.data import clean_ride, make_windows, normalize  # noqa: E402
from ride_classifier.evaluate import compute_metrics  # noqa: E402

BATCH_SIZES = [9, 64, 256, 1024]  # windows; 9 ≈ one ride. The full test set is added at run time.


def synchronize(device: torch.device) -> None:
    """GPU calls are asynchronous: wait for them before reading the clock."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def load_wrapper(model_uri: str):
    """The RideClassifier inside the MLflow pyfunc (network weights + normaliser statistics)."""
    local = model_uri if Path(model_uri).exists() else mlflow.artifacts.download_artifacts(model_uri)
    return mlflow.pyfunc.load_model(str(local)).unwrap_python_model()


def device_label(device: torch.device) -> str:
    name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    return f"PyTorch {torch.__version__} on {device.type} ({name}, {torch.get_num_threads()} CPU threads)"


def benchmark_device(ctx: Context, wrapper, rides, device: torch.device, warmup: int, repetitions: int) -> None:
    frames = [ride for _, ride, _ in rides]
    prepared = [normalize(make_windows(clean_ride(frame)), wrapper.stats) for frame in frames]
    runtime = device_label(device)

    def add(metric, value, unit, **kwargs):
        ctx.add(metric, value, unit, runtime=runtime, **kwargs)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # First prediction: network to the device + first forward (CUDA context, kernels, cuDNN autotuning)
    start = time.perf_counter()
    net = type(wrapper.model)().to(device).eval()
    net.load_state_dict(wrapper.model.state_dict())
    with torch.no_grad():
        net(torch.from_numpy(prepared[0]).to(device))
    synchronize(device)
    add("first_prediction_time", (time.perf_counter() - start) * 1000, "ms", notes="to(device) + first forward")

    def predict(frame) -> int:
        windows = normalize(make_windows(clean_ride(frame)), wrapper.stats)
        with torch.no_grad():
            proba = torch.softmax(net(torch.from_numpy(windows).to(device)), dim=1).mean(dim=0)
        return int(proba.argmax().item())  # .item() copies to the CPU, so it waits for the GPU

    latencies = time_calls(predict, frames, warmup, repetitions)
    for name, value in latency_stats(latencies).items():
        add(f"ride_latency_{name}", value, "ms", repetitions=repetitions, notes="1 ride: preprocess + transfer + CNN")

    def transfer(windows):
        tensor = torch.from_numpy(windows).to(device)
        synchronize(device)
        return tensor

    def forward(tensor):
        with torch.no_grad():
            output = net(tensor)
        synchronize(device)
        return output

    on_device = [transfer(windows) for windows in prepared]
    preprocess = time_calls(
        lambda f: normalize(make_windows(clean_ride(f)), wrapper.stats), frames, warmup, repetitions
    )
    add("ride_preprocessing_mean", preprocess.mean(), "ms", repetitions=repetitions, notes="CPU, same on every device")
    add("ride_transfer_mean", time_calls(transfer, prepared, warmup, repetitions).mean(), "ms", notes="host -> device")
    add("ride_network_mean", time_calls(forward, on_device, warmup, repetitions).mean(), "ms", notes="CNN forward")

    # Batch-size sweep: where (if anywhere) does the GPU start to pay off?
    all_windows = np.concatenate(prepared)
    for size in [*BATCH_SIZES, len(all_windows)]:
        batch = all_windows[:size]
        reps = max(5, min(repetitions, 20_000 // size))
        timings = time_calls(lambda b: forward(transfer(b)), [batch], warmup, reps)
        notes = f"transfer + forward of {size} windows, {reps} repetitions"
        add("batch_latency_mean", timings.mean(), "ms", concurrency=size, repetitions=reps, notes=notes)
        add("batch_throughput", size / (timings.mean() / 1000), "windows/s", concurrency=size, notes=notes)

    y_true = np.array([label for _, _, label in rides])
    y_pred = np.array([predict(frame) for frame in frames])
    add("macro_f1", compute_metrics(y_true, y_pred)["macro_f1"], "", repetitions=len(rides), notes="150 test rides")

    if device.type == "cuda":
        add("peak_vram_allocated", torch.cuda.max_memory_allocated(device) / 2**20, "MB", notes="torch allocator peak")
    add("peak_process_rss", peak_rss_mb(), "MB", notes="whole benchmark process so far")


def write_rows(ctx: Context, out: Path | None, print_csv: bool) -> None:
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(ctx.rows)
        print(f"-> {out}")
    if print_csv:  # on Vertex AI the results are read back from the job logs
        writer = csv.DictWriter(sys.stdout, fieldnames=COLUMNS)
        print("CSV-BEGIN")
        writer.writeheader()
        writer.writerows(ctx.rows)
        print("CSV-END", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-uri", default="models:/ride-classifier@champion", help="registry URI or local folder")
    parser.add_argument("--model-version", default="ride-classifier v1 (@champion)", help="label stored in the CSV")
    parser.add_argument("--devices", nargs="+", default=["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=200)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--print-csv", action="store_true", help="also print the results to stdout")
    args = parser.parse_args()

    wrapper = load_wrapper(args.model_uri)
    rides = test_rides()
    ctx = Context(model_version=args.model_version, runtime="")
    print(f"{ctx.hardware} | devices {args.devices}", flush=True)
    for name in args.devices:
        print(f"Benchmarking on {name}", flush=True)
        benchmark_device(ctx, wrapper, rides, torch.device(name), args.warmup, args.repetitions)
    for row in ctx.rows:
        print(
            f"  {row['runtime'][:40]:<40} {row['metric']:<26} {row['value']:>12} {row['unit']:<9} {row['concurrency']}"
        )
    write_rows(ctx, args.out, args.print_csv)


if __name__ == "__main__":
    main()
