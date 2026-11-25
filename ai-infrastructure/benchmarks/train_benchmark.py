"""CPU vs GPU training benchmark: the training loop of ride_classifier.train, on a chosen device.

Same model, class-weighted loss, Adam, batch size, epochs and seed as train.train(); only the device changes.
Nothing is logged to MLflow: this measures time and memory, plus macro F1 to check the model is as good.

Usage (from the project root, after `python -m ride_classifier.data`):
    uv run python ai-infrastructure/benchmarks/train_benchmark.py --devices cpu --out ai-infrastructure/benchmarks/x.csv
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark import Context, peak_rss_mb  # noqa: E402
from device_benchmark import device_label, synchronize, write_rows  # noqa: E402

from ride_classifier.data import PROCESSED_DIR, SEED  # noqa: E402
from ride_classifier.evaluate import compute_metrics, ride_level  # noqa: E402
from ride_classifier.model import RideCNN  # noqa: E402
from ride_classifier.train import PARAMS, class_weights  # noqa: E402


def train_on(ctx: Context, device: torch.device, data, params: dict) -> None:
    runtime = device_label(device)

    def add(metric, value, unit, **kwargs):
        ctx.add(metric, value, unit, runtime=runtime, **kwargs)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    torch.manual_seed(SEED)
    X_train, y_train = data["X_train"], data["y_train"]
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=params["batch_size"],
        shuffle=True,
    )

    start = time.perf_counter()
    model = RideCNN(dropout=params["dropout"]).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights(y_train).to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
    synchronize(device)
    add("train_setup_time", (time.perf_counter() - start) * 1000, "ms", notes="model + loss to device")

    epoch_times = []
    for _ in range(params["epochs"]):
        model.train()
        epoch_start = time.perf_counter()
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss_fn(model(X_batch), y_batch).backward()
            optimizer.step()
        synchronize(device)
        epoch_times.append(time.perf_counter() - epoch_start)

    epochs = np.asarray(epoch_times)
    notes = f"{params['epochs']} epochs, batch {params['batch_size']}, {len(y_train)} windows"
    add("train_total_time", epochs.sum(), "s", repetitions=params["epochs"], notes=notes)
    add("train_first_epoch_time", epochs[0], "s", notes="includes warm-up (CUDA kernels, allocator)")
    add("train_epoch_time_median", np.median(epochs[1:]), "s", repetitions=params["epochs"] - 1, notes="epochs 2..N")
    add("train_throughput", len(y_train) / np.median(epochs[1:]), "windows/s", notes="median epoch")

    model.eval()
    X_test = data["X_test"]
    with torch.no_grad():
        batches = [model(torch.from_numpy(X_test[i : i + 256]).to(device)).cpu() for i in range(0, len(X_test), 256)]
    proba = torch.softmax(torch.cat(batches), dim=1).numpy()
    y_true, y_pred = ride_level(proba, data["ride_test"], data["y_test"])
    add("train_macro_f1", compute_metrics(y_true, y_pred)["macro_f1"], "", notes="150 test rides, ride level")

    if device.type == "cuda":
        add("train_peak_vram_allocated", torch.cuda.max_memory_allocated(device) / 2**20, "MB", notes="allocator peak")
    add("train_peak_process_rss", peak_rss_mb(), "MB", notes="whole process so far")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--devices", nargs="+", default=["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
    parser.add_argument("--epochs", type=int, default=PARAMS["epochs"])
    parser.add_argument("--out", type=Path)
    parser.add_argument("--print-csv", action="store_true")
    args = parser.parse_args()

    data = np.load(PROCESSED_DIR / "dataset.npz")
    ctx = Context(model_version="RideCNN, train.PARAMS (not registered)", runtime="")
    print(f"{ctx.hardware} | devices {args.devices}", flush=True)
    for name in args.devices:
        print(f"Training on {name}", flush=True)
        train_on(ctx, torch.device(name), data, {**PARAMS, "epochs": args.epochs})
    for row in ctx.rows:
        print(f"  {row['runtime'][:40]:<40} {row['metric']:<26} {row['value']:>12} {row['unit']}")
    write_rows(ctx, args.out, args.print_csv)


if __name__ == "__main__":
    main()
