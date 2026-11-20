# AI Infrastructure — Ride Classifier

AI Infrastructure contribution for the existing MLOps project (see the [main README](../README.md)).
The goal is not a new model. It explains what the model needs to run, measures inference, tests one
optimization and designs a scalable, cost-aware serving system.

> **Measure, do not claim.** Every number below is tagged **[measured]** (with how), **[estimate]** (with the
> assumption), or **To measure**.

| Deliverable | File | Status |
|---|---|---|
| Current system diagram | [architecture/current-system.md](architecture/current-system.md) | Draft (Mermaid, export to PNG) |
| Proposed system diagram | [architecture/proposed-system.md](architecture/proposed-system.md) | To do |
| Baseline benchmark | [benchmarks/benchmark.ipynb](benchmarks/benchmark.ipynb) · [benchmark.py](benchmarks/benchmark.py) · [baseline-results.csv](benchmarks/baseline-results.csv) | Done (3 runs) |
| Optimization comparison | [optimization/comparison.ipynb](optimization/comparison.ipynb) | To do |
| Distributed training decision | [distributed-training/technical-analysis.md](distributed-training/technical-analysis.md) | Draft |
| Scaling, cost and resilience | [cost-resilience/infrastructure-plan.md](cost-resilience/infrastructure-plan.md) | To do |

## 1. Inventory of the current MLOps system

| Component | Implementation | Where it runs |
|---|---|---|
| Preprocessing | `ride_classifier.data`: cleaning, 60 s windows (600 × 6), normalisation fitted on train | Local (Mac), Airflow (Docker) |
| Feature store | Feast, local (parquet offline, SQLite online), `feature_repo/` | Local, Airflow |
| Training | PyTorch 1D CNN, class-weighted loss, 20 epochs, CPU | Local (Mac), Airflow, GitHub self-hosted runner |
| Tracking / registry | MLflow 3.18 + PostgreSQL + MinIO (S3), alias `champion` | Docker Compose (dev), Kubernetes (prod) |
| Serving | FastAPI loading the MLflow pyfunc (preprocessing + CNN), 2 replicas | Docker Compose, Kubernetes (Docker Desktop) |
| CI/CD | GitHub Actions: ruff + pytest, multi-arch images on Docker Hub, deploy with rollback | GitHub runners + self-hosted Mac runner |
| Monitoring | Prometheus + Grafana (API, model, data layers), 4 alert rules | Docker Compose (dev only) |

The model does **not** run on Colab or GCP: everything runs on one Apple Silicon Mac, in prod on the local
Kubernetes cluster of Docker Desktop.

## 2. Resource sizing

| Item | Value | Source |
|---|---|---|
| Framework | PyTorch 2.15.0 (CPU build on Linux images), Python 3.12 | [measured] `uv.lock`, `torch.__version__` |
| Model | 1D CNN, 3 conv blocks (32-64-64) + linear, **24,677 parameters** | [measured] `sum(p.numel())` |
| Serialized weights | **107,550 bytes (≈ 0.1 MB)** `state_dict` | [measured] `torch.save` to memory |
| MLflow model artifact (weights + normalizer + code + env files) | **1.27 MB** | [measured] `benchmark.py`, downloaded artifact |
| Dataset used | GBG500: **30.2 MB** parquet, **500 rides**, **1,529,547 rows** (1,807–4,767 rows per ride, median 2,906) | [measured] file size, pandas |
| Labels | 500 rides, 5 classes (Safe 416, Safe overall 38, Bad weather 22, Tandem 18, Reckless 6) | [measured] `GBG500_labels.csv` |
| Unused dataset | SRV3490: 268.2 MB parquet, 3,490 rides | [measured] file size |
| Model input | windows of shape (600 time steps, 6 sensors); one ride ≈ 9 windows | [measured] `data.py` constants |
| Batch size (training) | 64 windows | `train.PARAMS` |
| Hardware | Apple M4, 10 CPU cores, 16 GB unified memory, macOS 15.7.2; MPS available but **not used** (training is CPU) | [measured] `sysctl`, `torch.backends.mps` |
| GPU / VRAM | None used | — |
| Preprocessing duration | **4.8 s** wall, peak RSS 731 MB (350 train rides → 3,026 windows, 150 test rides → 1,323 windows) | [measured] `/usr/bin/time -l python -m ride_classifier.data`, [log](benchmarks/raw/preprocess.log) |
| Training duration | **82.8 s** wall (20 epochs, CPU, includes baselines, feature store model and MLflow logging) | [measured] `/usr/bin/time -l python -m ride_classifier.train`, [log](benchmarks/raw/training.log) |
| Peak training RAM | **970 MB** (max RSS) | [measured] same command |
| API process RAM | **≈ 225 MB** after load (median of 3 runs; one run reached 439 MB). k8s request 768 Mi, limit 2 Gi | [measured] `ps rss` of the uvicorn process |

## 3. Baseline benchmark

Procedure, charts and interpretation: [benchmarks/benchmark.ipynb](benchmarks/benchmark.ipynb). All values come from
[baseline-results.csv](benchmarks/baseline-results.csv) (raw per-request latencies in [benchmarks/raw/](benchmarks/raw/)):
**median of 3 runs, min–max in brackets**, 10 warm-up calls, 200 timed calls per run, 1 ride (≈ 9 windows) per call.

| Metric | Value | Measurement conditions |
|---|---|---|
| Model and version | `ride-classifier` v1 (@champion), macro F1 0.272 (same as the README champion v1) | PyTorch 2.15.0 CPU, MLflow 3.18 pyfunc; trained with `train.py` into a local SQLite MLflow registry |
| Hardware | Apple M4, 10 cores (4 torch threads), 16 GB, macOS 15.7 | shared with other Docker stacks ([background load](benchmarks/raw/background-load.txt)); no GPU |
| Model size (MB) | 0.11 (weights) · 1.27 (MLflow artifact) | `ride_cnn.pt` state_dict / whole pyfunc folder |
| Mean latency (ms) | **model 2.04** [1.83–2.25] · **API 4.23** [3.67–6.11] | model: `pyfunc.predict` in-process; API: `POST /predict` (CSV upload), 1 uvicorn worker, concurrency 1 |
| p95 latency (ms) | **model 2.97** [2.64–4.83] · **API 6.00** [4.48–10.70] | same |
| Throughput (requests/s) | **API 206** at 1 client [204–208] · 206 at 4 [187–269] · 232 at 8 [183–243] | 20 s closed loop per level, 0 errors; model alone: 491 rides/s sequential |
| Memory usage (MB) | API process ≈ 225 [213–439] | RSS of the uvicorn process after each load test |
| Classification quality | macro F1 **0.272**, accuracy 0.607 | 150 held-out rides (`data.split_rides`) |

Main findings:

- **Where the time goes** (model level, 2.0 ms): preprocessing 0.50 ms, CNN forward 0.76 ms, the rest is pyfunc
  overhead (schema check, DataFrame output). The API adds ≈ 2 ms (HTTP, multipart, CSV parsing, validation).
- **One worker serves one request at a time**: throughput stays ≈ 200–230 requests/s from 1 to 8 clients, while p95
  latency grows from 8 ms to 32 ms (4 clients) and 58 ms (8 clients). Cause: `/predict` is `async def` but calls the
  model synchronously, which blocks the uvicorn event loop. Scaling = more workers / replicas, not more concurrency.
- **Not tested**: GPU, the Kubernetes deployment (2 replicas), the production MLflow. The benchmark can be re-run
  against prod by setting `MLFLOW_TRACKING_URI` (see the notebook).
- During the first training attempt PyTorch raised a transient internal error
  (`Could not find schema for aten::_to_copy`); the identical command succeeded on the second try and the error could
  not be reproduced.

## 4. Optimization

See [optimization/](optimization/).

| Metric | Baseline | Optimized | Interpretation |
|---|---|---|---|
| Mean latency (ms) | — | — | — |
| p95 latency (ms) | — | — | — |
| Throughput (requests/s) | — | — | — |
| Memory usage (MB) | — | — | — |
| Classification quality | — | — | — |

## 5. Distributed training, scaling, cost and resilience

- Distributed training: [distributed-training/technical-analysis.md](distributed-training/technical-analysis.md)
- Serving architecture, scaling, monitoring, costs, rollback: [cost-resilience/infrastructure-plan.md](cost-resilience/infrastructure-plan.md)

## Owners and next actions

| Task | Owner | Due |
|---|---|---|
| Resource sizing (training time, RAM) | To assign | |
| Baseline benchmark | To assign | |
| Optimization | To assign | |
| Distributed training analysis | To assign | |
| Architecture, cost and resilience | To assign | |
