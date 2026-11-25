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
| CPU vs GPU (GCP, T4) | [benchmarks/device_benchmark.py](benchmarks/device_benchmark.py) · [train_benchmark.py](benchmarks/train_benchmark.py) · [vertex-t4/](benchmarks/vertex-t4/) · [gpu/](gpu/) | Done (1 run) |
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
| GPU / VRAM | None used locally. CPU vs GPU measured on GCP: 1 NVIDIA T4 (15 GB); peak VRAM 327 MB (inference, 1,323 windows) and 60 MB (training) | [measured] [§ 3.1](#31-cpu-vs-gpu-on-gcp-vertex-ai-t4) |
| Preprocessing duration | **4.8 s** wall, peak RSS 731 MB (350 train rides → 3,026 windows, 150 test rides → 1,323 windows) | [measured] `/usr/bin/time -l python -m ride_classifier.data` |
| Training duration | **82.8 s** wall (20 epochs, CPU, includes baselines, feature store model and MLflow logging) | [measured] `/usr/bin/time -l python -m ride_classifier.train` |
| Peak training RAM | **970 MB** (max RSS) | [measured] same command |
| API process RAM | **≈ 218 MB** after load (median of 3 runs; one run reached 435 MB). k8s request 768 Mi, limit 2 Gi | [measured] `ps rss` of the uvicorn process |

## 3. Baseline benchmark

Procedure, charts and interpretation: [benchmarks/benchmark.ipynb](benchmarks/benchmark.ipynb). All values come from
[baseline-results.csv](benchmarks/baseline-results.csv) (raw per-request latencies in [benchmarks/raw/](benchmarks/raw/)):
**median of 3 runs, min–max in brackets**, 10 warm-up calls, 200 timed calls per run, 1 ride (≈ 9 windows) per call.

| Metric | Value | Measurement conditions |
|---|---|---|
| Model and version | `ride-classifier` v1 (@champion), macro F1 0.272 (same as the README champion v1) | PyTorch 2.15.0 CPU, MLflow 3.18 pyfunc; trained with `train.py` into a local SQLite MLflow registry |
| Hardware | Apple M4, 10 cores (4 torch threads), 16 GB, macOS 15.7 | shared with other Docker stacks ([background load](benchmarks/raw/background-load.txt)); no GPU |
| Model size (MB) | 0.11 (weights) · 1.27 (MLflow artifact) | `ride_cnn.pt` state_dict / whole pyfunc folder |
| Model loading time | **cold ≈ 5.0 s** (imports 2,225 ms + `load_model` 2,733 ms) · **warm reload 8 ms** · download 5 ms | cold: new Python process, median of 5 = API pod start; warm: `load_model` again in the same process = `POST /model/reload`; download from the local registry (a file copy; with the prod MinIO it adds network time) |
| Mean latency (ms) | **model 1.78** [1.76–2.03] · **API 4.48** [3.91–5.43] | model: `pyfunc.predict` in-process; API: `POST /predict` (CSV upload), 1 uvicorn worker, concurrency 1 |
| p95 latency (ms) | **model 2.34** [2.33–3.26] · **API 6.51** [5.29–11.23] | same |
| Throughput (requests/s) | **API 219** at 1 client [158–223] · 226 at 4 [148–253] · 245 at 8 [245–246] | 20 s closed loop per level, 0 errors; model alone: 562 rides/s sequential |
| Memory usage (MB) | API process ≈ 218 [218–435] | RSS of the uvicorn process after each load test |
| Classification quality | macro F1 **0.272**, accuracy 0.607 | 150 held-out rides (`data.split_rides`) |

Main findings:

- **Where the time goes** (model level, 1.8 ms): preprocessing 0.49 ms, CNN forward 0.77 ms, the rest is pyfunc
  overhead (schema check, DataFrame output). The API adds ≈ 2 ms (HTTP, multipart, CSV parsing, validation).
- **One worker serves one request at a time**: throughput stays ≈ 220–245 requests/s from 1 to 8 clients, while p95
  latency grows from 7 ms to 29 ms (4 clients) and 49 ms (8 clients). Cause: `/predict` is `async def` but calls the
  model synchronously, which blocks the uvicorn event loop. Scaling = more workers / replicas, not more concurrency.
- **Loading dominates start-up, not the model**: a cold start costs ≈ 5 s, almost all Python / MLflow / PyTorch
  imports (the weights are 0.1 MB), while a warm reload takes 8 ms. Hot-reloading a new champion with
  `POST /model/reload` is therefore much cheaper than restarting pods; on Kubernetes the 5 s cold start is covered by
  the startup probe (up to 5 min) and the rolling update keeps the old pods serving meanwhile.
- **Not tested here**: GPU (see [§ 3.1](#31-cpu-vs-gpu-on-gcp-vertex-ai-t4)), the Kubernetes deployment (2 replicas), the production MLflow. The benchmark can be re-run
  against prod by setting `MLFLOW_TRACKING_URI` (see the notebook).
- During the first training attempt PyTorch raised a transient internal error
  (`Could not find schema for aten::_to_copy`); the identical command succeeded on the second try and the error could
  not be reproduced.

### 3.1 CPU vs GPU on GCP (Vertex AI, T4)

**Same model, same library versions, same machine; only the device changes.** One Vertex AI custom job
(project `advanced-cloud-ai-i2`, `europe-west4`) runs the inference and training benchmarks on the machine's CPU, then
on its GPU:

| Item | Value |
|---|---|
| Machine | `n1-standard-4`: Intel Xeon @ 2.00 GHz, 4 vCPU (PyTorch uses 2 threads), 15 GB RAM |
| GPU | 1 × NVIDIA Tesla T4, 15 GB, driver 580.178.04, **spot** |
| Software | PyTorch **2.15.0** (CUDA 13.2 build, same version as the Mac baseline), all other packages pinned from `uv.lock` |
| Model | the same `ride-classifier` v1 as the CPU baseline (macro F1 0.272), bundled in the image |
| Workload | the 150 held-out test rides; 10 warm-up + 200 timed calls; `torch.cuda.synchronize()` before every clock read |
| Code | [device_benchmark.py](benchmarks/device_benchmark.py) (inference), [train_benchmark.py](benchmarks/train_benchmark.py) (training: the loop of `train.train()`), [gpu/](gpu/) (image, job) |
| Raw results | [vertex-t4/inference-results.csv](benchmarks/vertex-t4/inference-results.csv), [vertex-t4/training-results.csv](benchmarks/vertex-t4/training-results.csv) — **1 run**, 2026-11-21 |

Why a Vertex AI job and not a VM: the project's Compute Engine GPU quota (`GPUS_ALL_REGIONS`) is 0, and so are the
on-demand Vertex AI T4 / L4 quotas in the regions we tried (europe-west4, europe-west1, us-central1); spot GPUs on
Vertex AI are allowed. A job also stops by itself (hard cap 45 min), so nothing
keeps billing.

**Inference**

| Metric | CPU | GPU (T4) | GPU vs CPU |
|---|---|---|---|
| **1 ride (≈ 9 windows), mean latency** | 5.10 ms | **3.86 ms** | 1.3× faster |
| 1 ride, p95 latency | 6.45 ms | 4.83 ms | 1.3× |
| — preprocessing (numpy/pandas, always on CPU) | 2.36 ms | 2.37 ms | same |
| — host → device transfer | 0.00 ms | 0.09 ms | GPU-only cost |
| — CNN forward | 2.40 ms | 1.05 ms | 2.3× |
| **First prediction** (network to device + first forward) | 13 ms | **461 ms** | 35× slower (CUDA initialisation) |
| Batch of 64 windows | 10.1 ms (6,343 windows/s) | 1.35 ms (47,327 windows/s) | 7.5× |
| Batch of 256 windows | 115 ms (2,232 windows/s) | 3.67 ms (69,685 windows/s) | 31× |
| **Batch of 1,323 windows** (whole test set) | 768 ms (1,723 windows/s) | **17.1 ms (77,158 windows/s)** | **45×** |
| Peak memory | 1,476 MB RAM | 1,632 MB RAM + **327 MB VRAM** | |
| macro F1 (150 rides) | 0.272 | 0.272 | identical |

**Training** (20 epochs, batch 64, 3,026 windows, same seed)

| Metric | CPU | GPU (T4) | GPU vs CPU |
|---|---|---|---|
| **Total training time** | 74.3 s | **5.0 s** | **15×** |
| Median epoch (2..20) | 3.70 s | 0.22 s | 17× |
| First epoch | 3.90 s | 0.79 s | |
| Throughput | 819 windows/s | 14,036 windows/s | 17× |
| Peak memory | 1,215 MB RAM | 1,688 MB RAM + **60 MB VRAM** | |
| macro F1 (150 rides) | 0.298 | 0.180 | see below |

**Interpretation and decision**

- **Serving stays on CPU.** The API scores one ride per request: there the GPU is only 1.3× faster, because half
  the time is preprocessing (on the CPU in both cases) and the CNN (24,677 parameters, ≈ 9 windows) is too small to
  keep a GPU busy; the transfer adds 0.09 ms and the first prediction pays ≈ 0.5 s of CUDA initialisation per pod. A
  gain of ≈ 1.2 ms per request does not justify a GPU per replica.
- **The GPU only pays off for large batches**: 31–45× from 256 windows up (CPU throughput falls for large batches,
  the GPU keeps rising until ≈ 1,000 windows). Useful for scoring many rides offline, not for the API.
- **Training**: 15× faster on the T4, but the CPU needs only 74 s, so the gain is about one minute per training run;
  CPU training on one machine stays sufficient.
- **No distributed training (DDP / FSDP / ZeRO)**: one T4 trains the model in 5 s using 60 MB of its 15 GB.
  There is nothing to split; splitting it over 2 GPUs would add All-Reduce communication for no gain.
- **Training quality differs between devices (0.298 vs 0.180)**: not a GPU defect. With the same seed, the GPU draws
  different random numbers (dropout) and uses different cuDNN algorithms, so it trains a different model; with 150
  test rides (2 Reckless) macro F1 varies a lot between runs. Inference with the same weights gives exactly the same
  score on both devices (0.272), which shows the computation is equivalent. Comparing training quality fairly needs
  several seeds per device (not done).
- **Limits**: one run (no min–max); spot GPU; the cloud CPU uses 2 threads, slower than the Mac (4 threads), so CPU
  vs GPU is compared on the **same** cloud machine and the Mac stays the local baseline.

**Reproduce** (from the project root, the data in `data/raw/`, the model in the MLflow registry):

```bash
bash ai-infrastructure/gpu/build.sh        # builds and pushes the image, prints its name
sed "s#imageUri: IMAGE.*#imageUri: <image printed above>#" ai-infrastructure/gpu/job.yaml > /tmp/job.yaml
gcloud ai custom-jobs create --project=advanced-cloud-ai-i2 --region=europe-west4 \
  --display-name=ride-classifier-cpu-vs-gpu --config=/tmp/job.yaml
# results: the CSV blocks between CSV-BEGIN / CSV-END in the job logs
gcloud logging read 'resource.type="ml_job" AND resource.labels.job_id="<job id>"' \
  --project=advanced-cloud-ai-i2 --order=asc --format='value(textPayload)'
```

Two image settings were needed for the GPU to be visible in the container (see [gpu/Dockerfile](gpu/Dockerfile)):
`NVIDIA_VISIBLE_DEVICES` / `NVIDIA_DRIVER_CAPABILITIES`, and `/usr/local/nvidia/{bin,lib64}` in `PATH` /
`LD_LIBRARY_PATH`, where Vertex AI mounts the driver.

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
