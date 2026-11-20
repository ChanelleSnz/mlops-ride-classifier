# Ride Classifier: MLOps and AI Infrastructure

[![CI](https://github.com/ChanelleSnz/mlops-ride-classifier/actions/workflows/ci.yml/badge.svg)](https://github.com/ChanelleSnz/mlops-ride-classifier/actions/workflows/ci.yml)
[![Docker images](https://github.com/ChanelleSnz/mlops-ride-classifier/actions/workflows/docker.yml/badge.svg)](https://github.com/ChanelleSnz/mlops-ride-classifier/actions/workflows/docker.yml)

An end-to-end MLOps pipeline that classifies micromobility rides from their IMU sensor data
(accelerometer and gyroscope) into 5 classes: **Safe, Safe overall, Bad weather, Tandem, Reckless**.

Group project at Albert School (M2) shared by two courses:

- **MLOps & Model Deployment**: training, tracking, registry, API, web app, Docker, CI/CD, Kubernetes, tests.
- **Advanced Cloud Infrastructure / AI Infrastructure**: resource sizing, benchmarking, optimization,
  distributed training decision, scaling, costs and resilience (see [AI Infrastructure](#ai-infrastructure)).

**Team:** Rania, Luli, Wei-Ling, Chanelle.

## Table of contents

1. [Problem and data](#problem-and-data)
2. [Architecture](#architecture)
3. [Technical choices](#technical-choices)
4. [Repository structure](#repository-structure)
5. [Getting started (dev)](#getting-started-dev)
6. [Tests](#tests)
7. [CI/CD](#cicd)
8. [Production on local Kubernetes](#production-on-local-kubernetes)
9. [AI Infrastructure](#ai-infrastructure)
10. [Screenshots](#screenshots)
11. [Roadmap](#roadmap)
12. [Contributors](#contributors)

## Problem and data

Each ride is a time series of 6 IMU signals (`ax, ay, az, rx, ry, rz`) recorded at 10 measurements per second.
The goal is to predict the class of the whole ride. *Reckless* is the class we most want to detect; *Safe overall*,
*Bad weather* and *Tandem* are borderline cases that must not be confused with it.

| Item | Value (measured on `GBG500`) |
| --- | --- |
| Raw file | `GBG500.parquet`, 28.8 MB on disk |
| Measurements | 1,529,547 rows, 8 columns (`ride_id`, `time_index`, 6 sensors) |
| Rides | 500 (Safe 416, Safe overall 38, Bad weather 22, Tandem 18, Reckless 6) |
| Ride length | 1,807 to 4,767 measurements (mean 3,059, i.e. about 5 minutes) |
| Labels | `GBG500_labels.csv` (`RIDE_ID`, `LABEL`) |

The data is not committed (see `.gitignore`). Put `GBG500.parquet` and `GBG500_labels.csv` in `data/raw/`.

**Preprocessing** (`src/ride_classifier/data.py`), shared by training and the API:

1. Sort by time, drop duplicate timestamps, interpolate missing values.
2. Cut each ride into windows of 600 measurements (60 s) with a stride of 300 (50 % overlap).
3. Split **by ride** (70 % train / 30 % test, stratified, seed 42), so windows of one ride are never in both sets.
4. Normalise each sensor with the mean and standard deviation of the training set only.

**Evaluation** (`src/ride_classifier/evaluate.py`): the window probabilities of a ride are averaged to give one
prediction per ride. Because classes are very imbalanced, we track macro F1, balanced accuracy and the F1 of
*Reckless*, and compare them with naive baselines (`baseline.py`).

## Architecture

Current system (what is implemented in this repository):

```mermaid
flowchart LR
    subgraph Data
        RAW[(data/raw<br/>GBG500.parquet)]
    end
    subgraph Training
        PRE[data.py<br/>windows + normalisation] --> TRAIN[train.py<br/>1D CNN, PyTorch]
    end
    subgraph Tracking
        MLF[MLflow server]
        PG[(PostgreSQL<br/>runs, registry)]
        S3[(MinIO / S3<br/>models, figures)]
        MLF --- PG
        MLF --- S3
    end
    subgraph Serving
        API[FastAPI<br/>/predict /health /model]
        WEB[Streamlit web app]
    end
    RAW --> PRE
    TRAIN -- params, metrics, model --> MLF
    MLF -- "models:/ride-classifier@champion" --> API
    USER((User)) --> WEB -- CSV of one ride --> API
```

- `train.py` logs parameters, metrics, the confusion matrix and the model (weights + normaliser + preprocessing,
  packaged as an MLflow `pyfunc`) and registers a new version of `ride-classifier`.
- The new version gets the `champion` alias only if its macro F1 beats the current champion.
- The API loads `ride-classifier@champion` at startup and can reload it without restarting (`POST /model/reload`).
- The web app never loads the model: it calls the API like any other client.

## Technical choices

| Need | Choice | Why |
| --- | --- | --- |
| Dependencies | **uv** + `pyproject.toml` + `uv.lock` | Reproducible installs, fast, one lock file for dev, CI and Docker. PyTorch CPU build on Linux (no CUDA, much smaller images). |
| Model | **1D CNN** in **PyTorch** (`RideCNN`) | Learns patterns directly from raw signals; small (24,677 parameters); class-weighted loss for the rare classes. |
| Tracking and registry | **MLflow 3.18** | Runs, metrics, artifacts and Model Registry with aliases (`champion`). |
| MLflow backend | **PostgreSQL** | Reliable metadata store, required for the Model Registry. |
| Model storage | **MinIO** (S3-compatible) | Same S3 API as AWS, runs locally. Chainguard image, because the official `minio/minio` images left Docker Hub in September 2026. |
| API | **FastAPI** | Validation with Pydantic, automatic Swagger UI (`/docs`), health endpoint for Kubernetes. |
| Input format | **CSV of one ride** | The model works on IMU time series, not images: the API takes the raw measurements of one ride (`time_index, ax, ay, az, rx, ry, rz`). |
| Web app | **Streamlit** | Upload a ride or pick an example, see the predicted class and the probabilities. |
| Dev environment | **Docker Compose** | One command starts MinIO, PostgreSQL, MLflow, the API and the web app. |
| Images | **Docker Hub**, multi-arch (amd64 + arm64) | Built natively on GitHub runners, tagged with the commit SHA and `latest`. |
| Prod | **Kubernetes** (Docker Desktop) + **Kustomize** | Close to a real cluster; manifests stay plain YAML. |
| CD | **GitHub Actions** on a **self-hosted runner** (Mac) | The runner can reach the local cluster; automatic rollback on failure. |
| Quality | **ruff**, **pytest**, **pytest-cov** | Lint + format checks and unit / integration / e2e tests in CI. |

## Repository structure

```text
.
├── src/ride_classifier/
│   ├── data.py            # loading, cleaning, windows, split, normalisation
│   ├── model.py           # RideCNN (1D CNN)
│   ├── train.py           # training + MLflow tracking + registry + champion alias
│   ├── evaluate.py        # ride-level metrics
│   ├── baseline.py        # naive baselines
│   ├── pyfunc_model.py    # MLflow model: raw ride in, class out
│   └── api.py             # FastAPI service
├── webapp/app.py          # Streamlit web app
├── notebooks/             # data exploration
├── examples/              # two real rides to test the API
├── tests/                 # unit/, integration/, e2e/
├── docker/                # Dockerfiles (api, webapp, mlflow) + bucket creation
├── docker-compose.yml     # dev stack
├── k8s/                   # prod manifests (Kustomize)
├── .github/workflows/     # ci.yml, docker.yml, deploy.yml
├── pyproject.toml, uv.lock
└── requirements.txt       # pip alternative, exported from uv.lock
```

## Getting started (dev)

Prerequisites: Docker Desktop, [uv](https://docs.astral.sh/uv/) (or Python 3.12 + pip), the data in `data/raw/`.

```bash
# 1. Install the dependencies
uv sync --all-groups
#    or, without uv:
#    python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt && pip install -e . --no-deps

# 2. Configure the secrets, then start the stack
cp .env.example .env          # change the passwords
docker compose up -d --build

# 3. Build the dataset and train (the first run becomes the champion)
uv run python -m ride_classifier.data
uv run python -m ride_classifier.train

# 4. Load the new champion in the API (it started before the model existed)
curl -X POST http://localhost:8000/model/reload
```

| Service | URL |
| --- | --- |
| Web app | http://localhost:8501 |
| API (Swagger) | http://localhost:8000/docs |
| MLflow | http://localhost:5001 (5000 is used by macOS AirPlay) |
| MinIO console | http://localhost:9001 |

Example prediction:

```bash
curl -F "file=@examples/ride_safe.csv" http://localhost:8000/predict
```

`requirements.txt` is generated from `uv.lock`, so both always contain the same versions.
It holds the runtime and web app dependencies, not the dev tools. After any change to `pyproject.toml`, run:

```bash
uv lock
uv export --frozen --no-hashes --no-emit-project --no-dev --group webapp -o requirements.txt
```

## Tests

```bash
uv run pytest                      # unit + integration (no running stack needed)
uv run pytest --cov                # with coverage
docker compose up -d && uv run pytest -m e2e   # end-to-end, against the real stack
uv run ruff check . && uv run ruff format --check .
```

- **Unit**: preprocessing, evaluation, model output shapes.
- **Integration**: the API with a fake model (validation errors, 503 without a model, predictions).
- **End-to-end**: the real stack serves the champion and classifies a real ride.

## CI/CD

| Workflow | Trigger | What it does |
| --- | --- | --- |
| `ci.yml` | every PR and push to `main` | ruff lint and format check, unit + integration tests with coverage |
| `docker.yml` | push to `main` | builds `api`, `webapp`, `mlflow` for amd64 and arm64, pushes to Docker Hub (`<sha>` and `latest`), smoke-tests the API image |
| `deploy.yml` | after `docker.yml` succeeds on `main` | on the self-hosted runner: deploys the images of that exact commit to Kubernetes, waits for the rollout, smoke-tests the API, **rolls back** on failure |

Actions are pinned to commit SHAs. Docker Hub credentials are a repository variable (`DOCKERHUB_USERNAME`)
and a secret (`DOCKERHUB_TOKEN`).

## Production on local Kubernetes

```bash
kubectl config use-context docker-desktop
kubectl create namespace ride-classifier
kubectl create secret generic ride-classifier-secrets -n ride-classifier --from-env-file=.env
kubectl apply -k k8s/
```

| Component | Replicas | Resources (request / limit) | Access |
| --- | --- | --- | --- |
| API | 2, rolling update with `maxUnavailable: 0` | 250m CPU, 768Mi / 2Gi | http://localhost:30800 |
| Web app | 1 | 100m CPU, 256Mi / 768Mi | http://localhost:30851 |
| MLflow | 1 | 200m CPU, 512Mi / 1536Mi | http://localhost:30500 |
| MinIO | 1, PVC 5Gi | 100m CPU, 256Mi / 512Mi | console http://localhost:30901 |
| PostgreSQL | 1, PVC 2Gi | 100m CPU, 256Mi / 512Mi | internal |

The API pod is *ready* only when a model is loaded (`/health` returns 503 otherwise), so no traffic reaches a pod
without a model.

## AI Infrastructure

This section extends the existing MLOps project; it is not a new ML project. Its rule is
**measure, do not claim**: every value below is either measured (with its conditions), computed from the code,
or marked **to measure** / **TBD**. Designs and estimates are labelled as such.

The detailed work lives in `ai-infrastructure/` (planned structure):

```text
ai-infrastructure/
├── README.md
├── architecture/
│   ├── current-system.png
│   └── proposed-system.png
├── benchmarks/
│   ├── benchmark.ipynb
│   └── baseline-results.csv
├── optimization/
│   └── comparison.ipynb
├── distributed-training/
│   └── technical-analysis.md
└── cost-resilience/
    └── infrastructure-plan.md
```

### 1. Inventory of the current system

| Component | Status | Where |
| --- | --- | --- |
| Preprocessing | done | `data.py` |
| Training (PyTorch, CPU) | done | `train.py` |
| Tracking, model storage, versioning | done | MLflow + PostgreSQL + MinIO |
| Prediction API | done | `api.py` (FastAPI) |
| Docker images | done | `docker/`, Docker Hub |
| CI/CD | done | `.github/workflows/` |
| Kubernetes (local) | done | `k8s/` |
| Monitoring (Prometheus + Grafana) | not yet | planned |

**Where it runs:** training and serving run locally on a developer Mac (Apple Silicon, CPU only) and on the local
Kubernetes cluster of Docker Desktop. Google Colab and GCP are optional and not used yet. The current-system
diagram is in [Architecture](#architecture).

### 2. Resource sizing

| Item | Value | Source |
| --- | --- | --- |
| Framework | PyTorch 2.15.0 (CPU build on Linux), MLflow 3.18.0 | `uv.lock` |
| Model | `RideCNN`: 3 conv blocks (32, 64, 64 channels) + global average pooling + linear layer | `model.py` |
| Parameters | 24,677 (about 96 KB in float32) | computed from the architecture |
| Model file size (MB) | to measure | `ride_cnn.pt` and the full MLflow artifact |
| Dataset | 28.8 MB on disk, 1,529,547 rows, 500 rides | measured |
| Input shape | `(batch, 600, 6)`: 600 time steps, 6 sensors, float32 | `data.py` |
| Samples (windows) | to measure (printed by `python -m ride_classifier.data`) | `data.py` |
| Train / test split | 350 / 150 rides (70 / 30, stratified) | `data.py` |
| Training hyperparameters | 20 epochs, batch size 64, Adam, learning rate 1e-3, dropout 0.3 | `train.py` |
| Training hardware | CPU of the developer Mac (exact model, cores, RAM: TBD) | to complete |
| Training duration | to measure | wall-clock time of `train.py` |
| Peak RAM during training | to measure | |
| GPU / VRAM | none used | `train.py` has no `.to("cuda")` |
| Serving resources | API pod: request 250m CPU / 768Mi, limit 2Gi | `k8s/api.yaml` |
| MLflow server memory | about 800 MB with 2 workers | measured, noted in `k8s/mlflow.yaml` |

**Is one machine enough?** Our assessment, to confirm with the measurements above: yes. The model is tiny, the
full dataset fits in memory many times over, and training runs on a laptop CPU.

### 3. Baseline benchmark

**Procedure** (to implement in `ai-infrastructure/benchmarks/benchmark.ipynb`):

- Fixed model: one registered version of `ride-classifier` (version number recorded with the results).
- Fixed workload: the test rides of the ride-level split, with the same preprocessing as training.
- Two levels: (a) the model alone (`pyfunc.predict` on one ride, in-process), (b) the HTTP API (`POST /predict`).
- Warm-up runs first, then repeated predictions; report mean and p95 latency, throughput, peak memory.
- Record the CPU, the number of PyTorch threads, batch size (one ride = all its windows) and concurrency.
- No GPU is used, so no CUDA synchronisation is needed; if a GPU is used later, call `torch.cuda.synchronize()`
  before reading the clock.
- Raw measurements saved to `ai-infrastructure/benchmarks/baseline-results.csv`.

| Metric | Value | Measurement conditions |
| --- | --- | --- |
| Model and version | `ride-classifier` vTBD | PyTorch 2.15.0, MLflow 3.18.0 pyfunc |
| Hardware | TBD | CPU / RAM (no GPU) |
| Model size (MB) | to measure | serialized model file |
| Mean latency (ms) | to measure | one ride per request, N repetitions after warm-up |
| p95 latency (ms) | to measure | same |
| Throughput (requests/s) | to measure | concurrency and test duration TBD |
| Memory usage (MB) | to measure | CPU RAM of the API process |
| Classification quality | to measure | macro F1 on the 150 test rides |

### 4. Optimization

**Candidate:** export the network to **ONNX** and run it with **ONNX Runtime** (CPU), keeping the same
preprocessing. Dynamic quantization in PyTorch only covers `Linear` / recurrent layers, so it would barely touch
this convolutional model; static quantization of the `Conv1d` layers is a possible second experiment.

**Hypothesis:** ONNX Runtime lowers the per-ride latency on CPU with the same predictions (macro F1 unchanged).

**Fair comparison:** same model version, same test rides, same hardware, same number of threads, same warm-up and
repetitions. A slower optimized model is still a valid result if it is explained.

| Metric | Baseline | Optimized | Interpretation |
| --- | --- | --- | --- |
| Mean latency (ms) | to measure | to measure | TBD |
| p95 latency (ms) | to measure | to measure | TBD |
| Throughput (requests/s) | to measure | to measure | TBD |
| Memory usage (MB) | to measure | to measure | TBD |
| Classification quality | to measure | to measure | TBD |

Status: not started. Notebook: `ai-infrastructure/optimization/comparison.ipynb`.

### 5. Distributed training decision

**Decision: no distributed training (no DDP, FSDP or ZeRO).** Reasons, from the facts above:

- The model has 24,677 parameters: it fits on any single device, so model sharding (FSDP, ZeRO) has nothing to shard.
- The dataset is 28.8 MB on disk and fits in RAM: data parallelism (DDP) would mostly add communication overhead
  (gradient synchronisation at every step) to a computation that is already small.
- No GPU is needed for this workload, and the course does not require multi-GPU training.

This decision should be checked again if the training duration measured in section 2 becomes a bottleneck, or if
the much larger `SRV3490` dataset (3,490 rides, 256 MB) is used. Even then, a single machine with a GPU or more
data-loader workers would be the first step. Details: `ai-infrastructure/distributed-training/technical-analysis.md`.

### 6. Scaling, cost and resilience

**Proposed serving architecture (design, not deployed):**

- CPU-only API pods behind a Kubernetes Service, stateless, model loaded from the MLflow registry.
- **Horizontal Pod Autoscaler** on CPU usage (min 2 replicas), with thresholds chosen from the benchmark.
  *Not implemented and not tested yet.*
- **Monitoring:** Prometheus metrics from the API (request rate, latency p95, errors, predictions per class) and
  Grafana dashboards. *Planned.*
- MLflow, PostgreSQL and MinIO stay single-replica with persistent volumes; in a cloud setup they would move to
  managed services (Cloud SQL and Cloud Storage on GCP, for example).

**Cost estimate:** to compute with explicit assumptions, using the provider's pricing page.

| Assumption | Value |
| --- | --- |
| Provider and region | TBD (for example GCP, `europe-west1`) |
| Machine type for the API | TBD (CPU only, sized from the benchmark) |
| Number of replicas and usage hours per month | TBD |
| Request volume per month | TBD |
| Storage (MLflow artifacts, database) | TBD |
| **Approximate monthly cost** | **TBD** |

If GCP is used, remaining credits are monitored and unused resources are shut down after each session.

**Recovery and rollback (implemented):**

- **Bad deployment:** `deploy.yml` smoke-tests the API after each rollout and runs `kubectl rollout undo`
  automatically if any step fails. Images are tagged with the commit SHA, so any previous version can be redeployed.
- **Bad model:** a new model only becomes `champion` if its macro F1 is better. To roll back by hand, point the
  `champion` alias to the previous version in MLflow, then call `POST /model/reload`.
- **Pod failure:** readiness, startup and liveness probes; 2 API replicas with `maxUnavailable: 0`.
- **Data loss:** PostgreSQL and MinIO use persistent volumes. Backups: TBD.

### Responsibilities and next actions

| Task | Owner | Status |
| --- | --- | --- |
| Current and proposed architecture diagrams | TBD | to do |
| Resource inventory (hardware, training time, model size) | TBD | to do |
| Baseline benchmark notebook and raw results | TBD | to do |
| ONNX Runtime optimization and comparison | TBD | to do |
| Distributed training analysis | TBD | to do |
| Cost and resilience plan | TBD | to do |

## Screenshots

*To add: web app prediction, Swagger UI, MLflow runs and registry, GitHub Actions runs, `kubectl get pods`.*

## Roadmap

- Retraining pipeline with GitHub Actions (bonus).
- Monitoring with Prometheus and Grafana (bonus).
- Orchestration with Airflow (`dags/`) and a feature store (bonus).
- AI Infrastructure measurements and notebooks (`ai-infrastructure/`).

## Contributors

| Member | Main areas |
| --- | --- |
| Rania | project structure, tests and CI, monitoring, documentation |
| Luli | data exploration, preprocessing, model, Airflow, feature store |
| Wei-Ling | Docker, Docker Hub, Kubernetes, continuous deployment |
| Chanelle | MLflow tracking and registry, API, web app, retraining |
