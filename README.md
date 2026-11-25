# Ride Classifier: MLOps and AI Infrastructure

[![CI](https://github.com/ChanelleSnz/mlops-ride-classifier/actions/workflows/ci.yml/badge.svg)](https://github.com/ChanelleSnz/mlops-ride-classifier/actions/workflows/ci.yml)
[![Docker images](https://github.com/ChanelleSnz/mlops-ride-classifier/actions/workflows/docker.yml/badge.svg)](https://github.com/ChanelleSnz/mlops-ride-classifier/actions/workflows/docker.yml)

An end-to-end MLOps pipeline that classifies micromobility rides from their IMU sensor data
(accelerometer and gyroscope) into 5 classes: **Safe, Safe overall, Bad weather, Tandem, Reckless**.

Group project at Albert School (M2) shared by two courses:

- **MLOps & Model Deployment**: training, tracking, registry, API, web app, Docker, CI/CD, Kubernetes, tests.
- **Advanced Cloud AI Infrastructure**: resource sizing, CPU vs GPU benchmarking, optimization (ONNX Runtime),
  distributed training decision, scaling, costs and resilience (see [AI Infrastructure](#ai-infrastructure)).

**Team:** Rania, Luli, Wei-Ling, Chanelle.

| | |
|---|---|
| **Docker images** | [`raniamehria/ride-classifier-api`](https://hub.docker.com/r/raniamehria/ride-classifier-api) · [`raniamehria/ride-classifier-webapp`](https://hub.docker.com/r/raniamehria/ride-classifier-webapp) · [`raniamehria/ride-classifier-mlflow`](https://hub.docker.com/r/raniamehria/ride-classifier-mlflow) (amd64 + arm64, tagged `latest` and `<commit sha>`) |
| **Stack** | PyTorch · MLflow 3.18 · PostgreSQL · MinIO · FastAPI · Streamlit · Docker Compose · Kubernetes (Docker Desktop) · GitHub Actions · Feast · Airflow 3.4 · Prometheus · Grafana · uv |

## Table of contents

1. [Problem and data](#problem-and-data)
2. [Architecture](#architecture)
3. [Technical choices](#technical-choices)
4. [Results](#results)
5. [Repository structure](#repository-structure)
6. [Getting started (dev)](#getting-started-dev)
7. [Feature store](#feature-store)
8. [Orchestration (Airflow)](#orchestration-airflow)
9. [Monitoring](#monitoring)
10. [API](#api)
11. [Tests](#tests)
12. [CI/CD](#cicd)
13. [Production on local Kubernetes](#production-on-local-kubernetes)
14. [AI Infrastructure](#ai-infrastructure)
15. [Problems met and how we solved them](#problems-met-and-how-we-solved-them)
16. [Limits and next steps](#limits-and-next-steps)
17. [Screenshots](#screenshots)
18. [Contributors](#contributors)

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

Findings from the exploration (`notebooks/01_exploration.py`): 0.7% of rows have no sensor value (439 rides),
one duplicated timestamp, gyroscope values ~30× larger than accelerometer values, similar ride durations across classes
(so duration cannot be used as a shortcut).

![One ride per class](docs/eda_one_ride_per_class.png)

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

Around this core: a **Feast feature store** (ride-level features for a second, feature-based model), an **Airflow**
training pipeline, **Prometheus + Grafana** monitoring, and a **retraining workflow** that redeploys only a better model.
Each one has its own section below.

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
| Feature store | **Feast** (local: parquet offline store, SQLite online store) | Same ride features for training and lookup, defined once; no server to run. |
| Orchestration | **Airflow 3.4** | Scheduled and manual training runs, with clear task-level logs and retries. |
| Monitoring | **Prometheus** + **Grafana** | API, model and data metrics, alert rules, provisioned dashboard. |

## Results

Test set: 150 rides never seen during training.

| Model | Accuracy | Balanced accuracy | **Macro F1** | F1 Reckless |
|---|---|---|---|---|
| Baseline: always "Safe" | 0.833 | 0.200 | 0.182 | 0.000 |
| Baseline: stratified random | 0.827 | 0.198 | 0.181 | 0.000 |
| **1D CNN (champion v1)** | 0.607 | 0.342 | **0.272** | 0.000 |

The CNN beats the baselines on macro F1 (+49%). Accuracy goes down on purpose: class weights trade some "Safe"
precision for the rare classes (Bad weather: 4/7 rides found). **Reckless is not detected**: only 6 rides in total
(4 for training, 2 for testing) — a data limitation, see [Limits](#limits-and-next-steps).

![Confusion matrix](docs/screenshots/confusion-matrix.png)

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
│   ├── features.py        # ride-level features written for the feature store
│   ├── feature_model.py   # random forest trained on the feature store's training set
│   └── api.py             # FastAPI service + Prometheus metrics
├── webapp/app.py          # Streamlit web app
├── notebooks/             # data exploration
├── examples/              # two real rides to test the API
├── tests/                 # unit/, integration/, e2e/
├── feature_repo/          # Feast feature store (definitions, training set, tests)
├── dags/                  # Airflow training DAG
├── monitoring/            # Prometheus config + alerts, Grafana dashboard
├── scripts/               # send_traffic.py, champion_version.py
├── ai-infrastructure/     # benchmarks, optimization, AI Infrastructure analysis
├── docker/                # Dockerfiles (api, webapp, mlflow, airflow) + bucket creation
├── docker-compose.yml     # dev stack
├── k8s/                   # prod manifests (Kustomize)
├── .github/workflows/     # ci.yml, docker.yml, deploy.yml, retrain.yml
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

## Feature store

A local **Feast** repository in `feature_repo/` (its own environment: Feast and the ML libraries have different
dependencies). Everything is a file, nothing to run:

- **Entity** `ride` (key `ride_id`), **feature view** `ride_stats`: 33 summary statistics of the whole ride
  (duration, missing ratio, mean / std / min / max of each sensor, acceleration and rotation magnitudes, jerk).
- **Offline store**: the parquet file written by `python -m ride_classifier.features`, used to build the training set
  with a point-in-time join. **Online store**: SQLite, filled by `materialize`, for low-latency lookup of one ride.
- **Feature service** `ride_classifier_v1`: the exact features used at training.
- `src/ride_classifier/feature_model.py` trains a random forest on this training set, on the same test rides as the CNN,
  so the CNN is compared with a feature-based model and not only with naive baselines. Scores are logged in the same MLflow run.

```bash
uv run python -m ride_classifier.features                                   # compute the ride features
uv run --project feature_repo python feature_repo/build_training_set.py     # register, build the training set, materialize
uv run python -m ride_classifier.feature_model                              # train the feature-based model
```

Labels are not stored in Feast: they stay in `GBG500_labels.csv` and are joined by the training code.
`feature_repo/tests/test_feature_repo.py` checks that the feature list matches `features.py`.

## Orchestration (Airflow)

```bash
docker compose --profile airflow up -d --build airflow    # http://localhost:8080
```

DAG `ride_classifier_training` (weekly + manual): `check_raw_data → preprocess → train_and_register → reload_api`.
Airflow and the ML libraries have conflicting dependencies, so the project lives in its own virtual env inside the
Airflow image and the DAG calls that Python.

## Monitoring

```bash
docker compose up -d prometheus grafana
uv run python scripts/send_traffic.py --n 60     # real rides + 10% invalid files
```

The API exposes `/metrics`. Grafana (http://localhost:3000) shows four layers:

| Layer | Metrics |
|---|---|
| API | requests per route and status, errors, p50/p95 latency |
| Model | predicted class mix and mean confidence (drift signals), served model version |
| Data | length of the received rides |

Alert rules (`monitoring/prometheus/alerts.yml`): `ApiDown`, `HighErrorRate` (>5% 5xx), `SlowPredictions`
(p95 > 2 s), `LowModelConfidence` (mean < 50%: incoming rides may differ from training data).

![Grafana](docs/screenshots/grafana.png)

## API

| Route | Purpose |
|---|---|
| `POST /predict` | CSV of one ride (`time_index, ax, ay, az, rx, ry, rz`) → class, probabilities, model version and run id |
| `GET /health` | 200 when a model is loaded, 503 otherwise (Kubernetes readiness) |
| `GET /model` | served model name, alias, version, MLflow run id |
| `POST /model/reload` | load the current champion without restarting |
| `GET /metrics` | Prometheus metrics |

Clear errors: 422 (missing columns, unreadable or empty file), 413 (too many rows), 503 (no model).

![Swagger](docs/screenshots/swagger.png)
![Web app](docs/screenshots/webapp.png)

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
- **Feature store**: `tests/unit/test_features.py` and `feature_repo/tests/` (run with `uv run --project feature_repo pytest feature_repo/tests`).

Coverage ~60%: training code needs the data and an MLflow server, it is exercised by the e2e test, Airflow and the
retraining workflow rather than by CI.

## CI/CD

| Workflow | Trigger | What it does |
| --- | --- | --- |
| `ci.yml` | every PR and push to `main` | ruff lint and format check, unit + integration tests with coverage |
| `docker.yml` | push to `main` | builds `api`, `webapp`, `mlflow` for amd64 and arm64, pushes to Docker Hub (`<sha>` and `latest`), smoke-tests the API image |
| `deploy.yml` | after `docker.yml` succeeds on `main` | on the self-hosted runner: deploys the images of that exact commit to Kubernetes, waits for the rollout, smoke-tests the API, **rolls back** on failure |
| `retrain.yml` | weekly + manual | on the self-hosted runner: retrain, register in the prod MLflow, rolling restart of the API **only if a new champion** was promoted |

Actions are pinned to commit SHAs. Docker Hub credentials are a repository variable (`DOCKERHUB_USERNAME`)
and a secret (`DOCKERHUB_TOKEN`).

![Deployment](docs/screenshots/github-actions-deploy.png)

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

Contribution for the **Advanced Cloud AI Infrastructure** course: what the model needs to run, how fast it is, one
optimization, and a scalable, cost-aware serving plan. It extends this MLOps project (same model, same data).
Rule: **measure, do not claim**. Every number is tagged in the detailed report as measured, computed, estimated or
design only.

**Full report, code and raw results: [`ai-infrastructure/README.md`](ai-infrastructure/README.md).**

| Step | Main result | Details |
| --- | --- | --- |
| Resource sizing | 24,677 parameters (0.1 MB of weights); training 82.8 s on the Mac CPU, peak RAM 970 MB; API process ≈ 218 MB. **One machine is enough.** | [§ 2](ai-infrastructure/README.md#2-resource-sizing) |
| CPU baseline (Mac M4) | model 1.8 ms and API 4.5 ms per ride (mean); ≈ 220–245 requests/s per worker, but one request at a time (p95 7 → 49 ms from 1 to 8 clients); cold start ≈ 5 s, hot reload 8 ms | [§ 3](ai-infrastructure/README.md#3-baseline-benchmark), [`benchmarks/`](ai-infrastructure/benchmarks/) |
| CPU vs GPU (Vertex AI, 1 spot NVIDIA T4) | one ride: GPU only **1.3× faster**, + 0.5 s CUDA start per pod; big batches up to 45×; training 74 s (CPU) vs 5 s (T4) | [§ 3.1](ai-infrastructure/README.md#31-cpu-vs-gpu-on-gcp-vertex-ai-t4), [`gpu/`](ai-infrastructure/gpu/) |
| Optimization: ONNX Runtime | same predictions (150 / 150 rides); start 1.4 s → 0.12 s, memory −46 %, big batches 15× faster; no stable gain for one ride (hypothesis written before measuring) | [§ 4](ai-infrastructure/README.md#4-optimization-onnx-runtime), [`optimization/`](ai-infrastructure/optimization/) |
| Distributed training | **no DDP, FSDP or ZeRO**: one T4 trains in 5 s using 60 MB of its 15 GB; communication would cost more than it saves | [technical-analysis.md](ai-infrastructure/distributed-training/technical-analysis.md) |
| Scaling, cost, resilience | serving stays on **CPU** (a T4 node ≈ $447 / month vs ≈ $54 for a CPU node, for 1.3×); proposed GKE system ≈ $211 / month (≈ $136 with the free tier); our GPU experiments cost ≈ $0.10; 8 failure scenarios with recovery | [infrastructure-plan.md](ai-infrastructure/cost-resilience/infrastructure-plan.md) |

| Current system (implemented and tested) | Proposed system (design, not tested) |
| --- | --- |
| [![Current system](ai-infrastructure/architecture/current-system.png)](ai-infrastructure/architecture/current-system.png) | [![Proposed system](ai-infrastructure/architecture/proposed-system.png)](ai-infrastructure/architecture/proposed-system.png) |

**Not tested:** the autoscaler, a node failure, backups and a managed cluster are designs; the CD rollback, the
probes and the model hot reload exist and run today. Owners of each part: see
[`ai-infrastructure/README.md`](ai-infrastructure/README.md#owners).

## Problems met and how we solved them

| Problem | Diagnosis | Fix |
|---|---|---|
| `minio/minio` images removed from Docker Hub (Sept. 2026) | `pull access denied` | `cgr.dev/chainguard/minio` (MinIO rebuilt from source), bucket created with boto3 instead of `minio/mc` |
| MLflow options silently ignored | the bucket stayed empty although runs were logged | YAML folded block: extra indentation kept the line breaks → rewritten as a list |
| MLflow crash with PostgreSQL | `No module named 'psycopg'` | SQLAlchemy 2.1 uses psycopg 3 by default |
| MLflow **OOMKilled** in Kubernetes | measured memory per process: 4 workers + a GenAI job runner = 2.7 GB | `--workers 2`, `MLFLOW_SERVER_ENABLE_JOB_EXECUTION=false` → 0.8 GB |
| MLflow crashing when starting before PostgreSQL | `CrashLoopBackOff` | initContainer `wait-for-postgres` |
| Apple Silicon Mac vs amd64 GitHub runners | images must run on both | multi-arch build on native amd64 and arm64 runners |
| Docker Hub login failing after fixing the variable | re-runs reuse the variables of the first attempt | start a new run instead of re-running |
| API image `Permission denied` | local files were `600`, container runs as non-root | `chmod -R a+rX` in the Dockerfiles |

## Limits and next steps

- **Reckless is not detected** (6 rides in total). Next: use the 3,490 unlabelled rides of `SRV3490` for
  self-supervised pre-training, or treat Reckless as anomaly detection, as `data.txt` suggests.
- Test set of 150 rides only: scores vary between runs and machines → cross-validation.
- Monitoring runs in the dev environment; next step: deploy Prometheus and Grafana in the cluster with the official Helm charts.
- MinIO image only available as `latest` for free (Chainguard) → mirror a fixed version in our own registry.
- Horizontal Pod Autoscaler, managed cluster and backups: designed and costed, not deployed (see [AI Infrastructure](#ai-infrastructure)).

## Screenshots

| | |
|---|---|
| ![MLflow](docs/screenshots/mlflow-run.png) | ![Kubernetes](docs/screenshots/k8s-pods.png) |
| ![Airflow](docs/screenshots/airflow-dag.png) | ![Grafana](docs/screenshots/grafana.png) |
| ![Prometheus alerts](docs/screenshots/prometheus-alerts.png) | ![MinIO](docs/screenshots/minio.png) |

## Contributors

| Member | Main areas |
| --- | --- |
| Rania | project structure, tests and CI, monitoring, documentation |
| Luli | data exploration, preprocessing, model, Airflow, feature store |
| Wei-Ling | Docker, Docker Hub, Kubernetes, continuous deployment |
| Chanelle | MLflow tracking and registry, API, web app, retraining |
