#!/usr/bin/env bash
# Build the benchmark image (linux/amd64) and push it to Artifact Registry.
# Run from the project root:  bash ai-infrastructure/gpu/build.sh
# Needs: Docker, gcloud logged in, the data in data/raw/, and the model in the MLflow registry
# (MLFLOW_TRACKING_URI; default: the local SQLite registry used for the CPU baseline).
set -euo pipefail

PROJECT=${PROJECT:-advanced-cloud-ai-i2}
REGISTRY=${REGISTRY:-europe-west1-docker.pkg.dev/$PROJECT/cloud-ai}
TAG=${TAG:-$(date +%Y%m%d-%H%M)}
IMAGE=${IMAGE:-$REGISTRY/ride-classifier-bench:$TAG}
MODEL_URI=${MODEL_URI:-models:/ride-classifier@champion}
export MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI:-sqlite:///$PWD/mlruns/mlflow.db}

context=$(mktemp -d)
trap 'rm -rf "$context"' EXIT

# Code, data (copied, not symlinked) and the exact model version used for the CPU baseline
cp -R src "$context/"
mkdir -p "$context/ai-infrastructure/benchmarks" "$context/data/raw"
cp ai-infrastructure/benchmarks/*.py "$context/ai-infrastructure/benchmarks/"
cp -L data/raw/GBG500.parquet data/raw/GBG500_labels.csv "$context/data/raw/"
cp ai-infrastructure/gpu/Dockerfile ai-infrastructure/gpu/run.sh "$context/"
uv run python -c "import mlflow; mlflow.artifacts.download_artifacts('$MODEL_URI', dst_path='$context/model')" >/dev/null

# Same pinned versions as uv.lock, except torch: the default PyPI build (CUDA on Linux) instead of +cpu
uv export --frozen --no-hashes --no-dev --no-emit-project \
  | sed -E -e "/^torch==[^ ]+ ; sys_platform != 'linux'/d" -e "s/^torch==([0-9.]+)\+cpu ; sys_platform == 'linux'/torch==\1/" \
  > "$context/requirements-gpu.txt"
grep -q "^torch==[0-9.]*$" "$context/requirements-gpu.txt" || { echo "torch pin not found"; exit 1; }

# Short-lived login (1 h access token), no permanent Docker credential helper
gcloud auth print-access-token | docker login -u oauth2accesstoken --password-stdin "https://${REGISTRY%%/*}" >/dev/null
docker buildx build --platform linux/amd64 -t "$IMAGE" --push "$context"
echo "$IMAGE"
