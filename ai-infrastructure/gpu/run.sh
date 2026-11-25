#!/usr/bin/env bash
# Entry point of the benchmark job: install, check the GPU, then the same benchmarks on CPU and GPU.
# Results are printed between CSV-BEGIN / CSV-END and read back from the job logs (see README.md).
set -euo pipefail
cd /app

echo "== Installing pinned dependencies"
start=$(date +%s)
pip install -q -r requirements-gpu.txt
echo "install took $(( $(date +%s) - start )) s"

nvidia-smi || {
  echo "no GPU visible. Diagnostics:"
  ls -la /dev/nvidia* /usr/local/nvidia /usr/local/nvidia/lib64 2>&1 | head -20 || true
}
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'gpu available', torch.cuda.is_available())"
DEVICES="cpu"
if python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)"; then DEVICES="cpu cuda"; fi

echo "== Preprocessing"
python -m ride_classifier.data

echo "== Inference benchmark ($DEVICES)"
python -W ignore ai-infrastructure/benchmarks/device_benchmark.py --model-uri /app/model --devices $DEVICES --print-csv

echo "== Training benchmark ($DEVICES)"
python -W ignore ai-infrastructure/benchmarks/train_benchmark.py --devices $DEVICES --print-csv
