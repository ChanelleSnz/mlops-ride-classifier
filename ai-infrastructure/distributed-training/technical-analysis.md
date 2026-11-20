# Distributed training decision (DDP / FSDP / ZeRO)

## Evidence

| Factor | Value | Source |
|---|---|---|
| Model size | 24,677 parameters, ≈ 0.1 MB of weights | [measured] |
| Dataset | 30.2 MB parquet, 500 rides; training windows fit in RAM | [measured] |
| Training time on one CPU | 82.8 s for 20 epochs, including MLflow logging | [measured] `/usr/bin/time -l` |
| Peak training RAM | 970 MB | [measured] same |
| Training windows | 3,026 × 600 × 6 float32 ≈ 44 MB | [measured] `data.py` output |
| Hardware | 1 Apple M4 (10 cores, 16 GB), no GPU used | [measured] |

## Analysis

To write: compare the training time with the overhead of starting workers and synchronizing gradients;
FSDP / ZeRO shard parameters and optimizer states, which only helps when the model does not fit in one device's memory.

## Decision

To write (expected: no distributed training, justified by the numbers above).
