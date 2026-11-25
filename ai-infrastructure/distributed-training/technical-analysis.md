# Distributed training decision (DDP / FSDP / ZeRO)

**Decision: no distributed training.** One machine is enough: 74–83 s on a CPU, 5 s on one GPU. Splitting the
training over several GPUs would add communication and start-up costs larger than the time it could save.

Tags: **[measured]** = measured in this project (source given), **[computed]** = derived from the code or from
measured numbers (formula given), **[estimate]** = order of magnitude, not measured.

## Evidence

| Factor | Value | Source |
|---|---|---|
| Model size | 24,677 parameters, ≈ 0.1 MB of weights (99 KB in float32) | [measured] `sum(p.numel())` |
| Dataset | GBG500: 30.2 MB parquet, 500 rides | [measured] |
| Training data | 3,026 windows × 600 × 6 float32 ≈ 44 MB: fits in RAM and in GPU memory many times over | [measured] `data.py` output |
| Training on CPU (Mac M4) | **82.8 s** for 20 epochs, including baselines, feature store model and MLflow logging; peak RAM 970 MB | [measured] `/usr/bin/time -l`, [§ 2](../README.md#2-resource-sizing) |
| Training on CPU (cloud, 4 vCPU) | **74.3 s**, 3.70 s per epoch | [measured] [training-results.csv](../benchmarks/vertex-t4/training-results.csv) |
| Training on 1 GPU (NVIDIA T4) | **5.0 s**, 0.22 s per epoch (15× faster than the same cloud CPU) | [measured] same file |
| GPU memory used by training | **60 MB** of the T4's 15 GB (0.4 %) | [measured] `torch.cuda.max_memory_allocated` |
| Steps per epoch | 48 (3,026 windows / batch 64) → 960 steps in 20 epochs | [computed] |
| Time per training step on the T4 | ≈ 4.5 ms (0.216 s / 48) | [computed] from the measurements |
| Gradient exchanged per step with DDP | ≈ 0.1 MB (24,677 × 4 bytes) | [computed] |
| Parameters + gradients + Adam states | ≈ 0.4 MB in total (4 × 99 KB) | [computed] |
| GPUs available | 1 spot T4 per job; the project's on-demand GPU quotas are 0 | [measured] quota checks, [§ 3.1](../README.md#31-cpu-vs-gpu-on-gcp-vertex-ai-t4) |

## Analysis

**What distributed training would do.** With DDP on 2 GPUs, each GPU gets half of every batch (32 windows instead
of 64), computes its gradients, and the GPUs then **average their gradients (All-Reduce)** before every optimizer
step. FSDP and ZeRO go further: they also split (shard) the parameters, gradients and optimizer states across GPUs,
to train models that do not fit in one GPU's memory.

**1. The work per step is too small to split.** One step takes ≈ 4.5 ms on one T4 and uses 60 MB of its 15 GB.
With 32 windows per GPU, each GPU would be even less busy; GPUs are efficient only with enough work per kernel.
At best, DDP could save ≈ 2.5 s over the whole training (half of 5 s) **[estimate]**.

**2. The communication would cost about as much as the computation.** DDP needs one All-Reduce per step: 960 times
during the training, each ≈ 0.1 MB. The data is small, but each All-Reduce has a fixed cost (synchronization of the
GPUs, NCCL calls, PCIe transfers) that is not negligible next to a 4.5 ms step **[estimate]**. This is the trade-off
seen in class: with a small model and a small dataset, communication can make 2 GPUs slower than 1.

**3. The fixed start-up cost is larger than the possible gain.** DDP starts one process per GPU and initializes
the communication (NCCL) before the first step: typically a few seconds **[estimate]**, which is already as long as
the whole training on one T4 (5 s).

**4. FSDP / ZeRO solve a problem we do not have.** All the training state (parameters + gradients + Adam) is
≈ 0.4 MB. Sharding it across GPUs would save nothing and add more communication.

**5. One machine is already enough.** The CPU trains the model in 74–83 s, once a week. Even the simplest
acceleration (one GPU) only saves about one minute per training run.

**6. Practical limit.** The project can get one spot T4 per job; two GPUs on one machine would need a quota
increase that only the project owners can request.

## Decision

- **No DDP, FSDP or ZeRO.** Training stays on one machine.
- **Default: CPU** (74–83 s, no extra cost, same environment as today's pipeline).
- **Optional: one GPU** (one spot T4 on Vertex AI, 5 s) for experiments that need many training runs.
- **We did not measure DDP on 2 GPUs.** The decision is justified by the measured single-device numbers and the
  computed communication volume above, as the brief allows ("a well-supported decision not to use distributed
  training is acceptable").

## When would we reconsider?

| Change | Effect | Better first step than DDP |
|---|---|---|
| More data, e.g. adding SRV3490 (268 MB, 3,490 rides ≈ 7× GBG500) | ≈ 7× longer training: ≈ 9 min on CPU, ≈ 35 s on one T4 **[estimate]** | one GPU |
| Hyperparameter search or cross-validation (many runs) | many independent trainings | run them **in parallel, one per device** (no gradient exchange needed) |
| A much larger model (millions of parameters, e.g. a transformer) | training > tens of minutes on one GPU, more GPU memory | then DDP (model fits in one GPU) or FSDP / ZeRO (model does not fit) |
