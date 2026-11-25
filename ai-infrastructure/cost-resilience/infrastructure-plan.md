# Scaling, cost and resilience plan

Proposed serving architecture: [../architecture/proposed-system.png](../architecture/proposed-system.png).
Tags: **[measured]** in this project, **[price]** official Google Cloud price (source and date below),
**[computed]** from those two, **[estimate]** assumption or order of magnitude, **[design]** proposed, not tested.

## 1. Serving architecture

| Layer | Choice | Why |
|---|---|---|
| Entry | Ingress / load balancer (TLS) **[design]** | one stable entry point; today the API is a NodePort on the Mac |
| API | FastAPI pods on **CPU**, 2 replicas minimum | GPU only 1.3× faster for one ride, +0.5 s CUDA start per pod **[measured]** |
| Model runtime | MLflow pyfunc (PyTorch) today; **ONNX Runtime** proposed **[design]** | start 1.4 s → 0.12 s, −250 MB per process, same predictions **[measured]** |
| Registry | MLflow + PostgreSQL + S3 (MinIO), alias `@champion` | exists; the alias only moves if macro F1 improves |
| Offline | weekly training on CPU (74–83 s), batch scoring with ONNX Runtime on CPU | [§ 3.1, § 4](../README.md) **[measured]** |

## 2. Request scaling

**Measured capacity** (Mac M4, [§ 3](../README.md#3-baseline-benchmark)): one uvicorn worker serves **≈ 220–245
requests/s**, but **one request at a time**: from 1 to 8 concurrent clients the throughput stays flat and the p95
latency grows from 7 ms to 49 ms. Cause: `/predict` is `async def` and calls the model synchronously.

| Step | Action | Status |
|---|---|---|
| 1 | Make `/predict` a sync endpoint (FastAPI runs it in a thread pool) or start N uvicorn workers per pod | **[design]**, to measure with `benchmarks/benchmark.py` |
| 2 | Horizontal Pod Autoscaler: 2 → N replicas on CPU usage (or p95 latency with a custom metric) | **[design]**, needs metrics-server; **not tested** |
| 3 | Keep the pods small (1 vCPU, ≈ 250 MB RAM with ONNX Runtime) so that scaling out is cheap | **[estimate]** from the measured RSS (218 MB today) |

**Expected load vs capacity [estimate].** Assume 10,000 rides scored per day with peaks at 10× the average:
average 0.12 request/s, peak ≈ 1.2 requests/s. A cloud CPU is slower than the Mac (one ride: 5.1 ms on the 2-thread
`n1-standard-4` vs 1.5 ms on the Mac **[measured]**), so one worker on a cloud vCPU would serve roughly 60–70
requests/s **[estimate]**: still ≈ 50× the assumed peak. **Two replicas are needed for availability, not for
capacity.** The real cloud throughput must be measured before sizing a real deployment.

## 3. Monitoring

| What | Today | Proposed |
|---|---|---|
| API: requests, errors, p50 / p95 latency | Prometheus + Grafana, **dev only** | same dashboards in the cluster (Helm charts) **[design]** |
| Model: predicted class mix, mean confidence, served version | dev only | idem; drift signal when the class mix or the confidence shifts |
| Data: ride length | dev only | idem |
| Alerts | `ApiDown`, `HighErrorRate` (> 5 % 5xx), `SlowPredictions` (p95 > 2 s), `LowModelConfidence` (< 50 %) | route to e-mail / chat; tighten `SlowPredictions` to the measured scale (p95 ≈ 7 ms today, alert at 100 ms) **[design]** |
| Infrastructure | Kubernetes probes (startup, readiness on `/health`, liveness) | + node and pod metrics (CPU, memory, restarts) for the autoscaler |

## 4. Costs

### 4.1 What our GPU experiments cost

Source: start and end times of our four Vertex AI jobs (europe-west4, 1 spot T4 + `n1-standard-4`), Vertex AI
billing rule "charged in 30-second increments; Spot VMs are billed according to Compute Engine Spot VMs pricing".

| Job | Running time [measured] | Billed time [computed] |
|---|---|---|
| GPU smoke test | 32 s | 60 s |
| Benchmark run 1 (GPU not visible in the container) | 302 s | 330 s |
| Benchmark run 2 (GPU not visible in the container) | 302 s | 330 s |
| Benchmark run 3 (CPU vs GPU results) | 332 s | 360 s |
| **Total** | **968 s** | **1,080 s = 0.30 h** |

| Price assumption | Hourly price | Cost |
|---|---|---|
| Spot: `n1-standard-4` $0.125185 + T4 $0.1959 **[price]** | $0.321 / h | **≈ $0.10** |
| Upper bound: on-demand Vertex training price ($0.2405 + $0.4025 **[price]**) and the waiting time before the start counted too (2,352 s) | $0.643 / h | ≤ $0.42 |

Not included: Artifact Registry storage of the three benchmark images (≈ 50 MB each, cents per month) and logs.
**The exact amount must be read in the billing report** (Billing → Reports, filter "Vertex AI"): our accounts do
not have access to the billing account, so this is computed from the job times, not read from the invoice.

### 4.2 Monthly cost of the proposed system [estimate]

Assumptions: region europe-west4, 730 h per month, everything inside one **zonal GKE Standard** cluster (same
components as today: API ×2, web app, MLflow, PostgreSQL, MinIO, Prometheus + Grafana), **2 × `e2-standard-2`**
nodes (2 vCPU, 8 GB each; today's pod requests ≈ 2.8 GiB + monitoring fit with room for a node failure), 50 GB
balanced boot disk per node, 7 GiB of volumes (PostgreSQL 2 Gi + MinIO 5 Gi), one load balancer.

| Item | Unit price [price] | Quantity | Per month [computed] |
|---|---|---|---|
| GKE cluster management fee | $0.10 / h per cluster | 730 h | $73.00 |
| `e2-standard-2` nodes, on-demand | $0.07377204 / h | 2 × 730 h | $107.71 |
| Balanced persistent disk | $0.000150685 / GiB-h | 107 GiB × 730 h | $11.77 |
| Load balancer forwarding rule (first 5) | $0.025 / h | 730 h | $18.25 |
| **Total** | | | **≈ $211 / month** |
| GKE free tier (one zonal or Autopilot cluster per billing account: $74.40 / month) | | if still unused on the billing account | **≈ $136 / month** |
| Nodes on Spot (`e2-standard-2` $0.04196 / h) instead of on-demand | | 2 × 730 h: $61.26 | **≈ $90–164 / month**, but pods are stopped when Google reclaims the node: fine for dev, risky for prod |

Not included: network egress (small: one CSV of ≈ 100 KB per request), Cloud Logging above the free quota, backups.

**Offline workloads [computed]:** weekly CPU training (≈ 80 s) and batch scoring (93 ms per 1,323 windows) run on
the same nodes: ≈ $0. Optional training on a spot T4 job: ≈ 5.5 min billed per run (mostly installing packages) →
≈ $0.03 per run.

**Why not a GPU for serving [computed, estimate]:** one `n1-standard-4` + T4 around the clock costs
$0.209168 (VM, on-demand) + ≈ $0.40 (T4; Vertex AI training price used as a proxy, the Compute Engine on-demand T4
price was not read) ≈ **$0.61 / h ≈ $447 / month per GPU node**, against **$54 / month** for one `e2-standard-2`,
for a 1.3× gain on one ride.

**Price sources (read on 2026-11-21):**
[Vertex AI / Gemini platform pricing](https://cloud.google.com/vertex-ai/pricing) (custom-trained models, Netherlands),
[Spot VM pricing](https://cloud.google.com/spot-vms/pricing) (Netherlands),
[General-purpose VM pricing](https://cloud.google.com/products/compute/pricing/general-purpose) (Netherlands),
[GKE pricing](https://cloud.google.com/kubernetes-engine/pricing),
[Disk pricing](https://cloud.google.com/compute/disks-image-pricing) (Netherlands),
[Network pricing](https://cloud.google.com/vpc/network-pricing) (forwarding rules).
Spot prices change over time.

## 5. Resilience: failure scenarios and recovery

| Failure | What happens today | Recovery | Status |
|---|---|---|---|
| An API pod crashes or hangs | the other replica keeps serving; liveness probe restarts the pod; readiness (`/health`) keeps traffic away until a model is loaded | automatic, ≈ 5 s cold start **[measured]** (the startup probe allows up to 5 min) | exists |
| A new image is broken | rolling update with `maxUnavailable: 0`: old pods serve until new ones are ready | `deploy.yml` runs smoke tests (`/health`, `/model`, a real `/predict`) and runs `kubectl rollout undo` on failure | exists, used in CI |
| A new model is worse | retraining promotes `@champion` only if macro F1 improves | to roll back a model: point `@champion` to the previous version in MLflow, then `POST /model/reload` (8 ms, no restart) **[measured]** | exists (alias + reload); the alias change is manual |
| MLflow is down | running pods keep the model in memory and keep serving; **new pods cannot load a model** (they stay not-ready, `/health` = 503) | restart MLflow (initContainer waits for PostgreSQL) | exists; single MLflow replica = single point of failure |
| PostgreSQL or MinIO volume lost | registry or model artifacts lost; serving pods keep their loaded model | today: no backup → retrain (≈ 80 s) and re-register | **gap**: add a daily `pg_dump` CronJob and object versioning / copy of the bucket **[design]** |
| A node fails | today: one Mac = everything stops | with 2 nodes + 2 API replicas + PodDisruptionBudget, one node can fail **[design]** | **gap** today |
| The Mac / self-hosted runner is off | CD and retraining cannot run; nothing is deployed | manual | **gap**: a managed cluster and a cloud runner remove this |
| Spot GPU job pre-empted | the benchmark or training job stops | re-submit the job (results are only written at the end) | accepted (experiments only) |

**Targets [estimate]:** pod failure → back in ≈ 5–10 s (cold start + probes); bad deployment → rollback in
≈ 1 min (CD smoke tests); bad model → ≈ 1 min (alias change + reload); lost volumes → today up to a full retraining
(≈ 2 min of compute, but manual), after backups ≤ 1 day of registry history lost.

**Not tested here:** autoscaling, node failure, backup / restore, a managed cluster. Only the CD rollback, the
probes and the model reload exist and run today.
