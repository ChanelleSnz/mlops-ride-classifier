# Scaling, cost and resilience plan

## Serving architecture
To write (link `../architecture/proposed-system.png`).

## Request scaling
To write. Today: fixed 2 API replicas, no autoscaling tested.

## Monitoring
To write. Today: Prometheus + Grafana in dev, alerts ApiDown, HighErrorRate, SlowPredictions, LowModelConfidence.

## Cost estimate
State every assumption: region, machine type, hours per month, requests per day, price source and date.

| Option | Machine | Hours / month | Price | Monthly cost |
|---|---|---|---|---|

## Recovery and rollback
To write. Today: `deploy.yml` rolls back automatically (`kubectl rollout undo`) when smoke tests fail;
the `champion` alias only moves when macro F1 improves.
