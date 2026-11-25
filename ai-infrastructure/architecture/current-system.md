# Current system (as implemented)

Diagram of what exists **today** (not the proposed system).

![Current system](current-system.png)

The PNG is drawn by [draw.py](draw.py) (`uv run python ai-infrastructure/architecture/draw.py`). The same system as
Mermaid, rendered directly by GitHub:

```mermaid
flowchart LR
    subgraph MAC[One Apple M4 Mac, 16 GB]
        subgraph DEV[Docker Compose: dev]
            AF[Airflow DAG<br/>preprocess · feature store · train] --> MLF[MLflow]
            PROM[Prometheus] -->|scrape /metrics| API1[API]
            GRAF[Grafana] --> PROM
        end
        subgraph PROD[Kubernetes: Docker Desktop, prod]
            WEB[Streamlit web app] --> API[FastAPI x2 replicas<br/>pyfunc: preprocessing + CNN, CPU]
            API -->|load champion| MLF2[MLflow]
            MLF2 --> PG[(PostgreSQL<br/>runs, registry)]
            MLF2 --> S3[(MinIO<br/>model artifacts)]
        end
        RUNNER[GitHub self-hosted runner] -->|kubectl apply, rollback| PROD
    end
    GH[GitHub Actions<br/>CI · images · deploy · retrain] --> RUNNER
    GH --> HUB[(Docker Hub<br/>amd64 + arm64 images)]
    HUB --> PROD
```
