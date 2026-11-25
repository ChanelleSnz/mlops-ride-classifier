"""Draw the current and the proposed architecture as PNG (diagrams as code, no external tool).

Usage (from the project root): uv run python ai-infrastructure/architecture/draw.py
Writes current-system.png and proposed-system.png next to this file.
Numbers in the boxes are measurements from ai-infrastructure/ (benchmarks/, optimization/).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch  # noqa: E402

HERE = Path(__file__).resolve().parent

# Colours: existing and tested (blue/green), proposed design (orange), storage (grey), groups (light)
EXISTS = "#dbe9f6"
EXISTS_EDGE = "#2f6ea5"
PROPOSED = "#fde7cf"
PROPOSED_EDGE = "#c96a12"
STORE = "#e8e8e8"
STORE_EDGE = "#6b6b6b"
MEASURE = "#1f7a3a"


class Canvas:
    def __init__(self, title: str, subtitle: str, width: float = 18, height: float = 10.5):
        self.fig, self.ax = plt.subplots(figsize=(width, height))
        self.ax.set_xlim(0, width)
        self.ax.set_ylim(0, height)
        self.ax.axis("off")
        self.ax.text(0.3, height - 0.35, title, fontsize=19, weight="bold", va="top")
        self.ax.text(0.3, height - 0.95, subtitle, fontsize=11, color="#444", va="top")
        self.boxes: dict[str, tuple[float, float, float, float]] = {}

    def group(self, x, y, w, h, label, color="#f4f6f8", edge="#9aa5b1", dashed=False):
        self.ax.add_patch(
            FancyBboxPatch(
                (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.15", fc=color, ec=edge, lw=1.4,
                ls="--" if dashed else "-", zorder=0,
            )
        )  # fmt: skip
        self.ax.text(x + 0.15, y + h - 0.12, label, fontsize=11, weight="bold", color="#333", va="top", zorder=1)

    def box(self, key, x, y, w, h, title, detail="", kind="exists", measure=""):
        fc, ec, ls = {
            "exists": (EXISTS, EXISTS_EDGE, "-"),
            "proposed": (PROPOSED, PROPOSED_EDGE, "--"),
            "store": (STORE, STORE_EDGE, "-"),
        }[kind]
        self.ax.add_patch(
            FancyBboxPatch(
                (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.1", fc=fc, ec=ec, lw=1.6, ls=ls, zorder=2,
            )
        )  # fmt: skip
        lines = 1 + bool(detail) + bool(measure)
        top = y + h / 2 + (lines - 1) * 0.17
        self.ax.text(x + w / 2, top, title, ha="center", va="center", fontsize=10.5, weight="bold", zorder=3)
        offset = 0.34
        if detail:
            self.ax.text(
                x + w / 2, top - offset, detail, ha="center", va="center", fontsize=8.6, color="#333", zorder=3
            )
            offset += 0.32
        if measure:
            self.ax.text(x + w / 2, top - offset, measure, ha="center", va="center", fontsize=8.4, color=MEASURE,
                         style="italic", zorder=3)  # fmt: skip
        self.boxes[key] = (x, y, w, h)

    def _anchor(self, key, side):
        x, y, w, h = self.boxes[key]
        return {
            "l": (x, y + h / 2), "r": (x + w, y + h / 2), "t": (x + w / 2, y + h), "b": (x + w / 2, y),
            "tl": (x + w * 0.25, y + h), "tr": (x + w * 0.75, y + h), "bl": (x + w * 0.25, y), "br": (x + w * 0.75, y),
            "rt": (x + w, y + h * 0.75), "rb": (x + w, y + h * 0.25), "lt": (x, y + h * 0.75), "lb": (x, y + h * 0.25),
        }[side]  # fmt: skip

    def arrow(self, a, b, label="", sa="r", sb="l", dashed=False, rad=0.0, color="#3d4752", label_pos=0.5, dy=0.14):
        start, end = self._anchor(a, sa), self._anchor(b, sb)
        self.ax.add_patch(
            FancyArrowPatch(
                start, end, arrowstyle="-|>", mutation_scale=13, lw=1.3, color=color, ls="--" if dashed else "-",
                connectionstyle=f"arc3,rad={rad}", zorder=1.5, shrinkA=2, shrinkB=2,
            )
        )  # fmt: skip
        if label:
            mx = start[0] + (end[0] - start[0]) * label_pos
            my = start[1] + (end[1] - start[1]) * label_pos
            self.ax.text(mx, my + dy, label, ha="center", va="bottom", fontsize=8.3, color=color, zorder=4,
                         bbox={"fc": "white", "ec": "none", "pad": 0.6, "alpha": 0.85})  # fmt: skip

    def legend(self, items):
        self.ax.legend(handles=items, loc="upper right", fontsize=9, frameon=True, ncol=len(items))

    def save(self, name: str) -> None:
        self.fig.tight_layout()
        path = HERE / name
        self.fig.savefig(path, dpi=150)
        plt.close(self.fig)
        print(f"-> {path.name}")


def current_system() -> None:
    c = Canvas(
        "Current system (as implemented and tested)",
        "Everything runs on one Apple M4 Mac (10 cores, 16 GB). CPU only: no GPU in serving or training. "
        "Green italic = measured (ai-infrastructure/benchmarks).",
        width=20.4,
    )
    # GitHub (left column)
    c.group(0.8, 4.6, 4.6, 4.7, "GitHub")
    c.box("ci", 1.1, 8.0, 4.0, 0.95, "CI on every pull request", "ruff + pytest (+ Feast tests)")
    c.box("img", 1.1, 6.85, 4.0, 0.95, "Image build on merge to main", "API · web app · MLflow, amd64 + arm64")
    c.box("deploy", 1.1, 5.7, 4.0, 0.95, "Deploy workflow", "smoke tests, automatic rollback")
    c.box("retrain", 1.1, 4.8, 4.0, 0.75, "Retrain workflow", "weekly / manual")
    c.box("hub", 1.1, 3.1, 4.0, 0.85, "Docker Hub", "images pinned to the commit SHA", kind="store")

    # The Mac
    c.group(5.8, 0.3, 12.2, 9.0, "Apple M4 Mac (developer machine)")
    c.box("runner", 6.1, 6.95, 3.0, 1.1, "Self-hosted runner", "kubectl apply / rollout undo")

    c.group(9.3, 4.3, 8.4, 4.8, "Kubernetes (Docker Desktop) = production", color="#eef4fb", edge=EXISTS_EDGE)
    c.box("api", 9.6, 6.9, 4.9, 1.35, "Prediction API, 2 replicas", "FastAPI + MLflow pyfunc (PyTorch CPU)",
          measure="4.5 ms / ride · ≈ 220 req/s per pod · cold start ≈ 5 s")  # fmt: skip
    c.box("web", 14.9, 7.05, 2.6, 1.05, "Web app", "Streamlit, 1 pod")
    c.box("mlflow", 9.6, 4.6, 4.9, 1.1, "MLflow server", "runs + registry, alias @champion")
    c.box("pg", 14.9, 5.6, 2.6, 0.85, "PostgreSQL", "runs, registry", kind="store")
    c.box("minio", 14.9, 4.55, 2.6, 0.85, "MinIO (S3)", "model artifacts, 1.3 MB", kind="store")

    c.box("user", 18.4, 7.05, 1.8, 1.05, "User", "browser")

    c.group(6.1, 0.55, 11.6, 3.45, "Docker Compose = development", color="#f2f7f0", edge="#5b8f4e")
    c.box("data", 6.4, 2.35, 2.8, 1.0, "Raw data", "GBG500: 30 MB, 500 rides", kind="store")
    c.box("airflow", 9.6, 2.2, 4.0, 1.3, "Airflow DAG (weekly)", "preprocess · Feast features · train · reload",
          measure="training 83 s, 970 MB RAM (CPU)")  # fmt: skip
    c.box("feast", 9.6, 0.75, 4.0, 1.0, "Feast feature store", "offline parquet + SQLite online", kind="store")
    c.box("mon", 14.0, 0.75, 3.5, 2.75, "Prometheus + Grafana", "API, model and data metrics,\n4 alert rules",
          measure="dev only (not in Kubernetes)")  # fmt: skip

    c.arrow("ci", "img", sa="b", sb="t")
    c.arrow("img", "deploy", sa="b", sb="t")
    c.arrow("img", "hub", "push", sa="l", sb="l", rad=0.2, label_pos=0.5, dy=-0.1)
    c.arrow("deploy", "runner", "triggers", sa="r", sb="lb", rad=-0.15)
    c.arrow("retrain", "runner", sa="r", sb="b", rad=-0.25)
    c.arrow("runner", "api", sa="r", sb="l")
    c.arrow("hub", "api", "pull images", sa="r", sb="bl", rad=0.0, label_pos=0.3)
    c.arrow("user", "web", sa="l", sb="r")
    c.arrow("web", "api", "POST /predict", sa="l", sb="r", label_pos=0.5)
    c.arrow("api", "mlflow", "load champion", sa="b", sb="t")
    c.arrow("mlflow", "pg", sa="rt", sb="l")
    c.arrow("mlflow", "minio", sa="rb", sb="l")
    c.arrow("data", "airflow", sa="r", sb="l")
    c.arrow("airflow", "feast", sa="b", sb="t")
    c.arrow("airflow", "mlflow", "log + register", sa="t", sb="b")
    c.legend([
        Patch(fc=EXISTS, ec=EXISTS_EDGE, label="component (exists, tested)"),
        Patch(fc=STORE, ec=STORE_EDGE, label="storage"),
        Line2D([], [], color=MEASURE, marker="", ls="", label="green italic = measured"),
    ])  # fmt: skip
    c.save("current-system.png")


def proposed_system() -> None:
    c = Canvas(
        "Proposed system (design)",
        "Orange dashed = proposed and NOT tested. Blue = exists today. Choices justified by the measurements: "
        "CPU serving, ONNX Runtime, no distributed training.",
    )
    c.box("user", 0.4, 7.25, 2.6, 1.0, "Clients", "web app, other services")

    c.group(3.4, 3.35, 9.2, 5.95, "Kubernetes cluster (local or managed, e.g. GKE)", color="#eef4fb", edge=EXISTS_EDGE)
    c.box("ingress", 3.7, 7.25, 2.5, 1.0, "Ingress / LB", "TLS, routing", kind="proposed")
    c.box("api", 6.6, 6.85, 5.7, 1.8, "Prediction API (CPU pods)", "FastAPI, sync /predict + N uvicorn workers;\n"
          "model = ONNX Runtime (no PyTorch)", kind="proposed",
          measure="ONNX: start 1.4 s → 0.12 s, −250 MB / pod (measured)")  # fmt: skip
    c.box("hpa", 3.7, 5.35, 2.5, 1.15, "Autoscaler (HPA)", "2 → N replicas,\non CPU / p95 latency", kind="proposed")
    c.box("mlflow", 6.6, 5.0, 5.7, 1.1, "MLflow registry", "@champion only moves if macro F1 improves")
    prom_detail = "in the cluster (Helm);\nscrapes API /metrics,\nalerts → on-call"
    c.box("prom", 3.7, 3.55, 2.5, 1.5, "Prometheus + Grafana", prom_detail, kind="proposed")

    c.group(3.4, 0.3, 9.2, 2.7, "Storage", color="#f6f6f6", edge=STORE_EDGE)
    c.box("fs", 3.7, 0.65, 2.5, 1.1, "Feast", "ride features", kind="store")
    c.box("pg", 6.6, 0.65, 2.6, 1.1, "PostgreSQL", "runs, registry (backups)", kind="store")
    c.box("s3", 9.6, 0.65, 2.7, 1.1, "S3 / MinIO", "PyTorch + ONNX artifacts", kind="store")

    c.group(13.0, 0.3, 4.7, 9.0, "Offline workloads and decisions", color="#fff8f0", edge=PROPOSED_EDGE, dashed=True)
    c.box("train", 13.3, 6.85, 4.1, 1.6, "Training (1 machine, CPU)", "weekly; reads Feast features;\n"
          "optional: 1 spot T4 for experiments", measure="CPU 74–83 s · T4 5 s · no DDP needed")  # fmt: skip
    c.box("batch", 13.3, 4.65, 4.1, 1.65, "Batch scoring job", "CronJob over the ride history,\nONNX Runtime on CPU",
          kind="proposed", measure="1,323 windows: 93 ms (PyTorch 1.4 s)")  # fmt: skip
    c.box("gpu", 13.3, 2.45, 4.1, 1.6, "GPU: not used in serving", "1 ride: T4 only 1.3× faster;\n"
          "+0.5 s CUDA start per pod", kind="store", measure="measured on Vertex AI (spot T4)")  # fmt: skip
    c.box("rollback", 13.3, 0.6, 4.1, 1.5, "Recovery", "rolling update (maxUnavailable 0), probes,\n"
          "auto rollback in CD, reload previous champion", measure="model reload 8 ms vs pod start ≈ 5 s")  # fmt: skip

    c.arrow("user", "ingress", "HTTPS", sa="r", sb="l")
    c.arrow("ingress", "api", sa="r", sb="lt")
    c.arrow("hpa", "api", "scale", sa="r", sb="lb", label_pos=0.45)
    c.arrow("api", "mlflow", "load / hot reload", sa="b", sb="t")
    c.arrow("mlflow", "pg", sa="bl", sb="t")
    c.arrow("mlflow", "s3", sa="br", sb="t")
    c.arrow("train", "mlflow", "register", sa="l", sb="rt", label_pos=0.5)
    c.arrow("batch", "mlflow", "load champion", sa="l", sb="rb", dashed=True, color=PROPOSED_EDGE, label_pos=0.5,
            dy=-0.4)  # fmt: skip
    c.legend([
        Patch(fc=EXISTS, ec=EXISTS_EDGE, label="exists today"),
        Patch(fc=PROPOSED, ec=PROPOSED_EDGE, ls="--", label="proposed, not tested"),
        Patch(fc=STORE, ec=STORE_EDGE, label="storage / decision"),
        Line2D([], [], color=MEASURE, ls="", label="green italic = measured"),
    ])  # fmt: skip
    c.save("proposed-system.png")


if __name__ == "__main__":
    current_system()
    proposed_system()
