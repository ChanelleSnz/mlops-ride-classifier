"""Training pipeline: check data -> preprocess -> train, log and register in MLflow -> reload the API.

Each step calls the project code (ride_classifier package) with the project's own Python
environment, so Airflow and the ML libraries never conflict.
"""

from __future__ import annotations

import json
import os
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import dag, task

PROJECT_DIR = os.environ.get("RIDE_PROJECT_DIR", "/opt/project")
PROJECT_PYTHON = os.environ.get("RIDE_PROJECT_PYTHON", f"{PROJECT_DIR}/.venv/bin/python")
API_URL = os.environ.get("RIDE_API_URL", "http://api:8000")
RAW_FILES = ["GBG500.parquet", "GBG500_labels.csv"]

default_args = {
    "owner": "ride-classifier",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}


@dag(
    dag_id="ride_classifier_training",
    description="Preprocess the rides, train the CNN, register it in MLflow and reload the API",
    schedule="@weekly",
    start_date=datetime(2026, 11, 1),
    catchup=False,
    max_active_runs=1,
    default_args=default_args,
    tags=["mlops", "training"],
)
def ride_classifier_training():
    @task
    def check_raw_data() -> dict:
        """Fail early with a clear message if the raw data is missing."""
        raw_dir = Path(PROJECT_DIR) / "data" / "raw"
        missing = [name for name in RAW_FILES if not (raw_dir / name).exists()]
        if missing:
            raise FileNotFoundError(f"Missing raw files in {raw_dir}: {missing}")
        sizes = {name: (raw_dir / name).stat().st_size for name in RAW_FILES}
        print(f"Raw data OK: {sizes}")
        return sizes

    preprocess = BashOperator(
        task_id="preprocess",
        bash_command=f"cd {PROJECT_DIR} && {PROJECT_PYTHON} -m ride_classifier.data",
    )

    train_and_register = BashOperator(
        task_id="train_and_register",
        bash_command=f"cd {PROJECT_DIR} && {PROJECT_PYTHON} -m ride_classifier.train",
        execution_timeout=timedelta(minutes=30),
    )

    @task
    def reload_api() -> dict:
        """Ask the API to load the current champion (new one if training improved it)."""
        request = urllib.request.Request(f"{API_URL}/model/reload", method="POST")
        with urllib.request.urlopen(request, timeout=120) as response:
            model = json.load(response)
        print(f"API now serves {model['name']} version {model['version']} (@{model['alias']})")
        return model

    check_raw_data() >> preprocess >> train_and_register >> reload_api()


ride_classifier_training()
