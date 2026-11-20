# Airflow (orchestration) + the project in its OWN virtual environment.
# Airflow and MLflow/PyTorch have conflicting dependencies, so the DAG calls
# /opt/project/.venv/bin/python instead of importing the ML code into Airflow,
# and /opt/project/feature_repo/.venv/bin/python for the Feast feature store.
FROM apache/airflow:3.4.0-python3.12

USER root
COPY --from=ghcr.io/astral-sh/uv:0.12.23 /uv /usr/local/bin/uv
WORKDIR /opt/project
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never

COPY pyproject.toml uv.lock .python-version README.md ./
RUN uv sync --locked --no-dev --no-install-project
COPY src ./src
RUN uv sync --locked --no-dev

# Feature store: Feast in a THIRD environment (it needs pandas < 3, the project uses pandas 3)
COPY feature_repo/pyproject.toml feature_repo/uv.lock feature_repo/.python-version ./feature_repo/
RUN uv sync --locked --no-dev --project feature_repo
COPY feature_repo ./feature_repo
RUN mkdir -p feature_repo/data && chown -R airflow:0 /opt/project

USER airflow
WORKDIR /opt/airflow
