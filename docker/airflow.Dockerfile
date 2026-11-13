# Airflow (orchestration) + the project in its OWN virtual environment.
# Airflow and MLflow/PyTorch have conflicting dependencies, so the DAG calls
# /opt/project/.venv/bin/python instead of importing the ML code into Airflow.
FROM apache/airflow:3.4.0-python3.12

USER root
COPY --from=ghcr.io/astral-sh/uv:0.12.23 /uv /usr/local/bin/uv
WORKDIR /opt/project
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never

COPY pyproject.toml uv.lock .python-version README.md ./
RUN uv sync --locked --no-dev --no-install-project
COPY src ./src
RUN uv sync --locked --no-dev && chown -R airflow:0 /opt/project

USER airflow
WORKDIR /opt/airflow
