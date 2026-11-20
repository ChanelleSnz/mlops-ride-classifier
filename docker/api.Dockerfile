# API image: FastAPI + the ride_classifier package (PyTorch CPU, no Streamlit)
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.23 /uv /usr/local/bin/uv

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PATH="/app/.venv/bin:$PATH"

# 1) Dependencies only: this layer is cached as long as pyproject.toml / uv.lock do not change
COPY pyproject.toml uv.lock .python-version README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project

# 2) The project code (changes often, so it comes last)
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

# Do not run as root (and make sure that user can read the code, whatever the local file permissions)
RUN useradd --create-home app && chmod -R a+rX /app/src
USER app

EXPOSE 8000
CMD ["uvicorn", "ride_classifier.api:app", "--host", "0.0.0.0", "--port", "8000"]
