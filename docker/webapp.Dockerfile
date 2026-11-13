# Web app image: Streamlit only (no PyTorch, no model: it calls the API)
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.23 /uv /usr/local/bin/uv

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PATH="/app/.venv/bin:$PATH"

# Only the "webapp" dependency group (streamlit, requests)
COPY pyproject.toml uv.lock .python-version ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --only-group webapp

COPY webapp ./webapp
COPY examples ./examples

RUN useradd --create-home app && chmod -R a+rX /app/webapp /app/examples
USER app

EXPOSE 8501
CMD ["streamlit", "run", "webapp/app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
