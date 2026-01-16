FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS base

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY uv.lock uv.lock
COPY README.md README.md
COPY pyproject.toml pyproject.toml
COPY src src/

RUN uv sync --locked --no-cache --no-install-project

ENTRYPOINT ["uv", "run", "uvicorn", "src.mlopsproject.api:app", "--host", "0.0.0.0", "--port", "8000"]
