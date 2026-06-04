# syntax=docker/dockerfile:1.7
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    UV_SYSTEM_PYTHON=1 \
    UV_NO_PROGRESS=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl gcc procps \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -Ls https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

COPY pyproject.toml .

# Install all groups (model + test + core)
RUN --mount=type=cache,target=/root/.uv \
    uv sync --all-groups --no-install-project

COPY . .