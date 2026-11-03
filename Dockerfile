# syntax=docker/dockerfile:1
# Stage 1: web build (Node 22 + pnpm)
FROM node:22-slim AS web
RUN npm install -g pnpm@10.34.9
WORKDIR /web
COPY web/package.json web/pnpm-lock.yaml ./
RUN pnpm install --frozen-lockfile
COPY web/ ./
RUN pnpm build

# Stage 2: python runtime (uv); FastAPI serves web/dist as a SPA
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /usr/local/bin/uv
# libgl/glib: required by opencv-python at import time
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy MTMCT_HOME=/app PATH="/app/.venv/bin:$PATH"
COPY pyproject.toml uv.lock README.md ./
COPY third_party third_party
COPY src src
RUN uv sync --frozen --no-dev
COPY configs configs
COPY --from=web /web/dist web/dist
# data/ cache/ models/ outputs/ db/ are bind-mounted by docker-compose.yml
EXPOSE 8000
CMD ["mtmct", "api", "--host", "0.0.0.0", "--port", "8000"]
