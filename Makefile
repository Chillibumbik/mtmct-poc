SHELL := /bin/bash
export PATH := $(HOME)/.local/opt/node/bin:$(PATH)
UV := uv run

.PHONY: setup models dev api worker web check lint type test e2e openapi up down

setup:            ## install python + web deps and model weights
	uv sync --all-groups
	cd web && pnpm install
	$(UV) python -m mtmct.models
	uv run --group export python scripts/export_osnet.py

dev:              ## api + worker + vite dev server
	@trap 'kill 0' EXIT; \
	$(UV) mtmct api & \
	$(UV) mtmct worker & \
	(cd web && pnpm dev) & \
	wait

api:
	$(UV) mtmct api --reload
worker:
	$(UV) mtmct worker
web:
	cd web && pnpm dev

lint:
	$(UV) ruff check src tests
	$(UV) ruff format --check src tests
type:
	$(UV) mypy src/mtmct/io src/mtmct/config.py src/mtmct/registry.py src/mtmct/associate src/mtmct/eval src/mtmct/track src/mtmct/refine src/mtmct/api src/mtmct/worker src/mtmct/db.py src/mtmct/pipeline src/mtmct/runtime
test:
	$(UV) pytest -q -m "not data and not slow"
check: lint type test
	@if [ -f web/package.json ]; then cd web && pnpm tsc --noEmit && pnpm vitest run && pnpm lint; fi

e2e:              ## smoke end-to-end on epfl_terrace (needs data + models)
	$(UV) pytest -q -m "e2e"

openapi:
	$(UV) python -c "import json; from mtmct.api.app import app; print(json.dumps(app.openapi(), indent=1))" > web/openapi.json

up:
	docker compose up --build -d
down:
	docker compose down
