.PHONY: help install services up down logs ingest dream eval test lint format clean

help:
	@echo "NOESIS — make targets"
	@echo "  install        Install Python dependencies (with train + interp extras)"
	@echo "  services       Start Docker services (Qdrant, Postgres, Grafana)"
	@echo "  up             Start services + Python API"
	@echo "  down           Stop everything"
	@echo "  logs           Tail service logs"
	@echo "  ingest         Run daily ingestion pipeline"
	@echo "  dream          Run dream cycle (manual trigger)"
	@echo "  eval           Run eval harness against holdout"
	@echo "  test           Run pytest"
	@echo "  lint           Run ruff + mypy"
	@echo "  format         Run ruff format"
	@echo "  clean          Remove caches"

install:
	uv sync --extra train --extra interp --extra dev || pip install -e ".[train,interp,dev]"

services:
	docker compose up -d qdrant postgres grafana

up: services
	uvicorn noesis.api.server:app --reload --host 0.0.0.0 --port 8000

down:
	docker compose down

logs:
	docker compose logs -f --tail=100

ingest:
	python scripts/ingest_daily.py

dream:
	python scripts/dream_cycle.py --manual

eval:
	python scripts/eval_harness.py

test:
	pytest -v --cov=noesis --cov-report=term-missing

lint:
	ruff check noesis tests scripts
	mypy noesis

format:
	ruff format noesis tests scripts
	ruff check --fix noesis tests scripts

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov dist build
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
