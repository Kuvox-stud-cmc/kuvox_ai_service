.DEFAULT_GOAL := help
SHELL := /bin/sh

PYTHON ?= python
PY ?= $(PYTHON)
PIP ?= pip
VENV ?= .venv

.PHONY: help install dev test lint typecheck format worker-media-optimization worker-media up down clean

help: ## Show this help.
	@awk 'BEGIN {FS = ":.*##"; printf "Targets:\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  %-12s %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

install: ## Create venv and install all dependencies (incl. dev extras).
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"

dev: ## Run the FastAPI service with auto-reload on localhost:8000.
	$(PY) -m uvicorn kuvox_ai.main:app --reload --host 0.0.0.0 --port 8000

test: ## Run the pytest suite.
	$(PY) -m pytest

lint: ## Lint with ruff.
	$(PY) -m ruff check src tests
	$(PY) -m ruff format --check src tests

typecheck: ## Static-type check with mypy.
	$(PY) -m mypy

format: ## Auto-format with ruff.
	$(PY) -m ruff format src tests
	$(PY) -m ruff check --fix src tests

worker-media-optimization: ## Run the media optimization worker.
	$(PY) -m kuvox_ai.workers.media_optimization_worker

worker-media: worker-media-optimization ## Alias for worker-media-optimization.

up: ## Start local infra (Qdrant, Redis, RabbitMQ, SeaweedFS) via docker-compose.
	docker compose up -d

down: ## Stop local infra.
	docker compose down

clean: ## Remove build artifacts and caches.
	rm -rf build dist *.egg-info .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov
