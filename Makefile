.DEFAULT_GOAL := help
SHELL := /bin/sh

PYTHON ?= python
VENV ?= .venv
ifeq ($(OS),Windows_NT)
    VENV_BIN := $(VENV)/Scripts
else
    VENV_BIN := $(VENV)/bin
endif
PIP := $(VENV_BIN)/pip
PY := $(VENV_BIN)/python

.PHONY: help install dev test lint typecheck format up down clean

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

up: ## Start local infra (Qdrant, Redis, RabbitMQ, MinIO) via docker-compose.
	docker compose up -d

down: ## Stop local infra.
	docker compose down

clean: ## Remove build artifacts and caches.
	rm -rf build dist *.egg-info .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov
