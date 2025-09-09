.DEFAULT_GOAL := help

# Prefer Python 3.11 if available; fallback to python3
PY ?= $(shell command -v python3.11 >/dev/null 2>&1 && echo python3.11 || echo python3)
VENV := .venv
PIP := $(VENV)/bin/pip
PYBIN := $(VENV)/bin/python

help: ## Show this help
	@awk 'BEGIN {FS = ":.*?## "}; /^[a-zA-Z0-9_\-]+:.*?## / {printf "\033[36m%-24s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST) | sort

$(VENV): ## Create virtual environment
	$(PY) -m venv $(VENV)
	$(PIP) install --upgrade pip

setup: $(VENV) ## Create venv and install dev deps
	@$(PY) -c "import sys; assert sys.version_info[:2]>=(3,11), f'Python 3.11+ required, got {sys.version.split()[0]}'"
	$(PIP) install -e ".[dev]"
	$(VENV)/bin/pre-commit install

lint: ## Run ruff and black (check only)
	$(VENV)/bin/ruff check .
	$(VENV)/bin/black --check .

fix: ## Auto-fix lint issues
	$(VENV)/bin/ruff check --fix .
	$(VENV)/bin/black .

type: ## Run mypy type checks
	$(VENV)/bin/mypy src

test: ## Run tests
	$(VENV)/bin/pytest -q

format: ## Format codebase
	$(VENV)/bin/black .
	$(VENV)/bin/ruff format .

clean: ## Remove caches and build artifacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache build dist *.egg-info

.PHONY: help setup lint fix type test format clean

# --- Optional RDKit setup helpers ---
chem-setup: $(VENV) ## Install RDKit and NumPy 1.x compatible wheels
	$(PIP) install --no-cache-dir -r constraints/chem.txt
	@echo "RDKit + NumPy (ABI-compatible) installed."

chem-test: ## Run only RDKit-dependent tests
	$(VENV)/bin/pytest -q tests/chem

.PHONY: chem-setup chem-test
