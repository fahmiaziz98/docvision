.PHONY: lint format check test test-unit test-integration build

# Variables
PYTHON := python3
RUFF := ruff
BLACK := black
PYTEST := pytest
UV := uv

# Default target
all: lint format test build

# Linting
lint:
	$(UV) run $(RUFF) check . --fix

# Formatting
format:
	$(UV) run $(RUFF) format .

# Check without fixing (CI style)
check:
	$(UV) run $(RUFF) check .
	$(UV) run $(RUFF) format --check .
	$(UV) run $(BLACK) src/ tests/

# Testing
test:
	$(UV) run $(PYTEST)

test-unit:
	$(UV) run $(PYTEST) -m unit

test-integration:
	$(UV) run $(PYTEST) -m integration

# Building
build:
	$(UV) build
