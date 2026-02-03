.PHONY: lint format check test test-unit test-integration

# Variables
PYTHON := python3
RUFF := ruff
PYTEST := pytest

# Default target
all: lint format test

# Linting
lint:
	uv run $(RUFF) check . --fix

# Formatting
format:
	uv run $(RUFF) format .

# Check without fixing (CI style)
check:
	uv run $(RUFF) check .
	uv run $(RUFF) format --check .

# Testing
test:
	uv run $(PYTEST)

test-unit:
	uv run $(PYTEST) -m unit

test-integration:
	uv run $(PYTEST) -m integration
