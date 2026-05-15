-include .env
export

.PHONY: help lint lf test tests commit cz bump

.DEFAULT_GOAL := help

help:
	@echo "Available commands:"
	@echo "  make lint        Check code with ruff linting and format checks"
	@echo "  make lf          Fix code with ruff linting and formatting"
	@echo "  make test        Run the test suite with pytest"
	@echo "  make tests       Alias for 'make test'"
	@echo "  make commit      Run lint checks, then create a commitizen commit"
	@echo "  make cz          Alias for 'make commit'"
	@echo "  make bump        Bump version, update CHANGELOG, and tag via commitizen"

lint:
	uv run ruff check .
	uv run ruff format --check .

lf:
	uv run ruff check --fix .
	uv run ruff format .

test tests:
	uv run pytest

commit: lint
	uv run cz commit

cz: commit

bump:
	uv run cz bump
