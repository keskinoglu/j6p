export

.PHONY: help lint fix test tests secrets commit cz bump

.DEFAULT_GOAL := help

help:
	@echo "Available commands:"
	@echo "  make lint        Check code with ruff linting and format checks"
	@echo "  make fix          Fix code with ruff linting and formatting"
	@echo "  make test        Run the test suite with pytest"
	@echo "  make tests       Alias for 'make test'"
	@echo "  make secrets     Scan tracked files and git history for secrets and sensitive data"
	@echo "  make commit      Run lint checks, then create a commitizen commit"
	@echo "  make cz          Alias for 'make commit'"
	@echo "  make bump        Bump version, update CHANGELOG, and tag via commitizen"

lint:
	uv run ruff check .
	uv run ruff format --check .

fix:
	uv run ruff check --fix .
	uv run ruff format .

test tests:
	uv run pytest

secrets:
	uv run pre-commit run --all-files
	uv run pre-commit run --hook-stage manual gitleaks-history
	uv run pre-commit run --hook-stage manual check-sensitive-data-history

commit: lint
	uv run cz commit

cz: commit

bump:
	uv run cz bump
