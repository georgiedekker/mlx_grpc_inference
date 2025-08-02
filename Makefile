# Makefile for MLX Distributed Inference System

.PHONY: help install test test-unit test-integration test-e2e test-performance test-fast test-coverage clean lint format check-format setup-dev

help: ## Show this help message
	@echo 'Usage: make <target>'
	@echo ''
	@echo 'Targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install dependencies with UV
	uv sync --dev

setup-dev: install ## Setup development environment
	uv run pre-commit install
	@echo "Development environment setup complete!"

test: ## Run all tests
	uv run python tests/run_tests.py --suite all

test-unit: ## Run unit tests only
	uv run python tests/run_tests.py --suite unit

test-integration: ## Run integration tests only
	uv run python tests/run_tests.py --suite integration

test-e2e: ## Run end-to-end tests only
	uv run python tests/run_tests.py --suite e2e

test-performance: ## Run performance benchmarks
	uv run python tests/run_tests.py --suite performance

test-fast: ## Run tests excluding slow ones
	uv run python tests/run_tests.py --fast

test-coverage: ## Run tests with coverage report
	uv run python tests/run_tests.py --coverage

test-parallel: ## Run tests in parallel
	uv run python tests/run_tests.py --parallel

test-verbose: ## Run tests with verbose output
	uv run python tests/run_tests.py --verbose

test-specific: ## Run specific test (usage: make test-specific TEST=test_name)
	uv run python -m pytest $(TEST) -v

lint: ## Run linting with ruff
	uv run ruff check src tests

lint-fix: ## Fix linting issues automatically
	uv run ruff check --fix src tests

format: ## Format code with black and ruff
	uv run black src tests
	uv run ruff format src tests

check-format: ## Check code formatting without making changes
	uv run black --check src tests
	uv run ruff format --check src tests

type-check: ## Run type checking with mypy
	uv run mypy src --ignore-missing-imports

quality: lint format type-check ## Run all code quality checks

clean: ## Clean up generated files
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage
	rm -rf coverage.xml
	rm -rf dist
	rm -rf build
	rm -rf *.egg-info
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

clean-logs: ## Clean up log files
	find . -name "*.log" -delete
	find . -name "*.pid" -delete

build: ## Build package
	uv build

check-deps: ## Check for dependency issues
	uv sync --check

update-deps: ## Update dependencies
	uv sync --upgrade

generate-protos: ## Generate protobuf files
	cd protos && bash generate_protos.sh

# CI/CD targets
ci-test: install lint test-coverage ## Run CI test suite
	@echo "CI tests completed successfully!"

ci-lint: install lint check-format type-check ## Run CI linting
	@echo "CI linting completed successfully!"

# Development workflow targets
dev-test: test-fast ## Quick development test run
	@echo "Development tests completed!"

dev-check: lint-fix format test-unit ## Quick development check
	@echo "Development check completed!"

# Performance testing targets
benchmark: test-performance ## Run performance benchmarks
	@echo "Benchmarks completed!"

benchmark-report: ## Generate performance report
	uv run python tests/run_tests.py --suite performance --verbose > benchmark_report.txt
	@echo "Benchmark report saved to benchmark_report.txt"

# Container targets (if using Docker)
docker-test: ## Run tests in Docker container
	docker run --rm -v $(PWD):/app -w /app python:3.11 make install test

# Documentation targets
docs: ## Generate documentation
	@echo "Documentation generation not yet implemented"

# Monitoring targets
test-watch: ## Watch for changes and run tests continuously
	uv run pytest-watch -- tests/unit

# Quick start targets
quick-start: install dev-check ## Quick start for new developers
	@echo "🚀 Quick start completed!"
	@echo "Run 'make test' to run all tests"
	@echo "Run 'make dev-test' for quick testing during development"

# Target for running specific test patterns
test-pattern: ## Run tests matching pattern (usage: make test-pattern PATTERN=test_orchestrator)
	uv run python -m pytest -k "$(PATTERN)" -v

# Integration with IDE
ide-setup: setup-dev ## Setup for IDE integration
	@echo "IDE setup completed!"
	@echo "Configure your IDE to use the virtual environment created by UV"

# Deployment preparation
pre-deploy: clean ci-test benchmark ## Prepare for deployment
	@echo "Pre-deployment checks completed successfully!"