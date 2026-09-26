.PHONY: test lint typecheck check format install dev clean benchmark

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

test:
	pytest tests/ -v --tb=short -m "not slow and not benchmark"

test-all:
	pytest tests/ -v --tb=short

test-unit:
	pytest tests/unit/ -v --tb=short

test-integration:
	pytest tests/integration/ -v --tb=short

test-adversarial:
	pytest tests/adversarial/ -v --tb=short

test-cov:
	pytest tests/ -v --tb=short --cov=contextos --cov-report=term-missing --cov-report=html -m "not slow and not benchmark"

benchmark:
	pytest tests/benchmarks/ -v --tb=short -m benchmark

lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/
	ruff check --fix src/ tests/

typecheck:
	mypy src/contextos/

check: lint typecheck test

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .mypy_cache .ruff_cache htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
