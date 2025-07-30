.PHONY: help install test lint format security docs clean build docker health dev
.DEFAULT_GOAL := help

# Variables
PYTHON := python
PIP := pip
PROJECT_NAME := slack-kb-agent
DOCKER_IMAGE := $(PROJECT_NAME):latest

help: ## Show this help message
	@echo "Slack KB Agent - Development Commands"
	@echo "====================================="
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies
	$(PIP) install -e ".[llm]"
	$(PIP) install pytest pytest-cov pytest-mock black ruff mypy bandit safety pre-commit
	pre-commit install

test: ## Run test suite
	$(PYTHON) -m pytest tests/ -v --cov=src/slack_kb_agent --cov-report=html --cov-report=term

test-fast: ## Run tests without coverage
	$(PYTHON) -m pytest tests/ -v

test-integration: ## Run integration tests only
	$(PYTHON) -m pytest tests/ -v -m integration

test-unit: ## Run unit tests only
	$(PYTHON) -m pytest tests/ -v -m "not integration"

lint: ## Run linting checks
	$(PYTHON) -m ruff check src/ tests/
	$(PYTHON) -m mypy src/slack_kb_agent/
	$(PYTHON) -m bandit -r src/ -f json

format: ## Format code
	$(PYTHON) -m black src/ tests/
	$(PYTHON) -m ruff check --fix src/ tests/

security: ## Run security scans
	$(PYTHON) -m bandit -r src/ -f json -o security-report.json
	$(PYTHON) -m safety check --json --output security-deps.json
	@echo "Security reports generated: security-report.json, security-deps.json"

security-full: security ## Run comprehensive security scans
	@echo "🔒 Running comprehensive security scans..."
	$(PYTHON) scripts/generate-sbom.py --output sbom.json --summary
	@if command -v trivy >/dev/null 2>&1; then \
		echo "Scanning Docker image with Trivy..."; \
		trivy image --format json --output trivy-report.json $(DOCKER_IMAGE); \
	else \
		echo "Trivy not installed, skipping container scan"; \
	fi
	@echo "Security scan complete. Check: security-report.json, security-deps.json, sbom.json"

docs: ## Generate documentation
	@echo "📚 Generating documentation..."
	@echo "API documentation would be generated here"

clean: ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete

build: clean ## Build package
	$(PYTHON) -m build

docker: ## Build Docker image
	docker build -t $(DOCKER_IMAGE) .

docker-run: ## Run Docker container
	docker run -d --name $(PROJECT_NAME) -p 3000:3000 -p 9090:9090 $(DOCKER_IMAGE)

docker-stop: ## Stop Docker container
	docker stop $(PROJECT_NAME) || true
	docker rm $(PROJECT_NAME) || true

health: ## Check application health
	@echo "🏥 Checking application health..."
	@curl -f http://localhost:9090/health || echo "❌ Health check failed"
	@curl -f http://localhost:9090/metrics || echo "❌ Metrics endpoint failed"

dev: ## Start development servers
	@echo "🚀 Starting development environment..."
	@echo "Starting monitoring server in background..."
	$(PYTHON) monitoring_server.py &
	@echo "Starting Slack bot server..."
	$(PYTHON) bot.py

setup-db: ## Setup database
	slack-kb-db init

backup-db: ## Backup database
	slack-kb-db backup data/backups/backup-$(shell date +%Y%m%d-%H%M%S).json.gz

verify: lint test security ## Run all verification checks

ci: verify ## Run CI pipeline locally

pre-commit: ## Run pre-commit hooks on all files
	pre-commit run --all-files

update-deps: ## Update dependencies
	$(PIP) install --upgrade pip
	$(PIP) install --upgrade -e ".[llm]"
	pre-commit autoupdate

benchmark: ## Run performance benchmarks
	@echo "🏃 Running performance benchmarks..."
	$(PYTHON) -m pytest tests/ -v -m benchmark

performance: ## Run performance tests
	@echo "⚡ Running performance test suite..."
	$(PYTHON) -m pytest tests/performance/ -v --tb=short

load-test: ## Run load tests
	@echo "📈 Running load tests..."
	$(PYTHON) -m pytest tests/ -v -m load_test

stress-test: ## Run stress tests
	@echo "💪 Running stress tests..."
	$(PYTHON) -m pytest tests/ -v -m stress_test

monitor: ## Show monitoring dashboard
	@echo "📊 Monitoring endpoints:"
	@echo "Health: http://localhost:9090/health"
	@echo "Metrics: http://localhost:9090/metrics"
	@echo "Status: http://localhost:9090/status"

secrets-scan: ## Scan for secrets in codebase
	@echo "🔍 Scanning for secrets..."
	@if command -v detect-secrets >/dev/null 2>&1; then \
		detect-secrets scan --baseline .secrets.baseline; \
	else \
		echo "detect-secrets not installed, skipping secrets scan"; \
	fi

license-check: ## Check license compliance
	@echo "⚖️  Checking license compliance..."
	@if command -v pip-licenses >/dev/null 2>&1; then \
		pip-licenses --format=table --with-urls; \
	else \
		echo "pip-licenses not installed, install with: pip install pip-licenses"; \
	fi

container-scan: docker ## Scan container image for vulnerabilities
	@echo "🐳 Scanning container image..."
	@if command -v trivy >/dev/null 2>&1; then \
		trivy image --exit-code 0 --severity HIGH,CRITICAL $(DOCKER_IMAGE); \
	else \
		echo "Trivy not installed, skipping container scan"; \
	fi

audit: ## Run comprehensive audit
	@echo "🔍 Running comprehensive audit..."
	@$(MAKE) security-full
	@$(MAKE) secrets-scan
	@$(MAKE) license-check
	@echo "✅ Audit complete"

compliance: ## Run compliance checks
	@echo "📋 Running compliance checks..."
	@echo "- Security scan: $(shell $(MAKE) security >/dev/null 2>&1 && echo '✅' || echo '❌')"
	@echo "- Code quality: $(shell $(MAKE) lint >/dev/null 2>&1 && echo '✅' || echo '❌')"
	@echo "- Test coverage: $(shell $(MAKE) test >/dev/null 2>&1 && echo '✅' || echo '❌')"
	@echo "- License check: $(shell $(MAKE) license-check >/dev/null 2>&1 && echo '✅' || echo '❌')"

release-check: ## Validate release readiness
	@echo "🚀 Checking release readiness..."
	@$(MAKE) audit
	@$(MAKE) compliance
	@$(MAKE) test
	@$(MAKE) build
	@echo "✅ Release validation complete"