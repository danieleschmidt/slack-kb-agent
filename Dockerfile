# Multi-stage build for Slack KB Agent
FROM python:3.11-slim as base

# Security: Create non-root user
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Build stage for dependencies
FROM base as builder

# Copy requirements and install dependencies
COPY pyproject.toml ./
RUN pip install --user build && \
    python -m build --wheel

# Production stage
FROM base as production

# Copy built wheel from builder stage
COPY --from=builder /app/dist/*.whl /tmp/
RUN pip install /tmp/*.whl && rm /tmp/*.whl

# Copy application code
COPY src/ ./src/
COPY bot.py monitoring_server.py ./
COPY migrations/ ./migrations/
COPY alembic.ini ./

# Create necessary directories
RUN mkdir -p logs data/backups && \
    chown -R appuser:appuser /app

# Security: Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:9090/health || exit 1

# Expose ports
EXPOSE 3000 9090

# Default command
CMD ["python", "bot.py"]

# Development stage
FROM base as development

# Install development dependencies
RUN pip install pytest pytest-cov pytest-mock black ruff mypy bandit safety pre-commit

# Copy application code
COPY . .

# Install in development mode
RUN pip install -e ".[llm]"

# Create directories
RUN mkdir -p logs data/backups && \
    chown -R appuser:appuser /app

USER appuser

# Development command
CMD ["python", "bot.py"]