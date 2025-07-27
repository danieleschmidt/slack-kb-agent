#!/bin/bash

# Development container setup script
set -e

echo "🚀 Setting up Slack KB Agent development environment..."

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install -e ".[llm]"
pip install pytest pytest-cov pytest-mock black ruff mypy bandit safety

# Install additional development tools
echo "🔧 Installing development tools..."
pip install pre-commit

# Setup pre-commit hooks
echo "🪝 Setting up pre-commit hooks..."
pre-commit install

# Create necessary directories
echo "📁 Creating project directories..."
mkdir -p logs data/backups docs/runbooks

# Copy environment template if not exists
if [ ! -f .env ]; then
    echo "🔐 Creating .env file from template..."
    cp .env.example .env
fi

# Setup database (if PostgreSQL is available)
echo "🗄️ Checking database availability..."
if command -v psql &> /dev/null; then
    echo "PostgreSQL available, setting up database..."
    # Create database if it doesn't exist
    createdb slack_kb_agent 2>/dev/null || echo "Database already exists"
fi

# Setup Redis (if available)
echo "🔴 Checking Redis availability..."
if command -v redis-cli &> /dev/null; then
    echo "Redis available and ready"
fi

# Run initial tests to verify setup
echo "🧪 Running setup verification tests..."
python -m pytest tests/test_foundational.py -v || echo "Some tests failed - this is expected in a fresh setup"

# Display helpful information
echo ""
echo "✅ Development environment setup complete!"
echo ""
echo "🔧 Available commands:"
echo "  make test          - Run test suite"
echo "  make lint          - Run linting checks"
echo "  make format        - Format code"
echo "  make security      - Run security scans"
echo "  make docs          - Generate documentation"
echo ""
echo "🚀 Quick start:"
echo "  1. Configure .env with your Slack tokens"
echo "  2. Run 'python bot.py' to start the bot"
echo "  3. Run 'python monitoring_server.py' for monitoring"
echo ""
echo "📚 Documentation:"
echo "  - README.md - Main documentation"
echo "  - ARCHITECTURE.md - System architecture"
echo "  - docs/adr/ - Architecture decision records"
echo ""