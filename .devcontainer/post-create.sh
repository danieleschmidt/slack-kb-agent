#!/bin/bash
set -e

echo "🚀 Setting up Slack KB Agent development environment..."

# Update package lists
sudo apt-get update

# Install additional system dependencies
sudo apt-get install -y \
    postgresql-client \
    redis-tools \
    curl \
    wget \
    jq \
    vim \
    htop

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install --upgrade pip setuptools wheel
pip install -e ".[llm]"
pip install pre-commit pytest-cov coverage[toml]

# Install development tools
echo "🔧 Installing development tools..."
pip install \
    black \
    ruff \
    mypy \
    pytest-xdist \
    pytest-mock \
    bandit[toml] \
    safety

# Setup pre-commit hooks
echo "🪝 Setting up pre-commit hooks..."
pre-commit install
pre-commit install --hook-type commit-msg

# Create logs directory
mkdir -p logs
touch logs/.gitkeep

# Create cache directories  
mkdir -p .cache/vectors
mkdir -p .cache/embeddings
touch .cache/.gitkeep

# Setup environment file if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env from template..."
    cp .env.example .env
    echo "⚠️  Please edit .env with your actual configuration values"
fi

# Setup database directory for local development
mkdir -p data/postgres
mkdir -p data/redis

# Install Claude Code CLI for enhanced development
echo "🤖 Installing Claude Code CLI..."
if ! command -v claude &> /dev/null; then
    npm install -g @anthropic-ai/claude-code
fi

# Setup VS Code settings if not exists
mkdir -p .vscode
if [ ! -f .vscode/settings.json ]; then
    cat > .vscode/settings.json << 'EOF'
{
    "python.terminal.activateEnvironment": true,
    "python.defaultInterpreterPath": "/usr/local/bin/python",
    "editor.rulers": [88],
    "files.exclude": {
        "**/__pycache__": true,
        "**/.pytest_cache": true,
        "**/.mypy_cache": true,
        "**/.ruff_cache": true,
        "**/logs/*.log": true
    },
    "search.exclude": {
        "**/logs": true,
        "**/.cache": true,
        "**/data": true
    }
}
EOF
fi

echo "✅ Development environment setup complete!"
echo ""
echo "🎯 Next steps:"
echo "1. Edit .env with your configuration"
echo "2. Start PostgreSQL and Redis: docker-compose up -d postgres redis"
echo "3. Run database migrations: slack-kb-db init"
echo "4. Run tests: pytest"
echo "5. Start the bot: python bot.py"
echo ""
echo "📚 Useful commands:"
echo "  make test          # Run all tests"
echo "  make lint          # Run linting"
echo "  make format        # Format code"
echo "  make dev           # Start development server"
echo "  make clean         # Clean cache files"