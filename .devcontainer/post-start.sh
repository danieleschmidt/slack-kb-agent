#!/bin/bash
set -e

echo "🌟 Starting Slack KB Agent development session..."

# Check if services are running
check_service() {
    local service=$1
    local port=$2
    local name=$3
    
    if nc -z localhost $port 2>/dev/null; then
        echo "✅ $name is running on port $port"
    else
        echo "⚠️  $name is not running on port $port"
        echo "   Start with: docker-compose up -d $service"
    fi
}

# Check database connection
echo "🔍 Checking service availability..."
check_service "postgres" 5432 "PostgreSQL"
check_service "redis" 6379 "Redis"

# Check if .env is configured
if [ -f .env ]; then
    if grep -q "your-.*-token-here" .env; then
        echo "⚠️  .env file needs configuration"
        echo "   Please update tokens and API keys in .env"
    else
        echo "✅ .env appears to be configured"
    fi
else
    echo "⚠️  .env file not found"
    echo "   Copy from .env.example and configure"
fi

# Check Python environment
echo "🐍 Checking Python environment..."
if python -c "import slack_kb_agent" 2>/dev/null; then
    echo "✅ Python package is properly installed"
else
    echo "⚠️  Python package not installed"
    echo "   Run: pip install -e ."
fi

# Display quick commands
echo ""
echo "🚀 Quick commands for this session:"
echo "  make dev           # Start development server with hot reload"
echo "  make test-watch    # Run tests in watch mode"
echo "  make logs          # View application logs"
echo "  make shell         # Start Python shell with imports"
echo ""
echo "🔧 Development tools:"
echo "  make clean         # Clean all cache files"
echo "  make reset-db      # Reset database (destructive)"
echo "  make backup        # Create database backup"
echo ""
echo "📊 Monitoring:"
echo "  make status        # Check service health"
echo "  make metrics       # View current metrics"
echo "  make dashboard     # Open monitoring dashboard"
echo ""

# Set up helpful aliases for the session
alias ll='ls -la'
alias kb='slack-kb-agent'
alias kbdb='slack-kb-db'
alias logs='tail -f logs/app.log'
alias pytest-watch='pytest-watch --runner "python -m pytest"'

echo "🎯 Session ready! Happy coding! 🚀"