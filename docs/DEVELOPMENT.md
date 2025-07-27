# Development Guide

This guide covers the development workflow, setup, and best practices for the Slack KB Agent project.

## Quick Start

### Prerequisites

- Python 3.8+
- Docker and Docker Compose
- Git
- VS Code (recommended) or your preferred IDE

### Development Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd slack-kb-agent
   ```

2. **Set up development environment**:
   ```bash
   # Option 1: Using Dev Container (Recommended)
   # Open in VS Code and use "Reopen in Container"
   
   # Option 2: Local setup
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -e ".[llm]"
   pip install pytest pytest-cov black ruff mypy bandit safety pre-commit
   pre-commit install
   ```

3. **Configure environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

4. **Start development services**:
   ```bash
   # Start PostgreSQL and Redis
   docker-compose -f docker-compose.dev.yml up -d postgres redis
   
   # Initialize database
   make setup-db
   ```

5. **Run the application**:
   ```bash
   # Start the bot
   python bot.py
   
   # In another terminal, start monitoring
   python monitoring_server.py
   ```

## Development Workflow

### Code Quality

We maintain high code quality standards through automated tooling:

```bash
# Format code
make format

# Run linting
make lint

# Run type checking
make mypy

# Run security checks
make security

# Run all quality checks
make verify
```

### Testing

We use pytest for testing with comprehensive coverage:

```bash
# Run all tests
make test

# Run unit tests only
make test-unit

# Run integration tests only
make test-integration

# Run with coverage
pytest tests/ --cov=src/slack_kb_agent --cov-report=html
```

### Pre-commit Hooks

Pre-commit hooks automatically run code quality checks:

```bash
# Install hooks
pre-commit install

# Run hooks manually
pre-commit run --all-files
```

## Project Structure

```
slack-kb-agent/
├── src/slack_kb_agent/          # Main application code
│   ├── __init__.py
│   ├── slack_bot.py            # Slack bot server
│   ├── knowledge_base.py       # Knowledge storage and retrieval
│   ├── query_processor.py      # Query understanding and routing
│   ├── vector_search.py        # Vector similarity search
│   ├── llm.py                  # LLM integration
│   ├── ingestion.py            # Knowledge source ingestion
│   ├── monitoring.py           # Monitoring and metrics
│   └── ...
├── tests/                       # Test suite
│   ├── conftest.py             # Test configuration and fixtures
│   ├── test_*.py              # Test modules
│   └── ...
├── docs/                        # Documentation
│   ├── adr/                    # Architecture Decision Records
│   ├── guides/                 # User and developer guides
│   └── runbooks/               # Operational runbooks
├── .github/workflows/          # CI/CD workflows
├── .devcontainer/              # Development container configuration
├── monitoring/                 # Monitoring configuration
└── migrations/                 # Database migrations
```

## Architecture Overview

### Core Components

1. **Slack Bot Server** (`slack_bot.py`): Handles Slack Events API integration
2. **Knowledge Base** (`knowledge_base.py`): Core knowledge storage and retrieval
3. **Query Processor** (`query_processor.py`): Advanced query understanding
4. **Vector Search** (`vector_search.py`): Semantic similarity search
5. **LLM Integration** (`llm.py`): AI-powered response generation
6. **Ingestion Pipeline** (`ingestion.py`): Multi-source knowledge ingestion

### Data Flow

```
User Query → Slack Bot → Query Processor → Knowledge Base → Response Generator → User
                              ↓
                        Vector Search + LLM
```

## Coding Standards

### Python Style

- **Formatter**: Black (line length 88)
- **Linter**: Ruff with strict settings
- **Type Hints**: Required for all public functions
- **Docstrings**: Google style for all public functions and classes

### Example Code Style

```python
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

class ExampleClass:
    """Example class demonstrating coding standards.
    
    This class shows the preferred coding style including type hints,
    docstrings, and error handling.
    """
    
    def __init__(self, name: str, config: Optional[dict] = None) -> None:
        """Initialize the example class.
        
        Args:
            name: The name of the instance
            config: Optional configuration dictionary
        """
        self.name = name
        self.config = config or {}
        
    def process_data(self, data: List[str]) -> List[str]:
        """Process a list of data items.
        
        Args:
            data: List of strings to process
            
        Returns:
            Processed list of strings
            
        Raises:
            ValueError: If data is empty
        """
        if not data:
            raise ValueError("Data cannot be empty")
            
        try:
            result = [item.strip().lower() for item in data]
            logger.info(f"Processed {len(result)} items")
            return result
        except Exception as e:
            logger.error(f"Error processing data: {e}")
            raise
```

### Git Workflow

1. **Branch Naming**:
   - `feature/description` for new features
   - `bugfix/description` for bug fixes
   - `hotfix/description` for critical fixes

2. **Commit Messages**:
   ```
   type(scope): description
   
   feat(auth): add JWT authentication
   fix(search): resolve vector indexing issue
   docs(api): update endpoint documentation
   test(integration): add Slack bot tests
   ```

3. **Pull Request Process**:
   - Create feature branch from `main`
   - Implement changes with tests
   - Ensure all CI checks pass
   - Request review from maintainers
   - Squash and merge after approval

## Testing Guidelines

### Test Structure

```python
import pytest
from unittest.mock import MagicMock, patch

from slack_kb_agent.knowledge_base import KnowledgeBase

class TestKnowledgeBase:
    """Test suite for KnowledgeBase class."""
    
    def test_add_document_success(self, sample_document_data):
        """Test successful document addition."""
        kb = KnowledgeBase()
        result = kb.add_document(**sample_document_data)
        
        assert result is not None
        assert kb.document_count() == 1
        
    def test_search_with_results(self, mock_knowledge_base):
        """Test search returning results."""
        results = mock_knowledge_base.search("test query")
        
        assert len(results) > 0
        assert all(hasattr(r, 'score') for r in results)
        
    @pytest.mark.integration
    def test_database_integration(self, database_url):
        """Test database integration."""
        # Integration test code here
        pass
```

### Test Categories

- **Unit Tests**: Test individual functions and classes
- **Integration Tests**: Test component interactions
- **End-to-End Tests**: Test complete workflows
- **Performance Tests**: Test performance characteristics

## Environment Configuration

### Development Environment Variables

```bash
# Application Configuration
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/slack_kb_agent_dev
REDIS_URL=redis://localhost:6379/1
LOG_LEVEL=DEBUG

# Slack Configuration (for testing)
SLACK_BOT_TOKEN=xoxb-test-token
SLACK_SIGNING_SECRET=test-secret
SLACK_APP_TOKEN=xapp-test-token

# Optional Features
OPENAI_API_KEY=sk-your-test-key
VECTOR_SEARCH_ENABLED=true
CACHE_ENABLED=true
```

### Testing Configuration

```bash
# Test Environment
TESTING=true
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/slack_kb_agent_test
REDIS_URL=redis://localhost:6379/2
LOG_LEVEL=WARNING
```

## Debugging

### Logging

The application uses structured logging:

```python
import logging

logger = logging.getLogger(__name__)

# Use appropriate log levels
logger.debug("Detailed diagnostic information")
logger.info("General information about application flow")
logger.warning("Something unexpected happened")
logger.error("Error occurred but application continues")
logger.critical("Serious error that may prevent the application from continuing")
```

### Debug Configuration

```bash
# Enable debug mode
export LOG_LEVEL=DEBUG
export SLACK_DEBUG=true

# Run with debug logging
python bot.py
```

### VS Code Debug Configuration

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Debug Slack Bot",
            "type": "python",
            "request": "launch",
            "program": "bot.py",
            "console": "integratedTerminal",
            "env": {
                "LOG_LEVEL": "DEBUG",
                "TESTING": "true"
            }
        }
    ]
}
```

## Performance Optimization

### Profiling

```bash
# Profile the application
python -m cProfile -o profile.stats bot.py

# Analyze profile results
python -c "import pstats; pstats.Stats('profile.stats').sort_stats('cumulative').print_stats(20)"
```

### Memory Monitoring

```bash
# Monitor memory usage
python -m memory_profiler bot.py

# Track memory over time
mprof run bot.py
mprof plot
```

## Deployment

### Local Deployment

```bash
# Build and run with Docker Compose
docker-compose up -d

# Check application health
curl http://localhost:9090/health
```

### Production Deployment

```bash
# Build production image
docker build -t slack-kb-agent:latest .

# Deploy with production configuration
docker-compose -f docker-compose.yml up -d
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure you've installed the package in development mode: `pip install -e .`
2. **Database Connection**: Check PostgreSQL is running and DATABASE_URL is correct
3. **Redis Connection**: Verify Redis is running and accessible
4. **Slack Integration**: Validate tokens and check network connectivity

### Getting Help

- **Documentation**: Check the `docs/` directory
- **Issues**: Search existing GitHub issues
- **Discussions**: Use GitHub Discussions for questions
- **Community**: Join our Discord server

## Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) for detailed contribution guidelines.

## Resources

- [Architecture Documentation](ARCHITECTURE.md)
- [API Usage Guide](../API_USAGE_GUIDE.md)
- [Security Policy](../SECURITY.md)
- [Changelog](../CHANGELOG.md)