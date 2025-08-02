# Testing Documentation

This directory contains comprehensive testing documentation for the Slack KB Agent project.

## Test Structure

```
tests/
├── conftest.py                 # Global test configuration and fixtures
├── fixtures/                   # Test data and utilities
│   ├── __init__.py
│   └── sample_data.py         # Sample documents, events, and configurations
├── integration/                # Integration tests
│   ├── __init__.py
│   └── test_slack_integration.py
├── e2e/                       # End-to-end tests
│   ├── __init__.py
│   └── test_complete_workflow.py
├── performance/               # Performance and load tests
│   ├── conftest.py
│   └── test_search_performance.py
└── test_*.py                  # Unit tests for individual components
```

## Test Categories

### Unit Tests
- **Location**: `tests/test_*.py`
- **Purpose**: Test individual components in isolation
- **Run with**: `pytest tests/test_*.py`
- **Coverage**: >90% for core functionality

### Integration Tests
- **Location**: `tests/integration/`
- **Purpose**: Test component interactions
- **Run with**: `pytest tests/integration/ -m integration`
- **Focus**: Slack bot + knowledge base, database + cache interactions

### End-to-End Tests
- **Location**: `tests/e2e/`
- **Purpose**: Test complete user workflows
- **Run with**: `pytest tests/e2e/ -m e2e`
- **Scenarios**: Knowledge ingestion → search → Slack response

### Performance Tests
- **Location**: `tests/performance/`
- **Purpose**: Validate performance requirements
- **Run with**: `pytest tests/performance/ -m benchmark`
- **Metrics**: Response time, memory usage, throughput

## Test Markers

Use pytest markers to categorize and run specific test types:

```bash
# Run only unit tests
pytest -m "not integration and not e2e and not benchmark"

# Run integration tests
pytest -m integration

# Run end-to-end tests
pytest -m e2e

# Run performance tests
pytest -m benchmark

# Run slow tests
pytest -m slow

# Run all except slow tests
pytest -m "not slow"
```

## Test Configuration

### Environment Variables
```bash
# Test database (use separate DB for testing)
TEST_DATABASE_URL=postgresql://test:test@localhost:5432/slack_kb_agent_test

# Test Redis instance
TEST_REDIS_URL=redis://localhost:6379/15

# Mock external services in tests
TESTING=true
MOCK_EXTERNAL_APIS=true

# Test Slack credentials (use test workspace)
TEST_SLACK_BOT_TOKEN=xoxb-test-token
TEST_SLACK_APP_TOKEN=xapp-test-token
TEST_SLACK_SIGNING_SECRET=test-signing-secret
```

### pytest Configuration
See `pytest.ini` for test discovery, markers, and coverage settings.

## Writing Tests

### Test Naming Convention
- Unit tests: `test_<component>_<function>.py`
- Integration tests: `test_<component>_integration.py`
- E2E tests: `test_<workflow>_workflow.py`

### Fixture Usage
```python
# Use existing fixtures from conftest.py
def test_search_functionality(knowledge_base, sample_documents):
    # Test implementation
    pass

# Create component-specific fixtures
@pytest.fixture
def slack_bot_with_knowledge(knowledge_base):
    from slack_kb_agent.slack_bot import SlackBot
    bot = SlackBot(config=test_config)
    bot.kb = knowledge_base
    return bot
```

### Mocking External Services
```python
# Mock Slack API calls
@patch('slack_kb_agent.slack_bot.App')
def test_slack_integration(mock_slack_app):
    # Test implementation
    pass

# Mock database connections
@patch('slack_kb_agent.database.create_engine')
def test_database_operations(mock_engine):
    # Test implementation
    pass
```

## Test Data Management

### Using Sample Data
```python
from tests.fixtures.sample_data import (
    SAMPLE_DOCUMENTS,
    SAMPLE_SLACK_EVENTS,
    create_sample_documents
)

def test_document_processing():
    documents = create_sample_documents()
    # Use documents in test
```

### Creating Test-Specific Data
```python
def test_custom_scenario():
    custom_doc = {
        'content': 'Test content',
        'source': 'test.md',
        'metadata': {'test': True}
    }
    # Use custom data
```

## Running Tests

### Local Development
```bash
# Run all tests
make test

# Run with coverage
make test-coverage

# Run specific test file
pytest tests/test_knowledge_base.py -v

# Run with specific marker
pytest -m integration -v

# Run and drop into debugger on failure
pytest --pdb tests/test_failing.py
```

### Continuous Integration
```bash
# Full test suite for CI
make test-ci

# Security and quality checks
make security
make lint

# Performance benchmarks
make benchmark
```

### Docker Testing
```bash
# Run tests in Docker container
docker-compose -f docker-compose.test.yml run tests

# Run specific test category
docker-compose -f docker-compose.test.yml run tests pytest -m integration
```

## Test Coverage

### Coverage Requirements
- **Unit tests**: >90% line coverage
- **Integration tests**: >80% path coverage
- **E2E tests**: >70% user journey coverage

### Generating Coverage Reports
```bash
# HTML coverage report
pytest --cov=src/slack_kb_agent --cov-report=html

# Terminal coverage report
pytest --cov=src/slack_kb_agent --cov-report=term

# Coverage with missing lines
pytest --cov=src/slack_kb_agent --cov-report=term-missing
```

### Coverage Exclusions
See `pyproject.toml` for coverage exclusions:
- Abstract methods
- Debug code
- Error handling for unreachable conditions

## Performance Testing

### Benchmark Tests
```python
@pytest.mark.benchmark
def test_search_performance(benchmark, knowledge_base):
    result = benchmark(knowledge_base.search, "test query")
    assert len(result) > 0
```

### Load Testing
```bash
# Run load tests with multiple concurrent users
pytest tests/performance/ -m load_test --workers=10

# Stress testing with memory monitoring
pytest tests/performance/ -m stress_test --memory-limit=512MB
```

### Memory Profiling
```bash
# Profile memory usage during tests
pytest --profile tests/test_memory_intensive.py

# Monitor memory leaks
pytest --memray tests/test_potential_leaks.py
```

## Debugging Tests

### Common Debugging Techniques
```bash
# Run single test with verbose output
pytest tests/test_specific.py::test_function -v -s

# Drop into debugger on failure
pytest --pdb tests/test_failing.py

# Show local variables on failure
pytest --tb=long tests/test_failing.py

# Capture stdout during tests
pytest -s tests/test_with_prints.py
```

### Test Isolation Issues
```bash
# Run tests in random order to find dependencies
pytest --random-order tests/

# Run tests in parallel to find race conditions
pytest -n auto tests/
```

## Best Practices

### Test Design Principles
1. **Arrange-Act-Assert**: Clear test structure
2. **Independent Tests**: No test dependencies
3. **Descriptive Names**: Clear test purpose
4. **Single Responsibility**: One concept per test
5. **Fast Execution**: Mock external dependencies

### Code Quality
- Use type hints in test functions
- Add docstrings for complex test scenarios
- Keep test data in fixtures, not inline
- Use parametrized tests for multiple scenarios
- Mock external services and databases

### Maintenance
- Update tests when adding new features
- Remove obsolete tests when removing features
- Keep test dependencies up to date
- Monitor test execution time and optimize slow tests
- Regular review of test coverage and quality

## Troubleshooting

### Common Issues

#### Tests Failing in CI but Passing Locally
- Check environment differences
- Verify Docker container configuration
- Review test isolation and cleanup

#### Slow Test Execution
- Identify slow tests with `pytest --durations=10`
- Mock external services
- Use faster test data fixtures
- Consider parallel test execution

#### Flaky Tests
- Review test dependencies and cleanup
- Check for race conditions
- Improve test assertions and timeouts
- Use deterministic test data

#### Memory Issues in Tests
- Monitor test memory usage
- Clean up fixtures properly
- Avoid circular references
- Use memory profiling tools

### Getting Help
- Review test logs and error messages
- Check fixture configuration
- Validate test environment setup
- Consult team knowledge base
- File issues with detailed reproduction steps