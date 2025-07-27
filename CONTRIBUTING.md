# Contributing to Slack KB Agent

Thank you for your interest in contributing to the Slack KB Agent! This document provides guidelines and information for contributors.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Contribution Types](#contribution-types)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Community](#community)

## Code of Conduct

This project adheres to a [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainers.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Docker and Docker Compose
- Git
- Basic knowledge of Python, Slack APIs, and machine learning concepts

### Development Setup

1. **Fork and clone the repository**:
   ```bash
   git clone https://github.com/your-username/slack-kb-agent.git
   cd slack-kb-agent
   ```

2. **Set up development environment**:
   ```bash
   # Option 1: Dev Container (Recommended)
   # Open in VS Code and select "Reopen in Container"
   
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
   # Edit .env with your test configuration
   ```

4. **Run tests to verify setup**:
   ```bash
   make test
   ```

## Development Workflow

### Branch Strategy

- **main**: Production-ready code
- **develop**: Integration branch for features
- **feature/**: Feature development branches
- **bugfix/**: Bug fix branches
- **hotfix/**: Critical production fixes

### Workflow Steps

1. **Create a feature branch**:
   ```bash
   git checkout develop
   git pull origin develop
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**:
   - Write code following our [coding standards](#coding-standards)
   - Add or update tests
   - Update documentation if needed

3. **Test your changes**:
   ```bash
   make verify  # Runs linting, tests, and security checks
   ```

4. **Commit your changes**:
   ```bash
   git add .
   git commit -m "feat(component): add new feature description"
   ```

5. **Push and create a Pull Request**:
   ```bash
   git push origin feature/your-feature-name
   ```

## Contribution Types

### 🐛 Bug Reports

Before creating a bug report:
- Check existing issues to avoid duplicates
- Use the latest version
- Provide minimal reproduction steps

Use our [bug report template](.github/ISSUE_TEMPLATE/bug_report.yml).

### ✨ Feature Requests

Before suggesting a feature:
- Check the roadmap and existing feature requests
- Consider if it aligns with project goals
- Provide clear use cases

Use our [feature request template](.github/ISSUE_TEMPLATE/feature_request.yml).

### 🔧 Code Contributions

We welcome contributions in these areas:

#### High Priority
- **Core Features**: Knowledge base improvements, search enhancements
- **Performance**: Optimization and scalability improvements
- **Security**: Security enhancements and vulnerability fixes
- **Testing**: Improved test coverage and test automation

#### Medium Priority
- **Integrations**: New knowledge source integrations
- **Documentation**: User guides, API documentation
- **Developer Experience**: Development tooling, debugging aids

#### Low Priority
- **UI/UX**: Interface improvements (Slack interactions)
- **Analytics**: Enhanced monitoring and reporting features

### 📚 Documentation

Documentation improvements are always welcome:
- API documentation
- User guides and tutorials
- Developer documentation
- Code comments and docstrings

## Pull Request Process

### Before Submitting

1. **Ensure your code follows our standards**:
   ```bash
   make lint      # Check code style
   make format    # Format code
   make test      # Run tests
   make security  # Security checks
   ```

2. **Update documentation**:
   - Update relevant documentation files
   - Add docstrings to new functions/classes
   - Update CHANGELOG.md if applicable

3. **Test thoroughly**:
   - Add unit tests for new features
   - Add integration tests if applicable
   - Ensure all tests pass

### Pull Request Template

When creating a PR, include:

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests added and passing
```

### Review Process

1. **Automated Checks**: CI/CD pipeline runs automatically
2. **Code Review**: Maintainers review code for:
   - Functionality and correctness
   - Code quality and style
   - Security implications
   - Performance impact
   - Documentation completeness

3. **Feedback**: Address reviewer feedback
4. **Approval**: Once approved, maintainers will merge

## Coding Standards

### Python Style

We use strict code quality standards:

```python
# Example of good code style
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Process documents for knowledge base ingestion.
    
    This class handles the processing and validation of documents
    before they are added to the knowledge base.
    """
    
    def __init__(self, config: Optional[dict] = None) -> None:
        """Initialize the document processor.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        
    def process_documents(self, documents: List[str]) -> List[dict]:
        """Process a list of documents.
        
        Args:
            documents: List of document content strings
            
        Returns:
            List of processed document dictionaries
            
        Raises:
            ValueError: If documents list is empty
        """
        if not documents:
            raise ValueError("Documents list cannot be empty")
            
        processed = []
        for doc in documents:
            try:
                result = self._process_single_document(doc)
                processed.append(result)
                logger.debug(f"Processed document with {len(doc)} characters")
            except Exception as e:
                logger.error(f"Failed to process document: {e}")
                raise
                
        return processed
        
    def _process_single_document(self, document: str) -> dict:
        """Process a single document (private method)."""
        # Implementation here
        return {"content": document.strip(), "processed": True}
```

### Code Quality Tools

- **Formatter**: Black (line length 88)
- **Linter**: Ruff with strict configuration
- **Type Checker**: MyPy
- **Security**: Bandit
- **Import Sorting**: isort (via Ruff)

### Commit Message Format

We use [Conventional Commits](https://www.conventionalcommits.org/):

```
type(scope): description

[optional body]

[optional footer]
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `test`: Test changes
- `ci`: CI/CD changes
- `chore`: Maintenance tasks
- `security`: Security improvements

**Examples**:
```
feat(search): add vector similarity search
fix(auth): resolve token validation issue
docs(api): update endpoint documentation
test(integration): add Slack bot integration tests
```

## Testing Guidelines

### Test Structure

```python
import pytest
from unittest.mock import MagicMock, patch

class TestDocumentProcessor:
    """Test suite for DocumentProcessor class."""
    
    def test_process_documents_success(self, sample_documents):
        """Test successful document processing."""
        processor = DocumentProcessor()
        result = processor.process_documents(sample_documents)
        
        assert len(result) == len(sample_documents)
        assert all(doc["processed"] for doc in result)
        
    def test_process_documents_empty_list(self):
        """Test error handling for empty document list."""
        processor = DocumentProcessor()
        
        with pytest.raises(ValueError, match="cannot be empty"):
            processor.process_documents([])
            
    @pytest.mark.integration
    def test_end_to_end_processing(self, knowledge_base):
        """Test complete processing workflow."""
        # Integration test implementation
        pass
```

### Test Categories

Use pytest markers to categorize tests:

```python
@pytest.mark.unit
def test_unit_functionality():
    """Unit test for isolated functionality."""
    pass

@pytest.mark.integration
def test_component_integration():
    """Integration test for component interaction."""
    pass

@pytest.mark.slow
def test_performance_heavy():
    """Test that takes significant time to run."""
    pass
```

### Coverage Requirements

- **Minimum Coverage**: 80% overall
- **New Code**: 90% coverage required
- **Critical Components**: 95% coverage (authentication, security)

## Documentation

### Docstring Format

Use Google-style docstrings:

```python
def search_knowledge_base(query: str, limit: int = 10) -> List[SearchResult]:
    """Search the knowledge base for relevant documents.
    
    This function performs a hybrid search combining keyword matching
    and vector similarity to find the most relevant documents.
    
    Args:
        query: The search query string
        limit: Maximum number of results to return (default: 10)
        
    Returns:
        List of SearchResult objects ordered by relevance
        
    Raises:
        ValueError: If query is empty or limit is invalid
        SearchError: If search operation fails
        
    Example:
        >>> results = search_knowledge_base("Python programming", limit=5)
        >>> print(f"Found {len(results)} results")
    """
```

### Documentation Updates

When contributing, update relevant documentation:

- **API changes**: Update API documentation
- **New features**: Add user guide sections
- **Configuration**: Update configuration documentation
- **Deployment**: Update deployment guides

## Community

### Communication Channels

- **GitHub Discussions**: General questions and feature discussions
- **GitHub Issues**: Bug reports and feature requests
- **Pull Requests**: Code review and collaboration

### Getting Help

1. **Documentation**: Check existing documentation first
2. **Search Issues**: Look for similar questions/problems
3. **Create Discussion**: For questions not covered in docs
4. **Create Issue**: For bugs or specific feature requests

### Recognition

We recognize contributors in several ways:

- **Contributors List**: Updated automatically in README
- **Release Notes**: Contributors mentioned in release notes
- **Hall of Fame**: Special recognition for significant contributions

## Security

### Security Considerations

When contributing, consider security implications:

- **Input Validation**: Validate all user inputs
- **Authentication**: Follow authentication best practices
- **Secrets**: Never commit secrets or credentials
- **Dependencies**: Use secure, up-to-date dependencies

### Reporting Security Issues

**Do not** create public issues for security vulnerabilities. Instead:

1. Email security@slack-kb-agent.com
2. Include detailed description and reproduction steps
3. Allow time for assessment and fix
4. Coordinate disclosure timeline

## License

By contributing to this project, you agree that your contributions will be licensed under the project's [MIT License](LICENSE).

## Questions?

If you have questions about contributing:

1. Check this document and other documentation
2. Search existing GitHub issues and discussions
3. Create a new discussion with your question
4. Reach out to maintainers if needed

Thank you for contributing to Slack KB Agent! 🎉