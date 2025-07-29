# CI/CD Workflow Setup Guide

This document provides templates and setup instructions for implementing comprehensive CI/CD workflows for the Slack-KB-Agent project.

## Required GitHub Actions Workflows

### 1. Continuous Integration (`ci.yml`)

```yaml
name: CI
on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, "3.10", "3.11"]
    
    services:
      postgres:
        image: postgres:13
        env:
          POSTGRES_PASSWORD: postgres
          POSTGRES_DB: test_db
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
      
      redis:
        image: redis:7
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5

    steps:
    - uses: actions/checkout@v4
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[llm]"
        pip install pytest pytest-cov pytest-xdist
    
    - name: Run tests with coverage
      env:
        DATABASE_URL: postgresql://postgres:postgres@localhost:5432/test_db
        REDIS_URL: redis://localhost:6379/0
      run: |
        pytest --cov=src/slack_kb_agent --cov-report=xml --cov-report=term-missing -n auto
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        fail_ci_if_error: true

  lint:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: "3.11"
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install ruff black mypy bandit
    
    - name: Run linting
      run: |
        ruff check src/ tests/
        black --check src/ tests/
        mypy src/
        bandit -r src/ -f json -o bandit-report.json
    
    - name: Upload security scan results
      uses: github/codeql-action/upload-sarif@v2
      if: always()
      with:
        sarif_file: bandit-report.json
```

### 2. Security Scanning (`security.yml`)

```yaml
name: Security
on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]
  schedule:
    - cron: '0 6 * * 1'  # Weekly on Monday

jobs:
  dependency-check:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - name: Run Safety check
      uses: pyupio/safety@v2.3.4
      with:
        api-key: ${{ secrets.SAFETY_API_KEY }}
    
    - name: Run pip-audit
      run: |
        pip install pip-audit
        pip-audit --desc --output pip-audit-report.json --format json

  codeql:
    runs-on: ubuntu-latest
    permissions:
      actions: read
      contents: read
      security-events: write
    
    steps:
    - uses: actions/checkout@v4
    - name: Initialize CodeQL
      uses: github/codeql-action/init@v2
      with:
        languages: python
    
    - name: Autobuild
      uses: github/codeql-action/autobuild@v2
    
    - name: Perform CodeQL Analysis
      uses: github/codeql-action/analyze@v2

  container-scan:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - name: Build Docker image
      run: docker build -t slack-kb-agent:test .
    
    - name: Run Trivy vulnerability scanner
      uses: aquasecurity/trivy-action@master
      with:
        image-ref: 'slack-kb-agent:test'
        format: 'sarif'
        output: 'trivy-results.sarif'
    
    - name: Upload Trivy scan results
      uses: github/codeql-action/upload-sarif@v2
      with:
        sarif_file: 'trivy-results.sarif'
```

### 3. Release Management (`release.yml`)

```yaml
name: Release
on:
  push:
    tags:
      - 'v*'

jobs:
  release:
    runs-on: ubuntu-latest
    permissions:
      contents: write
      packages: write
    
    steps:
    - uses: actions/checkout@v4
      with:
        fetch-depth: 0
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: "3.11"
    
    - name: Install build dependencies
      run: |
        python -m pip install --upgrade pip
        pip install build twine
    
    - name: Build package
      run: python -m build
    
    - name: Create GitHub Release
      uses: softprops/action-gh-release@v1
      with:
        generate_release_notes: true
        files: dist/*
    
    - name: Publish to PyPI
      env:
        TWINE_USERNAME: __token__
        TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
      run: twine upload dist/*
    
    - name: Build and push Docker image
      uses: docker/build-push-action@v4
      with:
        context: .
        push: true
        tags: |
          ghcr.io/${{ github.repository }}:${{ github.ref_name }}
          ghcr.io/${{ github.repository }}:latest
```

## Setup Instructions

### 1. Repository Secrets Configuration

Add these secrets in GitHub Settings > Secrets and variables > Actions:

```
PYPI_API_TOKEN=pypi-...           # PyPI publishing token
SAFETY_API_KEY=your-safety-key    # Safety.com API key (optional)
CODECOV_TOKEN=your-codecov-token  # Codecov integration token
```

### 2. Branch Protection Rules

Configure these branch protection rules for `main` branch:

- Require status checks to pass before merging
- Required status checks:
  - `test (3.8)`, `test (3.9)`, `test (3.10)`, `test (3.11)`
  - `lint`
  - `dependency-check`
  - `codeql`
- Require branches to be up to date before merging
- Require pull request reviews before merging (1 reviewer minimum)
- Dismiss stale PR approvals when new commits are pushed
- Restrict pushes that create files over 100MB

### 3. Integration Setup

#### Codecov Integration
1. Sign up at [codecov.io](https://codecov.io/)
2. Connect your GitHub repository
3. Add `CODECOV_TOKEN` to repository secrets

#### Container Registry Setup
1. Enable GitHub Container Registry in repository settings
2. Workflows will automatically push images to `ghcr.io`

## Workflow Features

### Advanced Testing
- **Multi-version testing**: Python 3.8-3.11 compatibility
- **Service dependencies**: PostgreSQL and Redis for integration tests
- **Parallel execution**: `pytest-xdist` for faster test runs
- **Coverage reporting**: XML and terminal coverage reports

### Security Integration
- **Static analysis**: Bandit security scanning
- **Dependency scanning**: Safety and pip-audit for vulnerability detection
- **Code analysis**: GitHub CodeQL semantic analysis
- **Container scanning**: Trivy for Docker image vulnerabilities

### Quality Gates
- **Code formatting**: Black and Ruff enforcement
- **Type checking**: MyPy static type analysis
- **Test coverage**: Minimum coverage requirements
- **Security baseline**: No high-severity vulnerabilities allowed

### Release Automation
- **Semantic versioning**: Git tag-based releases
- **Package publishing**: Automatic PyPI deployment
- **Container images**: Multi-tag Docker image publishing
- **Release notes**: Auto-generated from commit messages

## Monitoring and Observability

The workflows include monitoring capabilities:

- **Build metrics**: Execution time and success rates
- **Test metrics**: Coverage trends and failure patterns  
- **Security metrics**: Vulnerability counts and remediation tracking
- **Performance metrics**: Build duration and resource usage

## Rollback Procedures

### Failed Deployment Rollback
1. Revert to previous Git tag: `git tag -d v1.x.x && git push origin :refs/tags/v1.x.x`
2. Create new patch release with fixes
3. Monitor deployment health metrics

### Broken Build Recovery
1. Identify failing workflow in Actions tab
2. Check workflow logs for specific error details
3. Create hotfix branch for urgent fixes
4. Use workflow re-run capabilities for transient failures

## Customization Options

### Environment-Specific Workflows
- **Development**: Faster feedback with reduced test matrix
- **Staging**: Full test suite with performance benchmarks
- **Production**: Additional security scans and approval gates

### Integration Extensions
- **Slack notifications**: Build status updates to team channels
- **Jira integration**: Automatic ticket updates from PR links
- **Performance monitoring**: APM integration for production deployments

Refer to [GitHub Actions documentation](https://docs.github.com/en/actions) for additional configuration options.