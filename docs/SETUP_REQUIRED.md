# Manual Setup Requirements

This document outlines the manual setup steps required after the automated SDLC implementation due to GitHub App permission limitations.

## Critical: Repository Owner Actions Required

### 1. GitHub Workflows Setup (HIGH PRIORITY)

Due to GitHub App permission restrictions, the following workflow files must be manually created:

#### Create Workflow Directory
```bash
mkdir -p .github/workflows
```

#### Copy Workflow Templates
```bash
# Core CI/CD pipeline
cp docs/workflows/examples/ci.yml .github/workflows/ci.yml

# Security scanning workflow
cp docs/workflows/examples/security-scan.yml .github/workflows/security-scan.yml

# Dependency management workflow
cp docs/workflows/examples/dependency-update.yml .github/workflows/dependency-update.yml
```

### 2. Repository Settings Configuration

#### Branch Protection Rules (REQUIRED)
Navigate to Settings > Branches and configure:

**Main Branch Protection:**
- ✅ Require pull request reviews before merging (1 reviewer minimum)
- ✅ Require status checks to pass before merging
- ✅ Require branches to be up to date before merging
- ✅ Require linear history
- ✅ Include administrators
- ✅ Restrict pushes that create files larger than 100MB

**Required Status Checks:**
- `security-analysis`
- `code-quality`
- `test-suite`
- `docker-build`

#### Environment Configuration
Create the following environments in Settings > Environments:

**Staging Environment:**
- Required reviewers: 1
- Deployment branches: `develop`
- Environment secrets: `STAGING_*` variables

**Production Environment:**
- Required reviewers: 2
- Deployment branches: `main`
- Wait timer: 5 minutes
- Environment secrets: `PRODUCTION_*` variables

### 3. Secrets Management
Add repository secrets in Settings → Secrets and variables → Actions:
* `SLACK_BOT_TOKEN` - Production Slack bot token
* `OPENAI_API_KEY` - OpenAI API key for LLM features
* `DOCKER_REGISTRY_TOKEN` - Container registry authentication

## GitHub Actions Workflows

Manual creation required for workflow files in `.github/workflows/`:

1. **CI Pipeline** (`ci.yml`) - Test automation and code quality
2. **Security Scanning** (`security.yml`) - Vulnerability detection
3. **Release Automation** (`release.yml`) - Automated versioning and deployment

## External Integrations

### Monitoring Setup
* Configure Prometheus metrics collection endpoints
* Set up Grafana dashboards for visualization
* Configure alerting rules for critical system events

### Security Tools
* Enable Dependabot for automated dependency updates
* Configure CodeQL for security analysis
* Set up container registry scanning

## Documentation Hosting
* Deploy documentation to GitHub Pages or external platform
* Configure custom domain if required
* Set up automated documentation builds

## Deployment Configuration
* Configure production environment variables
* Set up container orchestration (Docker Compose/Kubernetes)
* Configure load balancing and health checks