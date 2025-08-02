# GitHub Workflows Implementation Guide

This guide provides comprehensive instructions for implementing GitHub Actions workflows for the Slack KB Agent project.

## Important Note

**Due to GitHub App permission limitations, these workflows cannot be automatically created.** Repository maintainers must manually create the workflow files in the `.github/workflows/` directory.

## Quick Implementation

### 1. Create Workflow Directory
```bash
mkdir -p .github/workflows
```

### 2. Copy Workflow Templates
Copy the workflow files from `docs/workflows/examples/` to `.github/workflows/`:

```bash
# Core CI/CD pipeline
cp docs/workflows/examples/ci.yml .github/workflows/ci.yml

# Security scanning
cp docs/workflows/examples/security-scan.yml .github/workflows/security-scan.yml

# Dependency management
cp docs/workflows/examples/dependency-update.yml .github/workflows/dependency-update.yml
```

### 3. Configure Secrets
Set up the following secrets in your GitHub repository settings:

**Required Secrets:**
- `GITGUARDIAN_API_KEY`: For secret detection
- `SLACK_WEBHOOK`: For deployment notifications
- `SECURITY_SLACK_WEBHOOK`: For security alerts

**Optional Secrets:**
- `CODECOV_TOKEN`: For code coverage reporting
- `SONAR_TOKEN`: For SonarCloud analysis

## Workflow Overview

### 1. CI/CD Pipeline (`ci.yml`)

**Triggers:**
- Push to `main` and `develop` branches
- Pull requests to `main` and `develop`
- Nightly scheduled runs for security

**Jobs:**
- **Security Analysis**: GitGuardian, Bandit, Safety, Semgrep
- **Code Quality**: Black, Ruff, MyPy formatting and linting
- **Test Suite**: Unit, integration, and performance tests
- **Docker Build**: Multi-stage builds with security scanning
- **SBOM Generation**: Software Bill of Materials
- **E2E Tests**: End-to-end workflow validation
- **Deployment**: Staging and production deployments

### 2. Security Scanning (`security-scan.yml`)

**Triggers:**
- Daily scheduled scans at 3 AM UTC
- Manual workflow dispatch
- Push to main branches

**Jobs:**
- **Secret Detection**: GitGuardian, TruffleHog, detect-secrets
- **Dependency Scanning**: Safety, pip-audit
- **SAST**: Bandit, Semgrep, CodeQL
- **Container Security**: Trivy, Grype, Docker Scout
- **License Compliance**: pip-licenses analysis
- **IaC Security**: Checkov, Hadolint
- **Compliance Check**: CIS, OWASP, NIST alignment

### 3. Dependency Management (`dependency-update.yml`)

**Triggers:**
- Weekly scheduled runs (Mondays at 9 AM UTC)
- Manual workflow dispatch

**Jobs:**
- **Security Updates**: High-priority vulnerability fixes
- **Regular Updates**: Patch and minor version updates
- **Major Updates**: Manual review required
- **Dependency Pinning**: Reproducible builds
- **Vulnerability Monitoring**: Continuous security monitoring

## Configuration Requirements

### Repository Settings

1. **Branch Protection Rules** (Required):
   ```yaml
   # Protect main branch
   Branch: main
   Settings:
     - Require pull request reviews before merging
     - Require status checks to pass before merging
     - Require branches to be up to date before merging
     - Require linear history
     - Include administrators
   
   Required Status Checks:
     - Security Analysis
     - Code Quality
     - Test Suite (python-3.8, 3.9, 3.10, 3.11)
     - Docker Build
   ```

2. **Environments**:
   ```yaml
   # Staging environment
   Environment: staging
   Protection Rules:
     - Required reviewers: 1
     - Deployment branches: develop
   
   # Production environment  
   Environment: production
   Protection Rules:
     - Required reviewers: 2
     - Deployment branches: main
     - Wait timer: 5 minutes
   ```

### Secret Configuration

1. **Navigate to Repository Settings > Secrets and Variables > Actions**

2. **Add Repository Secrets**:
   ```bash
   # Security scanning
   GITGUARDIAN_API_KEY=your_gitguardian_api_key
   
   # Notifications
   SLACK_WEBHOOK=https://hooks.slack.com/services/YOUR/WEBHOOK/URL
   SECURITY_SLACK_WEBHOOK=https://hooks.slack.com/services/YOUR/SECURITY/URL
   
   # Code coverage (optional)
   CODECOV_TOKEN=your_codecov_token
   
   # Container registry (if using private registry)
   REGISTRY_USERNAME=your_registry_username
   REGISTRY_PASSWORD=your_registry_password
   ```

## Workflow Customization

### Environment Variables

Customize workflows by modifying environment variables:

```yaml
env:
  PYTHON_VERSION: '3.11'          # Python version for CI
  NODE_VERSION: '18'              # Node.js for tools
  REGISTRY: ghcr.io               # Container registry
  IMAGE_NAME: ${{ github.repository }}
```

### Job Customization

#### Skip Jobs on Draft PRs
```yaml
if: github.event.pull_request.draft == false
```

#### Run Only on Specific File Changes
```yaml
if: contains(github.event.head_commit.message, '[security]')
```

#### Matrix Strategy for Multiple Versions
```yaml
strategy:
  matrix:
    python-version: ['3.8', '3.9', '3.10', '3.11']
    os: [ubuntu-latest, macos-latest, windows-latest]
```

### Notification Customization

#### Slack Notifications
```yaml
- name: Notify deployment
  uses: 8398a7/action-slack@v3
  with:
    status: ${{ job.status }}
    channel: '#deployments'
    webhook_url: ${{ secrets.SLACK_WEBHOOK }}
    fields: repo,message,commit,author,action,eventName,ref,workflow
```

#### Email Notifications
```yaml
- name: Send email notification
  uses: dawidd6/action-send-mail@v3
  with:
    server_address: smtp.gmail.com
    server_port: 587
    username: ${{ secrets.EMAIL_USERNAME }}
    password: ${{ secrets.EMAIL_PASSWORD }}
    subject: CI/CD Status for ${{ github.repository }}
    body: Build completed with status ${{ job.status }}
    to: team@company.com
```

## Advanced Configuration

### Conditional Deployments

```yaml
deploy-production:
  if: |
    github.ref == 'refs/heads/main' && 
    github.event_name == 'push' &&
    !contains(github.event.head_commit.message, '[skip-deploy]')
```

### Multi-Environment Deployments

```yaml
strategy:
  matrix:
    environment: [staging, production]
    include:
      - environment: staging
        branch: develop
        reviewers: 1
      - environment: production
        branch: main
        reviewers: 2
```

### Performance Optimization

#### Caching
```yaml
- name: Cache Python dependencies
  uses: actions/cache@v3
  with:
    path: ~/.cache/pip
    key: ${{ runner.os }}-pip-${{ hashFiles('**/pyproject.toml') }}
    restore-keys: |
      ${{ runner.os }}-pip-

- name: Cache Docker layers
  uses: actions/cache@v3
  with:
    path: /tmp/.buildx-cache
    key: ${{ runner.os }}-buildx-${{ github.sha }}
    restore-keys: |
      ${{ runner.os }}-buildx-
```

#### Parallel Job Execution
```yaml
# Run security and quality checks in parallel
jobs:
  security:
    # Security job definition
  quality:
    # Quality job definition
  
  tests:
    needs: [security, quality]  # Wait for both to complete
    # Test job definition
```

## Monitoring and Maintenance

### Workflow Health Monitoring

1. **Regular Review**: Monthly review of workflow performance
2. **Failure Analysis**: Track and analyze workflow failures
3. **Performance Metrics**: Monitor execution times and resource usage
4. **Cost Optimization**: Optimize for GitHub Actions minutes usage

### Maintenance Tasks

#### Weekly
- Review failed workflow runs
- Update workflow if needed
- Check secret expiration

#### Monthly
- Review workflow performance metrics
- Update workflow dependencies
- Security review of workflow permissions

#### Quarterly
- Comprehensive workflow audit
- Update to latest action versions
- Review and update security policies

## Troubleshooting

### Common Issues

#### 1. Workflow Not Triggering
**Cause**: Incorrect trigger configuration or branch protection
**Solution**: 
```yaml
# Verify trigger syntax
on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]
```

#### 2. Secret Not Found
**Cause**: Secret not configured or incorrect name
**Solution**: 
- Verify secret exists in repository settings
- Check exact name matching (case-sensitive)
- Ensure secret is available in environment

#### 3. Permission Denied
**Cause**: Insufficient GITHUB_TOKEN permissions
**Solution**:
```yaml
permissions:
  contents: read
  security-events: write
  pull-requests: write
```

#### 4. Cache Miss
**Cause**: Cache key changed or expired
**Solution**: Review cache key generation and add fallback keys

### Debug Mode

Enable debug logging:
```yaml
env:
  ACTIONS_STEP_DEBUG: true
  ACTIONS_RUNNER_DEBUG: true
```

### Workflow Validation

Before committing workflows:
1. Use GitHub's workflow syntax checker
2. Test with workflow dispatch on a test branch
3. Validate all required secrets are configured
4. Review permissions and security implications

## Security Best Practices

### Workflow Security

1. **Principle of Least Privilege**: Grant minimal required permissions
2. **Secret Management**: Use repository secrets, not environment variables
3. **Third-party Actions**: Pin to specific versions, review security
4. **Resource Limits**: Set appropriate timeouts and resource constraints

### Code Security

```yaml
# Example security configuration
permissions:
  contents: read
  security-events: write
  
timeout-minutes: 30

env:
  # Prevent credential exposure
  ACTIONS_ALLOW_UNSECURE_COMMANDS: false
```

### Action Pinning

```yaml
# Pin to specific commit SHA for security
- uses: actions/checkout@8e5e7e5ab8b370d6c329ec480221332ada57f0ab  # v4.1.1
- uses: actions/setup-python@65d7f2d534ac1bc67fcd62888c5f4f3d2cb2b236  # v4.7.1
```

## Migration Guide

### From Jenkins/Other CI Systems

1. **Audit Current Pipeline**: Document existing pipeline stages
2. **Map to GitHub Actions**: Identify equivalent actions
3. **Secrets Migration**: Migrate secrets to GitHub secrets
4. **Incremental Migration**: Migrate one workflow at a time
5. **Validation**: Parallel run during transition period

### Rollback Plan

1. **Backup**: Keep existing CI system until GitHub Actions is stable
2. **Feature Flags**: Use repository variables to enable/disable workflows
3. **Monitoring**: Monitor workflow success rates during transition
4. **Quick Revert**: Documented process to revert to previous CI system

## Support and Resources

### Documentation
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Workflow Syntax](https://docs.github.com/en/actions/using-workflows/workflow-syntax-for-github-actions)
- [Security Hardening](https://docs.github.com/en/actions/security-guides/security-hardening-for-github-actions)

### Community
- [GitHub Community Forum](https://github.community/)
- [GitHub Actions Marketplace](https://github.com/marketplace?type=actions)

### Internal Support
- Technical questions: Create GitHub issue
- Security concerns: Contact security team
- Workflow optimization: Discuss in team channel

This implementation guide provides everything needed to successfully deploy and maintain the GitHub Actions workflows for the Slack KB Agent project.