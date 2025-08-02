# Deployment Documentation

This directory contains comprehensive deployment guides and configurations for the Slack KB Agent.

## Deployment Options

### 1. Docker Compose (Recommended for Development/Small Teams)
- **File**: `docker-compose.yml`
- **Use Case**: Development, testing, small team deployments
- **Resources**: Low to medium resource requirements
- **Complexity**: Simple setup and management

### 2. Kubernetes (Recommended for Production)
- **Files**: `k8s/` directory (to be created)
- **Use Case**: Production deployments, high availability
- **Resources**: Scalable resource allocation
- **Complexity**: Advanced setup, requires Kubernetes knowledge

### 3. Cloud Platforms
- **AWS**: ECS, EKS, Lambda (serverless)
- **Google Cloud**: Cloud Run, GKE, Cloud Functions
- **Azure**: Container Instances, AKS, Functions
- **Heroku**: Simple container deployment

### 4. Traditional Servers
- **Systemd**: Service management on Linux servers
- **PM2**: Process management for Node.js-style deployment
- **Supervisor**: Python process management

## Quick Start Deployment

### Using Docker Compose

1. **Clone and Configure**:
   ```bash
   git clone https://github.com/your-org/slack-kb-agent.git
   cd slack-kb-agent
   cp .env.example .env
   # Edit .env with your Slack credentials
   ```

2. **Start Services**:
   ```bash
   # Start with basic services
   docker-compose up -d
   
   # Or start with monitoring
   docker-compose --profile monitoring up -d
   ```

3. **Verify Deployment**:
   ```bash
   # Check service health
   curl http://localhost:9090/health
   
   # View logs
   docker-compose logs -f slack-kb-agent
   ```

### Using Docker Only

1. **Build Image**:
   ```bash
   docker build -t slack-kb-agent:latest .
   ```

2. **Run Container**:
   ```bash
   docker run -d \\
     --name slack-kb-agent \\
     -p 3000:3000 \\
     -p 9090:9090 \\
     -e SLACK_BOT_TOKEN=your-token \\
     -e SLACK_APP_TOKEN=your-app-token \\
     -e SLACK_SIGNING_SECRET=your-secret \\
     slack-kb-agent:latest
   ```

## Environment Configuration

### Required Environment Variables
```bash
# Slack Integration (Required)
SLACK_BOT_TOKEN=xoxb-your-bot-token
SLACK_APP_TOKEN=xapp-your-app-token
SLACK_SIGNING_SECRET=your-signing-secret

# Database (Optional - defaults to SQLite)
DATABASE_URL=postgresql://user:pass@host:5432/dbname

# Redis Cache (Optional)
REDIS_URL=redis://localhost:6379/0
```

### Optional Configuration
```bash
# LLM Integration
OPENAI_API_KEY=sk-your-openai-key
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key

# Vector Search
VECTOR_MODEL=all-MiniLM-L6-v2
SIMILARITY_THRESHOLD=0.5

# Performance
MAX_MEMORY_MB=2048
CACHE_TTL=3600

# Security
API_KEY_HEADER=X-API-Key
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=3600

# Monitoring
MONITORING_ENABLED=true
METRICS_PORT=9090
LOG_LEVEL=INFO
```

## Security Considerations

### Container Security
- **Non-root user**: Application runs as `appuser` (UID 1000)
- **Minimal base image**: Uses Python slim image
- **No secrets in image**: All secrets via environment variables
- **Health checks**: Built-in container health monitoring

### Network Security
```yaml
# Example nginx reverse proxy config
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://localhost:3000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
    
    location /metrics {
        proxy_pass http://localhost:9090;
        # Add authentication for metrics endpoint
        auth_basic "Metrics";
        auth_basic_user_file /etc/nginx/.htpasswd;
    }
}
```

### Secrets Management
```bash
# Using Docker secrets
echo "your-slack-token" | docker secret create slack_bot_token -
docker service create \\
  --secret slack_bot_token \\
  --env SLACK_BOT_TOKEN_FILE=/run/secrets/slack_bot_token \\
  slack-kb-agent:latest

# Using Kubernetes secrets
kubectl create secret generic slack-credentials \\
  --from-literal=bot-token="your-token" \\
  --from-literal=app-token="your-app-token"
```

## Performance Tuning

### Resource Allocation
```yaml
# Docker Compose resource limits
services:
  slack-kb-agent:
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '1.0'
        reservations:
          memory: 512M
          cpus: '0.5'
```

### Database Optimization
```bash
# PostgreSQL tuning for knowledge base workload
shared_buffers = 256MB
effective_cache_size = 1GB
work_mem = 4MB
maintenance_work_mem = 64MB
max_connections = 100
```

### Redis Configuration
```bash
# Redis optimization for caching
maxmemory 512mb
maxmemory-policy allkeys-lru
save 900 1
save 300 10
save 60 10000
```

## Monitoring and Observability

### Health Checks
```bash
# Application health
curl http://localhost:9090/health
{
  "status": "healthy",
  "timestamp": "2025-08-02T10:00:00Z",
  "services": {
    "database": "healthy",
    "redis": "healthy",
    "slack": "connected"
  }
}

# Metrics endpoint
curl http://localhost:9090/metrics
# Returns Prometheus-format metrics
```

### Log Management
```yaml
# Docker Compose logging configuration
services:
  slack-kb-agent:
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
```

### Monitoring Stack
```bash
# Start with full monitoring
docker-compose --profile monitoring up -d

# Access monitoring interfaces
# Prometheus: http://localhost:9091
# Grafana: http://localhost:3001 (admin/admin)
```

## Backup and Recovery

### Database Backup
```bash
# PostgreSQL backup
docker-compose exec postgres pg_dump -U postgres slack_kb_agent > backup.sql

# Application backup using built-in tools
docker-compose exec slack-kb-agent slack-kb-db backup /app/data/backup.json.gz
```

### Knowledge Base Backup
```bash
# Export knowledge base
docker-compose exec slack-kb-agent slack-kb-db export /app/data/kb-export.json

# Backup persistent volumes
docker run --rm -v slack-kb-agent_postgres_data:/data -v $(pwd):/backup ubuntu tar czf /backup/postgres-backup.tar.gz -C /data .
```

### Disaster Recovery
1. **Regular backups**: Automated daily/weekly backups
2. **Multi-region deployment**: For high availability
3. **Configuration as code**: Infrastructure as code for quick restoration
4. **Recovery testing**: Regular disaster recovery drills

## Scaling Strategies

### Horizontal Scaling
```yaml
# Docker Swarm scaling
docker service scale slack-kb-agent=3

# Kubernetes scaling
kubectl scale deployment slack-kb-agent --replicas=3
```

### Vertical Scaling
```yaml
# Increase resource limits
services:
  slack-kb-agent:
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2.0'
```

### Database Scaling
- **Read replicas**: For read-heavy workloads
- **Connection pooling**: PgBouncer for PostgreSQL
- **Sharding**: For very large knowledge bases

## Troubleshooting

### Common Issues

#### Container Won't Start
```bash
# Check logs
docker-compose logs slack-kb-agent

# Common issues:
# - Missing environment variables
# - Database connection failed
# - Port conflicts
```

#### Memory Issues
```bash
# Monitor memory usage
docker stats slack-kb-agent

# Check for memory leaks
docker-compose exec slack-kb-agent python -c "
import psutil
print(f'Memory usage: {psutil.virtual_memory().percent}%')
"
```

#### Performance Problems
```bash
# Check application metrics
curl http://localhost:9090/metrics | grep slack_

# Database performance
docker-compose exec postgres psql -U postgres -c "
SELECT query, mean_time, calls 
FROM pg_stat_statements 
ORDER BY mean_time DESC LIMIT 10;
"
```

### Debug Mode
```bash
# Run in debug mode
docker-compose -f docker-compose.yml -f docker-compose.debug.yml up

# Or with environment override
docker run -e DEBUG=true -e LOG_LEVEL=DEBUG slack-kb-agent:latest
```

## CI/CD Integration

### GitHub Actions
```yaml
# Example workflow snippet
- name: Build and test
  run: |
    docker build -t slack-kb-agent:test .
    docker run --rm slack-kb-agent:test pytest

- name: Deploy to staging
  run: |
    docker tag slack-kb-agent:test slack-kb-agent:staging
    docker push slack-kb-agent:staging
```

### Deployment Automation
```bash
# Blue-green deployment script
#!/bin/bash
NEW_VERSION=$1
docker pull slack-kb-agent:$NEW_VERSION
docker-compose -f docker-compose.blue.yml up -d
# Health check new deployment
if curl -f http://localhost:3001/health; then
    docker-compose -f docker-compose.green.yml down
    mv docker-compose.blue.yml docker-compose.green.yml
fi
```

## Cost Optimization

### Resource Efficiency
- **Multi-stage builds**: Smaller production images
- **Shared volumes**: Reduce storage duplication
- **Auto-scaling**: Scale down during low usage
- **Spot instances**: Use spot instances in cloud deployments

### Monitoring Costs
```bash
# Monitor resource usage
docker system df
docker system prune -a --volumes

# Cloud cost monitoring
# Use cloud provider cost management tools
```

## Compliance and Governance

### Security Scanning
```bash
# Container vulnerability scanning
docker scout cves slack-kb-agent:latest

# Dependency scanning
docker run --rm -v $(pwd):/app anchore/grype:latest /app
```

### Audit Logging
```yaml
# Enable audit logging
services:
  slack-kb-agent:
    environment:
      - AUDIT_LOGGING=true
      - AUDIT_LOG_LEVEL=INFO
```

### Compliance Reports
- **SOC 2**: Security and availability controls
- **GDPR**: Data protection and privacy
- **HIPAA**: Healthcare information protection (if applicable)

## Additional Resources

### Documentation Links
- [Docker Best Practices](https://docs.docker.com/develop/best-practices/)
- [Kubernetes Deployment Guide](https://kubernetes.io/docs/concepts/workloads/controllers/deployment/)
- [Slack App Deployment](https://api.slack.com/start/building/bolt-python)

### Community
- [GitHub Discussions](https://github.com/your-org/slack-kb-agent/discussions)
- [Discord Channel](https://discord.gg/slack-kb-agent)
- [Stack Overflow Tag](https://stackoverflow.com/questions/tagged/slack-kb-agent)

### Support
- **Documentation**: Comprehensive guides and API docs
- **Community Support**: GitHub issues and discussions
- **Enterprise Support**: Commercial support options available