# Incident Response Runbook

This runbook provides step-by-step procedures for responding to incidents affecting the Slack KB Agent service.

## Incident Classification

### Severity Levels

| Severity | Description | Response Time | Examples |
|----------|-------------|---------------|----------|
| **Critical (P0)** | Service completely down or major data loss | 15 minutes | Application crash, database corruption |
| **High (P1)** | Significant feature impairment | 1 hour | Authentication failures, high error rates |
| **Medium (P2)** | Partial feature impairment | 4 hours | Slow responses, cache failures |
| **Low (P3)** | Minor issues or cosmetic problems | 24 hours | UI glitches, documentation errors |

## Initial Response

### 1. Acknowledge the Incident

- **Acknowledge** the alert within **5 minutes**
- **Assign** an incident commander
- **Create** incident tracking ticket
- **Notify** stakeholders via established channels

### 2. Initial Assessment

```bash
# Quick health check
curl -f http://slack-kb-agent:9090/health

# Check application logs
docker logs slack-kb-agent --tail=100

# Check system resources
docker stats slack-kb-agent
```

### 3. Communication

- **Internal**: Update incident channel
- **External**: Post status page update if customer-facing
- **Timeline**: Provide updates every 30 minutes for P0/P1

## Common Incident Scenarios

### Scenario 1: Application Down

**Symptoms**: Health check failures, 502/503 errors

**Diagnosis Steps**:
1. Check container status:
   ```bash
   docker ps | grep slack-kb-agent
   docker logs slack-kb-agent --tail=50
   ```

2. Check resource usage:
   ```bash
   docker stats slack-kb-agent
   free -h
   df -h
   ```

3. Check dependencies:
   ```bash
   docker logs slack-kb-postgres --tail=20
   docker logs slack-kb-redis --tail=20
   ```

**Resolution Steps**:
1. **Restart application**:
   ```bash
   docker-compose restart slack-kb-agent
   ```

2. **If restart fails**, check configuration:
   ```bash
   docker-compose config
   docker-compose logs slack-kb-agent
   ```

3. **Rollback** if recent deployment:
   ```bash
   docker-compose down
   docker pull slack-kb-agent:previous-tag
   docker-compose up -d
   ```

### Scenario 2: Database Connection Issues

**Symptoms**: Database connection errors, timeouts

**Diagnosis Steps**:
1. Check PostgreSQL status:
   ```bash
   docker logs slack-kb-postgres --tail=50
   docker exec slack-kb-postgres pg_isready -U postgres
   ```

2. Test database connectivity:
   ```bash
   docker exec slack-kb-agent python -c "
   import psycopg2
   conn = psycopg2.connect('postgresql://postgres:postgres@postgres:5432/slack_kb_agent')
   print('Connection successful')
   "
   ```

3. Check connection pool:
   ```bash
   # Monitor active connections
   docker exec slack-kb-postgres psql -U postgres -c "
   SELECT count(*) as active_connections 
   FROM pg_stat_activity 
   WHERE state = 'active';
   "
   ```

**Resolution Steps**:
1. **Restart PostgreSQL**:
   ```bash
   docker-compose restart postgres
   ```

2. **Check database integrity**:
   ```bash
   docker exec slack-kb-postgres pg_dump slack_kb_agent > backup_$(date +%Y%m%d_%H%M%S).sql
   ```

3. **Recreate database** if corrupted:
   ```bash
   slack-kb-db backup data/emergency_backup.json.gz
   docker-compose down postgres
   docker volume rm slack-kb-agent_postgres_data
   docker-compose up -d postgres
   slack-kb-db restore data/emergency_backup.json.gz
   ```

### Scenario 3: High Memory Usage

**Symptoms**: OOM kills, slow responses, memory alerts

**Diagnosis Steps**:
1. Check memory usage:
   ```bash
   docker stats slack-kb-agent
   docker exec slack-kb-agent python -c "
   import psutil
   print(f'Memory: {psutil.virtual_memory().percent}%')
   print(f'Available: {psutil.virtual_memory().available / 1024**3:.2f} GB')
   "
   ```

2. Check knowledge base size:
   ```bash
   curl -s http://slack-kb-agent:9090/metrics | grep kb_documents
   curl -s http://slack-kb-agent:9090/metrics | grep memory_usage
   ```

**Resolution Steps**:
1. **Increase memory limits**:
   ```yaml
   # docker-compose.yml
   services:
     slack-kb-agent:
       deploy:
         resources:
           limits:
             memory: 2G
   ```

2. **Reduce knowledge base size**:
   ```bash
   # Connect to application and reduce document limit
   docker exec slack-kb-agent python -c "
   from slack_kb_agent.knowledge_base import KnowledgeBase
   kb = KnowledgeBase(max_documents=5000)
   kb.save()
   "
   ```

3. **Restart with new limits**:
   ```bash
   docker-compose down
   docker-compose up -d
   ```

### Scenario 4: High Error Rates

**Symptoms**: 500 errors, exception alerts, user complaints

**Diagnosis Steps**:
1. Check error logs:
   ```bash
   docker logs slack-kb-agent 2>&1 | grep -i error | tail -20
   ```

2. Check error metrics:
   ```bash
   curl -s http://slack-kb-agent:9090/metrics | grep error_rate
   curl -s http://slack-kb-agent:9090/metrics | grep slack_messages
   ```

3. Check external dependencies:
   ```bash
   # Test Slack API
   curl -H "Authorization: Bearer $SLACK_BOT_TOKEN" \
        https://slack.com/api/auth.test
   
   # Test OpenAI API (if configured)
   curl -H "Authorization: Bearer $OPENAI_API_KEY" \
        https://api.openai.com/v1/models
   ```

**Resolution Steps**:
1. **Enable circuit breakers**:
   ```bash
   # Temporarily disable problematic features
   docker exec slack-kb-agent python -c "
   import os
   os.environ['LLM_ENABLED'] = 'false'
   os.environ['VECTOR_SEARCH_ENABLED'] = 'false'
   "
   docker-compose restart slack-kb-agent
   ```

2. **Check API rate limits**:
   ```bash
   curl -s http://slack-kb-agent:9090/metrics | grep rate_limit
   ```

3. **Implement graceful degradation**:
   ```bash
   # Update configuration to reduce external API calls
   docker exec slack-kb-agent python -c "
   # Update rate limits or disable features temporarily
   "
   ```

## Monitoring and Alerting

### Key Metrics to Monitor

```promql
# Application health
up{job="slack-kb-agent"}

# Error rate
rate(slack_kb_agent_errors_total[5m])

# Response time
histogram_quantile(0.95, rate(slack_kb_agent_query_duration_seconds_bucket[5m]))

# Memory usage
slack_kb_agent_memory_usage_bytes / slack_kb_agent_memory_limit_bytes

# Database connections
pg_stat_database_numbackends
```

### Alert Thresholds

```yaml
# prometheus/alert_rules.yml
- alert: SlackKBAgentDown
  expr: up{job="slack-kb-agent"} == 0
  for: 1m

- alert: HighErrorRate
  expr: rate(slack_kb_agent_errors_total[5m]) > 0.1
  for: 2m

- alert: HighResponseTime
  expr: histogram_quantile(0.95, rate(slack_kb_agent_query_duration_seconds_bucket[5m])) > 2
  for: 5m
```

## Post-Incident Procedures

### 1. Resolution Verification

```bash
# Verify application health
curl -f http://slack-kb-agent:9090/health

# Run smoke tests
pytest tests/smoke/ --target=production

# Check metrics return to normal
curl -s http://slack-kb-agent:9090/metrics | grep -E "(error_rate|response_time|memory_usage)"
```

### 2. Communication

- **Update** incident ticket with resolution
- **Notify** stakeholders of resolution
- **Post** final status page update
- **Schedule** post-incident review

### 3. Post-Incident Review

Within **72 hours** of resolution:

1. **Timeline Review**: Document incident timeline
2. **Root Cause Analysis**: Identify underlying causes
3. **Action Items**: Create follow-up tasks
4. **Process Improvements**: Update runbooks and monitoring

### 4. Documentation

Update this runbook with:
- New incident scenarios encountered
- Improved diagnostic procedures
- Additional resolution steps
- Lessons learned

## Emergency Contacts

| Role | Contact | Availability |
|------|---------|--------------|
| On-call Engineer | [Contact Info] | 24/7 |
| Incident Commander | [Contact Info] | Business hours |
| Database Administrator | [Contact Info] | On-call rotation |
| Security Team | [Contact Info] | 24/7 |

## Useful Commands Reference

### Health Checks
```bash
# Application health
curl http://slack-kb-agent:9090/health

# Database health
docker exec slack-kb-postgres pg_isready -U postgres

# Redis health
docker exec slack-kb-redis redis-cli ping

# Service status
docker-compose ps
```

### Log Access
```bash
# Application logs
docker logs slack-kb-agent -f

# Database logs
docker logs slack-kb-postgres -f

# System logs
journalctl -u docker -f
```

### Performance Monitoring
```bash
# Resource usage
docker stats

# Disk usage
df -h

# Memory usage
free -h

# Network connections
netstat -tulpn
```

### Backup and Recovery
```bash
# Create backup
slack-kb-db backup data/backups/emergency_$(date +%Y%m%d_%H%M%S).json.gz

# Restore from backup
slack-kb-db restore data/backups/backup_file.json.gz

# Database dump
docker exec slack-kb-postgres pg_dump -U postgres slack_kb_agent > backup.sql
```

---

**Remember**: During an incident, focus on restoration first, investigation second. Document everything for post-incident analysis.