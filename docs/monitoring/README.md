# Monitoring and Observability

Comprehensive monitoring and observability setup for Slack KB Agent, including metrics collection, alerting, logging, and performance monitoring.

## Overview

The monitoring stack includes:
- **Prometheus**: Metrics collection and storage
- **Grafana**: Visualization and dashboards
- **Alert Manager**: Alert routing and notification
- **Structured Logging**: Application and system logs
- **Health Checks**: Service availability monitoring
- **Performance Monitoring**: Application performance insights

## Quick Start

### Enable Monitoring
```bash
# Start with full monitoring stack
docker-compose --profile monitoring up -d

# Access monitoring interfaces
# Grafana: http://localhost:3001 (admin/admin)
# Prometheus: http://localhost:9091
# Application metrics: http://localhost:9090/metrics
```

### Basic Health Check
```bash
curl http://localhost:9090/health
```

## Metrics Collection

### Application Metrics

The Slack KB Agent exposes metrics at `/metrics` endpoint in Prometheus format:

#### Request Metrics
- `slack_messages_received_total`: Total Slack messages processed
- `slack_responses_sent_total`: Total responses sent to Slack
- `slack_commands_processed_total`: Total slash commands processed
- `http_requests_total`: HTTP requests by method and status

#### Performance Metrics
- `query_duration_seconds`: Query processing time distribution
- `search_response_time_seconds`: Search operation latency
- `llm_request_duration_seconds`: LLM API request latency
- `knowledge_base_operations_duration_seconds`: KB operation timing

#### Knowledge Base Metrics
- `kb_total_documents`: Total documents in knowledge base
- `kb_search_queries_total`: Total search queries executed
- `kb_vector_search_queries_total`: Vector search queries (if enabled)
- `kb_cache_hits_total`: Cache hit rate for search results

#### System Health Metrics
- `memory_usage_bytes`: Application memory consumption
- `disk_usage_bytes`: Disk space utilization
- `database_connections_active`: Active database connections
- `redis_connections_active`: Active Redis connections

### Custom Metrics Example
```python
from prometheus_client import Counter, Histogram, Gauge

# Custom business metrics
knowledge_discovery_events = Counter(
    'knowledge_discovery_events_total',
    'Total knowledge discovery events',
    ['source_type', 'user_id']
)

user_satisfaction_score = Gauge(
    'user_satisfaction_score',
    'User satisfaction score from feedback',
    ['time_period']
)

response_accuracy = Histogram(
    'response_accuracy_score',
    'Distribution of response accuracy scores',
    buckets=[0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0]
)
```

## Dashboards

### Grafana Dashboard Configuration

**Main Dashboard Panels**:
1. **Overview**: System health, uptime, error rates
2. **Performance**: Response times, throughput, latency percentiles
3. **Usage**: Active users, popular queries, knowledge gaps
4. **Infrastructure**: Database, Redis, memory, CPU usage
5. **Business Metrics**: Knowledge discovery, user satisfaction

### Key Performance Indicators (KPIs)
```yaml
# Example dashboard configuration
Dashboard: Slack KB Agent Overview
Panels:
  - System Health:
      - Uptime: 99.9%
      - Error Rate: < 0.1%
      - Response Time: P95 < 500ms
  
  - Usage Metrics:
      - Daily Active Users: 150
      - Queries per Hour: 45
      - Knowledge Coverage: 85%
  
  - Performance:
      - Search Latency: P50 < 100ms, P95 < 300ms
      - Memory Usage: < 80% of limit
      - Database Connections: < 50% of pool
```

## Alerting

### Alert Rules

**Critical Alerts** (PagerDuty/Slack):
- Application down for > 2 minutes
- Error rate > 5% for > 5 minutes
- Memory usage > 90% for > 10 minutes
- Database connections > 80% for > 5 minutes

**Warning Alerts** (Slack/Email):
- Response time P95 > 1 second for > 10 minutes
- Disk usage > 80%
- Redis connection failures
- Knowledge base not updated > 24 hours

### Alert Manager Configuration
```yaml
# monitoring/alertmanager.yml
global:
  smtp_smarthost: 'localhost:587'
  smtp_from: 'alerts@company.com'

route:
  group_by: ['alertname', 'cluster', 'service']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'web.hook'

receivers:
  - name: 'web.hook'
    slack_configs:
      - api_url: 'YOUR_SLACK_WEBHOOK_URL'
        channel: '#alerts'
        title: 'Slack KB Agent Alert'
        text: 'Alert: {{ range .Alerts }}{{ .Annotations.summary }}{{ end }}'
```

### Slack Integration for Alerts
```python
# Example alert handler
import requests

def send_alert_to_slack(alert_data):
    webhook_url = "YOUR_SLACK_WEBHOOK_URL"
    message = {
        "text": f"🚨 Alert: {alert_data['alertname']}",
        "attachments": [{
            "color": "danger" if alert_data['severity'] == 'critical' else "warning",
            "fields": [
                {"title": "Service", "value": alert_data.get('service', 'Unknown'), "short": True},
                {"title": "Severity", "value": alert_data['severity'], "short": True},
                {"title": "Description", "value": alert_data.get('description', ''), "short": False}
            ]
        }]
    }
    requests.post(webhook_url, json=message)
```

## Logging

### Structured Logging Configuration
```python
# Application logging setup
import structlog
import logging

logging.basicConfig(
    format="%(message)s",
    stream=sys.stdout,
    level=logging.INFO,
)

structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()
```

### Log Levels and Categories
```python
# Usage examples
logger.info("Query processed", 
           user_id="U123456", 
           query="deployment guide", 
           response_time=0.245,
           results_count=3)

logger.warning("Slow query detected", 
              query="complex search", 
              duration=2.1, 
              threshold=1.0)

logger.error("Database connection failed", 
            error_type="ConnectionTimeout",
            retry_count=3,
            database_url="postgresql://...")
```

### Log Aggregation
```yaml
# ELK Stack configuration
version: '3.8'
services:
  elasticsearch:
    image: elasticsearch:7.14.0
    environment:
      - discovery.type=single-node
    
  logstash:
    image: logstash:7.14.0
    volumes:
      - ./logstash.conf:/usr/share/logstash/pipeline/logstash.conf
    
  kibana:
    image: kibana:7.14.0
    ports:
      - "5601:5601"
    environment:
      - ELASTICSEARCH_HOSTS=http://elasticsearch:9200
```

## Performance Monitoring

### Application Performance Monitoring (APM)
```python
# Example APM integration
from slack_kb_agent.monitoring import APMTracer

tracer = APMTracer()

@tracer.trace("search_operation")
def search_knowledge_base(query: str):
    with tracer.span("vector_search") as span:
        span.set_tag("query_length", len(query))
        results = perform_vector_search(query)
        span.set_tag("results_count", len(results))
    return results
```

### Memory Profiling
```python
# Memory monitoring
import psutil
import tracemalloc

def monitor_memory():
    process = psutil.Process()
    memory_info = process.memory_info()
    
    return {
        "rss_mb": memory_info.rss / 1024 / 1024,
        "vms_mb": memory_info.vms / 1024 / 1024,
        "percent": process.memory_percent()
    }

# Enable tracemalloc for detailed memory tracking
tracemalloc.start()
```

### Performance Benchmarks
```bash
# Run performance benchmarks
make benchmark

# Memory profiling
python -m memory_profiler bot.py

# CPU profiling
python -m cProfile -o profile.stats bot.py
```

## Health Checks

### Application Health Endpoints
```python
# Health check implementation
@app.route('/health')
def health_check():
    checks = {
        "database": check_database_connection(),
        "redis": check_redis_connection(),
        "slack": check_slack_api_connection(),
        "memory": check_memory_usage(),
        "disk": check_disk_space()
    }
    
    overall_status = "healthy" if all(checks.values()) else "unhealthy"
    
    return {
        "status": overall_status,
        "timestamp": datetime.utcnow().isoformat(),
        "checks": checks,
        "version": get_application_version()
    }

@app.route('/ready')
def readiness_check():
    # Kubernetes readiness probe
    return {"status": "ready"} if is_application_ready() else ({"status": "not ready"}, 503)

@app.route('/live')
def liveness_check():
    # Kubernetes liveness probe
    return {"status": "alive"}
```

### External Service Monitoring
```python
# Monitor external dependencies
def check_external_services():
    services = {
        "openai_api": check_openai_connection(),
        "github_api": check_github_api(),
        "slack_api": check_slack_api()
    }
    
    for service, status in services.items():
        if not status:
            logger.warning(f"External service down: {service}")
    
    return services
```

## Observability Best Practices

### Monitoring Strategy
1. **Golden Signals**: Latency, Traffic, Errors, Saturation
2. **RED Method**: Rate, Errors, Duration
3. **USE Method**: Utilization, Saturation, Errors
4. **SLI/SLO**: Service Level Indicators and Objectives

### Service Level Objectives (SLOs)
```yaml
SLOs:
  Availability:
    Target: 99.9%
    Measurement: Uptime monitoring
    
  Response Time:
    Target: 95% of requests < 500ms
    Measurement: P95 latency
    
  Error Rate:
    Target: < 0.1% of requests
    Measurement: HTTP 5xx responses
    
  Knowledge Freshness:
    Target: Knowledge updated within 24 hours
    Measurement: Last ingestion timestamp
```

### Alert Fatigue Prevention
- **Meaningful Alerts**: Only alert on actionable issues
- **Alert Grouping**: Group related alerts together
- **Escalation Policies**: Tiered escalation for different severities
- **Alert Suppression**: Suppress known issues during maintenance

## Troubleshooting Monitoring

### Common Issues

#### Metrics Not Appearing
```bash
# Check metrics endpoint
curl http://localhost:9090/metrics

# Verify Prometheus scraping
curl http://localhost:9091/api/v1/targets

# Check application logs
docker-compose logs slack-kb-agent
```

#### High Memory Usage
```bash
# Monitor memory in real-time
docker stats slack-kb-agent

# Analyze memory allocation
python -m tracemalloc --leaks
```

#### Slow Queries
```sql
-- PostgreSQL slow query analysis
SELECT query, mean_time, calls, total_time
FROM pg_stat_statements 
ORDER BY mean_time DESC 
LIMIT 10;
```

### Performance Optimization
- **Metric Cardinality**: Avoid high-cardinality labels
- **Scraping Frequency**: Balance freshness vs. overhead
- **Retention Policies**: Configure appropriate data retention
- **Resource Allocation**: Right-size monitoring infrastructure

## Integration Examples

### Slack Integration
```python
# Send monitoring alerts to Slack
def send_monitoring_update():
    metrics = get_current_metrics()
    message = f"""
📊 *Daily Monitoring Report*
• Uptime: {metrics['uptime']}%
• Queries Processed: {metrics['queries_today']}
• Average Response Time: {metrics['avg_response_time']}ms
• Active Users: {metrics['active_users']}
    """
    send_slack_message(channel="#ops", message=message)
```

### PagerDuty Integration
```python
# Critical alert integration
import pypd

def trigger_pagerduty_alert(alert_data):
    pypd.api_key = "YOUR_API_KEY"
    
    incident = pypd.Incident.create(
        title=f"Slack KB Agent: {alert_data['title']}",
        service=pypd.Service.find_one(name="Slack KB Agent"),
        body={
            "type": "incident_body",
            "details": alert_data['description']
        }
    )
```

## Security Monitoring

### Security Metrics
- Authentication failures
- Rate limit violations
- Suspicious query patterns
- Data access patterns
- Security scan results

### Compliance Monitoring
```python
# Example security monitoring
def monitor_security_events():
    events = {
        "auth_failures": count_auth_failures_last_hour(),
        "rate_limit_hits": count_rate_limit_violations(),
        "suspicious_queries": detect_suspicious_patterns(),
        "data_access": audit_data_access()
    }
    
    for event_type, count in events.items():
        security_metric.labels(event_type=event_type).set(count)
```

## Cost Monitoring

### Resource Usage Tracking
```python
# Cost monitoring for cloud deployments
def track_resource_costs():
    return {
        "cpu_hours": get_cpu_usage_hours(),
        "memory_gb_hours": get_memory_usage_gb_hours(),
        "storage_gb": get_storage_usage_gb(),
        "network_gb": get_network_transfer_gb()
    }
```

### Cost Optimization Alerts
- Resource utilization below thresholds
- Unexpected cost spikes
- Inefficient resource allocation
- Scaling recommendations

This comprehensive monitoring setup ensures full observability of the Slack KB Agent, enabling proactive issue detection, performance optimization, and reliable operations.