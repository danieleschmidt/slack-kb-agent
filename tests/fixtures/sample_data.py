"""Sample test data and fixtures for Slack KB Agent tests."""

from typing import List, Dict, Any
from slack_kb_agent.models import Document


# Sample documents for testing
SAMPLE_DOCUMENTS = [
    {
        'content': '''# API Documentation

## Authentication
The API uses Bearer token authentication. Include your token in the Authorization header:

```
Authorization: Bearer your-api-token-here
```

## Rate Limiting
API requests are limited to 100 requests per minute per user.

## Error Handling
The API returns standard HTTP status codes:
- 200: Success
- 401: Unauthorized
- 404: Not Found
- 500: Internal Server Error
''',
        'source': 'docs/api.md',
        'metadata': {
            'type': 'documentation',
            'section': 'api',
            'last_updated': '2025-08-01'
        }
    },
    {
        'content': '''# Deployment Guide

## Docker Deployment

### Building the Image
```bash
docker build -t slack-kb-agent:latest .
```

### Running the Container
```bash
docker run -d \\
  --name slack-kb-agent \\
  -p 3000:3000 \\
  -p 9090:9090 \\
  -e SLACK_BOT_TOKEN=your-token \\
  slack-kb-agent:latest
```

## Kubernetes Deployment

Use the provided Helm chart:
```bash
helm install slack-kb-agent ./helm-chart
```

## Environment Variables
- SLACK_BOT_TOKEN: Required Slack bot token
- DATABASE_URL: PostgreSQL connection string
- REDIS_URL: Redis connection string
''',
        'source': 'docs/deployment.md',
        'metadata': {
            'type': 'documentation',
            'section': 'deployment',
            'last_updated': '2025-08-01'
        }
    },
    {
        'content': '''# Troubleshooting Guide

## Common Issues

### Bot Not Responding
1. Check if the bot token is valid
2. Verify Socket Mode is enabled
3. Check network connectivity
4. Review application logs

### Search Not Working
1. Verify knowledge base is populated
2. Check if vector search dependencies are installed
3. Restart the application

### Performance Issues
1. Check memory usage
2. Review database connections
3. Monitor Redis cache performance
4. Check for memory leaks

## Getting Help
- Check the logs in `/var/log/slack-kb-agent/`
- Review monitoring metrics at http://localhost:9090/metrics
- File issues on GitHub
''',
        'source': 'docs/troubleshooting.md',
        'metadata': {
            'type': 'documentation',
            'section': 'troubleshooting',
            'last_updated': '2025-08-01'
        }
    }
]

# Sample Slack events for testing
SAMPLE_SLACK_EVENTS = {
    'mention_event': {
        'type': 'app_mention',
        'text': '<@U123456789> How do I deploy the application?',
        'user': 'U987654321',
        'channel': 'C123456789',
        'ts': '1234567890.123456',
        'event_ts': '1234567890.123456'
    },
    'dm_event': {
        'type': 'message',
        'text': 'What is the API authentication method?',
        'user': 'U987654321',
        'channel': 'D123456789',
        'ts': '1234567890.123456'
    },
    'slash_command': {
        'token': 'verification-token',
        'team_id': 'T123456789',
        'team_domain': 'example',
        'channel_id': 'C123456789',
        'channel_name': 'general',
        'user_id': 'U987654321',
        'user_name': 'testuser',
        'command': '/kb',
        'text': 'search troubleshooting',
        'response_url': 'https://hooks.slack.com/commands/1234/5678',
        'trigger_id': '1234567890.123456.abcdef123456'
    }
}

# Sample search queries and expected results
SAMPLE_QUERIES = [
    {
        'query': 'API authentication',
        'expected_keywords': ['Bearer', 'token', 'Authorization', 'header'],
        'expected_sources': ['docs/api.md']
    },
    {
        'query': 'docker deployment',
        'expected_keywords': ['docker', 'build', 'run', 'container'],
        'expected_sources': ['docs/deployment.md']
    },
    {
        'query': 'bot not responding troubleshooting',
        'expected_keywords': ['bot', 'token', 'Socket Mode', 'logs'],
        'expected_sources': ['docs/troubleshooting.md']
    },
    {
        'query': 'kubernetes helm chart',
        'expected_keywords': ['helm', 'install', 'chart'],
        'expected_sources': ['docs/deployment.md']
    }
]

# Sample configuration for testing
SAMPLE_CONFIGS = {
    'minimal': {
        'slack_bot_token': 'xoxb-test-token',
        'slack_app_token': 'xapp-test-token',
        'slack_signing_secret': 'test-signing-secret'
    },
    'with_database': {
        'slack_bot_token': 'xoxb-test-token',
        'slack_app_token': 'xapp-test-token',
        'slack_signing_secret': 'test-signing-secret',
        'database_url': 'sqlite:///test.db',
        'enable_persistence': True
    },
    'with_vector_search': {
        'slack_bot_token': 'xoxb-test-token',
        'slack_app_token': 'xapp-test-token',
        'slack_signing_secret': 'test-signing-secret',
        'enable_vector_search': True,
        'vector_model': 'all-MiniLM-L6-v2',
        'similarity_threshold': 0.5
    },
    'production_like': {
        'slack_bot_token': 'xoxb-test-token',
        'slack_app_token': 'xapp-test-token',
        'slack_signing_secret': 'test-signing-secret',
        'database_url': 'postgresql://test:test@localhost:5432/test_db',
        'redis_url': 'redis://localhost:6379/1',
        'enable_vector_search': True,
        'enable_monitoring': True,
        'log_level': 'INFO'
    }
}


def create_sample_documents() -> List[Document]:
    """Create Document objects from sample data."""
    documents = []
    for doc_data in SAMPLE_DOCUMENTS:
        doc = Document(
            content=doc_data['content'],
            source=doc_data['source'],
            metadata=doc_data['metadata']
        )
        documents.append(doc)
    return documents


def get_sample_slack_event(event_type: str) -> Dict[str, Any]:
    """Get a sample Slack event by type."""
    return SAMPLE_SLACK_EVENTS.get(event_type, {})


def get_sample_query(query_type: str = None) -> Dict[str, Any]:
    """Get a sample query for testing."""
    if query_type:
        for query in SAMPLE_QUERIES:
            if query_type in query['query']:
                return query
    return SAMPLE_QUERIES[0]


def get_sample_config(config_type: str = 'minimal') -> Dict[str, Any]:
    """Get a sample configuration for testing."""
    return SAMPLE_CONFIGS.get(config_type, SAMPLE_CONFIGS['minimal'])


# Error scenarios for testing
ERROR_SCENARIOS = {
    'invalid_slack_token': {
        'config': {
            'slack_bot_token': 'invalid-token',
            'slack_app_token': 'xapp-test-token',
            'slack_signing_secret': 'test-signing-secret'
        },
        'expected_error': 'invalid_auth'
    },
    'missing_database': {
        'config': {
            'slack_bot_token': 'xoxb-test-token',
            'slack_app_token': 'xapp-test-token',
            'slack_signing_secret': 'test-signing-secret',
            'database_url': 'postgresql://nonexistent:host@localhost:5432/nonexistent'
        },
        'expected_error': 'connection_failed'
    },
    'corrupted_knowledge_base': {
        'knowledge_base_file': 'invalid_json_content',
        'expected_error': 'json_decode_error'
    }
}


# Performance test data
PERFORMANCE_TEST_DATA = {
    'large_document_count': 1000,
    'concurrent_users': 50,
    'queries_per_user': 10,
    'max_response_time_ms': 500,
    'memory_limit_mb': 512
}


# Mocking utilities
class MockSlackResponse:
    """Mock Slack API response for testing."""
    
    def __init__(self, success: bool = True, data: Dict[str, Any] = None):
        self.data = data or {}
        self['ok'] = success
    
    def __getitem__(self, key):
        return self.data.get(key)
    
    def get(self, key, default=None):
        return self.data.get(key, default)


class MockDocument:
    """Mock document for testing without full Document class."""
    
    def __init__(self, content: str, source: str, metadata: Dict[str, Any] = None):
        self.content = content
        self.source = source
        self.metadata = metadata or {}
        self.id = f"mock_{hash(content + source)}"
    
    def to_dict(self):
        return {
            'id': self.id,
            'content': self.content,
            'source': self.source,
            'metadata': self.metadata
        }