"""Custom exceptions for slack-kb-agent."""


class SlackKBAgentError(Exception):
    """Base exception for all slack-kb-agent errors."""
    pass


class MonitoringError(SlackKBAgentError):
    """Base exception for monitoring system errors."""
    pass


class MetricsCollectionError(MonitoringError):
    """Error occurred during metrics collection."""
    pass


class HealthCheckError(MonitoringError):
    """Error occurred during health check."""
    pass


class SystemResourceError(HealthCheckError):
    """Error accessing system resources (memory, disk, etc.)."""
    pass


class KnowledgeBaseHealthError(HealthCheckError):
    """Error checking knowledge base health."""
    pass


class QueryProcessingError(SlackKBAgentError):
    """Base exception for query processing errors."""
    pass


class LLMError(SlackKBAgentError):
    """Base exception for LLM-related errors."""
    pass


class LLMProviderError(LLMError):
    """Error from LLM provider (OpenAI, Anthropic, etc.)."""
    pass


class LLMConfigurationError(LLMError):
    """Error in LLM configuration."""
    pass


class IngestionError(SlackKBAgentError):
    """Base exception for data ingestion errors."""
    pass


class FileIngestionError(IngestionError):
    """Error during file ingestion."""
    pass


class GitHubIngestionError(IngestionError):
    """Error during GitHub data ingestion."""
    pass


class WebCrawlingError(IngestionError):
    """Error during web crawling."""
    pass


class SlackIngestionError(IngestionError):
    """Error during Slack data ingestion."""
    pass


class SearchError(SlackKBAgentError):
    """Base exception for search-related errors."""
    pass


class VectorSearchError(SearchError):
    """Error in vector search operations."""
    pass


class AuthenticationError(SlackKBAgentError):
    """Authentication-related errors."""
    pass


class RateLimitError(SlackKBAgentError):
    """Rate limiting errors."""
    pass


class SlackBotError(SlackKBAgentError):
    """Slack bot operation errors."""
    pass