"""Tests for custom exceptions."""

import pytest
from slack_kb_agent.exceptions import (
    SlackKBAgentError,
    MonitoringError,
    MetricsCollectionError,
    HealthCheckError,
    SystemResourceError,
    KnowledgeBaseHealthError,
    QueryProcessingError,
    LLMError,
    LLMProviderError,
    LLMConfigurationError,
    IngestionError,
    FileIngestionError,
    GitHubIngestionError,
    WebCrawlingError,
    SlackIngestionError,
    SearchError,
    VectorSearchError,
    AuthenticationError,
    RateLimitError,
    SlackBotError
)


def test_exception_hierarchy():
    """Test that exception hierarchy is correct."""
    # Test base exception
    assert issubclass(SlackKBAgentError, Exception)
    
    # Test monitoring exceptions
    assert issubclass(MonitoringError, SlackKBAgentError)
    assert issubclass(MetricsCollectionError, MonitoringError)
    assert issubclass(HealthCheckError, MonitoringError)
    assert issubclass(SystemResourceError, HealthCheckError)
    assert issubclass(KnowledgeBaseHealthError, HealthCheckError)
    
    # Test other domain exceptions
    assert issubclass(QueryProcessingError, SlackKBAgentError)
    assert issubclass(LLMError, SlackKBAgentError)
    assert issubclass(LLMProviderError, LLMError)
    assert issubclass(LLMConfigurationError, LLMError)
    assert issubclass(IngestionError, SlackKBAgentError)
    assert issubclass(FileIngestionError, IngestionError)
    assert issubclass(GitHubIngestionError, IngestionError)
    assert issubclass(WebCrawlingError, IngestionError)
    assert issubclass(SlackIngestionError, IngestionError)
    assert issubclass(SearchError, SlackKBAgentError)
    assert issubclass(VectorSearchError, SearchError)
    assert issubclass(AuthenticationError, SlackKBAgentError)
    assert issubclass(RateLimitError, SlackKBAgentError)
    assert issubclass(SlackBotError, SlackKBAgentError)


def test_exception_instantiation():
    """Test that exceptions can be instantiated with messages."""
    message = "Test error message"
    
    # Test base exception
    error = SlackKBAgentError(message)
    assert str(error) == message
    
    # Test monitoring exceptions
    error = MonitoringError(message)
    assert str(error) == message
    
    error = MetricsCollectionError(message)
    assert str(error) == message
    
    error = HealthCheckError(message)
    assert str(error) == message
    
    error = SystemResourceError(message)
    assert str(error) == message
    
    error = KnowledgeBaseHealthError(message)
    assert str(error) == message


def test_exception_catching():
    """Test that exceptions can be caught properly."""
    # Test catching specific exception
    with pytest.raises(MetricsCollectionError):
        raise MetricsCollectionError("Test metrics error")
    
    # Test catching by parent class
    with pytest.raises(MonitoringError):
        raise MetricsCollectionError("Test metrics error")
    
    # Test catching by base class
    with pytest.raises(SlackKBAgentError):
        raise MetricsCollectionError("Test metrics error")
    
    # Test catching by Exception
    with pytest.raises(Exception):
        raise MetricsCollectionError("Test metrics error")


def test_exception_chaining():
    """Test exception chaining functionality."""
    original_error = ValueError("Original error")
    
    # Test chaining with from clause
    with pytest.raises(MetricsCollectionError) as exc_info:
        try:
            raise original_error
        except ValueError as e:
            raise MetricsCollectionError("Metrics collection failed") from e
    
    assert exc_info.value.__cause__ is original_error


def test_llm_exceptions():
    """Test LLM-specific exceptions."""
    with pytest.raises(LLMProviderError):
        raise LLMProviderError("OpenAI API error")
    
    with pytest.raises(LLMError):
        raise LLMProviderError("OpenAI API error")
    
    with pytest.raises(LLMConfigurationError):
        raise LLMConfigurationError("Invalid API key")


def test_ingestion_exceptions():
    """Test ingestion-specific exceptions."""
    with pytest.raises(FileIngestionError):
        raise FileIngestionError("Cannot read file")
    
    with pytest.raises(IngestionError):
        raise FileIngestionError("Cannot read file")
    
    with pytest.raises(GitHubIngestionError):
        raise GitHubIngestionError("GitHub API error")
    
    with pytest.raises(WebCrawlingError):
        raise WebCrawlingError("Cannot fetch URL")
    
    with pytest.raises(SlackIngestionError):
        raise SlackIngestionError("Slack API error")


def test_search_exceptions():
    """Test search-specific exceptions."""
    with pytest.raises(VectorSearchError):
        raise VectorSearchError("FAISS index error")
    
    with pytest.raises(SearchError):
        raise VectorSearchError("FAISS index error")


def test_auth_exceptions():
    """Test authentication and rate limiting exceptions."""
    with pytest.raises(AuthenticationError):
        raise AuthenticationError("Invalid credentials")
    
    with pytest.raises(RateLimitError):
        raise RateLimitError("Rate limit exceeded")


def test_slack_bot_exceptions():
    """Test Slack bot exceptions."""
    with pytest.raises(SlackBotError):
        raise SlackBotError("Slack connection failed")