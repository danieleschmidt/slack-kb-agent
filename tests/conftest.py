"""Pytest configuration and shared fixtures."""

import json
import os
import tempfile
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Provide a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def sample_document_data():
    """Sample document data for testing."""
    return {
        "content": "This is a test document about Python programming",
        "source": "test_source",
        "metadata": {"author": "test_author", "type": "documentation"},
    }


@pytest.fixture
def mock_env_vars():
    """Mock environment variables for testing."""
    env_vars = {
        "SLACK_BOT_TOKEN": "xoxb-test-token",
        "SLACK_SIGNING_SECRET": "test-signing-secret",
        "SLACK_APP_TOKEN": "xapp-test-token",
        "DATABASE_URL": "sqlite:///:memory:",
        "REDIS_URL": "redis://localhost:6379/0",
        "OPENAI_API_KEY": "sk-test-key",
        "MODEL_NAME": "gpt-3.5-turbo",
    }
    with patch.dict(os.environ, env_vars):
        yield env_vars


@pytest.fixture
def mock_slack_client():
    """Mock Slack client for testing."""
    mock_client = MagicMock()
    mock_client.chat_postMessage.return_value = {"ok": True, "ts": "1234567890.123456"}
    return mock_client


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for testing."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Test LLM response"
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client


@pytest.fixture
def sample_knowledge_base_data():
    """Sample knowledge base data for testing."""
    return [
        {
            "content": "Python is a high-level programming language",
            "source": "python_docs",
            "metadata": {"type": "documentation", "category": "programming"},
        },
        {
            "content": "Flask is a web framework for Python",
            "source": "flask_docs",
            "metadata": {"type": "documentation", "category": "web"},
        },
        {
            "content": "Database migrations in Django",
            "source": "django_docs",
            "metadata": {"type": "tutorial", "category": "database"},
        },
    ]


@pytest.fixture
def mock_sentence_transformer():
    """Mock sentence transformer for testing."""
    import numpy as np

    mock_model = MagicMock()
    # Return consistent embeddings for testing
    mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    mock_model.get_sentence_embedding_dimension.return_value = 3
    return mock_model


@pytest.fixture
def mock_faiss_index():
    """Mock FAISS index for testing."""
    import numpy as np

    mock_index = MagicMock()
    # Mock search results
    mock_index.search.return_value = (
        np.array([[0.9, 0.8]]),  # scores
        np.array([[0, 1]]),  # indices
    )
    mock_index.ntotal = 2
    return mock_index


@pytest.fixture
def mock_redis_client():
    """Mock Redis client for testing."""
    mock_client = MagicMock()
    mock_client.ping.return_value = True
    mock_client.get.return_value = None
    mock_client.set.return_value = True
    mock_client.exists.return_value = False
    return mock_client


@pytest.fixture
def sample_slack_event():
    """Sample Slack event for testing."""
    return {
        "type": "message",
        "channel": "C1234567890",
        "user": "U1234567890",
        "text": "What is Python?",
        "ts": "1234567890.123456",
        "event_ts": "1234567890.123456",
    }


@pytest.fixture
def database_url():
    """Provide database URL for testing."""
    return "sqlite:///:memory:"


@pytest.fixture
def test_config():
    """Test configuration for the application."""
    return {
        "knowledge_base": {
            "max_documents": 1000,
            "vector_search_enabled": True,
            "similarity_threshold": 0.5,
        },
        "llm": {
            "provider": "openai",
            "model": "gpt-3.5-turbo",
            "temperature": 0.1,
            "max_tokens": 500,
        },
        "slack": {
            "socket_mode": True,
            "rate_limit": {"per_minute": 60, "per_hour": 1000},
        },
        "monitoring": {"enabled": True, "port": 9090, "host": "localhost"},
    }


class TestDataFactory:
    """Factory for creating test data."""

    @staticmethod
    def create_document(content: str = "Test content", source: str = "test"):
        """Create a test document."""
        from slack_kb_agent.models import Document

        return Document(content=content, source=source, metadata={})

    @staticmethod
    def create_search_result(content: str = "Test result", score: float = 0.9):
        """Create a test search result."""
        from slack_kb_agent.models import SearchResult

        return SearchResult(
            content=content,
            source="test",
            score=score,
            metadata={},
        )


@pytest.fixture
def test_data_factory():
    """Provide test data factory."""
    return TestDataFactory


# Performance testing fixtures
@pytest.fixture
def benchmark_data():
    """Large dataset for performance testing."""
    return [
        {
            "content": f"Test document {i} with some content about topic {i % 10}",
            "source": f"source_{i % 5}",
            "metadata": {"index": i, "category": f"cat_{i % 3}"},
        }
        for i in range(1000)
    ]


# Integration test fixtures
@pytest.fixture
def integration_env():
    """Environment setup for integration tests."""
    return {
        "TESTING": True,
        "DATABASE_URL": "sqlite:///:memory:",
        "REDIS_ENABLED": False,
        "LLM_ENABLED": False,
    }