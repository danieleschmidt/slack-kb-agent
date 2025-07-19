"""Tests for Slack bot with mocked dependencies."""

import pytest
from unittest.mock import MagicMock, patch
import os

from slack_kb_agent.knowledge_base import KnowledgeBase
from slack_kb_agent.models import Document
from slack_kb_agent.analytics import UsageAnalytics


class TestSlackBotMocked:
    """Test Slack bot functionality with mocked Slack dependencies."""

    @patch('slack_kb_agent.slack_bot.SLACK_DEPS_AVAILABLE', True)
    @patch('slack_kb_agent.slack_bot.App')
    @patch('slack_kb_agent.slack_bot.SocketModeHandler')
    def test_bot_initialization_with_mocks(self, mock_socket_handler, mock_app_class):
        """Test bot initializes correctly with mocked dependencies."""
        from slack_kb_agent.slack_bot import SlackBotServer
        
        mock_app = MagicMock()
        mock_app_class.return_value = mock_app
        
        kb = KnowledgeBase()
        analytics = UsageAnalytics()
        
        bot = SlackBotServer(
            knowledge_base=kb,
            slack_bot_token="xoxb-test-token",
            slack_app_token="xapp-test-token",
            signing_secret="test-secret-1234567890",
            analytics=analytics
        )
        
        assert bot.knowledge_base is kb
        assert bot.analytics is analytics
        assert bot.slack_bot_token == "xoxb-test-token"
        
        # Verify App was initialized
        mock_app_class.assert_called_once_with(
            token="xoxb-test-token",
            signing_secret="test-secret-1234567890"
        )

    @patch('slack_kb_agent.slack_bot.SLACK_DEPS_AVAILABLE', True)
    @patch('slack_kb_agent.slack_bot.App')
    def test_query_processing(self, mock_app_class):
        """Test query processing functionality."""
        from slack_kb_agent.slack_bot import SlackBotServer
        
        mock_app = MagicMock()
        mock_app_class.return_value = mock_app
        
        kb = KnowledgeBase(enable_vector_search=False)
        kb.add_document(Document(content="Python programming tutorial", source="docs"))
        kb.add_document(Document(content="JavaScript development guide", source="docs"))
        
        bot = SlackBotServer(
            knowledge_base=kb,
            slack_bot_token="xoxb-test",
            slack_app_token="xapp-test",
            signing_secret="test-secret-1234567890"
        )
        
        # Test query processing
        results = bot.process_query("Python")
        assert len(results) == 1
        assert "Python" in results[0].content

    @patch('slack_kb_agent.slack_bot.SLACK_DEPS_AVAILABLE', True)
    @patch('slack_kb_agent.slack_bot.App')
    def test_response_formatting(self, mock_app_class):
        """Test response formatting for Slack."""
        from slack_kb_agent.slack_bot import SlackBotServer
        
        mock_app = MagicMock()
        mock_app_class.return_value = mock_app
        
        kb = KnowledgeBase()
        bot = SlackBotServer(
            knowledge_base=kb,
            slack_bot_token="xoxb-test",
            slack_app_token="xapp-test",
            signing_secret="test-secret-1234567890"
        )
        
        # Test with results
        docs = [Document(content="Test content", source="docs")]
        response = bot.format_response(docs, "test query")
        
        assert isinstance(response, dict)
        assert "text" in response
        assert "test query" in response["text"]
        assert "Test content" in response["text"]
        
        # Test with no results
        response = bot.format_response([], "unknown query")
        assert "No results found" in response["text"]

    def test_token_validation(self):
        """Test that token validation works correctly."""
        from slack_kb_agent.slack_bot import SlackBotServer
        
        kb = KnowledgeBase()
        
        # Test invalid bot token
        with pytest.raises(ValueError, match="Invalid bot token format"):
            SlackBotServer(
                knowledge_base=kb,
                slack_bot_token="invalid-token",
                slack_app_token="xapp-test",
                signing_secret="test-secret-1234567890"
            )
        
        # Test invalid app token
        with pytest.raises(ValueError, match="Invalid app token format"):
            SlackBotServer(
                knowledge_base=kb,
                slack_bot_token="xoxb-test",
                slack_app_token="invalid-token",
                signing_secret="test-secret-1234567890"
            )
        
        # Test short signing secret
        with pytest.raises(ValueError, match="Signing secret too short"):
            SlackBotServer(
                knowledge_base=kb,
                slack_bot_token="xoxb-test",
                slack_app_token="xapp-test",
                signing_secret="short"
            )

    @patch.dict(os.environ, {
        'SLACK_BOT_TOKEN': 'xoxb-test-token',
        'SLACK_APP_TOKEN': 'xapp-test-token',
        'SLACK_SIGNING_SECRET': 'test-secret-1234567890'
    })
    @patch('slack_kb_agent.slack_bot.SLACK_DEPS_AVAILABLE', True)
    @patch('slack_kb_agent.slack_bot.App')
    def test_create_bot_from_env(self, mock_app_class):
        """Test creating bot from environment variables."""
        from slack_kb_agent.slack_bot import create_bot_from_env
        
        mock_app = MagicMock()
        mock_app_class.return_value = mock_app
        
        bot = create_bot_from_env()
        
        assert bot.slack_bot_token == "xoxb-test-token"
        assert bot.slack_app_token == "xapp-test-token"
        assert bot.signing_secret == "test-secret-1234567890"

    def test_create_bot_from_env_missing_vars(self):
        """Test error handling when environment variables are missing."""
        from slack_kb_agent.slack_bot import create_bot_from_env
        
        # Clear environment
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="Missing required environment variables"):
                create_bot_from_env()

    def test_availability_check(self):
        """Test availability check function."""
        from slack_kb_agent.slack_bot import is_slack_bot_available
        
        # Should return False in test environment (no real Slack dependencies)
        result = is_slack_bot_available()
        assert isinstance(result, bool)