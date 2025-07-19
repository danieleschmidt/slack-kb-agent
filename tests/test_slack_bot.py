"""Tests for Slack bot server functionality."""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
import json

from slack_kb_agent.models import Document
from slack_kb_agent.knowledge_base import KnowledgeBase


class TestSlackBotServer:
    """Test Slack bot server integration."""

    def test_bot_server_initialization(self):
        """Test that bot server initializes with proper configuration."""
        from slack_kb_agent.slack_bot import SlackBotServer
        
        kb = KnowledgeBase()
        bot = SlackBotServer(
            knowledge_base=kb,
            slack_bot_token="xoxb-test-token",
            slack_app_token="xapp-test-token",
            signing_secret="test-secret"
        )
        
        assert bot.knowledge_base is kb
        assert bot.slack_bot_token == "xoxb-test-token"
        assert bot.slack_app_token == "xapp-test-token"
        assert bot.signing_secret == "test-secret"

    @patch('slack_kb_agent.slack_bot.App')
    def test_bot_handles_app_mentions(self, mock_app_class):
        """Test bot responds to @mentions in channels."""
        from slack_kb_agent.slack_bot import SlackBotServer
        
        # Setup mocks
        mock_app = MagicMock()
        mock_app_class.return_value = mock_app
        
        kb = KnowledgeBase()
        kb.add_document(Document(content="Python programming guide", source="docs"))
        
        bot = SlackBotServer(
            knowledge_base=kb,
            slack_bot_token="xoxb-test",
            slack_app_token="xapp-test",
            signing_secret="secret"
        )
        
        # Verify app mention handler was registered
        mock_app.event.assert_called()
        call_args = mock_app.event.call_args_list
        event_types = [call[0][0] for call in call_args]
        assert "app_mention" in event_types

    @patch('slack_kb_agent.slack_bot.App')
    def test_bot_handles_direct_messages(self, mock_app_class):
        """Test bot responds to direct messages."""
        from slack_kb_agent.slack_bot import SlackBotServer
        
        mock_app = MagicMock()
        mock_app_class.return_value = mock_app
        
        kb = KnowledgeBase()
        bot = SlackBotServer(
            knowledge_base=kb,
            slack_bot_token="xoxb-test",
            slack_app_token="xapp-test", 
            signing_secret="secret"
        )
        
        # Verify message handler was registered
        mock_app.event.assert_called()
        call_args = mock_app.event.call_args_list
        event_types = [call[0][0] for call in call_args]
        assert "message" in event_types

    @patch('slack_kb_agent.slack_bot.App')
    def test_bot_handles_slash_commands(self, mock_app_class):
        """Test bot responds to slash commands."""
        from slack_kb_agent.slack_bot import SlackBotServer
        
        mock_app = MagicMock()
        mock_app_class.return_value = mock_app
        
        kb = KnowledgeBase()
        bot = SlackBotServer(
            knowledge_base=kb,
            slack_bot_token="xoxb-test",
            slack_app_token="xapp-test",
            signing_secret="secret"
        )
        
        # Verify slash command handlers were registered
        mock_app.command.assert_called()
        call_args = mock_app.command.call_args_list
        commands = [call[0][0] for call in call_args]
        assert "/kb" in commands

    def test_query_processing_with_semantic_search(self):
        """Test bot processes queries using semantic search."""
        from slack_kb_agent.slack_bot import SlackBotServer
        
        kb = KnowledgeBase(enable_vector_search=False)  # Use keyword fallback for testing
        kb.add_document(Document(content="Python programming tutorial", source="docs"))
        kb.add_document(Document(content="JavaScript development guide", source="docs"))
        
        bot = SlackBotServer(
            knowledge_base=kb,
            slack_bot_token="xoxb-test",
            slack_app_token="xapp-test",
            signing_secret="secret"
        )
        
        # Test query processing
        results = bot.process_query("Python coding help")
        assert len(results) >= 1
        assert any("Python" in doc.content for doc in results)

    def test_response_formatting(self):
        """Test bot formats responses properly for Slack."""
        from slack_kb_agent.slack_bot import SlackBotServer
        
        kb = KnowledgeBase()
        bot = SlackBotServer(
            knowledge_base=kb,
            slack_bot_token="xoxb-test",
            slack_app_token="xapp-test",
            signing_secret="secret"
        )
        
        # Test response formatting
        docs = [
            Document(content="Python programming guide", source="docs"),
            Document(content="Python best practices", source="wiki")
        ]
        
        response = bot.format_response(docs, "Python help")
        
        assert isinstance(response, dict)
        assert "text" in response or "blocks" in response
        assert len(response) > 0

    def test_empty_query_handling(self):
        """Test bot handles empty queries gracefully."""
        from slack_kb_agent.slack_bot import SlackBotServer
        
        kb = KnowledgeBase()
        bot = SlackBotServer(
            knowledge_base=kb,
            slack_bot_token="xoxb-test",
            slack_app_token="xapp-test",
            signing_secret="secret"
        )
        
        # Empty query should return helpful message
        results = bot.process_query("")
        assert results == []
        
        response = bot.format_response([], "")
        assert "help" in response.get("text", "").lower() or "try" in response.get("text", "").lower()

    def test_no_results_handling(self):
        """Test bot handles queries with no results."""
        from slack_kb_agent.slack_bot import SlackBotServer
        
        kb = KnowledgeBase()
        kb.add_document(Document(content="Python programming", source="docs"))
        
        bot = SlackBotServer(
            knowledge_base=kb,
            slack_bot_token="xoxb-test",
            slack_app_token="xapp-test",
            signing_secret="secret"
        )
        
        # Query with no matches
        results = bot.process_query("JavaScript frameworks")
        assert results == []
        
        response = bot.format_response([], "JavaScript frameworks")
        assert "no results" in response.get("text", "").lower() or "not found" in response.get("text", "").lower()

    @patch('slack_kb_agent.slack_bot.SocketModeHandler')
    def test_socket_mode_connection(self, mock_socket_handler):
        """Test Socket Mode connection for real-time events."""
        from slack_kb_agent.slack_bot import SlackBotServer
        
        kb = KnowledgeBase()
        bot = SlackBotServer(
            knowledge_base=kb,
            slack_bot_token="xoxb-test",
            slack_app_token="xapp-test",
            signing_secret="secret"
        )
        
        # Test starting the bot
        bot.start()
        
        # Verify Socket Mode handler was created and started
        mock_socket_handler.assert_called_once()
        handler_instance = mock_socket_handler.return_value
        handler_instance.start.assert_called_once()

    def test_error_handling_in_query_processing(self):
        """Test bot handles errors gracefully during query processing."""
        from slack_kb_agent.slack_bot import SlackBotServer
        
        kb = KnowledgeBase()
        bot = SlackBotServer(
            knowledge_base=kb,
            slack_bot_token="xoxb-test",
            slack_app_token="xapp-test",
            signing_secret="secret"
        )
        
        # Mock a knowledge base error
        with patch.object(kb, 'search_semantic', side_effect=Exception("Test error")):
            results = bot.process_query("test query")
            # Should fallback gracefully
            assert isinstance(results, list)

    def test_analytics_integration(self):
        """Test bot integrates with usage analytics."""
        from slack_kb_agent.slack_bot import SlackBotServer
        from slack_kb_agent.analytics import UsageAnalytics
        
        kb = KnowledgeBase()
        analytics = UsageAnalytics()
        
        bot = SlackBotServer(
            knowledge_base=kb,
            slack_bot_token="xoxb-test",
            slack_app_token="xapp-test",
            signing_secret="secret",
            analytics=analytics
        )
        
        # Test that analytics are recorded
        initial_count = analytics.total_queries
        bot.process_query("test query", user_id="U123", channel_id="C456")
        
        assert analytics.total_queries > initial_count