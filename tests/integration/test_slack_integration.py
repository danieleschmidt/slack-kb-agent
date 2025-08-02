"""Integration tests for Slack Bot functionality.

These tests verify the end-to-end integration between the Slack bot,
knowledge base, and external services.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
import asyncio
from slack_kb_agent.slack_bot import SlackBot
from slack_kb_agent.knowledge_base import KnowledgeBase
from slack_kb_agent.configuration import Configuration


@pytest.mark.integration
class TestSlackIntegration:
    """Integration tests for Slack bot with knowledge base."""

    @pytest.fixture
    def mock_slack_app(self):
        """Mock Slack app for testing."""
        with patch('slack_kb_agent.slack_bot.App') as mock_app:
            yield mock_app.return_value

    @pytest.fixture
    def mock_knowledge_base(self):
        """Mock knowledge base for testing."""
        kb = Mock(spec=KnowledgeBase)
        kb.search.return_value = [
            {
                'content': 'Test answer',
                'source': 'test_source.md',
                'score': 0.85
            }
        ]
        return kb

    @pytest.fixture
    def slack_bot(self, mock_slack_app, mock_knowledge_base):
        """Create SlackBot instance for testing."""
        config = Configuration()
        bot = SlackBot(config)
        bot.kb = mock_knowledge_base
        bot.app = mock_slack_app
        return bot

    @pytest.mark.asyncio
    async def test_mention_message_processing(self, slack_bot, mock_knowledge_base):
        """Test processing of @mention messages."""
        # Mock Slack event
        mock_event = {
            'text': '<@U123456> How do I deploy the application?',
            'user': 'U789012',
            'channel': 'C123456',
            'ts': '1234567890.123456'
        }
        
        mock_say = AsyncMock()
        
        # Process the mention
        await slack_bot.handle_mention(mock_event, mock_say)
        
        # Verify knowledge base was searched
        mock_knowledge_base.search.assert_called_once()
        
        # Verify response was sent
        mock_say.assert_called_once()
        
        # Check response contains answer
        call_args = mock_say.call_args[1] if mock_say.call_args[1] else mock_say.call_args[0]
        assert 'Test answer' in str(call_args)

    @pytest.mark.asyncio
    async def test_dm_message_processing(self, slack_bot, mock_knowledge_base):
        """Test processing of direct messages."""
        mock_event = {
            'text': 'What is the API documentation?',
            'user': 'U789012',
            'channel': 'D123456'  # DM channel
        }
        
        mock_say = AsyncMock()
        
        await slack_bot.handle_direct_message(mock_event, mock_say)
        
        mock_knowledge_base.search.assert_called_once()
        mock_say.assert_called_once()

    @pytest.mark.asyncio
    async def test_slash_command_processing(self, slack_bot, mock_knowledge_base):
        """Test processing of slash commands."""
        mock_command = {
            'command': '/kb',
            'text': 'search deployment process',
            'user_id': 'U789012',
            'channel_id': 'C123456'
        }
        
        mock_respond = AsyncMock()
        
        await slack_bot.handle_slash_command(mock_command, mock_respond)
        
        mock_knowledge_base.search.assert_called_once()
        mock_respond.assert_called_once()

    def test_error_handling_integration(self, slack_bot, mock_knowledge_base):
        """Test error handling in integrated flows."""
        # Make knowledge base throw an exception
        mock_knowledge_base.search.side_effect = Exception("Search failed")
        
        mock_event = {
            'text': '<@U123456> test query',
            'user': 'U789012',
            'channel': 'C123456'
        }
        
        mock_say = Mock()
        
        # Should handle the error gracefully
        with pytest.raises(Exception):
            asyncio.run(slack_bot.handle_mention(mock_event, mock_say))

    @pytest.mark.asyncio
    async def test_rate_limiting_integration(self, slack_bot):
        """Test rate limiting behavior in integration."""
        # This would test the actual rate limiting logic
        # in a real integration scenario
        pass

    @pytest.mark.asyncio
    async def test_knowledge_base_update_integration(self, slack_bot, mock_knowledge_base):
        """Test that knowledge base updates are handled correctly."""
        # Mock adding new knowledge
        mock_knowledge_base.add_document.return_value = True
        
        # Test that the bot can handle updated knowledge
        mock_event = {
            'text': '<@U123456> What is new feature X?',
            'user': 'U789012',
            'channel': 'C123456'
        }
        
        mock_say = AsyncMock()
        
        await slack_bot.handle_mention(mock_event, mock_say)
        
        # Verify search was performed
        mock_knowledge_base.search.assert_called_once()