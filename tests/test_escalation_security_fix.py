"""Test urllib to requests security fix integration."""

import unittest
from unittest.mock import Mock, patch
from src.slack_kb_agent.escalation import SlackNotifier
from src.slack_kb_agent.smart_routing import TeamMember


class TestEscalationSecurityFix(unittest.TestCase):
    """Test the security fix replacing urllib with requests."""

    def setUp(self):
        """Set up test fixtures."""
        self.token = "test-token"
        self.notifier = SlackNotifier(self.token)
        self.test_member = TeamMember(id="U12345", name="Test User", expertise=["test"])

    @patch('requests.post')
    def test_uses_requests_with_ssl_verification(self, mock_post):
        """Test that requests is used with proper SSL verification."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response
        
        result = self.notifier.notify("test-user", "test message")
        
        # Verify requests.post was called
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        
        # Verify SSL verification is enabled
        self.assertTrue(kwargs.get('verify', True))
        
        # Verify proper timeout
        self.assertEqual(kwargs['timeout'], 5)
        
        # Verify proper headers including User-Agent
        headers = kwargs['headers']
        self.assertIn('User-Agent', headers)
        self.assertEqual(headers['User-Agent'], 'slack-kb-agent/1.0')
        
        self.assertTrue(result)

    @patch('requests.post')
    def test_network_error_handling(self, mock_post):
        """Test proper handling of network errors."""
        from requests.exceptions import ConnectionError
        
        mock_post.side_effect = ConnectionError("Connection failed")
        
        result = self.notifier.notify("test-user", "test message")
        
        # Should return False on network error
        self.assertFalse(result)

    @patch('requests.post')
    def test_notify_all_functionality(self, mock_post):
        """Test that notify_all works with the new implementation."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response
        
        members = [
            TeamMember(id="U1", name="User 1", expertise=["test"]),
            TeamMember(id="U2", name="User 2", expertise=["test"])
        ]
        
        # This should call notify for each member
        self.notifier.notify_all(members, "test message")
        
        # Should have called requests.post twice
        self.assertEqual(mock_post.call_count, 2)

    def test_custom_sender_still_works(self):
        """Test that custom sender functionality is preserved."""
        custom_sender = Mock()
        notifier = SlackNotifier(self.token, sender=custom_sender)
        
        notifier.notify("test-user", "test message")
        
        # Custom sender should be called instead of default
        custom_sender.assert_called_once_with("test-user", "test message")


if __name__ == '__main__':
    unittest.main()