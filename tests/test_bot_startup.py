#!/usr/bin/env python3
"""
Test suite for bot startup error handling.

Tests the bot.py main function error handling without sys.exit().
"""

import sys
import unittest
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import bot.py as a module
sys.path.insert(0, str(Path(__file__).parent.parent))
import bot


class TestBotStartupErrorHandling(unittest.TestCase):
    """Test bot startup error handling without sys.exit()."""
    
    def setUp(self):
        """Set up test environment."""
        # Store original environment
        self.original_env = dict(os.environ)
    
    def tearDown(self):
        """Restore original environment."""
        os.environ.clear()
        os.environ.update(self.original_env)
    
    def test_missing_slack_dependencies_raises_exception(self):
        """Test that missing Slack dependencies raises exception instead of sys.exit()."""
        with patch('bot.is_slack_bot_available', return_value=False):
            with self.assertRaises(SystemExit) as context:
                bot.main()
            
            # Should raise SystemExit with code 1, not call sys.exit() directly
            self.assertEqual(context.exception.code, 1)
    
    def test_missing_environment_variables_raises_exception(self):
        """Test that missing environment variables raises exception instead of sys.exit()."""
        # Clear required environment variables
        for var in ["SLACK_BOT_TOKEN", "SLACK_APP_TOKEN", "SLACK_SIGNING_SECRET"]:
            os.environ.pop(var, None)
        
        with patch('bot.is_slack_bot_available', return_value=True):
            with self.assertRaises(SystemExit) as context:
                bot.main()
            
            # Should raise SystemExit with code 1
            self.assertEqual(context.exception.code, 1)
    
    def test_bot_startup_exception_raises_exception(self):
        """Test that bot startup exceptions raise exception instead of sys.exit()."""
        # Set up valid environment
        os.environ.update({
            "SLACK_BOT_TOKEN": "xoxb-test",
            "SLACK_APP_TOKEN": "xapp-test", 
            "SLACK_SIGNING_SECRET": "test-secret"
        })
        
        with patch('bot.is_slack_bot_available', return_value=True):
            with patch('bot.check_environment', return_value=True):
                with patch('bot.load_knowledge_base', side_effect=Exception("Test error")):
                    with self.assertRaises(SystemExit) as context:
                        bot.main()
                    
                    # Should raise SystemExit with code 1
                    self.assertEqual(context.exception.code, 1)
    
    def test_keyboard_interrupt_handled_gracefully(self):
        """Test that KeyboardInterrupt is handled without sys.exit()."""
        # Set up valid environment
        os.environ.update({
            "SLACK_BOT_TOKEN": "xoxb-test",
            "SLACK_APP_TOKEN": "xapp-test", 
            "SLACK_SIGNING_SECRET": "test-secret"
        })
        
        with patch('bot.is_slack_bot_available', return_value=True):
            with patch('bot.check_environment', return_value=True):
                with patch('bot.load_knowledge_base') as mock_kb:
                    with patch('bot.load_analytics') as mock_analytics:
                        with patch('bot.create_bot_from_env') as mock_bot:
                            mock_bot_instance = MagicMock()
                            mock_bot_instance.start.side_effect = KeyboardInterrupt()
                            mock_bot.return_value = mock_bot_instance
                            
                            # KeyboardInterrupt should be handled gracefully without raising SystemExit
                            try:
                                bot.main()
                            except SystemExit:
                                self.fail("KeyboardInterrupt should not cause SystemExit")
    
    def test_successful_startup_returns_none(self):
        """Test that successful bot startup returns None instead of calling sys.exit()."""
        # Set up valid environment
        os.environ.update({
            "SLACK_BOT_TOKEN": "xoxb-test",
            "SLACK_APP_TOKEN": "xapp-test", 
            "SLACK_SIGNING_SECRET": "test-secret"
        })
        
        with patch('bot.is_slack_bot_available', return_value=True):
            with patch('bot.check_environment', return_value=True):
                with patch('bot.load_knowledge_base') as mock_kb:
                    with patch('bot.load_analytics') as mock_analytics:
                        with patch('bot.create_bot_from_env') as mock_bot:
                            with patch('bot.setup_monitoring', return_value={"status": "disabled"}):
                                mock_bot_instance = MagicMock()
                                mock_bot_instance.start = MagicMock()  # Don't actually start
                                mock_bot.return_value = mock_bot_instance
                                
                                # Mock the monitoring and KB setup
                                mock_kb_obj = MagicMock()
                                mock_kb_obj.kb.documents = []
                                mock_kb_obj.kb.enable_vector_search = False
                                mock_kb.return_value = mock_kb_obj
                                
                                mock_analytics_obj = MagicMock()
                                mock_analytics_obj.total_queries = 0
                                mock_analytics.return_value = mock_analytics_obj
                                
                                # Should return None on success, not call sys.exit()
                                result = bot.main()
                                self.assertIsNone(result)


class TestBotUtilityFunctions(unittest.TestCase):
    """Test bot utility functions for proper error return patterns."""
    
    def test_check_environment_returns_bool(self):
        """Test that check_environment returns boolean instead of calling sys.exit()."""
        # Clear environment
        original_env = dict(os.environ)
        for var in ["SLACK_BOT_TOKEN", "SLACK_APP_TOKEN", "SLACK_SIGNING_SECRET"]:
            os.environ.pop(var, None)
        
        try:
            result = bot.check_environment()
            self.assertFalse(result, "check_environment should return False for missing vars")
            
            # Set valid environment
            os.environ.update({
                "SLACK_BOT_TOKEN": "xoxb-test",
                "SLACK_APP_TOKEN": "xapp-test", 
                "SLACK_SIGNING_SECRET": "test-secret"
            })
            
            result = bot.check_environment()
            self.assertTrue(result, "check_environment should return True for valid vars")
        finally:
            # Restore environment
            os.environ.clear()
            os.environ.update(original_env)


if __name__ == "__main__":
    unittest.main()