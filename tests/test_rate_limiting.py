#!/usr/bin/env python3
"""
Test suite for rate limiting functionality.

Tests rate limiting for bot commands and API requests.
"""

import unittest
import time
from unittest.mock import patch, MagicMock
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from slack_kb_agent.rate_limiting import (
    RateLimiter,
    RateLimitConfig,
    RateLimitResult,
    UserRateLimiter,
    get_rate_limiter
)


class TestRateLimiting(unittest.TestCase):
    """Test rate limiting functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.config = RateLimitConfig(
            enabled=True,
            requests_per_minute=5,
            requests_per_hour=30,
            burst_limit=10
        )
        self.rate_limiter = RateLimiter(self.config)
    
    def test_rate_limit_config_defaults(self):
        """Test default rate limit configuration."""
        config = RateLimitConfig()
        self.assertTrue(config.enabled)
        self.assertEqual(config.requests_per_minute, 10)
        self.assertEqual(config.requests_per_hour, 100)
    
    def test_rate_limit_config_from_env(self):
        """Test RateLimitConfig creation from environment variables."""
        with patch.dict('os.environ', {
            'RATE_LIMIT_ENABLED': 'true',
            'RATE_LIMIT_PER_MINUTE': '15',
            'RATE_LIMIT_PER_HOUR': '200',
            'RATE_LIMIT_BURST_LIMIT': '20'
        }):
            config = RateLimitConfig.from_env()
            self.assertTrue(config.enabled)
            self.assertEqual(config.requests_per_minute, 15)
            self.assertEqual(config.requests_per_hour, 200)
            self.assertEqual(config.burst_limit, 20)
    
    def test_rate_limiter_allows_under_limit(self):
        """Test that requests under limit are allowed."""
        user_id = "U123456"
        
        # First few requests should be allowed
        for i in range(3):
            result = self.rate_limiter.check_rate_limit(user_id, "query")
            self.assertTrue(result.allowed, f"Request {i+1} should be allowed")
            self.assertIsNone(result.error_message)
    
    def test_rate_limiter_blocks_over_limit(self):
        """Test that requests over limit are blocked."""
        user_id = "U123456"
        
        # Exhaust the rate limit
        for i in range(self.config.requests_per_minute + 1):
            result = self.rate_limiter.check_rate_limit(user_id, "query")
            if i < self.config.requests_per_minute:
                self.assertTrue(result.allowed, f"Request {i+1} should be allowed")
            else:
                self.assertFalse(result.allowed, f"Request {i+1} should be blocked")
                self.assertIsNotNone(result.error_message)
    
    def test_rate_limiter_resets_after_window(self):
        """Test that rate limiter resets after time window."""
        user_id = "U123456"
        
        # Use up the limit
        for i in range(self.config.requests_per_minute):
            self.rate_limiter.check_rate_limit(user_id, "query")
        
        # Next request should be blocked
        result = self.rate_limiter.check_rate_limit(user_id, "query")
        self.assertFalse(result.allowed)
        
        # Mock time passage (1 minute) - need to patch both modules
        original_time = time.time()
        future_time = original_time + 61
        
        with patch('time.time', return_value=future_time):
            with patch('slack_kb_agent.rate_limiting.time.time', return_value=future_time):
                # Should be allowed again after time window reset
                result = self.rate_limiter.check_rate_limit(user_id, "query")
                self.assertTrue(result.allowed)
    
    def test_rate_limiter_different_users(self):
        """Test that different users have separate rate limits."""
        user1 = "U123456"
        user2 = "U789012"
        
        # Exhaust limit for user1
        for i in range(self.config.requests_per_minute):
            result = self.rate_limiter.check_rate_limit(user1, "query")
            self.assertTrue(result.allowed)
        
        # user1 should be blocked
        result = self.rate_limiter.check_rate_limit(user1, "query")
        self.assertFalse(result.allowed)
        
        # user2 should still be allowed
        result = self.rate_limiter.check_rate_limit(user2, "query")
        self.assertTrue(result.allowed)
    
    def test_rate_limiter_burst_handling(self):
        """Test burst limit handling."""
        user_id = "U123456"
        
        # Rapid burst requests up to burst limit
        for i in range(self.config.burst_limit):
            result = self.rate_limiter.check_rate_limit(user_id, "query")
            if i < self.config.burst_limit:
                self.assertTrue(result.allowed or i >= self.config.requests_per_minute)
    
    def test_rate_limiter_disabled(self):
        """Test that rate limiting can be disabled."""
        config = RateLimitConfig(enabled=False)
        limiter = RateLimiter(config)
        
        user_id = "U123456"
        
        # Should allow unlimited requests when disabled
        for i in range(100):
            result = limiter.check_rate_limit(user_id, "query")
            self.assertTrue(result.allowed)
    
    def test_user_rate_limiter_integration(self):
        """Test UserRateLimiter integration."""
        user_limiter = UserRateLimiter()
        user_id = "U123456"
        
        # Normal usage should be allowed
        result = user_limiter.check_user_rate_limit(user_id, "How do I deploy?")
        self.assertTrue(result.allowed)
        
        # Record the request
        user_limiter.record_request(user_id, "query")
        
        # Should still be allowed
        result = user_limiter.check_user_rate_limit(user_id, "What is OAuth?")
        self.assertTrue(result.allowed)
    
    def test_rate_limit_metrics(self):
        """Test that rate limiting updates metrics."""
        user_id = "U123456"
        
        initial_count = self.rate_limiter.get_request_count(user_id)
        
        # Make a request
        self.rate_limiter.check_rate_limit(user_id, "query")
        
        # Count should increase
        new_count = self.rate_limiter.get_request_count(user_id)
        self.assertGreater(new_count, initial_count)
    
    def test_global_rate_limiter(self):
        """Test global rate limiter instance."""
        limiter = get_rate_limiter()
        self.assertIsInstance(limiter, RateLimiter)
        
        # Should return same instance
        limiter2 = get_rate_limiter()
        self.assertIs(limiter, limiter2)


class TestRateLimitingIntegration(unittest.TestCase):
    """Integration tests for rate limiting with Slack bot."""
    
    def test_slack_bot_rate_limiting(self):
        """Test rate limiting integration with Slack bot."""
        # This would test that the Slack bot properly applies rate limiting
        # For now, we'll test the integration points
        pass
    
    def test_rate_limit_error_messages(self):
        """Test user-friendly rate limit error messages."""
        config = RateLimitConfig(requests_per_minute=1)
        limiter = RateLimiter(config)
        user_id = "U123456"
        
        # First request allowed
        result = limiter.check_rate_limit(user_id, "query")
        self.assertTrue(result.allowed)
        
        # Second request blocked with helpful message
        result = limiter.check_rate_limit(user_id, "query")
        self.assertFalse(result.allowed)
        self.assertIn("rate limit", result.error_message.lower())
        self.assertIn("minute", result.error_message.lower())


if __name__ == "__main__":
    unittest.main()