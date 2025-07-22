#!/usr/bin/env python3
"""
Test to verify that cache exception handling improvements work correctly
and maintain graceful fallback behavior when Redis operations fail.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os
import tempfile
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from slack_kb_agent.cache import CacheManager, CacheConfig
except ImportError as e:
    print(f"Failed to import cache modules: {e}")
    print("This test requires the cache module to be available")
    sys.exit(1)

# Mock numpy if not available
try:
    import numpy as np
except ImportError:
    print("Warning: numpy not available, using mock array")
    class MockArray:
        def __init__(self, data):
            self.data = data
        def tobytes(self):
            return str(self.data).encode()
        @property
        def dtype(self):
            return "float64"
        @property 
        def shape(self):
            return (len(self.data),)
    
    class numpy:
        array = MockArray
        @staticmethod
        def frombuffer(buf, dtype):
            return MockArray([1.0, 2.0, 3.0])
    
    np = numpy


class TestCacheExceptionHandling(unittest.TestCase):
    """Test improved exception handling in cache operations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = CacheConfig(
            enabled=True,
            host="localhost",
            port=6379,
            password=None
        )
    
    def test_connection_error_handling(self):
        """Test that connection errors are handled gracefully."""
        with patch('slack_kb_agent.cache.redis') as mock_redis:
            # Mock Redis to raise connection error
            mock_redis.ConnectionPool.side_effect = ConnectionError("Redis connection failed")
            
            cache = CacheManager(self.config)
            
            # Should not raise exception, should disable caching
            self.assertFalse(cache.is_available())
    
    def test_timeout_error_handling(self):
        """Test that timeout errors are handled gracefully."""
        with patch('slack_kb_agent.cache.redis') as mock_redis:
            # Mock Redis to raise timeout error
            mock_redis.ConnectionPool.side_effect = TimeoutError("Redis timeout")
            
            cache = CacheManager(self.config)
            
            # Should not raise exception, should disable caching
            self.assertFalse(cache.is_available())
    
    def test_import_error_handling(self):
        """Test graceful handling when Redis is not available."""
        with patch('slack_kb_agent.cache.REDIS_AVAILABLE', False):
            cache = CacheManager(self.config)
            
            # Should not raise exception, should disable caching
            self.assertFalse(cache.is_available())
    
    def test_serialization_error_recovery(self):
        """Test that serialization errors are handled with cleanup."""
        with patch('slack_kb_agent.cache.redis') as mock_redis:
            mock_client = MagicMock()
            mock_redis.Redis.return_value = mock_client
            mock_client.ping.return_value = True
            
            cache = CacheManager(self.config)
            cache._is_connected = True
            cache._redis_client = mock_client
            
            # Mock deserialization to fail
            mock_client.get.return_value = b"invalid_json_data"
            
            # Should handle error gracefully and return None
            result = cache.get_embedding("test", "model")
            self.assertIsNone(result)
            
            # Should attempt cleanup of corrupted entry
            mock_client.delete.assert_called()
    
    def test_redis_operation_failures(self):
        """Test that Redis operation failures don't crash the application."""
        with patch('slack_kb_agent.cache.redis') as mock_redis:
            mock_client = MagicMock()
            mock_redis.Redis.return_value = mock_client
            mock_client.ping.return_value = True
            
            cache = CacheManager(self.config)
            cache._is_connected = True
            cache._redis_client = mock_client
            
            # Mock Redis operations to fail
            mock_client.get.side_effect = Exception("Redis server error")
            mock_client.setex.side_effect = Exception("Redis write error")
            
            # Operations should handle errors gracefully
            result = cache.get_embedding("test", "model")
            self.assertIsNone(result)
            
            # Set operations should not raise exceptions
            test_embedding = np.array([1.0, 2.0, 3.0])
            cache.set_embedding("test", "model", test_embedding)  # Should not crash
    
    def test_defensive_error_logging(self):
        """Test that errors are properly logged with context."""
        with patch('slack_kb_agent.cache.redis') as mock_redis, \
             patch('slack_kb_agent.cache.logger') as mock_logger:
            
            mock_redis.ConnectionPool.side_effect = ValueError("Invalid configuration")
            
            cache = CacheManager(self.config)
            
            # Should log configuration error with context
            mock_logger.warning.assert_called()
            logged_message = mock_logger.warning.call_args[0][0]
            self.assertIn("configuration error", logged_message)
    
    def test_error_categorization(self):
        """Test that different error types are properly categorized."""
        test_cases = [
            (ConnectionError("Connection failed"), "connection"),
            (TimeoutError("Operation timed out"), "timeout"), 
            (ValueError("Invalid value"), "configuration"),
            (TypeError("Type mismatch"), "configuration"),
            (MemoryError("Out of memory"), "resource"),
            (Exception("Unknown error"), "unexpected")
        ]
        
        for error, expected_category in test_cases:
            with patch('slack_kb_agent.cache.redis') as mock_redis, \
                 patch('slack_kb_agent.cache.logger') as mock_logger:
                
                mock_redis.ConnectionPool.side_effect = error
                
                cache = CacheManager(self.config)
                
                # Check that error was logged with appropriate category
                mock_logger.warning.assert_called()
                logged_message = mock_logger.warning.call_args[0][0]
                
                # For most specific errors, the category should be mentioned
                if expected_category != "unexpected":
                    self.assertIn(expected_category, logged_message.lower())


class TestCacheRecoveryBehavior(unittest.TestCase):
    """Test that cache failures don't affect main application functionality."""
    
    def test_graceful_degradation(self):
        """Test that cache failures result in graceful degradation."""
        config = CacheConfig(enabled=True, host="nonexistent-host")
        
        # Cache should initialize without crashing
        cache = CacheManager(config)
        
        # Should indicate unavailable
        self.assertFalse(cache.is_available())
        
        # Operations should work without crashing
        result = cache.get_embedding("test", "model")
        self.assertIsNone(result)
        
        # Set operations should not crash
        test_embedding = np.array([1.0, 2.0, 3.0])
        cache.set_embedding("test", "model", test_embedding)  # Should not raise
    
    def test_fallback_behavior_consistency(self):
        """Test that all cache operations have consistent fallback behavior."""
        with patch('slack_kb_agent.cache.redis') as mock_redis:
            mock_redis.Redis.side_effect = Exception("Redis unavailable")
            
            cache = CacheManager(CacheConfig(enabled=True, host="localhost"))
            
            # All get operations should return None on failure
            self.assertIsNone(cache.get_embedding("test", "model"))
            self.assertIsNone(cache.get_query_expansion("test", "synonyms"))
            self.assertIsNone(cache.get_search_results("test_hash"))
            
            # All set operations should complete without raising exceptions
            cache.set_embedding("test", "model", np.array([1.0]))
            cache.set_query_expansion("test", "synonyms", ["word"])
            cache.set_search_results("test_hash", [{"result": "data"}])
            
            # Cache should report as unavailable
            self.assertFalse(cache.is_available())


def run_tests():
    """Run all cache exception handling tests."""
    print("🧪 Testing Cache Exception Handling Improvements...")
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTest(loader.loadTestsFromTestCase(TestCacheExceptionHandling))
    suite.addTest(loader.loadTestsFromTestCase(TestCacheRecoveryBehavior))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return success status
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    if success:
        print("\n✅ All cache exception handling tests passed!")
    else:
        print("\n❌ Some tests failed. Cache exception handling needs improvement.")
    
    sys.exit(0 if success else 1)