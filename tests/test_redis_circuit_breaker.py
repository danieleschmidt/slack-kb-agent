#!/usr/bin/env python3
"""
Tests for Redis circuit breaker integration.

This test suite verifies that circuit breaker properly protects Redis
cache operations in CacheManager against Redis connection failures and outages.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import time
from typing import List, Dict, Any
import json

# Optional imports with fallbacks for test environment
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    redis = None
    REDIS_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    np = None
    NUMPY_AVAILABLE = False

from src.slack_kb_agent.cache import CacheManager, CacheConfig
from src.slack_kb_agent.circuit_breaker import (
    CircuitBreaker, CircuitBreakerConfig, CircuitState, CircuitOpenError
)
from src.slack_kb_agent.constants import CircuitBreakerDefaults


class TestCacheManagerCircuitBreaker(unittest.TestCase):
    """Test circuit breaker integration with CacheManager."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.cache_config = CacheConfig(
            enabled=True,
            host="localhost",
            port=6379,
            db=0,
            socket_timeout=5.0,
            socket_connect_timeout=5.0
        )
        
        # Circuit breaker config for Redis
        self.circuit_config = CircuitBreakerConfig(
            failure_threshold=CircuitBreakerDefaults.REDIS_FAILURE_THRESHOLD,
            success_threshold=CircuitBreakerDefaults.REDIS_SUCCESS_THRESHOLD,
            timeout_seconds=CircuitBreakerDefaults.REDIS_TIMEOUT_SECONDS,
            half_open_max_requests=CircuitBreakerDefaults.REDIS_HALF_OPEN_MAX_REQUESTS,
            service_name="redis"
        )

    def test_circuit_breaker_config_from_constants(self):
        """Test that circuit breaker uses proper configuration from constants."""
        config = CircuitBreakerConfig(
            failure_threshold=CircuitBreakerDefaults.REDIS_FAILURE_THRESHOLD,
            success_threshold=CircuitBreakerDefaults.REDIS_SUCCESS_THRESHOLD,
            timeout_seconds=CircuitBreakerDefaults.REDIS_TIMEOUT_SECONDS,
            half_open_max_requests=CircuitBreakerDefaults.REDIS_HALF_OPEN_MAX_REQUESTS,
            service_name="redis"
        )
        
        self.assertEqual(config.failure_threshold, 3)
        self.assertEqual(config.success_threshold, 2)
        self.assertEqual(config.timeout_seconds, 30.0)
        self.assertEqual(config.half_open_max_requests, 2)
        self.assertEqual(config.service_name, "redis")

    @patch('src.slack_kb_agent.cache.redis')
    def test_cache_manager_circuit_breaker_protects_initialization(self, mock_redis_module):
        """Test that circuit breaker protects Redis connection initialization."""
        # Mock successful Redis connection
        mock_pool = Mock()
        mock_client = Mock()
        mock_redis_module.ConnectionPool.return_value = mock_pool
        mock_redis_module.Redis.return_value = mock_client
        mock_client.ping.return_value = True
        
        with patch('src.slack_kb_agent.cache.CacheManager._get_circuit_breaker') as mock_cb:
            circuit_breaker = CircuitBreaker(self.circuit_config)
            mock_cb.return_value = circuit_breaker
            
            cache_manager = CacheManager(self.cache_config)
            
            # Circuit breaker should be initialized and in closed state
            self.assertEqual(circuit_breaker.state, CircuitState.CLOSED)
            self.assertEqual(circuit_breaker.failure_count, 0)
            self.assertTrue(cache_manager.is_available())

    @patch('src.slack_kb_agent.cache.redis')
    def test_cache_manager_circuit_breaker_handles_connection_failures(self, mock_redis_module):
        """Test that circuit breaker opens after repeated Redis connection failures."""
        # Mock Redis connection failure
        mock_redis_module.ConnectionPool.side_effect = ConnectionError("Redis connection refused")
        
        with patch('src.slack_kb_agent.cache.CacheManager._get_circuit_breaker') as mock_cb:
            circuit_breaker = CircuitBreaker(self.circuit_config)
            mock_cb.return_value = circuit_breaker
            
            # Generate failures to trigger circuit breaker
            for i in range(CircuitBreakerDefaults.REDIS_FAILURE_THRESHOLD):
                cache_manager = CacheManager(self.cache_config)
                self.assertFalse(cache_manager.is_available())
            
            # Circuit should now be open
            self.assertEqual(circuit_breaker.state, CircuitState.OPEN)

    @patch('src.slack_kb_agent.cache.redis')
    def test_cache_manager_circuit_breaker_blocks_when_open(self, mock_redis_module):
        """Test that circuit breaker blocks Redis operations when open."""
        with patch('src.slack_kb_agent.cache.CacheManager._get_circuit_breaker') as mock_cb:
            circuit_breaker = CircuitBreaker(self.circuit_config)
            mock_cb.return_value = circuit_breaker
            
            # Force circuit to open state
            circuit_breaker.state = CircuitState.OPEN
            circuit_breaker.failure_count = CircuitBreakerDefaults.REDIS_FAILURE_THRESHOLD
            circuit_breaker.last_failure_time = time.time()
            
            cache_manager = CacheManager(self.cache_config)
            
            # Operations should fail gracefully when circuit is open
            result = cache_manager.get_embedding("test text", "test-model")
            self.assertIsNone(result)  # Should return None instead of raising exception

    def test_cache_manager_has_circuit_breaker_method(self):
        """Test that CacheManager has _get_circuit_breaker method."""
        # This test will fail until we implement the method
        cache_manager = CacheManager(self.cache_config)
        self.assertTrue(hasattr(cache_manager, '_get_circuit_breaker'))
        self.assertTrue(hasattr(cache_manager, 'circuit_breaker'))


class TestRedisOperationsCircuitBreaker(unittest.TestCase):
    """Test circuit breaker integration with specific Redis operations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.cache_config = CacheConfig(
            enabled=True,
            host="localhost",
            port=6379,
            db=0
        )
        
        self.circuit_config = CircuitBreakerConfig(
            failure_threshold=CircuitBreakerDefaults.REDIS_FAILURE_THRESHOLD,
            success_threshold=CircuitBreakerDefaults.REDIS_SUCCESS_THRESHOLD,
            timeout_seconds=CircuitBreakerDefaults.REDIS_TIMEOUT_SECONDS,
            half_open_max_requests=CircuitBreakerDefaults.REDIS_HALF_OPEN_MAX_REQUESTS,
            service_name="redis"
        )

    @patch('src.slack_kb_agent.cache.REDIS_AVAILABLE', True)
    @patch('src.slack_kb_agent.cache.redis')
    def test_get_embedding_circuit_breaker_protection(self, mock_redis_module):
        """Test that get_embedding operations are protected by circuit breaker."""
        # Mock successful Redis operations
        mock_client = Mock()
        mock_client.get.return_value = b'{"dtype": "float32", "shape": [384], "data": "base64data"}'
        
        mock_pool = Mock()
        mock_redis_module.ConnectionPool.return_value = mock_pool
        mock_redis_module.Redis.return_value = mock_client
        mock_client.ping.return_value = True
        
        with patch('src.slack_kb_agent.cache.CacheManager._get_circuit_breaker') as mock_cb:
            circuit_breaker = CircuitBreaker(self.circuit_config)
            mock_cb.return_value = circuit_breaker
            
            cache_manager = CacheManager(self.cache_config)
            
            # Should succeed and circuit breaker should remain closed
            result = cache_manager.get_embedding("test text", "test-model")
            
            self.assertEqual(circuit_breaker.state, CircuitState.CLOSED)
            self.assertEqual(circuit_breaker.failure_count, 0)

    @patch('src.slack_kb_agent.cache.REDIS_AVAILABLE', True)
    @patch('src.slack_kb_agent.cache.redis')
    def test_set_embedding_circuit_breaker_protection(self, mock_redis_module):
        """Test that set_embedding operations are protected by circuit breaker."""
        # Mock successful Redis operations
        mock_client = Mock()
        mock_client.setex.return_value = True
        
        mock_pool = Mock()
        mock_redis_module.ConnectionPool.return_value = mock_pool
        mock_redis_module.Redis.return_value = mock_client
        mock_client.ping.return_value = True
        
        with patch('src.slack_kb_agent.cache.CacheManager._get_circuit_breaker') as mock_cb:
            circuit_breaker = CircuitBreaker(self.circuit_config)
            mock_cb.return_value = circuit_breaker
            
            cache_manager = CacheManager(self.cache_config)
            
            if NUMPY_AVAILABLE:
                test_embedding = np.array([0.1, 0.2, 0.3])
                cache_manager.set_embedding("test text", "test-model", test_embedding)
            
            # Circuit breaker should remain closed after successful operation
            self.assertEqual(circuit_breaker.state, CircuitState.CLOSED)

    @patch('src.slack_kb_agent.cache.REDIS_AVAILABLE', True)
    @patch('src.slack_kb_agent.cache.redis')
    def test_redis_operations_circuit_breaker_failure_handling(self, mock_redis_module):
        """Test that circuit breaker handles Redis operation failures."""
        # Mock Redis operation failure
        mock_client = Mock()
        mock_client.get.side_effect = ConnectionError("Redis connection lost")
        
        mock_pool = Mock()
        mock_redis_module.ConnectionPool.return_value = mock_pool
        mock_redis_module.Redis.return_value = mock_client
        mock_client.ping.return_value = True
        
        with patch('src.slack_kb_agent.cache.CacheManager._get_circuit_breaker') as mock_cb:
            circuit_breaker = CircuitBreaker(self.circuit_config)
            mock_cb.return_value = circuit_breaker
            
            cache_manager = CacheManager(self.cache_config)
            
            # Generate failures to trigger circuit breaker
            for i in range(CircuitBreakerDefaults.REDIS_FAILURE_THRESHOLD):
                result = cache_manager.get_embedding("test text", "test-model")
                self.assertIsNone(result)  # Should return None on failure
            
            # Circuit should now be open
            self.assertEqual(circuit_breaker.state, CircuitState.OPEN)

    @patch('src.slack_kb_agent.cache.REDIS_AVAILABLE', True)
    @patch('src.slack_kb_agent.cache.redis')
    def test_query_expansion_circuit_breaker_protection(self, mock_redis_module):
        """Test that query expansion operations are protected by circuit breaker."""
        # Mock successful Redis operations
        mock_client = Mock()
        mock_client.get.return_value = b'["expanded", "query", "terms"]'
        
        mock_pool = Mock()
        mock_redis_module.ConnectionPool.return_value = mock_pool
        mock_redis_module.Redis.return_value = mock_client
        mock_client.ping.return_value = True
        
        with patch('src.slack_kb_agent.cache.CacheManager._get_circuit_breaker') as mock_cb:
            circuit_breaker = CircuitBreaker(self.circuit_config)
            mock_cb.return_value = circuit_breaker
            
            cache_manager = CacheManager(self.cache_config)
            
            # Test both get and set operations
            result = cache_manager.get_query_expansion("test query", "expansion_type")
            self.assertEqual(result, ["expanded", "query", "terms"])
            
            cache_manager.set_query_expansion("test query", "expansion_type", ["new", "terms"])
            
            # Circuit breaker should remain closed
            self.assertEqual(circuit_breaker.state, CircuitState.CLOSED)

    @patch('src.slack_kb_agent.cache.REDIS_AVAILABLE', True)
    @patch('src.slack_kb_agent.cache.redis')
    def test_search_results_circuit_breaker_protection(self, mock_redis_module):
        """Test that search results operations are protected by circuit breaker."""
        # Mock successful Redis operations
        mock_client = Mock()
        mock_client.get.return_value = b'[{"content": "test", "score": 0.9}]'
        
        mock_pool = Mock()
        mock_redis_module.ConnectionPool.return_value = mock_pool
        mock_redis_module.Redis.return_value = mock_client
        mock_client.ping.return_value = True
        
        with patch('src.slack_kb_agent.cache.CacheManager._get_circuit_breaker') as mock_cb:
            circuit_breaker = CircuitBreaker(self.circuit_config)
            mock_cb.return_value = circuit_breaker
            
            cache_manager = CacheManager(self.cache_config)
            
            # Test search results operations
            query_hash = cache_manager.generate_search_hash("test query", {"param": "value"})
            result = cache_manager.get_search_results(query_hash)
            self.assertEqual(result, [{"content": "test", "score": 0.9}])
            
            cache_manager.set_search_results(query_hash, [{"new": "result"}])
            
            # Circuit breaker should remain closed
            self.assertEqual(circuit_breaker.state, CircuitState.CLOSED)


class TestRedisCircuitBreakerIntegration(unittest.TestCase):
    """Test circuit breaker integration across Redis cache components."""
    
    def test_circuit_breaker_metrics_tracking(self):
        """Test that Redis circuit breaker metrics are properly tracked."""
        circuit_config = CircuitBreakerConfig(
            failure_threshold=3,
            success_threshold=2,
            timeout_seconds=30.0,
            half_open_max_requests=2,
            service_name="redis"
        )
        
        with patch('src.slack_kb_agent.cache.CacheManager._get_circuit_breaker') as mock_cb:
            circuit_breaker = CircuitBreaker(circuit_config)
            mock_cb.return_value = circuit_breaker
            
            # Test metrics collection
            metrics = circuit_breaker.get_metrics()
            expected_keys = [
                'service_name', 'state', 'total_requests', 'total_successes', 'total_failures',
                'failure_rate', 'current_failure_count', 'failure_threshold', 'circuit_opened_count',
                'last_failure_time', 'last_state_change_time', 'time_since_last_failure', 'half_open_requests'
            ]
            
            for key in expected_keys:
                self.assertIn(key, metrics)
            
            self.assertEqual(metrics['service_name'], "redis")
            self.assertEqual(metrics['state'], CircuitState.CLOSED.value)
            self.assertEqual(metrics['current_failure_count'], 0)

    @patch('src.slack_kb_agent.cache.REDIS_AVAILABLE', True)
    @patch('src.slack_kb_agent.cache.redis')
    def test_circuit_breaker_recovery_after_outage(self, mock_redis_module):
        """Test circuit breaker recovery after Redis outage."""
        cache_config = CacheConfig(enabled=True, host="localhost", port=6379)
        
        with patch('src.slack_kb_agent.cache.CacheManager._get_circuit_breaker') as mock_cb:
            circuit_breaker = CircuitBreaker(self.circuit_config)
            mock_cb.return_value = circuit_breaker
            
            # Force circuit to open state
            circuit_breaker.state = CircuitState.OPEN
            circuit_breaker.failure_count = CircuitBreakerDefaults.REDIS_FAILURE_THRESHOLD
            circuit_breaker.last_failure_time = time.time() - (CircuitBreakerDefaults.REDIS_TIMEOUT_SECONDS + 1)
            
            # Mock successful Redis connection for recovery
            mock_client = Mock()
            mock_client.ping.return_value = True
            mock_pool = Mock()
            mock_redis_module.ConnectionPool.return_value = mock_pool
            mock_redis_module.Redis.return_value = mock_client
            
            # Create cache manager - should transition to half-open
            cache_manager = CacheManager(cache_config)
            
            # Should be in half-open state after timeout
            # (The actual state transition happens during the first operation)
            self.assertTrue(cache_manager.is_available())

    def setUp(self):
        """Set up test fixtures for integration tests."""
        self.circuit_config = CircuitBreakerConfig(
            failure_threshold=CircuitBreakerDefaults.REDIS_FAILURE_THRESHOLD,
            success_threshold=CircuitBreakerDefaults.REDIS_SUCCESS_THRESHOLD,
            timeout_seconds=CircuitBreakerDefaults.REDIS_TIMEOUT_SECONDS,
            half_open_max_requests=CircuitBreakerDefaults.REDIS_HALF_OPEN_MAX_REQUESTS,
            service_name="redis"
        )


class TestRedisCircuitBreakerPerformance(unittest.TestCase):
    """Test performance characteristics of Redis circuit breaker integration."""
    
    def setUp(self):
        """Set up performance test fixtures."""
        self.cache_config = CacheConfig(
            enabled=True,
            host="localhost",
            port=6379,
            socket_timeout=1.0,  # Short timeout for tests
            socket_connect_timeout=1.0
        )
        self.circuit_config = CircuitBreakerConfig(
            failure_threshold=3,
            success_threshold=2,
            timeout_seconds=1.0,  # Short timeout for tests
            half_open_max_requests=1,
            service_name="redis"
        )
    
    @patch('src.slack_kb_agent.cache.REDIS_AVAILABLE', True)
    @patch('src.slack_kb_agent.cache.redis')
    def test_circuit_breaker_overhead_minimal(self, mock_redis_module):
        """Test that circuit breaker adds minimal overhead to Redis operations."""
        # Mock successful Redis operations
        mock_client = Mock()
        mock_client.ping.return_value = True
        mock_client.get.return_value = None
        
        mock_pool = Mock()
        mock_redis_module.ConnectionPool.return_value = mock_pool
        mock_redis_module.Redis.return_value = mock_client
        
        with patch('src.slack_kb_agent.cache.CacheManager._get_circuit_breaker') as mock_cb:
            circuit_breaker = CircuitBreaker(self.circuit_config)
            mock_cb.return_value = circuit_breaker
            
            cache_manager = CacheManager(self.cache_config)
            
            # Measure time with circuit breaker
            start_time = time.time()
            result = cache_manager.get_embedding("test text", "test-model")
            elapsed_time = time.time() - start_time
            
            # Circuit breaker overhead should be minimal (< 10ms)
            self.assertLess(elapsed_time, 0.01)

    @patch('src.slack_kb_agent.cache.REDIS_AVAILABLE', True)
    @patch('src.slack_kb_agent.cache.redis')
    def test_circuit_breaker_fast_failure_when_open(self, mock_redis_module):
        """Test that circuit breaker provides fast failure when open."""
        with patch('src.slack_kb_agent.cache.CacheManager._get_circuit_breaker') as mock_cb:
            circuit_breaker = CircuitBreaker(self.circuit_config)
            mock_cb.return_value = circuit_breaker
            
            # Force circuit to open state
            circuit_breaker.state = CircuitState.OPEN
            circuit_breaker.failure_count = 3
            circuit_breaker.last_failure_time = time.time()
            
            cache_manager = CacheManager(self.cache_config)
            
            # Measure fast failure time
            start_time = time.time()
            result = cache_manager.get_embedding("test text", "test-model")
            elapsed_time = time.time() - start_time
            
            # Should fail very quickly (< 1ms) and return None
            self.assertLess(elapsed_time, 0.001)
            self.assertIsNone(result)


if __name__ == '__main__':
    unittest.main()