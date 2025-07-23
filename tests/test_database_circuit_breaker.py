#!/usr/bin/env python3
"""
Tests for database circuit breaker integration.

This test suite verifies that circuit breaker properly protects database
operations in DatabaseManager and DatabaseRepository classes against
PostgreSQL connection failures and outages.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import time
from typing import List

from src.slack_kb_agent.database import DatabaseManager, DatabaseRepository
from src.slack_kb_agent.circuit_breaker import (
    CircuitBreaker, CircuitBreakerConfig, CircuitState, CircuitOpenError
)
from src.slack_kb_agent.constants import CircuitBreakerDefaults
from src.slack_kb_agent.models import Document


class TestDatabaseManagerCircuitBreaker(unittest.TestCase):
    """Test circuit breaker integration with DatabaseManager."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_db_url = "postgresql://test:test@localhost/test_db"
        
        # Circuit breaker config for database
        self.circuit_config = CircuitBreakerConfig(
            failure_threshold=CircuitBreakerDefaults.DATABASE_FAILURE_THRESHOLD,
            success_threshold=CircuitBreakerDefaults.DATABASE_SUCCESS_THRESHOLD,
            timeout_seconds=CircuitBreakerDefaults.DATABASE_TIMEOUT_SECONDS,
            half_open_max_requests=CircuitBreakerDefaults.DATABASE_HALF_OPEN_MAX_REQUESTS,
            service_name="database"
        )

    def test_circuit_breaker_config_from_constants(self):
        """Test that circuit breaker uses proper configuration from constants."""
        config = CircuitBreakerConfig(
            failure_threshold=CircuitBreakerDefaults.DATABASE_FAILURE_THRESHOLD,
            success_threshold=CircuitBreakerDefaults.DATABASE_SUCCESS_THRESHOLD,
            timeout_seconds=CircuitBreakerDefaults.DATABASE_TIMEOUT_SECONDS,
            half_open_max_requests=CircuitBreakerDefaults.DATABASE_HALF_OPEN_MAX_REQUESTS,
            service_name="database"
        )
        
        self.assertEqual(config.failure_threshold, 5)
        self.assertEqual(config.success_threshold, 2)
        self.assertEqual(config.timeout_seconds, 60.0)
        self.assertEqual(config.half_open_max_requests, 3)
        self.assertEqual(config.service_name, "database")

    @patch('src.slack_kb_agent.database.create_engine')
    def test_database_manager_circuit_breaker_protects_connection(self, mock_create_engine):
        """Test that successful database connection establishment passes through circuit breaker."""
        # Mock successful engine creation
        mock_engine = Mock()
        mock_create_engine.return_value = mock_engine
        
        with patch('src.slack_kb_agent.database.DatabaseManager._get_circuit_breaker') as mock_cb:
            circuit_breaker = CircuitBreaker(self.circuit_config)
            mock_cb.return_value = circuit_breaker
            
            db_manager = DatabaseManager(self.test_db_url)
            
            # Circuit breaker should be initialized and in closed state
            self.assertEqual(circuit_breaker.state, CircuitState.CLOSED)
            self.assertEqual(circuit_breaker.failure_count, 0)
            mock_create_engine.assert_called_once()

    @patch('src.slack_kb_agent.database.create_engine')
    def test_database_manager_circuit_breaker_handles_connection_failures(self, mock_create_engine):
        """Test that circuit breaker opens after repeated database connection failures."""
        # Mock database connection failure
        mock_create_engine.side_effect = Exception("Connection refused")
        
        with patch('src.slack_kb_agent.database.DatabaseManager._get_circuit_breaker') as mock_cb:
            circuit_breaker = CircuitBreaker(self.circuit_config)
            mock_cb.return_value = circuit_breaker
            
            # Generate failures to trigger circuit breaker
            for i in range(CircuitBreakerDefaults.DATABASE_FAILURE_THRESHOLD):
                with self.assertRaises(Exception):
                    DatabaseManager(self.test_db_url)
            
            # Circuit should now be open
            self.assertEqual(circuit_breaker.state, CircuitState.OPEN)

    @patch('src.slack_kb_agent.database.create_engine')
    def test_database_manager_circuit_breaker_blocks_when_open(self, mock_create_engine):
        """Test that circuit breaker blocks requests when open."""
        with patch('src.slack_kb_agent.database.DatabaseManager._get_circuit_breaker') as mock_cb:
            circuit_breaker = CircuitBreaker(self.circuit_config)
            mock_cb.return_value = circuit_breaker
            
            # Force circuit to open state
            circuit_breaker.state = CircuitState.OPEN
            circuit_breaker.failure_count = CircuitBreakerDefaults.DATABASE_FAILURE_THRESHOLD
            circuit_breaker.last_failure_time = time.time()
            
            # Next call should be rejected by circuit breaker
            with self.assertRaises(CircuitOpenError):
                DatabaseManager(self.test_db_url)
            
            # Should not call the actual database connection
            mock_create_engine.assert_not_called()

    def test_database_manager_has_circuit_breaker_method(self):
        """Test that DatabaseManager has _get_circuit_breaker method."""
        # This test will fail until we implement the method
        with patch('src.slack_kb_agent.database.create_engine'):
            db_manager = DatabaseManager(self.test_db_url)
            self.assertTrue(hasattr(db_manager, '_get_circuit_breaker'))
            self.assertTrue(hasattr(db_manager, 'circuit_breaker'))


class TestDatabaseRepositoryCircuitBreaker(unittest.TestCase):
    """Test circuit breaker integration with DatabaseRepository CRUD operations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_session = Mock()
        self.test_document = Document(
            content="Test document content",
            source="test_source",
            metadata={"id": "test-1"}
        )
        
        # Circuit breaker config
        self.circuit_config = CircuitBreakerConfig(
            failure_threshold=CircuitBreakerDefaults.DATABASE_FAILURE_THRESHOLD,
            success_threshold=CircuitBreakerDefaults.DATABASE_SUCCESS_THRESHOLD,
            timeout_seconds=CircuitBreakerDefaults.DATABASE_TIMEOUT_SECONDS,
            half_open_max_requests=CircuitBreakerDefaults.DATABASE_HALF_OPEN_MAX_REQUESTS,
            service_name="database"
        )

    @patch('src.slack_kb_agent.database.DatabaseManager')
    def test_repository_circuit_breaker_protects_create_operations(self, mock_db_manager):
        """Test that circuit breaker protects document creation operations."""
        # Mock successful database operations
        mock_db_manager.return_value.get_session.return_value.__enter__.return_value = self.mock_session
        
        with patch('src.slack_kb_agent.database.DatabaseRepository._get_circuit_breaker') as mock_cb:
            circuit_breaker = CircuitBreaker(self.circuit_config)
            mock_cb.return_value = circuit_breaker
            
            repository = DatabaseRepository(mock_db_manager.return_value)
            
            # Should succeed and circuit breaker should remain closed
            result = repository.create_document(self.test_document)
            
            self.assertEqual(circuit_breaker.state, CircuitState.CLOSED)
            self.assertEqual(circuit_breaker.failure_count, 0)

    @patch('src.slack_kb_agent.database.DatabaseManager')
    def test_repository_circuit_breaker_handles_session_failures(self, mock_db_manager):
        """Test that circuit breaker handles database session failures."""
        # Mock database session failure
        mock_db_manager.return_value.get_session.side_effect = Exception("Database connection lost")
        
        with patch('src.slack_kb_agent.database.DatabaseRepository._get_circuit_breaker') as mock_cb:
            circuit_breaker = CircuitBreaker(self.circuit_config)
            mock_cb.return_value = circuit_breaker
            
            repository = DatabaseRepository(mock_db_manager.return_value)
            
            # Generate failures to trigger circuit breaker
            for i in range(CircuitBreakerDefaults.DATABASE_FAILURE_THRESHOLD):
                with self.assertRaises(Exception):
                    repository.create_document(self.test_document)
            
            # Circuit should now be open
            self.assertEqual(circuit_breaker.state, CircuitState.OPEN)

    @patch('src.slack_kb_agent.database.DatabaseManager')
    def test_repository_circuit_breaker_protects_read_operations(self, mock_db_manager):
        """Test that circuit breaker protects read operations."""
        # Mock successful read operation
        mock_query_result = Mock()
        mock_query_result.first.return_value = Mock()
        self.mock_session.query.return_value.filter.return_value = mock_query_result
        mock_db_manager.return_value.get_session.return_value.__enter__.return_value = self.mock_session
        
        with patch('src.slack_kb_agent.database.DatabaseRepository._get_circuit_breaker') as mock_cb:
            circuit_breaker = CircuitBreaker(self.circuit_config)
            mock_cb.return_value = circuit_breaker
            
            repository = DatabaseRepository(mock_db_manager.return_value)
            
            # Should succeed and circuit breaker should remain closed
            result = repository.get_document("test-id")
            
            self.assertEqual(circuit_breaker.state, CircuitState.CLOSED)

    @patch('src.slack_kb_agent.database.DatabaseManager')
    def test_repository_circuit_breaker_graceful_degradation(self, mock_db_manager):
        """Test graceful degradation when circuit breaker is open."""
        with patch('src.slack_kb_agent.database.DatabaseRepository._get_circuit_breaker') as mock_cb:
            circuit_breaker = CircuitBreaker(self.circuit_config)
            mock_cb.return_value = circuit_breaker
            
            # Force circuit to open state
            circuit_breaker.state = CircuitState.OPEN
            circuit_breaker.failure_count = CircuitBreakerDefaults.DATABASE_FAILURE_THRESHOLD
            circuit_breaker.last_failure_time = time.time()
            
            repository = DatabaseRepository(mock_db_manager.return_value)
            
            # Operations should be blocked by circuit breaker
            with self.assertRaises(CircuitOpenError):
                repository.create_document(self.test_document)
            
            with self.assertRaises(CircuitOpenError):
                repository.get_document("test-id")

    def test_repository_has_circuit_breaker_method(self):
        """Test that DatabaseRepository has _get_circuit_breaker method."""
        # This test will fail until we implement the method
        with patch('src.slack_kb_agent.database.DatabaseManager'):
            repository = DatabaseRepository(Mock())
            self.assertTrue(hasattr(repository, '_get_circuit_breaker'))
            self.assertTrue(hasattr(repository, 'circuit_breaker'))


class TestDatabaseCircuitBreakerIntegration(unittest.TestCase):
    """Test circuit breaker integration across database components."""
    
    def test_circuit_breaker_metrics_tracking(self):
        """Test that database circuit breaker metrics are properly tracked."""
        circuit_config = CircuitBreakerConfig(
            failure_threshold=5,
            success_threshold=2,
            timeout_seconds=60.0,
            half_open_max_requests=3,
            service_name="database"
        )
        
        with patch('src.slack_kb_agent.database.DatabaseManager._get_circuit_breaker') as mock_cb:
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
            
            self.assertEqual(metrics['service_name'], "database")
            self.assertEqual(metrics['state'], CircuitState.CLOSED.value)
            self.assertEqual(metrics['current_failure_count'], 0)

    @patch('src.slack_kb_agent.database.create_engine')
    def test_circuit_breaker_recovery_after_outage(self, mock_create_engine):
        """Test circuit breaker recovery after database outage."""
        with patch('src.slack_kb_agent.database.DatabaseManager._get_circuit_breaker') as mock_cb:
            circuit_breaker = CircuitBreaker(self.circuit_config)
            mock_cb.return_value = circuit_breaker
            
            # Force circuit to open state
            circuit_breaker.state = CircuitState.OPEN
            circuit_breaker.failure_count = CircuitBreakerDefaults.DATABASE_FAILURE_THRESHOLD
            circuit_breaker.last_failure_time = time.time() - (CircuitBreakerDefaults.DATABASE_TIMEOUT_SECONDS + 1)
            
            # Mock successful connection for recovery
            mock_engine = Mock()
            mock_create_engine.return_value = mock_engine
            
            # First call should transition to half-open and succeed
            db_manager = DatabaseManager(self.test_db_url)
            
            # Should be in half-open state after successful call
            self.assertEqual(circuit_breaker.state, CircuitState.HALF_OPEN)

    def setUp(self):
        """Set up test fixtures for integration tests."""
        self.test_db_url = "postgresql://test:test@localhost/test_db"
        self.circuit_config = CircuitBreakerConfig(
            failure_threshold=CircuitBreakerDefaults.DATABASE_FAILURE_THRESHOLD,
            success_threshold=CircuitBreakerDefaults.DATABASE_SUCCESS_THRESHOLD,
            timeout_seconds=CircuitBreakerDefaults.DATABASE_TIMEOUT_SECONDS,
            half_open_max_requests=CircuitBreakerDefaults.DATABASE_HALF_OPEN_MAX_REQUESTS,
            service_name="database"
        )


class TestDatabaseCircuitBreakerPerformance(unittest.TestCase):
    """Test performance characteristics of database circuit breaker integration."""
    
    def setUp(self):
        """Set up performance test fixtures."""
        self.test_db_url = "postgresql://test:test@localhost/test_db"
        self.circuit_config = CircuitBreakerConfig(
            failure_threshold=3,
            success_threshold=2,
            timeout_seconds=1.0,  # Short timeout for tests
            half_open_max_requests=1,
            service_name="database"
        )
    
    @patch('src.slack_kb_agent.database.create_engine')
    def test_circuit_breaker_overhead_minimal(self, mock_create_engine):
        """Test that circuit breaker adds minimal overhead to database operations."""
        # Mock successful engine creation
        mock_engine = Mock()
        mock_create_engine.return_value = mock_engine
        
        with patch('src.slack_kb_agent.database.DatabaseManager._get_circuit_breaker') as mock_cb:
            circuit_breaker = CircuitBreaker(self.circuit_config)
            mock_cb.return_value = circuit_breaker
            
            # Measure time with circuit breaker
            start_time = time.time()
            db_manager = DatabaseManager(self.test_db_url)
            elapsed_time = time.time() - start_time
            
            # Circuit breaker overhead should be minimal (< 10ms)
            self.assertLess(elapsed_time, 0.01)

    @patch('src.slack_kb_agent.database.create_engine')
    def test_circuit_breaker_fast_failure_when_open(self, mock_create_engine):
        """Test that circuit breaker provides fast failure when open."""
        with patch('src.slack_kb_agent.database.DatabaseManager._get_circuit_breaker') as mock_cb:
            circuit_breaker = CircuitBreaker(self.circuit_config)
            mock_cb.return_value = circuit_breaker
            
            # Force circuit to open state
            circuit_breaker.state = CircuitState.OPEN
            circuit_breaker.failure_count = 3
            circuit_breaker.last_failure_time = time.time()
            
            # Measure fast failure time
            start_time = time.time()
            with self.assertRaises(CircuitOpenError):
                DatabaseManager(self.test_db_url)
            elapsed_time = time.time() - start_time
            
            # Should fail very quickly (< 1ms)
            self.assertLess(elapsed_time, 0.001)
            
            # Should not call the actual database connection
            mock_create_engine.assert_not_called()


if __name__ == '__main__':
    unittest.main()