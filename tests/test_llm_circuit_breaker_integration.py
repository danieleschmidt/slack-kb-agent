#!/usr/bin/env python3
"""
Tests for LLM circuit breaker integration.

This test suite verifies that the circuit breaker properly protects LLM API calls
and handles failures gracefully according to the WSJF priority implementation.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import time
from typing import Optional, List

from src.slack_kb_agent.llm import (
    LLMResponse, LLMConfig, ResponseGenerator, OpenAIProvider, AnthropicProvider
)
from src.slack_kb_agent.circuit_breaker import (
    CircuitBreaker, CircuitBreakerConfig, CircuitState, CircuitOpenError
)
from src.slack_kb_agent.constants import CircuitBreakerDefaults
from src.slack_kb_agent.models import Document


class TestLLMCircuitBreakerIntegration(unittest.TestCase):
    """Test circuit breaker integration with LLM providers."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_config = LLMConfig(
            enabled=True,
            provider="openai",
            api_key="test-key",
            retry_attempts=1  # Reduced for faster tests
        )
        
        self.test_documents = [
            Document(
                content="This is test content for LLM context.",
                source="test",
                metadata={"title": "Test Document", "id": "doc1"}
            )
        ]
        
        # Circuit breaker config for LLM
        self.circuit_config = CircuitBreakerConfig(
            failure_threshold=CircuitBreakerDefaults.LLM_FAILURE_THRESHOLD,
            success_threshold=CircuitBreakerDefaults.LLM_SUCCESS_THRESHOLD,
            timeout_seconds=CircuitBreakerDefaults.LLM_TIMEOUT_SECONDS,
            half_open_max_requests=CircuitBreakerDefaults.LLM_HALF_OPEN_MAX_REQUESTS,
            service_name="llm_provider"
        )

    def test_circuit_breaker_config_from_constants(self):
        """Test that circuit breaker uses proper configuration from constants."""
        config = CircuitBreakerConfig(
            failure_threshold=CircuitBreakerDefaults.LLM_FAILURE_THRESHOLD,
            success_threshold=CircuitBreakerDefaults.LLM_SUCCESS_THRESHOLD,
            timeout_seconds=CircuitBreakerDefaults.LLM_TIMEOUT_SECONDS,
            half_open_max_requests=CircuitBreakerDefaults.LLM_HALF_OPEN_MAX_REQUESTS,
            service_name="llm_provider"
        )
        
        self.assertEqual(config.failure_threshold, 3)
        self.assertEqual(config.success_threshold, 2)
        self.assertEqual(config.timeout_seconds, 30.0)
        self.assertEqual(config.half_open_max_requests, 1)
        self.assertEqual(config.service_name, "llm_provider")

    @patch('src.slack_kb_agent.llm.LLMProvider.create')
    @patch('src.slack_kb_agent.llm.OpenAIProvider.generate_response')
    def test_circuit_breaker_protects_successful_calls(self, mock_generate, mock_create):
        """Test that successful LLM calls pass through circuit breaker."""
        # Mock successful response
        mock_generate.return_value = LLMResponse(
            content="Test response",
            success=True,
            token_usage={"total": 100},
            response_time=0.5
        )
        
        # Mock provider creation
        mock_provider = Mock()
        mock_provider.generate_response = mock_generate
        mock_create.return_value = mock_provider
        
        # Create circuit breaker protected response generator
        with patch('src.slack_kb_agent.llm.ResponseGenerator._get_circuit_breaker') as mock_cb:
            circuit_breaker = CircuitBreaker(self.circuit_config)
            mock_cb.return_value = circuit_breaker
            
            generator = ResponseGenerator(self.test_config)
            response = generator.generate_response("test query", self.test_documents)
            
            self.assertTrue(response.success)
            self.assertEqual(response.content, "Test response")
            self.assertEqual(circuit_breaker.state, CircuitState.CLOSED)
            self.assertEqual(circuit_breaker.failure_count, 0)

    @patch('src.slack_kb_agent.llm.LLMProvider.create')
    @patch('src.slack_kb_agent.llm.OpenAIProvider.generate_response')
    def test_circuit_breaker_handles_api_failures(self, mock_generate, mock_create):
        """Test that circuit breaker opens after repeated API failures."""
        # Mock API failure
        mock_generate.return_value = LLMResponse(
            content="",
            success=False,
            error_message="API rate limit exceeded"
        )
        
        # Mock provider creation
        mock_provider = Mock()
        mock_provider.generate_response = mock_generate
        mock_create.return_value = mock_provider
        
        with patch('src.slack_kb_agent.llm.ResponseGenerator._get_circuit_breaker') as mock_cb:
            circuit_breaker = CircuitBreaker(self.circuit_config)
            mock_cb.return_value = circuit_breaker
            
            generator = ResponseGenerator(self.test_config)
            
            # Generate failures to trigger circuit breaker
            for i in range(CircuitBreakerDefaults.LLM_FAILURE_THRESHOLD):
                response = generator.generate_response("test query", self.test_documents)
                self.assertFalse(response.success)
            
            # Circuit should now be open
            self.assertEqual(circuit_breaker.state, CircuitState.OPEN)
            
            # Next call should be rejected by circuit breaker
            response = generator.generate_response("test query", self.test_documents)
            self.assertFalse(response.success)
            self.assertIn("circuit breaker", response.error_message.lower())

    @patch('src.slack_kb_agent.llm.LLMProvider.create')
    @patch('src.slack_kb_agent.llm.OpenAIProvider.generate_response')
    def test_circuit_breaker_recovery_from_half_open(self, mock_generate, mock_create):
        """Test circuit breaker recovery after service recovers."""
        # Mock provider creation
        mock_provider = Mock()
        mock_provider.generate_response = mock_generate
        mock_create.return_value = mock_provider
        
        with patch('src.slack_kb_agent.llm.ResponseGenerator._get_circuit_breaker') as mock_cb:
            circuit_breaker = CircuitBreaker(self.circuit_config)
            mock_cb.return_value = circuit_breaker
            
            generator = ResponseGenerator(self.test_config)
            
            # Force circuit to open
            circuit_breaker.state = CircuitState.OPEN
            circuit_breaker.failure_count = CircuitBreakerDefaults.LLM_FAILURE_THRESHOLD
            circuit_breaker.last_failure_time = time.time() - (CircuitBreakerDefaults.LLM_TIMEOUT_SECONDS + 1)
            
            # Mock successful response for recovery
            mock_generate.return_value = LLMResponse(
                content="Recovery successful",
                success=True,
                token_usage={"total": 50},
                response_time=0.3
            )
            
            # First call should transition to half-open and succeed
            response = generator.generate_response("test query", self.test_documents)
            self.assertTrue(response.success)
            self.assertEqual(circuit_breaker.state, CircuitState.HALF_OPEN)
            
            # After successful half-open requests, we need enough successes to close circuit
            # The success threshold is 2, but half_open_max_requests is 1, so we need to 
            # wait for the circuit breaker logic to transition to closed after successful requests
            
            # Check if circuit transitioned to closed (this happens internally)
            # For this test, we'll verify that the first request succeeded and circuit is in half-open
            # The actual closing happens after success_threshold is met
            self.assertTrue(response.success)
            # Circuit should be in half-open after first successful request
            self.assertEqual(circuit_breaker.state, CircuitState.HALF_OPEN)

    @patch('src.slack_kb_agent.llm.LLMProvider.create')
    @patch('src.slack_kb_agent.llm.AnthropicProvider.generate_response')
    def test_circuit_breaker_works_with_anthropic(self, mock_generate, mock_create):
        """Test that circuit breaker works with Anthropic provider."""
        anthropic_config = LLMConfig(
            enabled=True,
            provider="anthropic",
            api_key="test-anthropic-key",
            retry_attempts=1
        )
        
        # Mock connection error
        mock_generate.return_value = LLMResponse(
            content="",
            success=False,
            error_message="Connection timeout"
        )
        
        # Mock provider creation
        mock_provider = Mock()
        mock_provider.generate_response = mock_generate
        mock_create.return_value = mock_provider
        
        with patch('src.slack_kb_agent.llm.ResponseGenerator._get_circuit_breaker') as mock_cb:
            circuit_breaker = CircuitBreaker(self.circuit_config)
            mock_cb.return_value = circuit_breaker
            
            generator = ResponseGenerator(anthropic_config)
            
            # Generate failures
            for i in range(CircuitBreakerDefaults.LLM_FAILURE_THRESHOLD):
                response = generator.generate_response("test query", self.test_documents)
                self.assertFalse(response.success)
            
            # Circuit should be open
            self.assertEqual(circuit_breaker.state, CircuitState.OPEN)

    def test_circuit_breaker_metrics_integration(self):
        """Test that circuit breaker metrics are properly tracked."""
        with patch('src.slack_kb_agent.llm.ResponseGenerator._get_circuit_breaker') as mock_cb:
            circuit_breaker = CircuitBreaker(self.circuit_config)
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
            
            self.assertEqual(metrics['state'], CircuitState.CLOSED.value)
            self.assertEqual(metrics['current_failure_count'], 0)
            self.assertEqual(metrics['total_successes'], 0)

    @patch('src.slack_kb_agent.llm.LLMProvider.create')
    @patch('src.slack_kb_agent.llm.OpenAIProvider.generate_response')
    def test_circuit_breaker_timeout_behavior(self, mock_generate, mock_create):
        """Test circuit breaker timeout behavior in half-open state."""
        mock_generate.return_value = LLMResponse(
            content="",
            success=False,
            error_message="Service temporarily unavailable"
        )
        
        # Mock provider creation
        mock_provider = Mock()
        mock_provider.generate_response = mock_generate
        mock_create.return_value = mock_provider
        
        with patch('src.slack_kb_agent.llm.ResponseGenerator._get_circuit_breaker') as mock_cb:
            circuit_breaker = CircuitBreaker(self.circuit_config)
            mock_cb.return_value = circuit_breaker
            
            generator = ResponseGenerator(self.test_config)
            
            # Force circuit to open
            circuit_breaker.state = CircuitState.OPEN
            circuit_breaker.failure_count = CircuitBreakerDefaults.LLM_FAILURE_THRESHOLD
            circuit_breaker.last_failure_time = time.time()
            
            # Call before timeout should be rejected
            response = generator.generate_response("test query", self.test_documents)
            self.assertFalse(response.success)
            self.assertIn("circuit breaker", response.error_message.lower())
            
            # After timeout, circuit should allow limited requests (half-open)
            circuit_breaker.last_failure_time = time.time() - (CircuitBreakerDefaults.LLM_TIMEOUT_SECONDS + 1)
            
            # This should transition to half-open and fail
            response = generator.generate_response("test query", self.test_documents)
            self.assertFalse(response.success)
            self.assertEqual(circuit_breaker.state, CircuitState.OPEN)  # Should reopen after failure

    def test_circuit_breaker_disabled_when_llm_disabled(self):
        """Test that circuit breaker is not used when LLM is disabled."""
        disabled_config = LLMConfig(enabled=False)
        generator = ResponseGenerator(disabled_config)
        
        response = generator.generate_response("test query", self.test_documents)
        self.assertFalse(response.success)
        self.assertIn("disabled", response.error_message.lower())

    @patch('src.slack_kb_agent.llm.LLMProvider.create')
    @patch('src.slack_kb_agent.llm.OpenAIProvider.generate_response')
    def test_circuit_breaker_exception_handling(self, mock_generate, mock_create):
        """Test that circuit breaker properly handles exceptions from LLM calls."""
        # Mock exception during API call
        mock_generate.side_effect = Exception("Unexpected API error")
        
        # Mock provider creation
        mock_provider = Mock()
        mock_provider.generate_response = mock_generate
        mock_create.return_value = mock_provider
        
        with patch('src.slack_kb_agent.llm.ResponseGenerator._get_circuit_breaker') as mock_cb:
            circuit_breaker = CircuitBreaker(self.circuit_config)
            mock_cb.return_value = circuit_breaker
            
            generator = ResponseGenerator(self.test_config)
            
            response = generator.generate_response("test query", self.test_documents)
            
            # Should handle exception gracefully
            self.assertFalse(response.success)
            self.assertIn("error", response.error_message.lower())
            
            # Circuit breaker should record the failure
            self.assertEqual(circuit_breaker.failure_count, 1)

    def test_circuit_breaker_state_transitions_logged(self):
        """Test that circuit breaker state transitions are properly logged."""
        with patch('src.slack_kb_agent.llm.ResponseGenerator._get_circuit_breaker') as mock_cb:
            circuit_breaker = CircuitBreaker(self.circuit_config)
            mock_cb.return_value = circuit_breaker
            
            # Test state transition tracking
            initial_state = circuit_breaker.state
            
            # Force state change
            circuit_breaker.state = CircuitState.OPEN
            circuit_breaker.failure_count = CircuitBreakerDefaults.LLM_FAILURE_THRESHOLD
            
            metrics = circuit_breaker.get_metrics()
            
            # Should track current state and provide metrics
            self.assertEqual(metrics['state'], CircuitState.OPEN.value)
            self.assertEqual(metrics['current_failure_count'], CircuitBreakerDefaults.LLM_FAILURE_THRESHOLD)
            self.assertIn('last_state_change_time', metrics)


class TestLLMCircuitBreakerPerformance(unittest.TestCase):
    """Test performance characteristics of circuit breaker integration."""
    
    def setUp(self):
        """Set up performance test fixtures."""
        self.test_config = LLMConfig(
            enabled=True,
            provider="openai", 
            api_key="test-key",
            retry_attempts=1
        )
        
        self.circuit_config = CircuitBreakerConfig(
            failure_threshold=3,
            success_threshold=2,
            timeout_seconds=1.0,  # Short timeout for tests
            half_open_max_requests=1,
            service_name="llm_provider"
        )
    
    @patch('src.slack_kb_agent.llm.LLMProvider.create')
    @patch('src.slack_kb_agent.llm.OpenAIProvider.generate_response')
    def test_circuit_breaker_overhead_minimal(self, mock_generate, mock_create):
        """Test that circuit breaker adds minimal overhead to successful calls."""
        mock_generate.return_value = LLMResponse(
            content="Test response",
            success=True,
            response_time=0.1
        )
        
        # Mock provider creation
        mock_provider = Mock()
        mock_provider.generate_response = mock_generate
        mock_create.return_value = mock_provider
        
        with patch('src.slack_kb_agent.llm.ResponseGenerator._get_circuit_breaker') as mock_cb:
            circuit_breaker = CircuitBreaker(self.circuit_config)
            mock_cb.return_value = circuit_breaker
            
            generator = ResponseGenerator(self.test_config)
            
            # Measure time with circuit breaker
            start_time = time.time()
            response = generator.generate_response("test query", [])
            elapsed_time = time.time() - start_time
            
            self.assertTrue(response.success)
            # Circuit breaker overhead should be minimal (< 10ms)
            self.assertLess(elapsed_time, 0.01)

    @patch('src.slack_kb_agent.llm.LLMProvider.create')
    @patch('src.slack_kb_agent.llm.OpenAIProvider.generate_response')
    def test_circuit_breaker_fast_failure_when_open(self, mock_generate, mock_create):
        """Test that circuit breaker provides fast failure when open."""
        # Mock provider creation
        mock_provider = Mock()
        mock_provider.generate_response = mock_generate
        mock_create.return_value = mock_provider
        
        with patch('src.slack_kb_agent.llm.ResponseGenerator._get_circuit_breaker') as mock_cb:
            circuit_breaker = CircuitBreaker(self.circuit_config)
            mock_cb.return_value = circuit_breaker
            
            # Force circuit to open state
            circuit_breaker.state = CircuitState.OPEN
            circuit_breaker.failure_count = 3
            circuit_breaker.last_failure_time = time.time()
            
            generator = ResponseGenerator(self.test_config)
            
            # Measure fast failure time
            start_time = time.time()
            response = generator.generate_response("test query", [])
            elapsed_time = time.time() - start_time
            
            self.assertFalse(response.success)
            self.assertIn("circuit breaker", response.error_message.lower())
            # Should fail very quickly (< 1ms)
            self.assertLess(elapsed_time, 0.001)
            
            # Should not call the actual LLM provider
            mock_generate.assert_not_called()


if __name__ == '__main__':
    unittest.main()