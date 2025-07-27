#!/usr/bin/env python3
"""
Test suite for constants module in the Slack KB Agent.

This module tests all configuration constants, default values,
utility functions, and environment configuration.
"""

import pytest
import os
from unittest.mock import patch
from typing import Dict, Any

# Import the constants module to test
from slack_kb_agent.constants import (
    NetworkDefaults, RateLimitDefaults, ValidationDefaults, SearchDefaults,
    LLMDefaults, CacheDefaults, CircuitBreakerDefaults, QueryProcessingDefaults,
    SlackBotDefaults, IngestionDefaults, DatabaseDefaults, MonitoringDefaults,
    DisplayDefaults, get_env_int, get_env_float, get_all_defaults,
    EnvironmentConfig
)


class TestDefaultConstantClasses:
    """Test all default constant classes have reasonable values."""
    
    def test_network_defaults(self):
        """Test NetworkDefaults constants."""
        # Test port numbers are valid
        assert 1024 <= NetworkDefaults.DEFAULT_MONITORING_PORT <= 65535
        assert 1 <= NetworkDefaults.DEFAULT_REDIS_PORT <= 65535
        assert 1 <= NetworkDefaults.DEFAULT_DATABASE_PORT <= 65535
        
        # Test timeouts are reasonable
        assert NetworkDefaults.DEFAULT_API_TIMEOUT > 0
        assert NetworkDefaults.DEFAULT_REDIS_SOCKET_TIMEOUT > 0
        assert NetworkDefaults.DEFAULT_REDIS_CONNECT_TIMEOUT > 0
        assert NetworkDefaults.DEFAULT_DATABASE_POOL_RECYCLE > 0
        
        # Test connection limits
        assert NetworkDefaults.DEFAULT_REDIS_MAX_CONNECTIONS > 0
        assert NetworkDefaults.DEFAULT_DATABASE_POOL_SIZE > 0
    
    def test_rate_limit_defaults(self):
        """Test RateLimitDefaults constants."""
        # Test request limits are reasonable
        assert RateLimitDefaults.REQUESTS_PER_MINUTE > 0
        assert RateLimitDefaults.REQUESTS_PER_HOUR > RateLimitDefaults.REQUESTS_PER_MINUTE
        assert RateLimitDefaults.REQUESTS_PER_DAY > RateLimitDefaults.REQUESTS_PER_HOUR
        
        # Test burst handling
        assert RateLimitDefaults.BURST_LIMIT >= RateLimitDefaults.REQUESTS_PER_MINUTE
        
        # Test intervals
        assert RateLimitDefaults.CLEANUP_INTERVAL_SECONDS > 0
        assert RateLimitDefaults.RATE_LIMIT_WINDOW_SECONDS > 0
    
    def test_validation_defaults(self):
        """Test ValidationDefaults constants."""
        # Test length limits are positive
        assert ValidationDefaults.MAX_QUERY_LENGTH > 0
        assert ValidationDefaults.MAX_USER_ID_LENGTH > 0
        assert ValidationDefaults.MAX_CHANNEL_ID_LENGTH > 0
        assert ValidationDefaults.MAX_CACHE_KEY_LENGTH > 0
        assert ValidationDefaults.MAX_DOCUMENT_CONTENT_DISPLAY > 0
        assert ValidationDefaults.MAX_SLACK_MESSAGE_LENGTH > 0
        
        # Test bcrypt constraints
        assert ValidationDefaults.MAX_BCRYPT_PASSWORD_BYTES == 72  # bcrypt limit
        assert ValidationDefaults.MIN_BCRYPT_COST >= 4
        assert ValidationDefaults.MAX_BCRYPT_COST <= 18
        assert ValidationDefaults.MIN_BCRYPT_COST < ValidationDefaults.MAX_BCRYPT_COST
    
    def test_search_defaults(self):
        """Test SearchDefaults constants."""
        assert SearchDefaults.MAX_INDEX_SIZE > 0
        assert SearchDefaults.MAX_CACHE_SIZE > 0
        assert SearchDefaults.DEFAULT_MAX_RESULTS > 0
        assert 0 <= SearchDefaults.DEFAULT_SIMILARITY_THRESHOLD <= 1
        assert SearchDefaults.MIN_TERM_LENGTH > 0
        assert SearchDefaults.INDEX_STATS_REPORTING_INTERVAL > 0
    
    def test_llm_defaults(self):
        """Test LLMDefaults constants."""
        assert LLMDefaults.DEFAULT_MAX_TOKENS > 0
        assert LLMDefaults.DEFAULT_MAX_CONTEXT_TOKENS > 0
        assert LLMDefaults.DEFAULT_TIMEOUT_SECONDS > 0
        assert LLMDefaults.DEFAULT_RETRY_ATTEMPTS >= 0
        assert LLMDefaults.DEFAULT_RETRY_DELAY_SECONDS >= 0
        assert 0 <= LLMDefaults.DEFAULT_TEMPERATURE <= 2
        assert 0 <= LLMDefaults.DEFAULT_SIMILARITY_THRESHOLD <= 1
        assert LLMDefaults.MAX_CONTEXT_DOCUMENTS > 0
    
    def test_cache_defaults(self):
        """Test CacheDefaults constants."""
        assert CacheDefaults.EMBEDDING_TTL_SECONDS > 0
        assert CacheDefaults.QUERY_EXPANSION_TTL_SECONDS > 0
        assert CacheDefaults.SEARCH_RESULTS_TTL_SECONDS > 0
        assert isinstance(CacheDefaults.DEFAULT_CACHE_ENABLED, bool)
        assert isinstance(CacheDefaults.CACHE_KEY_PREFIX, str)
        assert len(CacheDefaults.CACHE_KEY_PREFIX) > 0


class TestCircuitBreakerDefaults:
    """Test circuit breaker constants."""
    
    def test_llm_circuit_breaker_constants(self):
        """Test LLM circuit breaker constants."""
        assert CircuitBreakerDefaults.LLM_FAILURE_THRESHOLD > 0
        assert CircuitBreakerDefaults.LLM_SUCCESS_THRESHOLD > 0
        assert CircuitBreakerDefaults.LLM_TIMEOUT_SECONDS > 0
        assert CircuitBreakerDefaults.LLM_HALF_OPEN_MAX_REQUESTS > 0
        assert CircuitBreakerDefaults.LLM_FAILURE_WINDOW_SECONDS > 0
    
    def test_database_circuit_breaker_constants(self):
        """Test database circuit breaker constants."""
        assert CircuitBreakerDefaults.DATABASE_FAILURE_THRESHOLD > 0
        assert CircuitBreakerDefaults.DATABASE_SUCCESS_THRESHOLD > 0
        assert CircuitBreakerDefaults.DATABASE_TIMEOUT_SECONDS > 0
        assert CircuitBreakerDefaults.DATABASE_HALF_OPEN_MAX_REQUESTS > 0
        assert CircuitBreakerDefaults.DATABASE_FAILURE_WINDOW_SECONDS > 0
    
    def test_all_circuit_breaker_services_configured(self):
        """Test that all services have circuit breaker constants."""
        services = ['LLM', 'DATABASE', 'REDIS', 'SLACK', 'EXTERNAL_SERVICE']
        
        for service in services:
            # Each service should have all required constants
            assert hasattr(CircuitBreakerDefaults, f"{service}_FAILURE_THRESHOLD")
            assert hasattr(CircuitBreakerDefaults, f"{service}_SUCCESS_THRESHOLD")
            assert hasattr(CircuitBreakerDefaults, f"{service}_TIMEOUT_SECONDS")
            assert hasattr(CircuitBreakerDefaults, f"{service}_HALF_OPEN_MAX_REQUESTS")
            assert hasattr(CircuitBreakerDefaults, f"{service}_FAILURE_WINDOW_SECONDS")


class TestUtilityFunctions:
    """Test utility functions for environment variable handling."""
    
    def test_get_env_int_with_default(self):
        """Test get_env_int with default value."""
        with patch.dict(os.environ, {}, clear=True):
            result = get_env_int("TEST_INT", 42)
            assert result == 42
            assert isinstance(result, int)
    
    def test_get_env_int_from_environment(self):
        """Test get_env_int reading from environment."""
        with patch.dict(os.environ, {"TEST_INT": "123"}):
            result = get_env_int("TEST_INT", 42)
            assert result == 123
    
    def test_get_env_int_with_validation_bounds(self):
        """Test get_env_int with min/max validation."""
        with patch.dict(os.environ, {"TEST_INT": "50"}):
            # Within bounds
            result = get_env_int("TEST_INT", 42, min_val=10, max_val=100)
            assert result == 50
            
            # Below minimum
            with pytest.raises(ValueError, match="must be >= 60"):
                get_env_int("TEST_INT", 42, min_val=60)
            
            # Above maximum
            with pytest.raises(ValueError, match="must be <= 40"):
                get_env_int("TEST_INT", 42, max_val=40)
    
    def test_get_env_int_invalid_value(self):
        """Test get_env_int with invalid integer value."""
        with patch.dict(os.environ, {"TEST_INT": "not_an_integer"}):
            with pytest.raises(ValueError, match="Invalid integer value"):
                get_env_int("TEST_INT", 42)
    
    def test_get_env_float_with_default(self):
        """Test get_env_float with default value."""
        with patch.dict(os.environ, {}, clear=True):
            result = get_env_float("TEST_FLOAT", 3.14)
            assert result == 3.14
            assert isinstance(result, float)
    
    def test_get_env_float_from_environment(self):
        """Test get_env_float reading from environment."""
        with patch.dict(os.environ, {"TEST_FLOAT": "2.718"}):
            result = get_env_float("TEST_FLOAT", 3.14)
            assert result == 2.718
    
    def test_get_env_float_with_validation_bounds(self):
        """Test get_env_float with min/max validation."""
        with patch.dict(os.environ, {"TEST_FLOAT": "5.5"}):
            # Within bounds
            result = get_env_float("TEST_FLOAT", 3.14, min_val=1.0, max_val=10.0)
            assert result == 5.5
            
            # Below minimum
            with pytest.raises(ValueError, match="must be >= 6.0"):
                get_env_float("TEST_FLOAT", 3.14, min_val=6.0)
            
            # Above maximum
            with pytest.raises(ValueError, match="must be <= 5.0"):
                get_env_float("TEST_FLOAT", 3.14, max_val=5.0)
    
    def test_get_env_float_invalid_value(self):
        """Test get_env_float with invalid float value."""
        with patch.dict(os.environ, {"TEST_FLOAT": "not_a_float"}):
            with pytest.raises(ValueError, match="Invalid float value"):
                get_env_float("TEST_FLOAT", 3.14)


class TestGetAllDefaults:
    """Test the get_all_defaults function."""
    
    def test_get_all_defaults_returns_dict(self):
        """Test that get_all_defaults returns a dictionary."""
        defaults = get_all_defaults()
        assert isinstance(defaults, dict)
        assert len(defaults) > 0
    
    def test_get_all_defaults_contains_expected_categories(self):
        """Test that all expected categories are present."""
        defaults = get_all_defaults()
        expected_categories = [
            "network", "rate_limiting", "validation", "search",
            "llm", "cache", "circuit_breaker", "monitoring"
        ]
        
        for category in expected_categories:
            assert category in defaults
            assert isinstance(defaults[category], dict)
            assert len(defaults[category]) > 0
    
    def test_get_all_defaults_excludes_private_attributes(self):
        """Test that private attributes (starting with _) are excluded."""
        defaults = get_all_defaults()
        
        for category_dict in defaults.values():
            for attr_name in category_dict.keys():
                assert not attr_name.startswith('_')
    
    def test_get_all_defaults_values_are_serializable(self):
        """Test that all default values are JSON serializable."""
        import json
        
        defaults = get_all_defaults()
        
        # This should not raise an exception
        json_str = json.dumps(defaults)
        assert isinstance(json_str, str)
        assert len(json_str) > 0


class TestEnvironmentConfig:
    """Test the EnvironmentConfig class."""
    
    def test_get_monitoring_port_default(self):
        """Test get_monitoring_port with default value."""
        with patch.dict(os.environ, {}, clear=True):
            port = EnvironmentConfig.get_monitoring_port()
            assert port == NetworkDefaults.DEFAULT_MONITORING_PORT
            assert 1024 <= port <= 65535
    
    def test_get_monitoring_port_from_env(self):
        """Test get_monitoring_port from environment."""
        with patch.dict(os.environ, {"MONITORING_PORT": "8080"}):
            port = EnvironmentConfig.get_monitoring_port()
            assert port == 8080
    
    def test_get_monitoring_port_validation(self):
        """Test get_monitoring_port validation."""
        with patch.dict(os.environ, {"MONITORING_PORT": "80"}):
            with pytest.raises(ValueError, match="must be >= 1024"):
                EnvironmentConfig.get_monitoring_port()
    
    def test_get_redis_port_default(self):
        """Test get_redis_port with default value."""
        with patch.dict(os.environ, {}, clear=True):
            port = EnvironmentConfig.get_redis_port()
            assert port == NetworkDefaults.DEFAULT_REDIS_PORT
    
    def test_get_max_query_length_validation(self):
        """Test get_max_query_length validation."""
        with patch.dict(os.environ, {"MAX_QUERY_LENGTH": "5"}):
            with pytest.raises(ValueError, match="must be >= 10"):
                EnvironmentConfig.get_max_query_length()
        
        with patch.dict(os.environ, {"MAX_QUERY_LENGTH": "20000"}):
            with pytest.raises(ValueError, match="must be <= 10000"):
                EnvironmentConfig.get_max_query_length()
    
    def test_all_environment_config_methods(self):
        """Test that all EnvironmentConfig methods work."""
        with patch.dict(os.environ, {}, clear=True):
            # Test all methods return reasonable values
            assert EnvironmentConfig.get_monitoring_port() > 0
            assert EnvironmentConfig.get_redis_port() > 0
            assert EnvironmentConfig.get_max_query_length() > 0
            assert EnvironmentConfig.get_requests_per_minute() > 0
            assert EnvironmentConfig.get_requests_per_hour() > 0
            assert EnvironmentConfig.get_llm_timeout() > 0
            assert EnvironmentConfig.get_health_check_interval() > 0


class TestConstantValues:
    """Test specific constant values for reasonableness."""
    
    def test_http_status_codes(self):
        """Test HTTP status code constants."""
        assert MonitoringDefaults.HTTP_OK == 200
        assert MonitoringDefaults.HTTP_BAD_REQUEST == 400
        assert MonitoringDefaults.HTTP_UNAUTHORIZED == 401
        assert MonitoringDefaults.HTTP_FORBIDDEN == 403
        assert MonitoringDefaults.HTTP_NOT_FOUND == 404
        assert MonitoringDefaults.HTTP_INTERNAL_ERROR == 500
    
    def test_slack_bot_constants(self):
        """Test Slack bot specific constants."""
        assert SlackBotDefaults.MIN_SIGNING_SECRET_LENGTH > 0
        assert SlackBotDefaults.MAX_RESPONSE_SOURCES_FULL > 0
        assert SlackBotDefaults.MAX_RESPONSE_SOURCES_BRIEF > 0
        assert SlackBotDefaults.MAX_CONTENT_PREVIEW_LENGTH > 0
        assert SlackBotDefaults.SHUTDOWN_SLEEP_SECONDS >= 0
        
        # Error messages should be non-empty strings
        assert isinstance(SlackBotDefaults.ERROR_MESSAGE_GENERIC, str)
        assert isinstance(SlackBotDefaults.ERROR_MESSAGE_UNEXPECTED, str)
        assert len(SlackBotDefaults.ERROR_MESSAGE_GENERIC) > 0
        assert len(SlackBotDefaults.ERROR_MESSAGE_UNEXPECTED) > 0
    
    def test_ingestion_constants(self):
        """Test ingestion and data collection constants."""
        assert IngestionDefaults.MIN_API_KEY_LENGTH > 0
        assert IngestionDefaults.MIN_TOKEN_LENGTH > 0
        assert IngestionDefaults.MIN_SECRET_LENGTH > 0
        assert IngestionDefaults.GITHUB_API_PER_PAGE > 0
        assert IngestionDefaults.GITHUB_API_TIMEOUT_SECONDS > 0
        assert IngestionDefaults.DEFAULT_SLACK_HISTORY_DAYS > 0
        assert IngestionDefaults.SECONDS_PER_DAY == 24 * 60 * 60
    
    def test_display_constants(self):
        """Test display and formatting constants."""
        assert DisplayDefaults.MAX_ATTRIBUTE_DISPLAY_LENGTH > 0
        assert DisplayDefaults.MAX_ATTRIBUTES_DISPLAY > 0
        assert DisplayDefaults.PERCENT_MULTIPLIER == 100
        assert DisplayDefaults.BYTES_PER_MB == 1024 * 1024


class TestQueryProcessingDefaults:
    """Test query processing constants."""
    
    def test_classification_thresholds(self):
        """Test classification threshold values."""
        assert 0 <= QueryProcessingDefaults.CLASSIFICATION_HIGH_CONFIDENCE <= 1
        assert 0 <= QueryProcessingDefaults.CLASSIFICATION_LOW_CONFIDENCE <= 1
        assert (QueryProcessingDefaults.CLASSIFICATION_LOW_CONFIDENCE < 
                QueryProcessingDefaults.CLASSIFICATION_HIGH_CONFIDENCE)
        assert QueryProcessingDefaults.CLASSIFICATION_MIN_WORDS > 0
    
    def test_expansion_limits(self):
        """Test query expansion limits."""
        assert QueryProcessingDefaults.MIN_EXPANSION_TERM_LENGTH > 0
        assert QueryProcessingDefaults.MAX_EXPANSION_TERMS > 0
        assert QueryProcessingDefaults.MAX_EXPANDED_SEARCH_TERMS > 0
        assert QueryProcessingDefaults.MAX_BASIC_SUGGESTION_EXPANSIONS > 0
    
    def test_context_settings(self):
        """Test context processing settings."""
        assert QueryProcessingDefaults.DEFAULT_MAX_USER_CONTEXTS > 0
        assert QueryProcessingDefaults.CONTEXT_RECENT_WINDOW > 0
        assert QueryProcessingDefaults.CONTEXT_RELEVANCE_WINDOW > 0
        assert 0 <= QueryProcessingDefaults.CONTEXT_RELEVANCE_THRESHOLD <= 1
        assert QueryProcessingDefaults.MAX_CONTEXT_TOPICS > 0


class TestConstantIntegrity:
    """Test the integrity and consistency of constants."""
    
    def test_rate_limit_hierarchy(self):
        """Test that rate limits follow logical hierarchy."""
        # Per hour should be higher than per minute
        assert (RateLimitDefaults.REQUESTS_PER_HOUR > 
                RateLimitDefaults.REQUESTS_PER_MINUTE * 60 or
                RateLimitDefaults.REQUESTS_PER_HOUR >= 
                RateLimitDefaults.REQUESTS_PER_MINUTE)
        
        # Per day should be higher than per hour
        assert (RateLimitDefaults.REQUESTS_PER_DAY >= 
                RateLimitDefaults.REQUESTS_PER_HOUR)
    
    def test_timeout_consistency(self):
        """Test that timeout values are consistent."""
        # LLM timeout should be reasonable
        assert 5 <= LLMDefaults.DEFAULT_TIMEOUT_SECONDS <= 300
        
        # Circuit breaker timeouts should be reasonable
        assert CircuitBreakerDefaults.LLM_TIMEOUT_SECONDS > 0
        assert CircuitBreakerDefaults.DATABASE_TIMEOUT_SECONDS > 0
    
    def test_string_constants_not_empty(self):
        """Test that string constants are not empty."""
        assert len(CacheDefaults.CACHE_KEY_PREFIX) > 0
        assert len(SlackBotDefaults.ERROR_MESSAGE_GENERIC) > 0
        assert len(SlackBotDefaults.ERROR_MESSAGE_UNEXPECTED) > 0


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])