#!/usr/bin/env python3
"""
Configuration constants and default values for slack-kb-agent.

This module centralizes all hardcoded values, magic numbers, and configuration
constants to improve maintainability and make the system more configurable.

Constants are organized by functional area and include documentation
explaining their purpose and recommended ranges.
"""

import os
from typing import Dict, Any


# =============================================================================
# NETWORK & CONNECTIVITY CONSTANTS
# =============================================================================

class NetworkDefaults:
    """Network-related default values."""
    
    # Port numbers
    DEFAULT_MONITORING_PORT = 9090
    DEFAULT_REDIS_PORT = 6379
    DEFAULT_DATABASE_PORT = 5432
    
    # Timeouts (seconds)
    DEFAULT_API_TIMEOUT = 30
    DEFAULT_REDIS_SOCKET_TIMEOUT = 5.0
    DEFAULT_REDIS_CONNECT_TIMEOUT = 5.0
    DEFAULT_DATABASE_POOL_RECYCLE = 3600  # 1 hour
    
    # Connection limits
    DEFAULT_REDIS_MAX_CONNECTIONS = 20
    DEFAULT_DATABASE_POOL_SIZE = 5


# =============================================================================
# RATE LIMITING CONSTANTS  
# =============================================================================

class RateLimitDefaults:
    """Rate limiting default values and limits."""
    
    # Request limits per time window
    REQUESTS_PER_MINUTE = 10
    REQUESTS_PER_HOUR = 100
    REQUESTS_PER_DAY = 1000
    
    # Burst handling
    BURST_LIMIT = 15  # Allow short bursts above per-minute limit
    
    # Cleanup and maintenance
    CLEANUP_INTERVAL_SECONDS = 300  # 5 minutes
    RATE_LIMIT_WINDOW_SECONDS = 3600  # 1 hour for monitoring rate limits


# =============================================================================
# VALIDATION & INPUT LIMITS
# =============================================================================

class ValidationDefaults:
    """Input validation limits and constraints."""
    
    # Query constraints
    MAX_QUERY_LENGTH = 1000
    MAX_USER_ID_LENGTH = 20
    MAX_CHANNEL_ID_LENGTH = 20
    
    # Cache key limits
    MAX_CACHE_KEY_LENGTH = 200  # For Redis key length limits
    
    # Content limits
    MAX_DOCUMENT_CONTENT_DISPLAY = 200  # Characters shown in responses
    MAX_SLACK_MESSAGE_LENGTH = 4000     # Slack message limit
    
    # Password constraints
    MAX_BCRYPT_PASSWORD_BYTES = 72      # bcrypt truncation limit
    MIN_BCRYPT_COST = 4
    MAX_BCRYPT_COST = 18


# =============================================================================
# SEARCH & INDEXING CONSTANTS
# =============================================================================

class SearchDefaults:
    """Search engine and indexing default values."""
    
    # Index size limits
    MAX_INDEX_SIZE = 50000
    MAX_CACHE_SIZE = 1000
    
    # Search behavior
    DEFAULT_MAX_RESULTS = 10
    DEFAULT_SIMILARITY_THRESHOLD = 0.5
    MIN_TERM_LENGTH = 2  # Minimum characters for search terms
    
    # Performance tuning
    INDEX_STATS_REPORTING_INTERVAL = 1000  # Operations between stats


# =============================================================================
# LLM & AI CONSTANTS
# =============================================================================

class LLMDefaults:
    """Large Language Model configuration defaults."""
    
    # Token limits
    DEFAULT_MAX_TOKENS = 1000
    DEFAULT_MAX_CONTEXT_TOKENS = 3000
    
    # Request handling
    DEFAULT_TIMEOUT_SECONDS = 30
    DEFAULT_RETRY_ATTEMPTS = 3
    DEFAULT_RETRY_DELAY_SECONDS = 1.0
    DEFAULT_TEMPERATURE = 0.1
    
    # Safety and quality
    DEFAULT_SIMILARITY_THRESHOLD = 0.5
    MAX_CONTEXT_DOCUMENTS = 10


# =============================================================================
# CACHING CONSTANTS
# =============================================================================

class CacheDefaults:
    """Caching system default TTL values and limits."""
    
    # TTL values in seconds
    EMBEDDING_TTL_SECONDS = 3600 * 24 * 7   # 7 days
    QUERY_EXPANSION_TTL_SECONDS = 3600 * 24 # 1 day  
    SEARCH_RESULTS_TTL_SECONDS = 3600       # 1 hour
    
    # Cache behavior
    DEFAULT_CACHE_ENABLED = True
    CACHE_KEY_PREFIX = "slack_kb"


# =============================================================================
# CIRCUIT BREAKER CONSTANTS
# =============================================================================

class CircuitBreakerDefaults:
    """Circuit breaker configuration defaults for external services."""
    
    # OpenAI/Anthropic API circuit breaker
    LLM_FAILURE_THRESHOLD = 3
    LLM_SUCCESS_THRESHOLD = 2
    LLM_TIMEOUT_SECONDS = 30.0
    LLM_HALF_OPEN_MAX_REQUESTS = 1
    LLM_FAILURE_WINDOW_SECONDS = 600.0  # 10 minutes
    
    # Database circuit breaker  
    DATABASE_FAILURE_THRESHOLD = 5
    DATABASE_SUCCESS_THRESHOLD = 2
    DATABASE_TIMEOUT_SECONDS = 60.0
    DATABASE_HALF_OPEN_MAX_REQUESTS = 3
    DATABASE_FAILURE_WINDOW_SECONDS = 1800.0  # 30 minutes
    
    # Redis cache circuit breaker
    REDIS_FAILURE_THRESHOLD = 3
    REDIS_SUCCESS_THRESHOLD = 2
    REDIS_TIMEOUT_SECONDS = 30.0
    REDIS_HALF_OPEN_MAX_REQUESTS = 2
    REDIS_FAILURE_WINDOW_SECONDS = 600.0  # 10 minutes
    
    # Slack API circuit breaker
    SLACK_FAILURE_THRESHOLD = 3
    SLACK_SUCCESS_THRESHOLD = 2
    SLACK_TIMEOUT_SECONDS = 30.0
    SLACK_HALF_OPEN_MAX_REQUESTS = 1
    SLACK_FAILURE_WINDOW_SECONDS = 600.0  # 10 minutes
    
    # Generic external service circuit breaker
    EXTERNAL_SERVICE_FAILURE_THRESHOLD = 5
    EXTERNAL_SERVICE_SUCCESS_THRESHOLD = 2
    EXTERNAL_SERVICE_TIMEOUT_SECONDS = 60.0
    EXTERNAL_SERVICE_HALF_OPEN_MAX_REQUESTS = 2
    EXTERNAL_SERVICE_FAILURE_WINDOW_SECONDS = 1800.0  # 30 minutes


# =============================================================================
# QUERY PROCESSING CONSTANTS
# =============================================================================

class QueryProcessingDefaults:
    """Query processing and enhancement constants."""
    
    # Query classification
    CLASSIFICATION_HIGH_CONFIDENCE = 0.8
    CLASSIFICATION_LOW_CONFIDENCE = 0.6
    CLASSIFICATION_MIN_WORDS = 2
    
    # Query expansion
    MIN_EXPANSION_TERM_LENGTH = 2
    MAX_EXPANSION_TERMS = 5
    MAX_EXPANDED_SEARCH_TERMS = 5
    MAX_BASIC_SUGGESTION_EXPANSIONS = 3
    
    # Context and history
    DEFAULT_MAX_USER_CONTEXTS = 1000
    CONTEXT_RECENT_WINDOW = 3
    CONTEXT_RELEVANCE_WINDOW = 2
    CONTEXT_RELEVANCE_THRESHOLD = 0.2
    MAX_CONTEXT_TOPICS = 3
    MAX_CONTEXT_DOCUMENT_SOURCES = 5
    
    # Search behavior
    SEMANTIC_SEARCH_THRESHOLD = 0.7
    MAX_ENHANCED_SEARCH_RESULTS = 10
    MAX_QUERY_SUGGESTIONS = 4
    DEFAULT_SEMANTIC_SEARCH_TOP_K = 10
    
    # Content processing
    MAX_SANITIZED_QUERY_LENGTH = 500
    CHARS_PER_TOKEN_ESTIMATE = 4
    MIN_PARTIAL_DOCUMENT_CHARS = 100
    PARTIAL_CONTENT_SOURCE_BUFFER = 50


# =============================================================================
# SLACK BOT CONSTANTS
# =============================================================================

class SlackBotDefaults:
    """Slack bot configuration constants."""
    
    # Authentication validation
    MIN_SIGNING_SECRET_LENGTH = 10
    
    # Response formatting
    MAX_RESPONSE_SOURCES_FULL = 3
    MAX_RESPONSE_SOURCES_BRIEF = 2
    MAX_CONTENT_PREVIEW_LENGTH = 197
    ANALYTICS_TOP_QUERIES_COUNT = 5
    
    # Shutdown behavior
    SHUTDOWN_SLEEP_SECONDS = 0.1
    
    # Error messages
    ERROR_MESSAGE_GENERIC = "Sorry, I encountered an error processing your request."
    ERROR_MESSAGE_UNEXPECTED = "Sorry, I encountered an unexpected error. Please try again."


# =============================================================================
# INGESTION & DATA COLLECTION CONSTANTS
# =============================================================================

class IngestionDefaults:
    """Data ingestion and collection constants."""
    
    # Secret detection patterns
    MIN_API_KEY_LENGTH = 10
    MIN_TOKEN_LENGTH = 15
    MIN_SECRET_LENGTH = 10
    
    # API limits and timeouts
    GITHUB_API_PER_PAGE = 100
    GITHUB_API_TIMEOUT_SECONDS = 30
    WEB_CRAWL_TIMEOUT_SECONDS = 10
    DEFAULT_MAX_CRAWL_PAGES = 10
    
    # Slack ingestion
    DEFAULT_SLACK_HISTORY_DAYS = 30
    SLACK_API_MESSAGE_LIMIT = 1000
    SECONDS_PER_DAY = 24 * 60 * 60
    
    # Processing
    DEFAULT_BATCH_PROCESSOR_SIZE = 50
    DOCUMENT_HASH_PREFIX_LENGTH = 8


# =============================================================================
# DATABASE CONSTANTS
# =============================================================================

class DatabaseDefaults:
    """Database configuration constants."""
    
    # Connection pool settings
    DEFAULT_POOL_SIZE = 10
    DEFAULT_MAX_OVERFLOW = 20
    POOL_TIMEOUT_SECONDS = 30
    
    # Search limits
    DEFAULT_SEARCH_LIMIT = 100


# =============================================================================
# MONITORING & OBSERVABILITY CONSTANTS
# =============================================================================

class MonitoringDefaults:
    """Monitoring, metrics, and observability defaults."""
    
    # Health check intervals
    HEALTH_CHECK_INTERVAL_SECONDS = 30
    SYSTEM_RESOURCE_CHECK_INTERVAL = 60
    
    # Metrics retention
    METRICS_RETENTION_HOURS = 24
    METRICS_CLEANUP_INTERVAL_HOURS = 6
    
    # Thresholds for health checks
    MAX_MEMORY_USAGE_PERCENT = 90
    MEMORY_WARNING_THRESHOLD_PERCENT = 75
    MAX_DISK_USAGE_PERCENT = 95
    DISK_WARNING_THRESHOLD_PERCENT = 85
    MIN_FREE_DISK_MB = 1024  # 1GB
    MIN_KNOWLEDGE_BASE_DOCS_WARNING = 10
    
    # HTTP response codes
    HTTP_OK = 200
    HTTP_BAD_REQUEST = 400
    HTTP_UNAUTHORIZED = 401
    HTTP_FORBIDDEN = 403
    HTTP_NOT_FOUND = 404
    HTTP_INTERNAL_ERROR = 500
    
    # Authentication cleanup
    AUTH_CLEANUP_INTERVAL_SECONDS = 3600


# =============================================================================
# DISPLAY & FORMATTING CONSTANTS
# =============================================================================

class DisplayDefaults:
    """Constants for display and formatting purposes."""
    
    # Security utilities
    MAX_ATTRIBUTE_DISPLAY_LENGTH = 50
    MAX_ATTRIBUTES_DISPLAY = 5
    
    # Calculations
    PERCENT_MULTIPLIER = 100
    BYTES_PER_MB = 1024 * 1024


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_env_int(key: str, default: int, min_val: int = None, max_val: int = None) -> int:
    """
    Get integer value from environment with validation.
    
    Args:
        key: Environment variable name
        default: Default value if not set
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        
    Returns:
        Validated integer value
        
    Raises:
        ValueError: If value is outside allowed range
    """
    try:
        value = int(os.getenv(key, str(default)))
        
        if min_val is not None and value < min_val:
            raise ValueError(f"{key} must be >= {min_val}, got {value}")
        if max_val is not None and value > max_val:
            raise ValueError(f"{key} must be <= {max_val}, got {value}")
            
        return value
        
    except (ValueError, TypeError) as e:
        if "must be" in str(e):
            raise  # Re-raise validation errors
        raise ValueError(f"Invalid integer value for {key}: {os.getenv(key)}")


def get_env_float(key: str, default: float, min_val: float = None, max_val: float = None) -> float:
    """
    Get float value from environment with validation.
    
    Args:
        key: Environment variable name
        default: Default value if not set
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        
    Returns:
        Validated float value
        
    Raises:
        ValueError: If value is outside allowed range
    """
    try:
        value = float(os.getenv(key, str(default)))
        
        if min_val is not None and value < min_val:
            raise ValueError(f"{key} must be >= {min_val}, got {value}")
        if max_val is not None and value > max_val:
            raise ValueError(f"{key} must be <= {max_val}, got {value}")
            
        return value
        
    except (ValueError, TypeError) as e:
        if "must be" in str(e):
            raise  # Re-raise validation errors
        raise ValueError(f"Invalid float value for {key}: {os.getenv(key)}")


def get_all_defaults() -> Dict[str, Any]:
    """
    Get all default values as a dictionary for documentation/debugging.
    
    Returns:
        Dictionary of all constant values organized by category
    """
    return {
        "network": {
            attr: getattr(NetworkDefaults, attr)
            for attr in dir(NetworkDefaults)
            if not attr.startswith('_')
        },
        "rate_limiting": {
            attr: getattr(RateLimitDefaults, attr)
            for attr in dir(RateLimitDefaults)
            if not attr.startswith('_')
        },
        "validation": {
            attr: getattr(ValidationDefaults, attr)
            for attr in dir(ValidationDefaults)
            if not attr.startswith('_')
        },
        "search": {
            attr: getattr(SearchDefaults, attr)
            for attr in dir(SearchDefaults)
            if not attr.startswith('_')
        },
        "llm": {
            attr: getattr(LLMDefaults, attr)
            for attr in dir(LLMDefaults)
            if not attr.startswith('_')
        },
        "cache": {
            attr: getattr(CacheDefaults, attr)
            for attr in dir(CacheDefaults)
            if not attr.startswith('_')
        },
        "circuit_breaker": {
            attr: getattr(CircuitBreakerDefaults, attr)
            for attr in dir(CircuitBreakerDefaults)
            if not attr.startswith('_')
        },
        "monitoring": {
            attr: getattr(MonitoringDefaults, attr)
            for attr in dir(MonitoringDefaults)
            if not attr.startswith('_')
        }
    }


# =============================================================================
# ENVIRONMENT-AWARE CONFIGURATION
# =============================================================================

class EnvironmentConfig:
    """Environment-aware configuration that respects env vars but provides defaults."""
    
    @classmethod
    def get_monitoring_port(cls) -> int:
        return get_env_int("MONITORING_PORT", NetworkDefaults.DEFAULT_MONITORING_PORT, 1024, 65535)
    
    @classmethod
    def get_redis_port(cls) -> int:
        return get_env_int("REDIS_PORT", NetworkDefaults.DEFAULT_REDIS_PORT, 1, 65535)
    
    @classmethod
    def get_max_query_length(cls) -> int:
        return get_env_int("MAX_QUERY_LENGTH", ValidationDefaults.MAX_QUERY_LENGTH, 10, 10000)
    
    @classmethod
    def get_requests_per_minute(cls) -> int:
        return get_env_int("RATE_LIMIT_PER_MINUTE", RateLimitDefaults.REQUESTS_PER_MINUTE, 1, 1000)
    
    @classmethod
    def get_requests_per_hour(cls) -> int:
        return get_env_int("RATE_LIMIT_PER_HOUR", RateLimitDefaults.REQUESTS_PER_HOUR, 10, 10000)
    
    @classmethod
    def get_llm_timeout(cls) -> int:
        return get_env_int("LLM_TIMEOUT", LLMDefaults.DEFAULT_TIMEOUT_SECONDS, 5, 300)
    
    @classmethod
    def get_health_check_interval(cls) -> int:
        return get_env_int("HEALTH_CHECK_INTERVAL", MonitoringDefaults.HEALTH_CHECK_INTERVAL_SECONDS, 5, 3600)


if __name__ == "__main__":
    # Print all defaults for debugging
    import json
    defaults = get_all_defaults()
    print("Configuration Constants:")
    print(json.dumps(defaults, indent=2, sort_keys=True))