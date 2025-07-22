"""
Centralized configuration management for the Slack KB Agent.

This module consolidates all hardcoded values into configurable settings,
improving maintainability and deployment flexibility.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class AppConfig:
    """Application-wide configuration settings."""
    
    debug: bool = False
    environment: str = "production"
    log_level: str = "INFO"
    metrics_enabled: bool = True
    
    def __post_init__(self):
        """Validate configuration values."""
        valid_log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if self.log_level not in valid_log_levels:
            raise ValueError(f"Invalid log_level: {self.log_level}. Must be one of {valid_log_levels}")
        
        if not self.environment or not self.environment.strip():
            raise ValueError("Environment cannot be empty")
    
    @classmethod
    def from_env(cls) -> AppConfig:
        """Create configuration from environment variables."""
        return cls(
            debug=os.getenv("DEBUG", "false").lower() == "true",
            environment=os.getenv("ENVIRONMENT", "production"),
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
            metrics_enabled=os.getenv("METRICS_ENABLED", "true").lower() == "true"
        )


@dataclass(frozen=True)
class SearchConfig:
    """Search engine configuration settings."""
    
    min_word_length: int = 2
    max_index_size: int = 50000
    max_results_default: int = 100
    cache_size: int = 1000
    enable_indexing: bool = True
    
    def __post_init__(self):
        """Validate configuration values."""
        if self.min_word_length <= 0:
            raise ValueError("min_word_length must be positive")
        
        if self.max_index_size < 0:
            raise ValueError("max_index_size cannot be negative")
        
        if self.max_results_default <= 0:
            raise ValueError("max_results_default must be positive")
        
        if self.cache_size <= 0:
            raise ValueError("cache_size must be positive")
    
    @classmethod
    def from_env(cls) -> SearchConfig:
        """Create configuration from environment variables."""
        return cls(
            min_word_length=int(os.getenv("SEARCH_MIN_WORD_LENGTH", "2")),
            max_index_size=int(os.getenv("SEARCH_MAX_INDEX_SIZE", "50000")),
            max_results_default=int(os.getenv("SEARCH_MAX_RESULTS_DEFAULT", "100")),
            cache_size=int(os.getenv("SEARCH_CACHE_SIZE", "1000")),
            enable_indexing=os.getenv("SEARCH_ENABLE_INDEXING", "true").lower() == "true"
        )


# Note: ValidationConfig exists in validation.py - we'll extend it rather than replace it


@dataclass(frozen=True)
class SlackBotConfig:
    """Slack bot configuration settings."""
    
    max_results_default: int = 5
    response_timeout: int = 30
    max_history_length: int = 5
    
    def __post_init__(self):
        """Validate configuration values."""
        if self.max_results_default <= 0:
            raise ValueError("max_results_default must be positive")
        
        if self.response_timeout <= 0:
            raise ValueError("response_timeout must be positive")
        
        if self.max_history_length < 0:
            raise ValueError("max_history_length cannot be negative")
    
    @classmethod
    def from_env(cls) -> SlackBotConfig:
        """Create configuration from environment variables."""
        return cls(
            max_results_default=int(os.getenv("SLACK_BOT_MAX_RESULTS", "5")),
            response_timeout=int(os.getenv("SLACK_BOT_RESPONSE_TIMEOUT", "30")),
            max_history_length=int(os.getenv("SLACK_BOT_MAX_HISTORY", "5"))
        )


@dataclass(frozen=True)
class VectorSearchConfig:
    """Vector search configuration settings."""
    
    batch_size: int = 32
    top_k_default: int = 10
    similarity_threshold: float = 0.5
    
    def __post_init__(self):
        """Validate configuration values."""
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        
        if self.top_k_default <= 0:
            raise ValueError("top_k_default must be positive")
        
        if not (0 < self.similarity_threshold < 1):
            raise ValueError("similarity_threshold must be between 0 and 1")
    
    @classmethod
    def from_env(cls) -> VectorSearchConfig:
        """Create configuration from environment variables."""
        return cls(
            batch_size=int(os.getenv("VECTOR_SEARCH_BATCH_SIZE", "32")),
            top_k_default=int(os.getenv("VECTOR_SEARCH_TOP_K_DEFAULT", "10")),
            similarity_threshold=float(os.getenv("VECTOR_SEARCH_SIMILARITY_THRESHOLD", "0.5"))
        )


# Global configuration instances (initialized lazily)
_app_config: Optional[AppConfig] = None
_search_config: Optional[SearchConfig] = None
_slack_bot_config: Optional[SlackBotConfig] = None
_vector_search_config: Optional[VectorSearchConfig] = None


def get_app_config() -> AppConfig:
    """Get global application configuration."""
    global _app_config
    if _app_config is None:
        _app_config = AppConfig.from_env()
    return _app_config


def get_search_config() -> SearchConfig:
    """Get global search configuration."""
    global _search_config
    if _search_config is None:
        _search_config = SearchConfig.from_env()
    return _search_config


# Note: Validation config is managed in validation.py module


def get_slack_bot_config() -> SlackBotConfig:
    """Get global Slack bot configuration."""
    global _slack_bot_config
    if _slack_bot_config is None:
        _slack_bot_config = SlackBotConfig.from_env()
    return _slack_bot_config


def get_vector_search_config() -> VectorSearchConfig:
    """Get global vector search configuration."""
    global _vector_search_config
    if _vector_search_config is None:
        _vector_search_config = VectorSearchConfig.from_env()
    return _vector_search_config


def reset_config_cache() -> None:
    """Reset configuration cache. Useful for testing."""
    global _app_config, _search_config, _slack_bot_config, _vector_search_config
    _app_config = None
    _search_config = None
    _slack_bot_config = None
    _vector_search_config = None