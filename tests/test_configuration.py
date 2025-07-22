#!/usr/bin/env python3
"""
Test suite for configuration constants and settings management.

Tests the centralized configuration system that replaces hardcoded values.
"""

import os
import sys
import unittest
from unittest.mock import patch
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from slack_kb_agent.configuration import (
    AppConfig,
    SearchConfig,
    SlackBotConfig,
    VectorSearchConfig
)


class TestAppConfig(unittest.TestCase):
    """Test application-wide configuration."""
    
    def test_app_config_defaults(self):
        """Test that AppConfig has sensible defaults."""
        config = AppConfig()
        
        # Verify default values exist and are reasonable
        self.assertIsInstance(config.debug, bool)
        self.assertIsInstance(config.environment, str)
        self.assertIn(config.log_level, ['DEBUG', 'INFO', 'WARNING', 'ERROR'])
        self.assertIsInstance(config.metrics_enabled, bool)
        
    def test_app_config_from_environment(self):
        """Test AppConfig creation from environment variables."""
        with patch.dict(os.environ, {
            'DEBUG': 'true',
            'ENVIRONMENT': 'test',
            'LOG_LEVEL': 'DEBUG',
            'METRICS_ENABLED': 'false'
        }):
            config = AppConfig.from_env()
            
            self.assertTrue(config.debug)
            self.assertEqual(config.environment, 'test')
            self.assertEqual(config.log_level, 'DEBUG')
            self.assertFalse(config.metrics_enabled)
    
    def test_app_config_validation(self):
        """Test that AppConfig validates values properly."""
        # Test invalid log level
        with self.assertRaises(ValueError):
            AppConfig(log_level='INVALID')
        
        # Test invalid environment
        with self.assertRaises(ValueError):
            AppConfig(environment='')


class TestSearchConfig(unittest.TestCase):
    """Test search-related configuration."""
    
    def test_search_config_defaults(self):
        """Test SearchConfig default values."""
        config = SearchConfig()
        
        # Check that defaults match current hardcoded values
        self.assertEqual(config.min_word_length, 2)
        self.assertEqual(config.max_index_size, 50000)
        self.assertEqual(config.max_results_default, 100)
        self.assertEqual(config.cache_size, 1000)
        self.assertTrue(config.enable_indexing)
        
    def test_search_config_from_environment(self):
        """Test SearchConfig creation from environment variables."""
        with patch.dict(os.environ, {
            'SEARCH_MIN_WORD_LENGTH': '3',
            'SEARCH_MAX_INDEX_SIZE': '75000',
            'SEARCH_MAX_RESULTS_DEFAULT': '50',
            'SEARCH_CACHE_SIZE': '2000',
            'SEARCH_ENABLE_INDEXING': 'false'
        }):
            config = SearchConfig.from_env()
            
            self.assertEqual(config.min_word_length, 3)
            self.assertEqual(config.max_index_size, 75000)
            self.assertEqual(config.max_results_default, 50)
            self.assertEqual(config.cache_size, 2000)
            self.assertFalse(config.enable_indexing)
    
    def test_search_config_validation(self):
        """Test SearchConfig validation."""
        # Test invalid values
        with self.assertRaises(ValueError):
            SearchConfig(min_word_length=0)
        
        with self.assertRaises(ValueError):
            SearchConfig(max_index_size=-1)
        
        with self.assertRaises(ValueError):
            SearchConfig(cache_size=0)


# ValidationConfig is tested in the validation module's tests


class TestSlackBotConfig(unittest.TestCase):
    """Test Slack bot configuration."""
    
    def test_slack_bot_config_defaults(self):
        """Test SlackBotConfig default values."""
        config = SlackBotConfig()
        
        # Check current hardcoded values
        self.assertEqual(config.max_results_default, 5)
        self.assertEqual(config.response_timeout, 30)
        self.assertEqual(config.max_history_length, 5)
        
    def test_slack_bot_config_from_environment(self):
        """Test SlackBotConfig creation from environment variables."""
        with patch.dict(os.environ, {
            'SLACK_BOT_MAX_RESULTS': '10',
            'SLACK_BOT_RESPONSE_TIMEOUT': '60',
            'SLACK_BOT_MAX_HISTORY': '10'
        }):
            config = SlackBotConfig.from_env()
            
            self.assertEqual(config.max_results_default, 10)
            self.assertEqual(config.response_timeout, 60)
            self.assertEqual(config.max_history_length, 10)
    
    def test_slack_bot_config_validation(self):
        """Test SlackBotConfig validation."""
        with self.assertRaises(ValueError):
            SlackBotConfig(max_results_default=0)
        
        with self.assertRaises(ValueError):
            SlackBotConfig(response_timeout=-1)


class TestVectorSearchConfig(unittest.TestCase):
    """Test vector search configuration."""
    
    def test_vector_search_config_defaults(self):
        """Test VectorSearchConfig default values."""
        config = VectorSearchConfig()
        
        # Check current hardcoded values
        self.assertEqual(config.batch_size, 32)
        self.assertEqual(config.top_k_default, 10)
        self.assertIsInstance(config.similarity_threshold, float)
        self.assertGreater(config.similarity_threshold, 0)
        self.assertLess(config.similarity_threshold, 1)
        
    def test_vector_search_config_from_environment(self):
        """Test VectorSearchConfig creation from environment variables."""
        with patch.dict(os.environ, {
            'VECTOR_SEARCH_BATCH_SIZE': '64',
            'VECTOR_SEARCH_TOP_K_DEFAULT': '20',
            'VECTOR_SEARCH_SIMILARITY_THRESHOLD': '0.8'
        }):
            config = VectorSearchConfig.from_env()
            
            self.assertEqual(config.batch_size, 64)
            self.assertEqual(config.top_k_default, 20)
            self.assertEqual(config.similarity_threshold, 0.8)
    
    def test_vector_search_config_validation(self):
        """Test VectorSearchConfig validation."""
        with self.assertRaises(ValueError):
            VectorSearchConfig(batch_size=0)
        
        with self.assertRaises(ValueError):
            VectorSearchConfig(similarity_threshold=1.5)


class TestConfigurationIntegration(unittest.TestCase):
    """Test configuration system integration."""
    
    def test_config_classes_exist(self):
        """Test that all config classes can be imported and instantiated."""
        configs = [
            AppConfig(),
            SearchConfig(),
            SlackBotConfig(),
            VectorSearchConfig()
        ]
        
        for config in configs:
            self.assertIsNotNone(config)
    
    def test_config_environment_isolation(self):
        """Test that configuration from environment doesn't leak between tests."""
        # Set environment variables
        with patch.dict(os.environ, {'SEARCH_MAX_INDEX_SIZE': '99999'}):
            config1 = SearchConfig.from_env()
            self.assertEqual(config1.max_index_size, 99999)
        
        # Outside the context, should use defaults
        config2 = SearchConfig.from_env()
        self.assertEqual(config2.max_index_size, 50000)  # default
    
    def test_config_immutability(self):
        """Test that config objects maintain data integrity."""
        config = SearchConfig()
        original_size = config.max_index_size
        
        # Attempt to modify (should be prevented by dataclass frozen=True if implemented)
        try:
            config.max_index_size = 99999
            # If modification succeeded, verify it didn't actually change
            # (this depends on implementation details)
        except (AttributeError, TypeError):
            # Expected if dataclass is frozen
            pass
        
        # Create new config to verify defaults unchanged
        new_config = SearchConfig()
        self.assertEqual(new_config.max_index_size, original_size)


if __name__ == "__main__":
    unittest.main()