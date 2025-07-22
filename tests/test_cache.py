#!/usr/bin/env python3
"""
Test suite for Redis-based caching layer.

Tests the caching functionality for embeddings, query expansions, and search results.
"""

import os
import sys
import unittest
import json
import time
from unittest.mock import patch, MagicMock
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Mock redis before importing cache module
redis_mock = MagicMock()
with patch.dict(sys.modules, {'redis': redis_mock}):
    from slack_kb_agent.cache import CacheManager, CacheConfig, get_cache_manager, is_cache_available
    from slack_kb_agent.models import Document


class TestCacheConfig(unittest.TestCase):
    """Test cache configuration."""
    
    def test_cache_config_creation(self):
        """Test CacheConfig creation with defaults."""
        config = CacheConfig()
        self.assertTrue(config.enabled)
        self.assertIsNone(config.host)  # No default host for security
        self.assertEqual(config.port, 6379)
        self.assertEqual(config.db, 0)
        self.assertIsNone(config.password)
        self.assertEqual(config.embedding_ttl, 3600 * 24 * 7)  # 7 days
        self.assertEqual(config.query_expansion_ttl, 3600 * 24)  # 1 day
        self.assertEqual(config.search_results_ttl, 3600)  # 1 hour
    
    def test_cache_config_from_env(self):
        """Test CacheConfig creation from environment variables."""
        with patch.dict(os.environ, {
            'CACHE_ENABLED': 'false',
            'REDIS_HOST': 'redis.example.com',
            'REDIS_PORT': '6380',
            'REDIS_DB': '2',
            'REDIS_PASSWORD': 'secret',
            'CACHE_EMBEDDING_TTL': '86400',
            'CACHE_QUERY_EXPANSION_TTL': '7200',
            'CACHE_SEARCH_RESULTS_TTL': '1800'
        }):
            config = CacheConfig.from_env()
            self.assertFalse(config.enabled)
            self.assertEqual(config.host, "redis.example.com")
            self.assertEqual(config.port, 6380)
            self.assertEqual(config.db, 2)
            self.assertEqual(config.password, "secret")
            self.assertEqual(config.embedding_ttl, 86400)
            self.assertEqual(config.query_expansion_ttl, 7200)
            self.assertEqual(config.search_results_ttl, 1800)
    
    def test_cache_config_requires_host_when_enabled(self):
        """Test that Redis host is required when caching is enabled."""
        # Test with caching enabled but no host
        with patch.dict(os.environ, {'CACHE_ENABLED': 'true'}, clear=True):
            with self.assertRaises(ValueError) as cm:
                CacheConfig.from_env()
            self.assertIn("REDIS_HOST environment variable is required", str(cm.exception))
        
        # Test with caching disabled - should not require host
        with patch.dict(os.environ, {'CACHE_ENABLED': 'false'}, clear=True):
            config = CacheConfig.from_env()
            self.assertFalse(config.enabled)
            self.assertIsNone(config.host)
        
        # Test with caching enabled and host provided - should work
        with patch.dict(os.environ, {'CACHE_ENABLED': 'true', 'REDIS_HOST': 'localhost'}):
            config = CacheConfig.from_env()
            self.assertTrue(config.enabled)
            self.assertEqual(config.host, 'localhost')


class TestCacheManagerMocked(unittest.TestCase):
    """Test CacheManager with mocked Redis."""
    
    def setUp(self):
        """Set up test fixtures with mocked Redis."""
        # Create mock Redis client
        self.mock_redis = MagicMock()
        self.mock_pool = MagicMock()
        
        # Configure mocks
        redis_mock.ConnectionPool.return_value = self.mock_pool
        redis_mock.Redis.return_value = self.mock_redis
        self.mock_redis.ping.return_value = True
        
        # Create cache manager with test config
        config = CacheConfig(enabled=True, host="localhost", port=6379)
        
        # Mock the redis availability check
        with patch('slack_kb_agent.cache.REDIS_AVAILABLE', True):
            self.cache_manager = CacheManager(config)
    
    def test_cache_manager_initialization(self):
        """Test CacheManager initialization."""
        self.assertTrue(self.cache_manager.is_available())
        self.mock_redis.ping.assert_called_once()
    
    def test_generate_key(self):
        """Test cache key generation."""
        key = self.cache_manager._generate_key("test", "identifier")
        self.assertEqual(key, "slack_kb:test:identifier")
        
        # Test long identifier hashing
        long_id = "x" * 300
        key = self.cache_manager._generate_key("test", long_id)
        self.assertTrue(key.startswith("slack_kb:test:"))
        self.assertLess(len(key), 300)  # Should be hashed
    
    def test_embedding_cache_operations(self):
        """Test embedding caching operations."""
        import numpy as np
        
        text = "test document"
        model_name = "test-model"
        embedding = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        
        # Test cache miss
        self.mock_redis.get.return_value = None
        result = self.cache_manager.get_embedding(text, model_name)
        self.assertIsNone(result)
        
        # Test cache set
        self.cache_manager.set_embedding(text, model_name, embedding)
        self.mock_redis.setex.assert_called()
        
        # Test cache hit - simulate the new secure serialization format
        # Create the expected serialized format (JSON with base64 data)
        import base64
        metadata = {
            "dtype": str(embedding.dtype),
            "shape": embedding.shape,
            "data": base64.b64encode(embedding.tobytes()).decode('ascii')
        }
        serialized_data = json.dumps(metadata).encode('utf-8')
        self.mock_redis.get.return_value = serialized_data
        result = self.cache_manager.get_embedding(text, model_name)
        np.testing.assert_array_equal(result, embedding)
    
    def test_query_expansion_cache_operations(self):
        """Test query expansion caching operations."""
        query = "test query"
        expansion_type = "synonyms"
        expansion = ["test", "query", "search", "find"]
        
        # Test cache miss
        self.mock_redis.get.return_value = None
        result = self.cache_manager.get_query_expansion(query, expansion_type)
        self.assertIsNone(result)
        
        # Test cache set
        self.cache_manager.set_query_expansion(query, expansion_type, expansion)
        self.mock_redis.setex.assert_called()
        
        # Test cache hit
        self.mock_redis.get.return_value = json.dumps(expansion).encode()
        result = self.cache_manager.get_query_expansion(query, expansion_type)
        self.assertEqual(result, expansion)
    
    def test_search_results_cache_operations(self):
        """Test search results caching operations."""
        query_hash = "test_hash_123"
        results = [
            {
                "document": {
                    "id": "doc1",
                    "title": "Test Doc",
                    "content": "Test content",
                    "source": "test",
                    "metadata": {}
                },
                "score": 0.95
            }
        ]
        
        # Test cache miss
        self.mock_redis.get.return_value = None
        result = self.cache_manager.get_search_results(query_hash)
        self.assertIsNone(result)
        
        # Test cache set
        self.cache_manager.set_search_results(query_hash, results)
        self.mock_redis.setex.assert_called()
        
        # Test cache hit
        self.mock_redis.get.return_value = json.dumps(results).encode()
        result = self.cache_manager.get_search_results(query_hash)
        self.assertEqual(result, results)
    
    def test_search_hash_generation(self):
        """Test search hash generation."""
        query = "test query"
        params = {"model": "test", "top_k": 10, "threshold": 0.5}
        
        hash1 = self.cache_manager.generate_search_hash(query, params)
        hash2 = self.cache_manager.generate_search_hash(query, params)
        
        # Same input should produce same hash
        self.assertEqual(hash1, hash2)
        
        # Different input should produce different hash
        different_params = {"model": "different", "top_k": 10, "threshold": 0.5}
        hash3 = self.cache_manager.generate_search_hash(query, different_params)
        self.assertNotEqual(hash1, hash3)
    
    def test_cache_invalidation(self):
        """Test cache invalidation."""
        # Mock keys method
        test_keys = [b"slack_kb:search_results:key1", b"slack_kb:search_results:key2"]
        self.mock_redis.keys.return_value = test_keys
        self.mock_redis.delete.return_value = 2
        
        invalidated = self.cache_manager.invalidate_search_cache()
        self.assertEqual(invalidated, 2)
        self.mock_redis.keys.assert_called_with("slack_kb:search_results:*")
        self.mock_redis.delete.assert_called_with(*test_keys)
    
    def test_cache_stats(self):
        """Test cache statistics collection."""
        # Mock Redis info
        mock_info = {
            "redis_version": "6.2.0",
            "used_memory_human": "10M",
            "connected_clients": 5,
            "total_commands_processed": 1000,
            "keyspace_hits": 800,
            "keyspace_misses": 200
        }
        self.mock_redis.info.return_value = mock_info
        
        # Mock keys for different namespaces
        self.mock_redis.keys.side_effect = [
            [b"key1", b"key2"],  # embedding keys
            [b"key3"],           # query_expansion keys
            [b"key4", b"key5", b"key6"]  # search_results keys
        ]
        
        stats = self.cache_manager.get_cache_stats()
        
        self.assertEqual(stats["status"], "connected")
        self.assertEqual(stats["redis_version"], "6.2.0")
        self.assertEqual(stats["hit_rate"], 80.0)  # 800/(800+200)*100
        self.assertEqual(stats["key_counts"]["embedding"], 2)
        self.assertEqual(stats["key_counts"]["query_expansion"], 1)
        self.assertEqual(stats["key_counts"]["search_results"], 3)
    
    def test_cache_flush(self):
        """Test cache flushing."""
        # Test namespace-specific flush
        test_keys = [b"slack_kb:embedding:key1", b"slack_kb:embedding:key2"]
        self.mock_redis.keys.return_value = test_keys
        self.mock_redis.delete.return_value = 2
        
        flushed = self.cache_manager.flush_cache("embedding")
        self.assertEqual(flushed, 2)
        self.mock_redis.keys.assert_called_with("slack_kb:embedding:*")
        
        # Test full flush
        all_keys = [b"slack_kb:test:key1", b"slack_kb:test:key2", b"slack_kb:test:key3"]
        self.mock_redis.keys.return_value = all_keys
        self.mock_redis.delete.return_value = 3
        
        flushed = self.cache_manager.flush_cache()
        self.assertEqual(flushed, 3)
        self.mock_redis.keys.assert_called_with("slack_kb:*")
    
    def test_cache_error_handling(self):
        """Test cache error handling."""
        # Test Redis connection error
        self.mock_redis.get.side_effect = Exception("Redis connection error")
        
        result = self.cache_manager.get_embedding("test", "model")
        self.assertIsNone(result)  # Should fail gracefully
        
        # Test set error
        self.mock_redis.setex.side_effect = Exception("Redis write error")
        
        # Should not raise exception
        self.cache_manager.set_embedding("test", "model", [0.1, 0.2])


class TestCacheManagerDisabled(unittest.TestCase):
    """Test CacheManager when Redis is not available."""
    
    def setUp(self):
        """Set up test fixtures with disabled cache."""
        config = CacheConfig(enabled=False)
        
        # Mock Redis as unavailable
        with patch('slack_kb_agent.cache.REDIS_AVAILABLE', False):
            self.cache_manager = CacheManager(config)
    
    def test_cache_unavailable(self):
        """Test cache operations when Redis is unavailable."""
        self.assertFalse(self.cache_manager.is_available())
        
        # All operations should return None/empty gracefully
        self.assertIsNone(self.cache_manager.get_embedding("test", "model"))
        
        # Set operations should not crash
        self.cache_manager.set_embedding("test", "model", [0.1, 0.2])
        
        # Stats should indicate unavailable
        stats = self.cache_manager.get_cache_stats()
        self.assertEqual(stats["status"], "unavailable")


class TestGlobalCacheManager(unittest.TestCase):
    """Test global cache manager functions."""
    
    def test_get_cache_manager_singleton(self):
        """Test that get_cache_manager returns a singleton."""
        with patch('slack_kb_agent.cache.REDIS_AVAILABLE', True):
            manager1 = get_cache_manager()
            manager2 = get_cache_manager()
            self.assertIs(manager1, manager2)
    
    def test_is_cache_available(self):
        """Test cache availability check."""
        with patch('slack_kb_agent.cache.REDIS_AVAILABLE', False):
            self.assertFalse(is_cache_available())
        
        with patch('slack_kb_agent.cache.REDIS_AVAILABLE', True):
            # Would need actual Redis connection for True case
            pass


class TestCacheSecurityFeatures(unittest.TestCase):
    """Test security features of the cache implementation."""
    
    def setUp(self):
        """Set up security test fixtures."""
        self.mock_redis = MagicMock()
        redis_mock.ConnectionPool.return_value = MagicMock()
        redis_mock.Redis.return_value = self.mock_redis
        self.mock_redis.ping.return_value = True
        
        config = CacheConfig(enabled=True, host="localhost", port=6379)
        with patch('slack_kb_agent.cache.REDIS_AVAILABLE', True):
            self.cache_manager = CacheManager(config)
    
    def test_secure_embedding_serialization(self):
        """Test that embedding serialization uses secure JSON format instead of pickle."""
        import numpy as np
        import base64
        
        # Test various numpy array types and shapes
        test_cases = [
            np.array([0.1, 0.2, 0.3], dtype=np.float32),
            np.array([[1, 2], [3, 4]], dtype=np.int32),
            np.array([0.123456789], dtype=np.float64),
            np.zeros(100, dtype=np.float32),
        ]
        
        for embedding in test_cases:
            with self.subTest(embedding=embedding):
                # Test serialization
                serialized = self.cache_manager._serialize_embedding(embedding)
                self.assertIsInstance(serialized, bytes)
                
                # Verify it's JSON format (not pickle)
                try:
                    metadata = json.loads(serialized.decode('utf-8'))
                    self.assertIn('dtype', metadata)
                    self.assertIn('shape', metadata)
                    self.assertIn('data', metadata)
                except json.JSONDecodeError:
                    self.fail("Serialized data is not valid JSON")
                
                # Test deserialization
                deserialized = self.cache_manager._deserialize_embedding(serialized)
                np.testing.assert_array_equal(embedding, deserialized)
                self.assertEqual(embedding.dtype, deserialized.dtype)
                self.assertEqual(embedding.shape, deserialized.shape)
    
    def test_malicious_embedding_data_rejection(self):
        """Test that malicious embedding data is rejected during deserialization."""
        import base64
        
        # Test cases for malicious data
        malicious_data_cases = [
            # Invalid JSON
            b"invalid json data",
            
            # Missing required fields
            json.dumps({"dtype": "float32"}).encode(),
            json.dumps({"shape": [3]}).encode(),
            json.dumps({"data": "base64data"}).encode(),
            
            # Invalid base64 data
            json.dumps({
                "dtype": "float32",
                "shape": [3],
                "data": "invalid_base64!"
            }).encode(),
            
            # Mismatched data size
            json.dumps({
                "dtype": "float32", 
                "shape": [10],
                "data": base64.b64encode(b"short").decode()
            }).encode(),
            
            # Invalid dtype
            json.dumps({
                "dtype": "invalid_dtype",
                "shape": [3],
                "data": base64.b64encode(b"12345678901234567890").decode()
            }).encode(),
        ]
        
        for malicious_data in malicious_data_cases:
            with self.subTest(data=malicious_data):
                with self.assertRaises(ValueError):
                    self.cache_manager._deserialize_embedding(malicious_data)
    
    def test_embedding_cache_corruption_handling(self):
        """Test that corrupted cache entries are handled gracefully."""
        import numpy as np
        
        text = "test document"
        model_name = "test-model"
        
        # Test with corrupted cache data
        self.mock_redis.get.return_value = b"corrupted_data_not_json"
        result = self.cache_manager.get_embedding(text, model_name)
        self.assertIsNone(result)
        
        # Verify corrupted entry was deleted
        self.mock_redis.delete.assert_called()
    
    def test_no_code_execution_in_cache_data(self):
        """Test that cache data cannot execute arbitrary code."""
        import numpy as np
        
        # Create a normal embedding
        embedding = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        
        # Serialize it safely
        serialized = self.cache_manager._serialize_embedding(embedding)
        
        # Verify the serialized data contains no executable code patterns
        serialized_str = serialized.decode('utf-8')
        
        # Check for common code execution patterns
        dangerous_patterns = [
            'import', 'eval', 'exec', '__', 'pickle', 'subprocess',
            'os.system', 'compile', 'globals', 'locals'
        ]
        
        for pattern in dangerous_patterns:
            self.assertNotIn(pattern, serialized_str.lower(),
                           f"Serialized data contains dangerous pattern: {pattern}")
        
        # Verify it's just JSON with base64 data
        metadata = json.loads(serialized_str)
        self.assertIsInstance(metadata['dtype'], str)
        self.assertIsInstance(metadata['shape'], list)
        self.assertIsInstance(metadata['data'], str)
        
        # Verify base64 data contains only valid base64 characters
        import re
        base64_pattern = re.compile(r'^[A-Za-z0-9+/]*={0,2}$')
        self.assertTrue(base64_pattern.match(metadata['data']),
                       "Base64 data contains invalid characters")
    
    def test_embedding_type_validation(self):
        """Test that only numpy arrays are accepted for serialization."""
        invalid_inputs = [
            [0.1, 0.2, 0.3],  # Python list
            "not an array",    # String
            42,               # Integer
            {"array": [1, 2, 3]},  # Dictionary
            None,             # None
        ]
        
        for invalid_input in invalid_inputs:
            with self.subTest(input=invalid_input):
                with self.assertRaises(TypeError):
                    self.cache_manager._serialize_embedding(invalid_input)


class TestCacheIntegration(unittest.TestCase):
    """Integration tests for cache with other components."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        # Mock redis for integration tests
        self.mock_redis = MagicMock()
        redis_mock.ConnectionPool.return_value = MagicMock()
        redis_mock.Redis.return_value = self.mock_redis
        self.mock_redis.ping.return_value = True
    
    def test_document_serialization_roundtrip(self):
        """Test that Document objects can be serialized and deserialized."""
        document = Document(
            content="This is test content",
            source="test_source",
            metadata={"category": "test", "priority": 1}
        )
        
        # Simulate serialization (what would happen in cache)
        doc_dict = {
            "content": document.content,
            "source": document.source,
            "metadata": document.metadata
        }
        
        serialized = json.dumps(doc_dict)
        deserialized = json.loads(serialized)
        
        # Reconstruct Document
        reconstructed = Document(
            content=deserialized["content"],
            source=deserialized["source"],
            metadata=deserialized.get("metadata", {})
        )
        
        # Verify roundtrip integrity
        self.assertEqual(document.content, reconstructed.content)
        self.assertEqual(document.source, reconstructed.source)
        self.assertEqual(document.metadata, reconstructed.metadata)


if __name__ == "__main__":
    unittest.main()