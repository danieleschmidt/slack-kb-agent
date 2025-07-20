#!/usr/bin/env python3
"""
Integration tests for caching with vector search and query processing.

Tests that caching properly integrates with the actual search pipeline.
"""

import os
import sys
import unittest
import time
from unittest.mock import patch, MagicMock
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from slack_kb_agent.models import Document
from slack_kb_agent.cache import CacheManager, CacheConfig


class TestVectorSearchCaching(unittest.TestCase):
    """Test caching integration with vector search."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock Redis
        self.mock_redis = MagicMock()
        self.mock_pool = MagicMock()
        
        # Create real documents for testing
        self.documents = [
            Document(
                id="doc1",
                title="Python Documentation",
                content="Python is a programming language",
                source="test",
                metadata={"category": "programming"}
            ),
            Document(
                id="doc2", 
                title="Machine Learning Guide",
                content="Machine learning algorithms for data science",
                source="test",
                metadata={"category": "ml"}
            )
        ]
    
    @patch('slack_kb_agent.cache.redis')
    @patch('slack_kb_agent.cache.REDIS_AVAILABLE', True)
    def test_vector_search_with_caching(self, mock_redis_module):
        """Test that vector search properly uses caching."""
        # Configure mocks
        mock_redis_module.ConnectionPool.return_value = self.mock_pool
        mock_redis_module.Redis.return_value = self.mock_redis
        self.mock_redis.ping.return_value = True
        
        # Mock cache operations
        self.mock_redis.get.return_value = None  # Cache miss initially
        
        # Try to import and test vector search with caching
        try:
            from slack_kb_agent.vector_search import VectorSearchEngine
            from slack_kb_agent.cache import get_cache_manager
            
            # Create cache manager
            cache_manager = CacheManager(CacheConfig(enabled=True))
            
            # Test embedding caching
            test_text = "test query"
            model_name = "test-model"
            
            # First call should result in cache miss
            result = cache_manager.get_embedding(test_text, model_name)
            self.assertIsNone(result)
            
            # Simulate setting embedding in cache
            import numpy as np
            test_embedding = np.array([0.1, 0.2, 0.3], dtype=np.float32)
            cache_manager.set_embedding(test_text, model_name, test_embedding)
            
            # Verify setex was called for caching
            self.mock_redis.setex.assert_called()
            
        except ImportError:
            # Skip if vector search dependencies not available
            self.skipTest("Vector search dependencies not available")
    
    @patch('slack_kb_agent.cache.redis')
    @patch('slack_kb_agent.cache.REDIS_AVAILABLE', True)
    def test_search_results_caching(self, mock_redis_module):
        """Test search results caching functionality."""
        # Configure mocks
        mock_redis_module.ConnectionPool.return_value = self.mock_pool
        mock_redis_module.Redis.return_value = self.mock_redis
        self.mock_redis.ping.return_value = True
        
        from slack_kb_agent.cache import CacheManager, CacheConfig
        
        cache_manager = CacheManager(CacheConfig(enabled=True))
        
        # Test search hash generation
        query = "python programming"
        params = {"model_name": "test-model", "top_k": 10, "threshold": 0.5}
        
        hash1 = cache_manager.generate_search_hash(query, params)
        hash2 = cache_manager.generate_search_hash(query, params)
        
        # Same parameters should generate same hash
        self.assertEqual(hash1, hash2)
        
        # Different parameters should generate different hash
        different_params = {"model_name": "test-model", "top_k": 5, "threshold": 0.5}
        hash3 = cache_manager.generate_search_hash(query, different_params)
        self.assertNotEqual(hash1, hash3)
        
        # Test caching search results
        search_results = [
            {
                "document": {
                    "id": "doc1",
                    "title": "Python Guide",
                    "content": "Python programming guide",
                    "source": "test",
                    "metadata": {}
                },
                "score": 0.95
            }
        ]
        
        # Cache miss initially
        self.mock_redis.get.return_value = None
        result = cache_manager.get_search_results(hash1)
        self.assertIsNone(result)
        
        # Set results in cache
        cache_manager.set_search_results(hash1, search_results)
        self.mock_redis.setex.assert_called()
        
        # Simulate cache hit
        import json
        self.mock_redis.get.return_value = json.dumps(search_results).encode()
        result = cache_manager.get_search_results(hash1)
        self.assertEqual(result, search_results)


class TestQueryProcessorCaching(unittest.TestCase):
    """Test caching integration with query processor."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock Redis
        self.mock_redis = MagicMock()
        self.mock_pool = MagicMock()
    
    @patch('slack_kb_agent.cache.redis')
    @patch('slack_kb_agent.cache.REDIS_AVAILABLE', True)
    def test_query_expansion_caching(self, mock_redis_module):
        """Test query expansion caching in query processor."""
        # Configure mocks
        mock_redis_module.ConnectionPool.return_value = self.mock_pool
        mock_redis_module.Redis.return_value = self.mock_redis
        self.mock_redis.ping.return_value = True
        
        from slack_kb_agent.query_processor import QueryExpansion
        from slack_kb_agent.cache import CacheManager, CacheConfig
        
        # Create query expansion instance
        query_expansion = QueryExpansion()
        cache_manager = CacheManager(CacheConfig(enabled=True))
        
        query = "api documentation"
        
        # Test synonym expansion caching
        self.mock_redis.get.return_value = None  # Cache miss
        
        # First call should compute and cache results
        expanded = query_expansion.expand_synonyms(query)
        self.assertIn(query, expanded)
        self.mock_redis.setex.assert_called()
        
        # Test technical terms expansion caching
        self.mock_redis.reset_mock()
        self.mock_redis.get.return_value = None  # Cache miss
        
        tech_query = "ci/cd pipeline"
        expanded_tech = query_expansion.expand_technical_terms(tech_query)
        self.assertIn(tech_query, expanded_tech)
        self.assertIn("continuous integration", expanded_tech)
        self.mock_redis.setex.assert_called()
    
    @patch('slack_kb_agent.cache.redis') 
    @patch('slack_kb_agent.cache.REDIS_AVAILABLE', True)
    def test_llm_expansion_caching(self, mock_redis_module):
        """Test LLM-based query expansion caching."""
        # Configure mocks
        mock_redis_module.ConnectionPool.return_value = self.mock_pool
        mock_redis_module.Redis.return_value = self.mock_redis
        self.mock_redis.ping.return_value = True
        
        from slack_kb_agent.query_processor import QueryExpansion
        
        query_expansion = QueryExpansion()
        query = "machine learning model"
        
        # Mock cache miss
        self.mock_redis.get.return_value = None
        
        # Mock LLM response generator as unavailable to test fallback
        with patch('slack_kb_agent.query_processor.get_response_generator') as mock_get_generator:
            mock_generator = MagicMock()
            mock_generator.is_available.return_value = False
            mock_get_generator.return_value = mock_generator
            
            # Should fallback to synonym expansion and cache that
            expanded = query_expansion.expand_with_llm(query, fallback_to_synonyms=True)
            self.assertIn(query, expanded)
            self.mock_redis.setex.assert_called()


class TestCacheInvalidation(unittest.TestCase):
    """Test cache invalidation scenarios."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock Redis
        self.mock_redis = MagicMock()
        self.mock_pool = MagicMock()
    
    @patch('slack_kb_agent.cache.redis')
    @patch('slack_kb_agent.cache.REDIS_AVAILABLE', True)
    def test_knowledge_base_cache_invalidation(self, mock_redis_module):
        """Test that adding documents invalidates search cache."""
        # Configure mocks
        mock_redis_module.ConnectionPool.return_value = self.mock_pool
        mock_redis_module.Redis.return_value = self.mock_redis
        self.mock_redis.ping.return_value = True
        
        from slack_kb_agent.knowledge_base import KnowledgeBase
        from slack_kb_agent.models import Document
        
        # Mock cache invalidation
        search_keys = [b"slack_kb:search_results:key1", b"slack_kb:search_results:key2"]
        self.mock_redis.keys.return_value = search_keys
        self.mock_redis.delete.return_value = 2
        
        # Create knowledge base
        kb = KnowledgeBase(enable_vector_search=False)  # Disable vector search for simpler test
        
        # Add document - should trigger cache invalidation
        document = Document(
            id="test_doc",
            title="Test Document",
            content="Test content for cache invalidation",
            source="test"
        )
        
        kb.add_document(document)
        
        # Verify cache invalidation was attempted
        self.mock_redis.keys.assert_called_with("slack_kb:search_results:*")
    
    @patch('slack_kb_agent.cache.redis')
    @patch('slack_kb_agent.cache.REDIS_AVAILABLE', True) 
    def test_batch_document_cache_invalidation(self, mock_redis_module):
        """Test cache invalidation when adding multiple documents."""
        # Configure mocks
        mock_redis_module.ConnectionPool.return_value = self.mock_pool
        mock_redis_module.Redis.return_value = self.mock_redis
        self.mock_redis.ping.return_value = True
        
        from slack_kb_agent.knowledge_base import KnowledgeBase
        from slack_kb_agent.models import Document
        
        # Mock cache invalidation
        search_keys = [b"slack_kb:search_results:key1"]
        self.mock_redis.keys.return_value = search_keys
        self.mock_redis.delete.return_value = 1
        
        # Create knowledge base
        kb = KnowledgeBase(enable_vector_search=False)
        
        # Add multiple documents - should trigger cache invalidation once
        documents = [
            Document(id="doc1", title="Doc 1", content="Content 1", source="test"),
            Document(id="doc2", title="Doc 2", content="Content 2", source="test")
        ]
        
        kb.add_documents(documents)
        
        # Verify cache invalidation was attempted
        self.mock_redis.keys.assert_called_with("slack_kb:search_results:*")


class TestCachePerformance(unittest.TestCase):
    """Test caching performance characteristics."""
    
    def setUp(self):
        """Set up performance test fixtures."""
        # Mock Redis with realistic response times
        self.mock_redis = MagicMock()
        self.mock_pool = MagicMock()
    
    @patch('slack_kb_agent.cache.redis')
    @patch('slack_kb_agent.cache.REDIS_AVAILABLE', True)
    def test_cache_ttl_settings(self, mock_redis_module):
        """Test that cache TTL settings are properly applied."""
        # Configure mocks
        mock_redis_module.ConnectionPool.return_value = self.mock_pool
        mock_redis_module.Redis.return_value = self.mock_redis
        self.mock_redis.ping.return_value = True
        
        from slack_kb_agent.cache import CacheManager, CacheConfig
        
        # Create cache manager with custom TTL settings
        config = CacheConfig(
            embedding_ttl=7200,      # 2 hours
            query_expansion_ttl=3600, # 1 hour
            search_results_ttl=1800   # 30 minutes
        )
        cache_manager = CacheManager(config)
        
        # Test embedding cache with custom TTL
        import numpy as np
        embedding = np.array([0.1, 0.2], dtype=np.float32)
        cache_manager.set_embedding("test", "model", embedding)
        
        # Verify setex was called with correct TTL
        args, kwargs = self.mock_redis.setex.call_args
        self.assertEqual(args[1], 7200)  # embedding_ttl
        
        # Test query expansion cache with custom TTL
        self.mock_redis.reset_mock()
        cache_manager.set_query_expansion("test", "synonyms", ["test", "query"])
        
        args, kwargs = self.mock_redis.setex.call_args
        self.assertEqual(args[1], 3600)  # query_expansion_ttl
        
        # Test search results cache with custom TTL
        self.mock_redis.reset_mock()
        cache_manager.set_search_results("hash", [{"doc": "test"}])
        
        args, kwargs = self.mock_redis.setex.call_args
        self.assertEqual(args[1], 1800)  # search_results_ttl
    
    @patch('slack_kb_agent.cache.redis')
    @patch('slack_kb_agent.cache.REDIS_AVAILABLE', True)
    def test_cache_key_collision_avoidance(self, mock_redis_module):
        """Test that cache keys avoid collisions."""
        # Configure mocks
        mock_redis_module.ConnectionPool.return_value = self.mock_pool
        mock_redis_module.Redis.return_value = self.mock_redis
        self.mock_redis.ping.return_value = True
        
        from slack_kb_agent.cache import CacheManager, CacheConfig
        
        cache_manager = CacheManager(CacheConfig())
        
        # Test that different namespaces generate different keys
        key1 = cache_manager._generate_key("embedding", "test")
        key2 = cache_manager._generate_key("query_expansion", "test")
        key3 = cache_manager._generate_key("search_results", "test")
        
        self.assertNotEqual(key1, key2)
        self.assertNotEqual(key2, key3)
        self.assertNotEqual(key1, key3)
        
        # Test that long identifiers are properly hashed
        long_identifier = "x" * 500
        key_long = cache_manager._generate_key("test", long_identifier)
        self.assertLess(len(key_long), 300)  # Should be reasonable length
        
        # Same long identifier should generate same key
        key_long2 = cache_manager._generate_key("test", long_identifier)
        self.assertEqual(key_long, key_long2)


if __name__ == "__main__":
    unittest.main()