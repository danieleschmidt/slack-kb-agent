#!/usr/bin/env python3
"""
Simple cache tests that don't require external dependencies.
"""

import os
import sys
import unittest
import json
from unittest.mock import patch, MagicMock
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from slack_kb_agent.cache import CacheConfig, CacheManager
from slack_kb_agent.models import Document


class TestCacheBasics(unittest.TestCase):
    """Test basic cache functionality without Redis."""
    
    def test_cache_config_defaults(self):
        """Test cache configuration defaults."""
        config = CacheConfig()
        self.assertTrue(config.enabled)
        self.assertEqual(config.host, "localhost")
        self.assertEqual(config.port, 6379)
        self.assertEqual(config.db, 0)
        self.assertIsNone(config.password)
    
    def test_cache_config_from_env(self):
        """Test cache configuration from environment."""
        with patch.dict(os.environ, {
            'CACHE_ENABLED': 'false',
            'REDIS_HOST': 'testhost',
            'REDIS_PORT': '6380'
        }):
            config = CacheConfig.from_env()
            self.assertFalse(config.enabled)
            self.assertEqual(config.host, "testhost")
            self.assertEqual(config.port, 6380)
    
    def test_cache_disabled_fallback(self):
        """Test cache manager when Redis is disabled."""
        config = CacheConfig(enabled=False)
        
        # Mock Redis as unavailable
        with patch('slack_kb_agent.cache.REDIS_AVAILABLE', False):
            cache_manager = CacheManager(config)
            
            # Should not be available
            self.assertFalse(cache_manager.is_available())
            
            # Operations should return None gracefully
            self.assertIsNone(cache_manager.get_query_expansion("test", "synonyms"))
            
            # Set operations should not crash
            cache_manager.set_query_expansion("test", "synonyms", ["test", "query"])
            
            # Stats should show unavailable
            stats = cache_manager.get_cache_stats()
            self.assertEqual(stats["status"], "unavailable")
    
    def test_document_serialization(self):
        """Test Document serialization for caching."""
        document = Document(
            content="Test content for caching",
            source="test_source",
            metadata={"type": "test", "priority": 1}
        )
        
        # Serialize document (simulating cache storage)
        doc_dict = {
            "content": document.content,
            "source": document.source,
            "metadata": document.metadata
        }
        
        serialized = json.dumps(doc_dict)
        deserialized = json.loads(serialized)
        
        # Reconstruct document
        reconstructed = Document(
            content=deserialized["content"],
            source=deserialized["source"],
            metadata=deserialized.get("metadata", {})
        )
        
        # Verify integrity
        self.assertEqual(document.content, reconstructed.content)
        self.assertEqual(document.source, reconstructed.source)
        self.assertEqual(document.metadata, reconstructed.metadata)
    
    def test_cache_key_generation(self):
        """Test cache key generation logic."""
        # Create cache manager with disabled Redis
        config = CacheConfig(enabled=False)
        with patch('slack_kb_agent.cache.REDIS_AVAILABLE', False):
            cache_manager = CacheManager(config)
            
            # Test normal key generation
            key1 = cache_manager._generate_key("test", "identifier")
            self.assertEqual(key1, "slack_kb:test:identifier")
            
            # Test long identifier hashing
            long_id = "x" * 300
            key2 = cache_manager._generate_key("test", long_id)
            self.assertTrue(key2.startswith("slack_kb:test:"))
            self.assertLess(len(key2), 300)  # Should be hashed to shorter length
    
    def test_search_hash_consistency(self):
        """Test search hash generation consistency."""
        config = CacheConfig(enabled=False)
        with patch('slack_kb_agent.cache.REDIS_AVAILABLE', False):
            cache_manager = CacheManager(config)
            
            query = "test query"
            params1 = {"model": "test", "top_k": 10, "threshold": 0.5}
            params2 = {"model": "test", "top_k": 10, "threshold": 0.5}
            params3 = {"model": "different", "top_k": 10, "threshold": 0.5}
            
            # Same parameters should generate same hash
            hash1 = cache_manager.generate_search_hash(query, params1)
            hash2 = cache_manager.generate_search_hash(query, params2)
            self.assertEqual(hash1, hash2)
            
            # Different parameters should generate different hash
            hash3 = cache_manager.generate_search_hash(query, params3)
            self.assertNotEqual(hash1, hash3)


class TestCacheIntegrationBasic(unittest.TestCase):
    """Test cache integration without Redis dependencies."""
    
    def test_knowledge_base_cache_calls(self):
        """Test that knowledge base calls cache invalidation."""
        # Mock the cache manager
        mock_cache_manager = MagicMock()
        mock_cache_manager.invalidate_search_cache.return_value = 2
        
        with patch('slack_kb_agent.knowledge_base.get_cache_manager', return_value=mock_cache_manager):
            from slack_kb_agent.knowledge_base import KnowledgeBase
            
            # Create knowledge base without vector search to avoid dependencies
            kb = KnowledgeBase(enable_vector_search=False)
            
            # Add document
            document = Document(
                content="Test document content",
                source="test",
                metadata={"category": "test"}
            )
            
            kb.add_document(document)
            
            # Verify cache invalidation was called
            mock_cache_manager.invalidate_search_cache.assert_called_once()
    
    def test_query_processor_cache_integration(self):
        """Test query processor cache integration."""
        # Mock the cache manager
        mock_cache_manager = MagicMock()
        mock_cache_manager.get_query_expansion.return_value = None  # Cache miss
        mock_cache_manager.set_query_expansion.return_value = None
        
        with patch('slack_kb_agent.query_processor.get_cache_manager', return_value=mock_cache_manager):
            from slack_kb_agent.query_processor import QueryExpansion
            
            query_expansion = QueryExpansion()
            
            # Test synonym expansion
            result = query_expansion.expand_synonyms("api documentation")
            
            # Should have called cache get and set
            mock_cache_manager.get_query_expansion.assert_called()
            mock_cache_manager.set_query_expansion.assert_called()
            
            # Result should contain original query
            self.assertIn("api documentation", result)


if __name__ == "__main__":
    unittest.main()