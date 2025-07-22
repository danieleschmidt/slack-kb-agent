#!/usr/bin/env python3
"""
Test suite for inverted index size management and eviction policies.

Tests the index size limiting functionality to prevent memory issues.
"""

import sys
import unittest
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from slack_kb_agent.search_index import InvertedIndex
from slack_kb_agent.models import Document


class TestIndexSizeManagement(unittest.TestCase):
    """Test index size limiting and eviction policies."""
    
    def test_index_respects_max_size_limit(self):
        """Test that index respects max_index_size parameter."""
        # Create index with small limit for testing
        max_size = 5
        index = InvertedIndex(min_word_length=1, max_index_size=max_size)
        
        # Add documents that would exceed the limit
        docs = [
            Document(content=f"unique_term_{i} common_term", source=f"doc{i}")
            for i in range(10)  # 10 unique terms + 1 common term = 11 terms total
        ]
        
        for doc in docs:
            index.add_document(doc)
        
        # Index should not exceed max_size terms
        total_terms = len(index.index)
        self.assertLessEqual(total_terms, max_size,
                           f"Index size {total_terms} exceeds max_size {max_size}")
    
    def test_lru_eviction_policy(self):
        """Test that LRU (Least Recently Used) eviction works correctly."""
        max_size = 6  # Account for both content and source terms (2 terms per document)
        index = InvertedIndex(min_word_length=1, max_index_size=max_size)
        
        # Add documents with unique terms
        doc1 = Document(content="alpha", source="src1")
        doc2 = Document(content="beta", source="src2") 
        doc3 = Document(content="gamma", source="src3")
        
        index.add_document(doc1)
        index.add_document(doc2)
        index.add_document(doc3)
        
        # All terms should be present (6 terms total: alpha, src1, beta, src2, gamma, src3)
        self.assertIn("alpha", index.index)
        self.assertIn("beta", index.index)
        self.assertIn("gamma", index.index)
        
        # Access alpha to make it recently used
        results = index.search("alpha")
        self.assertTrue(len(results) > 0)
        
        # Add fourth document, should trigger eviction (8 terms > 6 max)
        doc4 = Document(content="delta", source="src4")
        index.add_document(doc4)
        
        # Should respect max size
        total_terms = len(index.index)
        self.assertLessEqual(total_terms, max_size, f"Index should not exceed max_size {max_size}")
        
        # Recently accessed alpha should be preserved
        self.assertIn("alpha", index.index, "Recently used term 'alpha' should not be evicted")
        
        # New term delta should be added
        self.assertIn("delta", index.index, "New term 'delta' should be added")
    
    def test_frequency_based_eviction(self):
        """Test that high-frequency terms are preserved during eviction."""
        max_size = 3
        index = InvertedIndex(min_word_length=1, max_index_size=max_size)
        
        # Add documents where some terms appear multiple times
        docs = [
            Document(content="common rare1", source="doc1"),
            Document(content="common rare2", source="doc2"),
            Document(content="common rare3", source="doc3"),
            Document(content="common rare4", source="doc4"),  # This should trigger eviction
        ]
        
        for doc in docs:
            index.add_document(doc)
        
        # 'common' appears 4 times, should be preserved
        self.assertIn("common", index.index, 
                     "High-frequency term 'common' should be preserved")
        
        # Should still respect size limit
        total_terms = len(index.index)
        self.assertLessEqual(total_terms, max_size)
    
    def test_index_size_metrics(self):
        """Test that index provides size metrics for monitoring."""
        max_size = 5
        index = InvertedIndex(max_index_size=max_size)
        
        # Add some documents
        for i in range(3):
            doc = Document(content=f"term{i} shared", source=f"doc{i}")
            index.add_document(doc)
        
        # Index should provide size information
        self.assertTrue(hasattr(index, 'get_size_stats'), 
                       "Index should provide get_size_stats method")
        
        stats = index.get_size_stats()
        self.assertIsInstance(stats, dict, "Stats should be a dictionary")
        self.assertIn('total_terms', stats, "Stats should include total_terms")
        self.assertIn('max_terms', stats, "Stats should include max_terms")
        self.assertIn('memory_usage_percent', stats, "Stats should include memory usage percent")
        
        # Verify stats accuracy
        self.assertEqual(stats['max_terms'], max_size)
        self.assertGreater(stats['total_terms'], 0)
        self.assertLessEqual(stats['total_terms'], max_size)
    
    def test_eviction_preserves_search_functionality(self):
        """Test that search still works correctly after evictions."""
        max_size = 4
        index = InvertedIndex(min_word_length=1, max_index_size=max_size)
        
        # Add documents that will cause eviction
        docs = [
            Document(content="important document", source="keep1"),
            Document(content="important information", source="keep2"),  
            Document(content="temporary content", source="evict1"),
            Document(content="temporary data", source="evict2"),
            Document(content="important result", source="keep3"),  # May cause eviction
        ]
        
        for doc in docs:
            index.add_document(doc)
        
        # Search should still work for remaining terms
        results = index.search("important")
        self.assertGreater(len(results), 0, "Search should return results for remaining terms")
        
        # Verify document content is preserved even if not all terms are indexed
        all_docs = [result.document for result in results]
        important_docs = [doc for doc in all_docs if "important" in doc.content]
        self.assertGreater(len(important_docs), 0, "Documents with searched terms should be found")
    
    def test_disabled_size_limiting(self):
        """Test that size limiting can be disabled."""
        # max_index_size of None or -1 should disable limiting
        index = InvertedIndex(max_index_size=None)
        
        # Add many documents
        for i in range(20):
            doc = Document(content=f"unique_term_{i}", source=f"doc{i}")
            index.add_document(doc)
        
        # Should have many terms (no artificial limit)
        self.assertGreater(len(index.index), 15, 
                          "Unlimited index should store all terms")
    
    def test_index_performance_with_limiting(self):
        """Test that index performance is acceptable with size limiting."""
        import time
        
        max_size = 100
        index = InvertedIndex(max_index_size=max_size)
        
        # Add documents and measure time
        start_time = time.time()
        
        for i in range(200):  # Add more docs than max_size
            doc = Document(content=f"term{i} common benchmark", source=f"doc{i}")
            index.add_document(doc)
        
        add_time = time.time() - start_time
        
        # Search performance
        start_time = time.time()
        results = index.search("common")
        search_time = time.time() - start_time
        
        # Performance should be reasonable (less than 1 second each)
        self.assertLess(add_time, 1.0, "Adding documents should be fast even with eviction")
        self.assertLess(search_time, 1.0, "Search should be fast")
        self.assertGreater(len(results), 0, "Search should return results")


class TestIndexSizeConfiguration(unittest.TestCase):
    """Test that index size management integrates with configuration system."""
    
    def test_default_config_integration(self):
        """Test that index uses configuration for max_index_size."""
        from slack_kb_agent.configuration import get_search_config
        
        config = get_search_config()
        index = InvertedIndex()  # Should use config defaults
        
        self.assertEqual(index.max_index_size, config.max_index_size,
                        "Index should use configuration max_index_size by default")
    
    def test_override_config_value(self):
        """Test that explicit max_index_size overrides configuration."""
        custom_size = 42
        index = InvertedIndex(max_index_size=custom_size)
        
        self.assertEqual(index.max_index_size, custom_size,
                        "Explicit max_index_size should override configuration")


if __name__ == "__main__":
    unittest.main()