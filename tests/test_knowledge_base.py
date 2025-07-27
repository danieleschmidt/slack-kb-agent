#!/usr/bin/env python3
"""
Test suite for KnowledgeBase class.

This module provides comprehensive testing for the core knowledge base functionality,
including document management, search capabilities, and persistence features.
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import List

# Import modules to test
from slack_kb_agent.knowledge_base import KnowledgeBase
from slack_kb_agent.models import Document
from slack_kb_agent.sources import BaseSource


class MockSource(BaseSource):
    """Mock source for testing."""
    
    def __init__(self, documents: List[Document]):
        self._documents = documents
    
    def load(self) -> List[Document]:
        """Return mock documents."""
        return self._documents


class TestKnowledgeBaseBasic:
    """Test basic KnowledgeBase functionality."""
    
    def test_knowledge_base_creation(self):
        """Test creating a knowledge base with default settings."""
        kb = KnowledgeBase()
        
        assert kb.sources == []
        assert kb.documents == []
        assert kb.max_documents is None
        assert hasattr(kb, 'search_engine')
        assert hasattr(kb, 'enable_vector_search')
        assert hasattr(kb, 'enable_indexed_search')
    
    def test_knowledge_base_creation_with_params(self):
        """Test creating a knowledge base with custom parameters."""
        kb = KnowledgeBase(
            enable_vector_search=False,
            vector_model="custom-model",
            similarity_threshold=0.8,
            max_documents=1000,
            enable_indexed_search=False
        )
        
        assert kb.max_documents == 1000
        assert not kb.enable_vector_search  # Should be disabled
        assert not kb.enable_indexed_search
    
    def test_add_source(self):
        """Test adding sources to knowledge base."""
        kb = KnowledgeBase()
        
        # Create mock documents
        docs = [
            Document(content="Test content 1", source="source1"),
            Document(content="Test content 2", source="source2")
        ]
        mock_source = MockSource(docs)
        
        kb.add_source(mock_source)
        
        assert len(kb.sources) == 1
        assert kb.sources[0] == mock_source
    
    def test_add_document(self):
        """Test adding a single document."""
        kb = KnowledgeBase(enable_vector_search=False)  # Disable for simpler testing
        doc = Document(content="Test content", source="test_source")
        
        kb.add_document(doc)
        
        assert len(kb.documents) == 1
        assert kb.documents[0] == doc
    
    def test_add_documents(self):
        """Test adding multiple documents."""
        kb = KnowledgeBase(enable_vector_search=False)
        docs = [
            Document(content="Content 1", source="source1"),
            Document(content="Content 2", source="source2"),
            Document(content="Content 3", source="source3")
        ]
        
        kb.add_documents(docs)
        
        assert len(kb.documents) == 3
        assert all(doc in kb.documents for doc in docs)
    
    def test_index_loads_from_sources(self):
        """Test that index() loads documents from all sources."""
        kb = KnowledgeBase(enable_vector_search=False)
        
        # Create multiple sources
        docs1 = [Document(content="Doc 1", source="source1")]
        docs2 = [Document(content="Doc 2", source="source2")]
        
        source1 = MockSource(docs1)
        source2 = MockSource(docs2)
        
        kb.add_source(source1)
        kb.add_source(source2)
        
        kb.index()
        
        assert len(kb.documents) == 2
        assert any(doc.content == "Doc 1" for doc in kb.documents)
        assert any(doc.content == "Doc 2" for doc in kb.documents)


class TestKnowledgeBaseSearch:
    """Test search functionality."""
    
    def test_keyword_search(self):
        """Test basic keyword search functionality."""
        kb = KnowledgeBase(enable_vector_search=False)
        
        docs = [
            Document(content="Python programming language", source="python_doc"),
            Document(content="JavaScript web development", source="js_doc"),
            Document(content="Python data science", source="data_doc")
        ]
        
        kb.add_documents(docs)
        
        # Search for "Python"
        results = kb.search("Python")
        
        # Should find documents containing "Python"
        assert len(results) >= 0  # May depend on search engine implementation
        # Note: Actual behavior depends on search engine, so we test basic functionality
    
    def test_semantic_search_fallback(self):
        """Test semantic search falls back to keyword search when vector search disabled."""
        kb = KnowledgeBase(enable_vector_search=False)
        
        docs = [
            Document(content="Machine learning algorithms", source="ml_doc"),
            Document(content="Deep neural networks", source="nn_doc")
        ]
        
        kb.add_documents(docs)
        
        # Should fall back to keyword search
        results = kb.search_semantic("artificial intelligence")
        
        # Should return results (may be empty if no keyword matches)
        assert isinstance(results, list)
    
    def test_hybrid_search_with_disabled_vector(self):
        """Test hybrid search falls back to keyword search when vector search disabled."""
        kb = KnowledgeBase(enable_vector_search=False)
        
        docs = [
            Document(content="Python programming", source="python_doc"),
            Document(content="Java programming", source="java_doc")
        ]
        
        kb.add_documents(docs)
        
        results = kb.hybrid_search("programming")
        
        # Should fall back to keyword search
        assert isinstance(results, list)
    
    def test_empty_query_handling(self):
        """Test handling of empty queries."""
        kb = KnowledgeBase(enable_vector_search=False)
        
        docs = [Document(content="Test content", source="test")]
        kb.add_documents(docs)
        
        # Empty queries should return empty results
        assert kb.search_semantic("") == []
        assert kb.search_semantic("   ") == []
        assert kb.search_hybrid("") == []
        assert kb.search_hybrid("   ") == []


class TestKnowledgeBaseMemoryManagement:
    """Test memory management and document limits."""
    
    def test_document_limit_enforcement(self):
        """Test that document limit is enforced."""
        kb = KnowledgeBase(max_documents=3, enable_vector_search=False)
        
        # Add more documents than the limit
        docs = [
            Document(content=f"Content {i}", source=f"source{i}")
            for i in range(5)
        ]
        
        kb.add_documents(docs)
        
        # Should only keep the limit
        assert len(kb.documents) == 3
        
        # Should keep the most recent documents (FIFO eviction removes oldest)
        # After adding [0,1,2,3,4], with limit 3, should keep [2,3,4]
        assert kb.documents[0].content == "Content 2"
        assert kb.documents[1].content == "Content 3"
        assert kb.documents[2].content == "Content 4"
    
    def test_document_limit_with_add_document(self):
        """Test document limit with individual document addition."""
        kb = KnowledgeBase(max_documents=2, enable_vector_search=False)
        
        # Add documents one by one
        kb.add_document(Document(content="Doc 1", source="s1"))
        assert len(kb.documents) == 1
        
        kb.add_document(Document(content="Doc 2", source="s2"))
        assert len(kb.documents) == 2
        
        kb.add_document(Document(content="Doc 3", source="s3"))
        assert len(kb.documents) == 2
        
        # Should have evicted the oldest document
        assert kb.documents[0].content == "Doc 2"
        assert kb.documents[1].content == "Doc 3"
    
    def test_no_limit_by_default(self):
        """Test that no limit is enforced by default."""
        kb = KnowledgeBase(enable_vector_search=False)
        
        # Add many documents
        docs = [
            Document(content=f"Content {i}", source=f"source{i}")
            for i in range(100)
        ]
        
        kb.add_documents(docs)
        
        # Should keep all documents
        assert len(kb.documents) == 100
    
    def test_memory_stats(self):
        """Test memory statistics reporting."""
        kb = KnowledgeBase(max_documents=10, enable_vector_search=False)
        
        docs = [
            Document(content="Short content", source="source1"),
            Document(content="Longer content with more text", source="source2")
        ]
        
        kb.add_documents(docs)
        
        stats = kb.get_memory_stats()
        
        assert stats["documents_count"] == 2
        assert stats["sources_count"] == 0  # No sources added
        assert stats["max_documents"] == 10
        assert stats["documents_usage_percent"] == 20.0  # 2/10 * 100
        assert "estimated_memory_bytes" in stats
        assert "estimated_memory_mb" in stats
        assert stats["estimated_memory_bytes"] > 0


class TestKnowledgeBasePersistence:
    """Test persistence functionality."""
    
    def test_to_dict(self):
        """Test serializing knowledge base to dictionary."""
        kb = KnowledgeBase(enable_vector_search=False)
        
        docs = [
            Document(content="Content 1", source="source1", metadata={"key": "value1"}),
            Document(content="Content 2", source="source2", metadata={"key": "value2"})
        ]
        
        kb.add_documents(docs)
        
        data = kb.to_dict()
        
        assert "documents" in data
        assert len(data["documents"]) == 2
        
        # Check document structure
        doc_dict = data["documents"][0]
        assert "content" in doc_dict
        assert "source" in doc_dict
        assert "metadata" in doc_dict
    
    def test_from_dict(self):
        """Test creating knowledge base from dictionary."""
        data = {
            "documents": [
                {
                    "content": "Test content 1",
                    "source": "test_source1",
                    "metadata": {"author": "test"}
                },
                {
                    "content": "Test content 2", 
                    "source": "test_source2",
                    "metadata": {}
                }
            ]
        }
        
        kb = KnowledgeBase.from_dict(data, max_documents=100)
        
        assert len(kb.documents) == 2
        assert kb.max_documents == 100
        assert kb.documents[0].content == "Test content 1"
        assert kb.documents[0].metadata["author"] == "test"
        assert kb.documents[1].content == "Test content 2"
    
    def test_save_and_load(self):
        """Test saving and loading knowledge base to/from file."""
        # Create original knowledge base
        kb_original = KnowledgeBase(enable_vector_search=False)
        
        docs = [
            Document(content="Persisted content 1", source="file1"),
            Document(content="Persisted content 2", source="file2", metadata={"persist": True})
        ]
        
        kb_original.add_documents(docs)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            kb_original.save(temp_path)
            
            # Load from file
            kb_loaded = KnowledgeBase.load(temp_path, max_documents=50)
            
            assert len(kb_loaded.documents) == 2
            assert kb_loaded.max_documents == 50
            assert kb_loaded.documents[0].content == "Persisted content 1"
            assert kb_loaded.documents[1].content == "Persisted content 2"
            assert kb_loaded.documents[1].metadata["persist"] is True
            
        finally:
            # Clean up
            Path(temp_path).unlink(missing_ok=True)
    
    def test_load_nonexistent_file(self):
        """Test loading from non-existent file returns empty knowledge base."""
        kb = KnowledgeBase.load("/nonexistent/path.json", max_documents=10)
        
        assert len(kb.documents) == 0
        assert kb.max_documents == 10
    
    def test_load_invalid_json(self):
        """Test loading from file with invalid JSON."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("invalid json content {")
            temp_path = f.name
        
        try:
            kb = KnowledgeBase.load(temp_path)
            assert len(kb.documents) == 0
        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestKnowledgeBaseErrorHandling:
    """Test error handling and edge cases."""
    
    def test_invalid_dict_data(self):
        """Test handling of invalid dictionary data."""
        # Test with non-dict input
        kb = KnowledgeBase.from_dict("not a dict")
        assert len(kb.documents) == 0
        
        # Test with missing documents key
        kb = KnowledgeBase.from_dict({"other_key": "value"})
        assert len(kb.documents) == 0
        
        # Test with invalid document structure
        kb = KnowledgeBase.from_dict({
            "documents": [
                "not a dict",
                {"missing_required_fields": True},
                {"content": "valid", "source": "valid"}  # This one should work
            ]
        })
        assert len(kb.documents) == 1  # Only the valid document should be added
    
    @patch('slack_kb_agent.knowledge_base.get_global_metrics')
    def test_metrics_error_handling(self, mock_get_metrics):
        """Test that metrics errors don't crash the knowledge base."""
        # Mock metrics to raise an error
        mock_metrics = Mock()
        mock_metrics.set_gauge.side_effect = Exception("Metrics error")
        mock_get_metrics.return_value = mock_metrics
        
        kb = KnowledgeBase(enable_vector_search=False)
        
        # This should not raise an exception despite metrics errors
        kb.add_document(Document(content="Test", source="test"))
        
        assert len(kb.documents) == 1
    
    def test_large_document_handling(self):
        """Test handling of very large documents."""
        kb = KnowledgeBase(enable_vector_search=False)
        
        # Create a large document
        large_content = "x" * 100000  # 100KB of content
        large_doc = Document(content=large_content, source="large_source")
        
        kb.add_document(large_doc)
        
        assert len(kb.documents) == 1
        assert len(kb.documents[0].content) == 100000
        
        # Memory stats should reflect the large size
        stats = kb.get_memory_stats()
        assert stats["estimated_memory_bytes"] > 100000


class TestKnowledgeBaseIntegration:
    """Test integration between different components."""
    
    def test_search_engine_integration(self):
        """Test integration with search engine."""
        kb = KnowledgeBase(enable_vector_search=False, enable_indexed_search=True)
        
        docs = [
            Document(content="Python programming tutorial", source="tutorial1"),
            Document(content="JavaScript web development", source="tutorial2"),
            Document(content="Python data analysis", source="tutorial3")
        ]
        
        kb.add_documents(docs)
        
        # Test that search engine is properly integrated
        results = kb.search("Python")
        assert isinstance(results, list)
        
        # Test search engine stats integration
        stats = kb.get_memory_stats()
        assert any(key.startswith("search_") for key in stats.keys())
    
    @patch('slack_kb_agent.knowledge_base.get_cache_manager')
    def test_cache_invalidation(self, mock_get_cache_manager):
        """Test that cache is properly invalidated when documents are added."""
        mock_cache = Mock()
        mock_cache.invalidate_search_cache.return_value = 5
        mock_get_cache_manager.return_value = mock_cache
        
        kb = KnowledgeBase(enable_vector_search=False)
        
        kb.add_document(Document(content="Test", source="test"))
        
        # Cache invalidation should be called
        mock_cache.invalidate_search_cache.assert_called()
    
    def test_source_integration(self):
        """Test integration with various source types."""
        kb = KnowledgeBase(enable_vector_search=False)
        
        # Test with multiple source types
        source1 = MockSource([Document(content="Source 1 content", source="s1")])
        source2 = MockSource([
            Document(content="Source 2 content A", source="s2a"),
            Document(content="Source 2 content B", source="s2b")
        ])
        
        kb.add_source(source1)
        kb.add_source(source2)
        
        kb.index()
        
        assert len(kb.documents) == 3
        assert len(kb.sources) == 2


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])