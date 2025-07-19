"""Tests for vector-based semantic search functionality."""

import pytest
from slack_kb_agent.knowledge_base import KnowledgeBase
from slack_kb_agent.models import Document


class TestVectorSearch:
    """Test vector-based semantic search capabilities."""

    def test_semantic_search_finds_related_documents(self):
        """Vector search should find semantically similar documents."""
        kb = KnowledgeBase()
        kb.add_document(Document(content="Python programming tutorial", source="docs"))
        kb.add_document(Document(content="JavaScript web development", source="docs"))
        kb.add_document(Document(content="Coding in Python language", source="docs"))
        
        # This should find both Python-related documents even without exact keyword match
        results = kb.search_semantic("programming with Python")
        
        assert len(results) >= 2
        python_results = [doc for doc in results if "Python" in doc.content]
        assert len(python_results) == 2

    def test_semantic_search_respects_similarity_threshold(self):
        """Vector search should filter results by similarity threshold."""
        kb = KnowledgeBase()
        kb.add_document(Document(content="Machine learning algorithms", source="docs"))
        kb.add_document(Document(content="Cooking recipes for dinner", source="docs"))
        
        # High threshold should only return very similar results
        results_high = kb.search_semantic("AI and machine learning", threshold=0.8)
        results_low = kb.search_semantic("AI and machine learning", threshold=0.3)
        
        assert len(results_high) <= len(results_low)
        assert len(results_high) >= 1  # Should find the ML document

    def test_semantic_search_with_empty_query(self):
        """Vector search should handle empty queries gracefully."""
        kb = KnowledgeBase()
        kb.add_document(Document(content="Some content", source="docs"))
        
        results = kb.search_semantic("")
        assert results == []

    def test_semantic_search_with_no_documents(self):
        """Vector search should handle empty knowledge base."""
        kb = KnowledgeBase()
        
        results = kb.search_semantic("any query")
        assert results == []

    def test_semantic_search_returns_sorted_by_similarity(self):
        """Vector search should return results sorted by similarity score."""
        kb = KnowledgeBase()
        kb.add_document(Document(content="Python programming guide", source="docs"))
        kb.add_document(Document(content="Python language tutorial", source="docs"))
        kb.add_document(Document(content="Web development with JavaScript", source="docs"))
        
        results = kb.search_semantic("Python coding tutorial")
        
        # Results should be sorted by relevance (most similar first)
        assert len(results) >= 2
        # The exact "Python language tutorial" should be more similar than the JS doc
        assert "Python" in results[0].content

    def test_vector_embedding_generation(self):
        """Test that documents generate consistent vector embeddings."""
        kb = KnowledgeBase()
        doc = Document(content="Test document for embedding", source="test")
        
        # This method should exist and return a vector
        embedding = kb._generate_embedding(doc.content)
        
        assert embedding is not None
        assert isinstance(embedding, (list, tuple)) or hasattr(embedding, '__array__')
        assert len(embedding) > 0  # Should have dimensions
        
        # Same content should generate same embedding
        embedding2 = kb._generate_embedding(doc.content)
        assert embedding == embedding2 or (
            hasattr(embedding, '__array__') and 
            hasattr(embedding2, '__array__') and
            (embedding == embedding2).all()
        )

    def test_fallback_to_keyword_search(self):
        """When vector search is disabled, should fallback to keyword search."""
        kb = KnowledgeBase(enable_vector_search=False)
        kb.add_document(Document(content="Python programming tutorial", source="docs"))
        
        # Should work even without vector search
        results = kb.search("Python")
        assert len(results) == 1
        assert "Python" in results[0].content

    def test_hybrid_search_combines_results(self):
        """Test hybrid search that combines vector and keyword search."""
        kb = KnowledgeBase()
        kb.add_document(Document(content="Python programming tutorial", source="docs"))
        kb.add_document(Document(content="Machine learning with PyTorch", source="docs"))
        kb.add_document(Document(content="The Python language guide", source="docs"))
        
        # Hybrid search should find both semantic and keyword matches
        results = kb.search_hybrid("Python coding")
        
        assert len(results) >= 2
        # Should include both exact matches and semantic matches