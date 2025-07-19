"""Tests for vector search with mocked dependencies."""

import pytest
from unittest.mock import MagicMock, patch
import numpy as np

from slack_kb_agent.knowledge_base import KnowledgeBase
from slack_kb_agent.models import Document


class TestVectorSearchMocked:
    """Test vector search functionality with mocked dependencies."""

    @patch('slack_kb_agent.vector_search.VECTOR_DEPS_AVAILABLE', True)
    @patch('slack_kb_agent.vector_search.SentenceTransformer')
    @patch('slack_kb_agent.vector_search.faiss')
    @patch('slack_kb_agent.vector_search.np')
    def test_vector_search_with_mocked_dependencies(self, mock_np, mock_faiss, mock_transformer, ):
        """Test vector search with all dependencies mocked."""
        # Mock numpy
        mock_np.array = np.array
        mock_np.zeros = np.zeros
        mock_np.float32 = np.float32
        
        # Mock sentence transformer
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        mock_model.get_sentence_embedding_dimension.return_value = 3
        mock_transformer.return_value = mock_model
        
        # Mock FAISS index
        mock_index = MagicMock()
        mock_index.search.return_value = (np.array([[0.9, 0.7]]), np.array([[0, 1]]))
        mock_index.ntotal = 2
        mock_faiss.IndexFlatIP.return_value = mock_index
        
        # Create knowledge base with vector search enabled
        kb = KnowledgeBase(enable_vector_search=True)
        assert kb.enable_vector_search is True
        
        # Add documents
        doc1 = Document(content="Python programming tutorial", source="docs")
        doc2 = Document(content="JavaScript web development", source="docs")
        kb.add_document(doc1)
        kb.add_document(doc2)
        
        # Test semantic search
        results = kb.search_semantic("programming with Python")
        
        # Verify the transformer was called
        mock_model.encode.assert_called()
        mock_index.search.assert_called()
        
        # Should return documents
        assert len(results) >= 1

    def test_vector_search_disabled_fallback(self):
        """Test fallback to keyword search when vector search is disabled."""
        kb = KnowledgeBase(enable_vector_search=False)
        assert kb.enable_vector_search is False
        
        kb.add_document(Document(content="Python programming tutorial", source="docs"))
        kb.add_document(Document(content="JavaScript web development", source="docs"))
        
        # Semantic search should fallback to keyword search
        results = kb.search_semantic("Python")
        assert len(results) == 1
        assert "Python" in results[0].content

    def test_hybrid_search_without_vector(self):
        """Test hybrid search falls back to keyword search when vector search unavailable."""
        kb = KnowledgeBase(enable_vector_search=False)
        
        kb.add_document(Document(content="Python programming tutorial", source="docs"))
        kb.add_document(Document(content="Python web framework", source="docs"))
        
        results = kb.search_hybrid("Python")
        assert len(results) == 2

    def test_empty_query_handling(self):
        """Test handling of empty queries."""
        kb = KnowledgeBase(enable_vector_search=False)
        kb.add_document(Document(content="Some content", source="docs"))
        
        # Empty queries should return empty results
        assert kb.search_semantic("") == []
        assert kb.search_semantic("   ") == []
        assert kb.search_hybrid("") == []

    def test_no_documents_handling(self):
        """Test handling when no documents are indexed."""
        kb = KnowledgeBase(enable_vector_search=False)
        
        assert kb.search_semantic("any query") == []
        assert kb.search_hybrid("any query") == []

    @patch('slack_kb_agent.vector_search.VECTOR_DEPS_AVAILABLE', True)
    @patch('slack_kb_agent.vector_search.SentenceTransformer')
    def test_generate_embedding_method(self, mock_transformer):
        """Test the _generate_embedding method."""
        # Mock transformer
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        mock_model.get_sentence_embedding_dimension.return_value = 3
        mock_transformer.return_value = mock_model
        
        kb = KnowledgeBase(enable_vector_search=True)
        
        # Test embedding generation
        embedding = kb._generate_embedding("test text")
        assert embedding is not None
        mock_model.encode.assert_called_with(["test text"], normalize_embeddings=True)

    def test_generate_embedding_without_vector_search(self):
        """Test _generate_embedding raises error when vector search disabled."""
        kb = KnowledgeBase(enable_vector_search=False)
        
        with pytest.raises(AttributeError):
            kb._generate_embedding("test text")