"""Tests for memory management features in KnowledgeBase."""

import pytest
from slack_kb_agent.knowledge_base import KnowledgeBase
from slack_kb_agent.models import Document


def test_knowledge_base_without_limit():
    """Test that KnowledgeBase works normally without document limit."""
    kb = KnowledgeBase(enable_vector_search=False)
    
    # Add many documents
    for i in range(100):
        kb.add_document(Document(content=f"Document {i}", source="test"))
    
    # All documents should be retained
    assert len(kb.documents) == 100


def test_knowledge_base_with_limit():
    """Test that KnowledgeBase enforces document limit."""
    max_docs = 5
    kb = KnowledgeBase(enable_vector_search=False, max_documents=max_docs)
    
    # Add documents up to limit
    for i in range(max_docs):
        kb.add_document(Document(content=f"Document {i}", source="test"))
    
    # Should have exactly max_docs
    assert len(kb.documents) == max_docs
    
    # Add one more document
    kb.add_document(Document(content="Document 5", source="test"))
    
    # Should still have max_docs, oldest should be evicted
    assert len(kb.documents) == max_docs
    assert kb.documents[0].content == "Document 1"  # First document evicted
    assert kb.documents[-1].content == "Document 5"  # New document is last


def test_knowledge_base_bulk_add_with_limit():
    """Test that add_documents respects document limit."""
    max_docs = 3
    kb = KnowledgeBase(enable_vector_search=False, max_documents=max_docs)
    
    # Add bulk documents exceeding limit
    documents = [Document(content=f"Document {i}", source="test") for i in range(5)]
    kb.add_documents(documents)
    
    # Should have only max_docs, keeping the last ones
    assert len(kb.documents) == max_docs
    assert kb.documents[0].content == "Document 2"  # First two evicted
    assert kb.documents[-1].content == "Document 4"


def test_knowledge_base_limit_with_vector_search_mocked(monkeypatch):
    """Test document limit with vector search enabled (mocked)."""
    # Mock vector search dependencies to be available
    monkeypatch.setattr("slack_kb_agent.knowledge_base.is_vector_search_available", lambda: True)
    
    # Mock the VectorSearchEngine to avoid import errors
    class MockVectorEngine:
        def __init__(self, *args, **kwargs):
            self.index = None
        
        def add_document(self, doc):
            pass
        
        def build_index(self, docs):
            self.index = "mock_index"
    
    monkeypatch.setattr("slack_kb_agent.knowledge_base.VectorSearchEngine", MockVectorEngine)
    
    max_docs = 3
    kb = KnowledgeBase(enable_vector_search=True, max_documents=max_docs)
    
    # Add documents exceeding limit
    for i in range(5):
        kb.add_document(Document(content=f"Document {i}", source="test"))
    
    # Should enforce limit
    assert len(kb.documents) == max_docs


def test_knowledge_base_index_with_limit():
    """Test that index() method respects document limit."""
    max_docs = 2
    
    # Create a mock source that returns many documents
    class MockSource:
        def load(self):
            return [Document(content=f"Source doc {i}", source="mock") for i in range(4)]
    
    kb = KnowledgeBase(enable_vector_search=False, max_documents=max_docs)
    kb.add_source(MockSource())
    
    # Index should enforce limit
    kb.index()
    assert len(kb.documents) == max_docs


def test_save_and_load_with_limit(tmp_path):
    """Test that save/load preserves document limit behavior."""
    max_docs = 3
    kb = KnowledgeBase(enable_vector_search=False, max_documents=max_docs)
    
    # Add documents
    for i in range(5):
        kb.add_document(Document(content=f"Document {i}", source="test"))
    
    # Save should save current state (after limit enforcement)
    path = tmp_path / "kb_limited.json"
    kb.save(path)
    
    # Load with same limit
    loaded_kb = KnowledgeBase.load(path, max_documents=max_docs)
    assert len(loaded_kb.documents) == max_docs
    
    # Add one more to test limit is enforced
    loaded_kb.add_document(Document(content="New document", source="test"))
    assert len(loaded_kb.documents) == max_docs


def test_load_with_different_limit(tmp_path):
    """Test loading with different document limit."""
    # Create KB with many documents
    kb = KnowledgeBase(enable_vector_search=False)
    for i in range(10):
        kb.add_document(Document(content=f"Document {i}", source="test"))
    
    # Save without limit
    path = tmp_path / "kb_unlimited.json"
    kb.save(path)
    
    # Load with strict limit
    limited_kb = KnowledgeBase.load(path, max_documents=3)
    assert len(limited_kb.documents) == 3


def test_from_dict_with_limit():
    """Test from_dict method with document limit."""
    data = {
        "documents": [
            {"content": f"Document {i}", "source": "test"} for i in range(5)
        ]
    }
    
    # Load with limit
    kb = KnowledgeBase.from_dict(data, max_documents=2)
    assert len(kb.documents) == 2


def test_memory_management_fifo_order():
    """Test that documents are evicted in FIFO order."""
    max_docs = 3
    kb = KnowledgeBase(enable_vector_search=False, max_documents=max_docs)
    
    # Add initial documents
    kb.add_document(Document(content="First", source="test"))
    kb.add_document(Document(content="Second", source="test"))
    kb.add_document(Document(content="Third", source="test"))
    
    # Verify initial state
    assert len(kb.documents) == 3
    assert [doc.content for doc in kb.documents] == ["First", "Second", "Third"]
    
    # Add one more - should evict "First"
    kb.add_document(Document(content="Fourth", source="test"))
    assert len(kb.documents) == 3
    assert [doc.content for doc in kb.documents] == ["Second", "Third", "Fourth"]
    
    # Add another - should evict "Second"
    kb.add_document(Document(content="Fifth", source="test"))
    assert len(kb.documents) == 3
    assert [doc.content for doc in kb.documents] == ["Third", "Fourth", "Fifth"]


def test_limit_zero():
    """Test behavior with zero document limit."""
    kb = KnowledgeBase(enable_vector_search=False, max_documents=0)
    
    # Should not be able to add any documents
    kb.add_document(Document(content="Test", source="test"))
    assert len(kb.documents) == 0


def test_limit_negative():
    """Test behavior with negative document limit (should be treated as no limit)."""
    kb = KnowledgeBase(enable_vector_search=False, max_documents=-1)
    
    # Should work like unlimited
    for i in range(5):
        kb.add_document(Document(content=f"Document {i}", source="test"))
    
    # Negative limit should be treated as no limit
    assert len(kb.documents) == 5