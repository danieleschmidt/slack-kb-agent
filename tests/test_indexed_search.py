"""Tests for indexed search functionality."""

import pytest
import time
from typing import List

from slack_kb_agent.models import Document
from slack_kb_agent.knowledge_base import KnowledgeBase


class TestIndexedSearch:
    """Test indexed search performance and functionality."""
    
    @pytest.fixture
    def large_document_set(self) -> List[Document]:
        """Create a large set of documents for performance testing."""
        documents = []
        
        # Create documents with different content patterns
        for i in range(1000):
            content = f"Document {i} contains information about "
            if i % 10 == 0:
                content += "python programming and machine learning algorithms"
            elif i % 10 == 1:
                content += "javascript development and web frameworks"
            elif i % 10 == 2:
                content += "database optimization and SQL queries"
            elif i % 10 == 3:
                content += "cloud computing and AWS services"
            elif i % 10 == 4:
                content += "data science and statistical analysis"
            elif i % 10 == 5:
                content += "artificial intelligence and neural networks"
            elif i % 10 == 6:
                content += "cybersecurity and threat detection"
            elif i % 10 == 7:
                content += "mobile development and React Native"
            elif i % 10 == 8:
                content += "DevOps practices and CI/CD pipelines"
            else:
                content += "system architecture and design patterns"
            
            documents.append(Document(
                content=content,
                source=f"test_source_{i % 5}",
                metadata={"index": i, "category": i % 10}
            ))
        
        return documents
    
    def test_search_correctness_with_small_dataset(self):
        """Test that search returns correct results with small dataset."""
        kb = KnowledgeBase(enable_vector_search=False)
        
        documents = [
            Document("Python is a programming language", "source1"),
            Document("JavaScript is used for web development", "source2"), 
            Document("Machine learning uses Python frequently", "source3"),
            Document("Java is also a programming language", "source4"),
        ]
        
        kb.add_documents(documents)
        
        # Test search functionality
        results = kb.search("python")
        assert len(results) == 2  # Should find documents 0 and 2
        
        # Verify correct documents are returned
        contents = [doc.content for doc in results]
        assert "Python is a programming language" in contents
        assert "Machine learning uses Python frequently" in contents
    
    def test_search_performance_baseline(self, large_document_set):
        """Measure baseline search performance with current implementation."""
        kb = KnowledgeBase(enable_vector_search=False)
        kb.add_documents(large_document_set)
        
        # Measure search time
        start_time = time.time()
        results = kb.search("python")
        search_time = time.time() - start_time
        
        # With 1000 documents, linear search should still be fast but measurable
        assert len(results) == 100  # Every 10th document should match
        assert search_time < 1.0  # Should complete within 1 second
        
        # Record baseline for comparison
        return search_time
    
    def test_search_case_insensitive(self):
        """Test that search is case insensitive."""
        kb = KnowledgeBase(enable_vector_search=False)
        
        documents = [
            Document("Python Programming", "source1"),
            Document("JAVASCRIPT development", "source2"),
            Document("machine learning", "source3"),
        ]
        
        kb.add_documents(documents)
        
        # Test different cases
        assert len(kb.search("python")) == 1
        assert len(kb.search("PYTHON")) == 1
        assert len(kb.search("Python")) == 1
        assert len(kb.search("javascript")) == 1
        assert len(kb.search("JAVASCRIPT")) == 1
    
    def test_search_partial_matches(self):
        """Test that search finds partial word matches."""
        kb = KnowledgeBase(enable_vector_search=False)
        
        documents = [
            Document("Programming with Python", "source1"),
            Document("Programmer's guide", "source2"),
            Document("Program execution", "source3"),
            Document("Random text", "source4"),
        ]
        
        kb.add_documents(documents)
        
        # Test partial matching
        results = kb.search("program")
        assert len(results) == 3  # Should match first 3 documents
    
    def test_search_empty_query(self):
        """Test search behavior with empty query."""
        kb = KnowledgeBase(enable_vector_search=False)
        
        documents = [
            Document("Some content", "source1"),
            Document("Other content", "source2"),
        ]
        
        kb.add_documents(documents)
        
        # Empty query should return all documents
        results = kb.search("")
        assert len(results) == 2
    
    def test_search_no_matches(self):
        """Test search behavior when no documents match."""
        kb = KnowledgeBase(enable_vector_search=False)
        
        documents = [
            Document("Python programming", "source1"),
            Document("JavaScript development", "source2"),
        ]
        
        kb.add_documents(documents)
        
        # Search for non-existent term
        results = kb.search("nonexistent")
        assert len(results) == 0
    
    def test_search_with_special_characters(self):
        """Test search with special characters and punctuation."""
        kb = KnowledgeBase(enable_vector_search=False)
        
        documents = [
            Document("API endpoints: /users/{id}", "source1"),
            Document("Regular expressions: [a-z]+", "source2"),
            Document("CSS selectors: .class#id", "source3"),
        ]
        
        kb.add_documents(documents)
        
        # Test search with special characters
        results = kb.search("api")
        assert len(results) == 1
        
        results = kb.search("{id}")
        assert len(results) == 1
    
    def test_search_multiple_words(self):
        """Test search behavior with multiple words."""
        kb = KnowledgeBase(enable_vector_search=False)
        
        documents = [
            Document("Machine learning with Python", "source1"),
            Document("Python web development", "source2"),
            Document("JavaScript machine learning", "source3"),
            Document("Data science", "source4"),
        ]
        
        kb.add_documents(documents)
        
        # Current implementation searches for the entire phrase
        results = kb.search("machine learning")
        assert len(results) == 2  # Documents 0 and 2 contain "machine learning"
    
    def test_memory_usage_with_large_dataset(self, large_document_set):
        """Test that search doesn't consume excessive memory."""
        kb = KnowledgeBase(enable_vector_search=False, max_documents=500)
        
        # Add documents - should respect max_documents limit
        kb.add_documents(large_document_set)
        
        # Should have enforced document limit
        assert len(kb.documents) == 500
        
        # Search should still work correctly
        results = kb.search("python")
        assert len(results) <= 50  # At most 50 matches (500 docs / 10 categories)
    
    def test_search_maintains_document_integrity(self):
        """Test that search doesn't modify original documents."""
        kb = KnowledgeBase(enable_vector_search=False)
        
        original_content = "Python Programming Tutorial"
        document = Document(original_content, "source1", {"key": "value"})
        kb.add_document(document)
        
        # Perform search
        results = kb.search("python")
        
        # Verify original document is unchanged
        assert document.content == original_content
        assert document.metadata == {"key": "value"}
        
        # Verify returned document is correct
        assert len(results) == 1
        assert results[0].content == original_content