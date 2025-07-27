#!/usr/bin/env python3
"""
Test suite for data models in the Slack KB Agent.

This module tests the core data structures and models used throughout the system.
"""

import pytest
from dataclasses import FrozenInstanceError
from typing import Dict, Any

# Import the models to test
from slack_kb_agent.models import Document


class TestDocument:
    """Test the Document model."""
    
    def test_document_creation_minimal(self):
        """Test creating a document with minimal required fields."""
        doc = Document(
            content="This is test content",
            source="test_source"
        )
        
        assert doc.content == "This is test content"
        assert doc.source == "test_source"
        assert doc.metadata == {}
        assert isinstance(doc.metadata, dict)
    
    def test_document_creation_with_metadata(self):
        """Test creating a document with custom metadata."""
        metadata = {
            "author": "test_author",
            "timestamp": "2025-01-01T00:00:00Z",
            "priority": "high",
            "tags": ["important", "documentation"]
        }
        
        doc = Document(
            content="Content with metadata",
            source="metadata_source",
            metadata=metadata
        )
        
        assert doc.content == "Content with metadata"
        assert doc.source == "metadata_source"
        assert doc.metadata == metadata
        assert doc.metadata["author"] == "test_author"
        assert doc.metadata["tags"] == ["important", "documentation"]
    
    def test_document_empty_content(self):
        """Test document with empty content."""
        doc = Document(
            content="",
            source="empty_source"
        )
        
        assert doc.content == ""
        assert doc.source == "empty_source"
        assert doc.metadata == {}
    
    def test_document_empty_source(self):
        """Test document with empty source."""
        doc = Document(
            content="Content without source",
            source=""
        )
        
        assert doc.content == "Content without source"
        assert doc.source == ""
        assert doc.metadata == {}
    
    def test_document_none_values_not_allowed(self):
        """Test that None values are not allowed for required fields."""
        with pytest.raises(TypeError):
            Document(content=None, source="test")
        
        with pytest.raises(TypeError):
            Document(content="test", source=None)
    
    def test_document_metadata_mutability(self):
        """Test that metadata dictionary is mutable."""
        doc = Document(
            content="Test content",
            source="test_source"
        )
        
        # Should be able to modify metadata
        doc.metadata["new_key"] = "new_value"
        assert doc.metadata["new_key"] == "new_value"
        
        # Should be able to update existing metadata
        doc.metadata.update({"author": "test", "version": 1})
        assert doc.metadata["author"] == "test"
        assert doc.metadata["version"] == 1
    
    def test_document_metadata_independence(self):
        """Test that metadata dictionaries are independent between instances."""
        doc1 = Document(content="Content 1", source="source1")
        doc2 = Document(content="Content 2", source="source2")
        
        doc1.metadata["key1"] = "value1"
        doc2.metadata["key2"] = "value2"
        
        assert "key1" in doc1.metadata
        assert "key1" not in doc2.metadata
        assert "key2" in doc2.metadata
        assert "key2" not in doc1.metadata
    
    def test_document_equality(self):
        """Test document equality comparison."""
        doc1 = Document(
            content="Same content",
            source="same_source",
            metadata={"key": "value"}
        )
        
        doc2 = Document(
            content="Same content", 
            source="same_source",
            metadata={"key": "value"}
        )
        
        doc3 = Document(
            content="Different content",
            source="same_source",
            metadata={"key": "value"}
        )
        
        assert doc1 == doc2  # Should be equal
        assert doc1 != doc3  # Should not be equal
    
    def test_document_repr(self):
        """Test document string representation."""
        doc = Document(
            content="Test content",
            source="test_source",
            metadata={"key": "value"}
        )
        
        repr_str = repr(doc)
        assert "Document" in repr_str
        assert "Test content" in repr_str
        assert "test_source" in repr_str
        assert "key" in repr_str
    
    def test_document_field_types(self):
        """Test that document fields accept expected types."""
        # Test with various string types
        doc = Document(
            content="Normal string",
            source="source"
        )
        assert isinstance(doc.content, str)
        assert isinstance(doc.source, str)
        
        # Test with unicode content
        doc_unicode = Document(
            content="Unicode: émojis 🚀 and special chars",
            source="unicode_source"
        )
        assert doc_unicode.content == "Unicode: émojis 🚀 and special chars"
        
        # Test with long content
        long_content = "x" * 10000
        doc_long = Document(
            content=long_content,
            source="long_source"
        )
        assert len(doc_long.content) == 10000
    
    def test_document_metadata_types(self):
        """Test that metadata accepts various data types."""
        complex_metadata = {
            "string": "value",
            "integer": 42,
            "float": 3.14,
            "boolean": True,
            "list": [1, 2, 3],
            "dict": {"nested": "value"},
            "none": None
        }
        
        doc = Document(
            content="Content with complex metadata",
            source="complex_source",
            metadata=complex_metadata
        )
        
        assert doc.metadata["string"] == "value"
        assert doc.metadata["integer"] == 42
        assert doc.metadata["float"] == 3.14
        assert doc.metadata["boolean"] is True
        assert doc.metadata["list"] == [1, 2, 3]
        assert doc.metadata["dict"]["nested"] == "value"
        assert doc.metadata["none"] is None
    
    def test_document_default_factory(self):
        """Test that default_factory creates independent metadata instances."""
        # Create multiple documents without explicit metadata
        docs = [
            Document(content=f"Content {i}", source=f"source_{i}")
            for i in range(3)
        ]
        
        # Modify metadata on each document
        for i, doc in enumerate(docs):
            doc.metadata[f"key_{i}"] = f"value_{i}"
        
        # Verify that metadata instances are independent
        for i, doc in enumerate(docs):
            assert f"key_{i}" in doc.metadata
            # Verify other keys are not present
            for j in range(3):
                if i != j:
                    assert f"key_{j}" not in doc.metadata
    
    def test_document_large_metadata(self):
        """Test document with large metadata dictionary."""
        large_metadata = {f"key_{i}": f"value_{i}" for i in range(1000)}
        
        doc = Document(
            content="Content with large metadata",
            source="large_metadata_source",
            metadata=large_metadata
        )
        
        assert len(doc.metadata) == 1000
        assert doc.metadata["key_0"] == "value_0"
        assert doc.metadata["key_999"] == "value_999"
    
    def test_document_immutable_after_frozen(self):
        """Test document behavior with frozen dataclass (if implemented)."""
        # This test checks if the dataclass is frozen
        # If not frozen, this test will pass; if frozen, it will test immutability
        doc = Document(
            content="Test content",
            source="test_source"
        )
        
        try:
            # Try to modify the content field
            doc.content = "Modified content"
            # If this succeeds, the dataclass is mutable
            assert doc.content == "Modified content"
        except (AttributeError, FrozenInstanceError):
            # If this fails, the dataclass is frozen (immutable)
            # This is actually desirable for value objects
            assert doc.content == "Test content"


class TestModelIntegration:
    """Test integration between different models."""
    
    def test_document_collection(self):
        """Test working with collections of documents."""
        docs = [
            Document(content=f"Content {i}", source=f"source_{i}", 
                    metadata={"index": i, "type": "test"})
            for i in range(5)
        ]
        
        # Test filtering
        test_docs = [doc for doc in docs if doc.metadata.get("type") == "test"]
        assert len(test_docs) == 5
        
        # Test sorting by metadata
        sorted_docs = sorted(docs, key=lambda d: d.metadata["index"])
        for i, doc in enumerate(sorted_docs):
            assert doc.metadata["index"] == i
    
    def test_document_serialization_compatibility(self):
        """Test that documents can be used with common serialization patterns."""
        import json
        
        doc = Document(
            content="Serializable content",
            source="json_source",
            metadata={"serializable": True, "count": 42}
        )
        
        # Test dictionary conversion
        doc_dict = {
            "content": doc.content,
            "source": doc.source,
            "metadata": doc.metadata
        }
        
        # Test JSON serialization of the dictionary
        json_str = json.dumps(doc_dict)
        loaded_dict = json.loads(json_str)
        
        # Reconstruct document
        reconstructed = Document(
            content=loaded_dict["content"],
            source=loaded_dict["source"],
            metadata=loaded_dict["metadata"]
        )
        
        assert reconstructed == doc


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])