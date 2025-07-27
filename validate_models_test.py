#!/usr/bin/env python3
"""
Simple validation script for models.py test coverage.
This script validates the models without requiring pytest.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from slack_kb_agent.models import Document

def test_document_basic_functionality():
    """Test basic Document functionality."""
    print("Testing Document basic functionality...")
    
    # Test minimal creation
    doc = Document(content="Test content", source="test_source")
    assert doc.content == "Test content"
    assert doc.source == "test_source"
    assert doc.metadata == {}
    print("✓ Basic document creation works")
    
    # Test with metadata
    metadata = {"author": "test", "priority": "high"}
    doc_with_meta = Document(
        content="Content with metadata",
        source="meta_source", 
        metadata=metadata
    )
    assert doc_with_meta.metadata["author"] == "test"
    assert doc_with_meta.metadata["priority"] == "high"
    print("✓ Document with metadata works")
    
    # Test metadata independence
    doc1 = Document(content="Doc1", source="source1")
    doc2 = Document(content="Doc2", source="source2")
    
    doc1.metadata["key1"] = "value1"
    doc2.metadata["key2"] = "value2"
    
    assert "key1" in doc1.metadata
    assert "key1" not in doc2.metadata
    assert "key2" in doc2.metadata
    assert "key2" not in doc1.metadata
    print("✓ Metadata independence works")
    
    # Test equality
    doc_a = Document(content="Same", source="same", metadata={"k": "v"})
    doc_b = Document(content="Same", source="same", metadata={"k": "v"})
    doc_c = Document(content="Different", source="same", metadata={"k": "v"})
    
    assert doc_a == doc_b
    assert doc_a != doc_c
    print("✓ Document equality works")
    
    print("All Document tests passed! ✓")

def test_error_conditions():
    """Test error conditions."""
    print("Testing error conditions...")
    
    # Test that empty strings are allowed
    empty_content_doc = Document(content="", source="source")
    assert empty_content_doc.content == ""
    print("✓ Empty content allowed")
    
    empty_source_doc = Document(content="content", source="")
    assert empty_source_doc.source == ""
    print("✓ Empty source allowed")
    
    print("Error condition tests passed! ✓")

if __name__ == "__main__":
    print("=== Validating models.py test coverage ===")
    
    try:
        test_document_basic_functionality()
        test_error_conditions()
        print("\n✅ All model validation tests passed!")
        print("The models.py module has good test coverage.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)