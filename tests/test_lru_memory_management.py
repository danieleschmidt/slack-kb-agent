"""Tests for LRU memory management in query processor."""

import pytest
from slack_kb_agent.query_processor import EnhancedQueryProcessor, QueryContext
from slack_kb_agent.knowledge_base import KnowledgeBase
from slack_kb_agent.models import Document


def test_query_context_lru_eviction():
    """Test that user contexts are evicted using LRU strategy."""
    kb = KnowledgeBase(enable_vector_search=False)
    processor = EnhancedQueryProcessor(kb, max_user_contexts=3)
    
    # Add contexts up to limit
    ctx1 = processor._get_user_context("user1")
    ctx2 = processor._get_user_context("user2") 
    ctx3 = processor._get_user_context("user3")
    
    # Should have exactly 3 contexts
    assert len(processor.user_contexts) == 3
    assert "user1" in processor.user_contexts
    assert "user2" in processor.user_contexts
    assert "user3" in processor.user_contexts
    
    # Add one more - should evict user1 (least recently used)
    ctx4 = processor._get_user_context("user4")
    
    # Should still have 3 contexts, user1 should be evicted
    assert len(processor.user_contexts) == 3
    assert "user1" not in processor.user_contexts
    assert "user2" in processor.user_contexts
    assert "user3" in processor.user_contexts
    assert "user4" in processor.user_contexts


def test_query_context_lru_access_updates():
    """Test that accessing a context moves it to end (most recently used)."""
    kb = KnowledgeBase(enable_vector_search=False)
    processor = EnhancedQueryProcessor(kb, max_user_contexts=3)
    
    # Add contexts
    processor._get_user_context("user1")
    processor._get_user_context("user2")
    processor._get_user_context("user3")
    
    # Access user1 (should move to end)
    processor._get_user_context("user1")
    
    # Add new context - should evict user2 (now least recently used)
    processor._get_user_context("user4")
    
    assert len(processor.user_contexts) == 3
    assert "user1" in processor.user_contexts  # Should not be evicted
    assert "user2" not in processor.user_contexts  # Should be evicted
    assert "user3" in processor.user_contexts
    assert "user4" in processor.user_contexts


def test_query_context_no_limit():
    """Test that processor works normally without user context limit."""
    kb = KnowledgeBase(enable_vector_search=False)
    processor = EnhancedQueryProcessor(kb, max_user_contexts=None)
    
    # Add many contexts
    for i in range(100):
        processor._get_user_context(f"user{i}")
    
    # All contexts should be retained
    assert len(processor.user_contexts) == 100


def test_query_context_zero_limit():
    """Test behavior with zero context limit."""
    kb = KnowledgeBase(enable_vector_search=False)
    processor = EnhancedQueryProcessor(kb, max_user_contexts=0)
    
    # Should not be able to store any contexts
    processor._get_user_context("user1")
    assert len(processor.user_contexts) == 0


def test_query_context_negative_limit():
    """Test behavior with negative context limit (should be treated as no limit)."""
    kb = KnowledgeBase(enable_vector_search=False)
    processor = EnhancedQueryProcessor(kb, max_user_contexts=-1)
    
    # Should work like unlimited
    for i in range(5):
        processor._get_user_context(f"user{i}")
    
    # Negative limit should be treated as no limit
    assert len(processor.user_contexts) == 5


def test_query_context_default_limit():
    """Test that default limit is reasonable."""
    kb = KnowledgeBase(enable_vector_search=False)
    processor = EnhancedQueryProcessor(kb)
    
    # Default should be 1000
    assert processor.max_user_contexts == 1000


def test_lru_maintains_order():
    """Test that LRU ordering is maintained correctly."""
    kb = KnowledgeBase(enable_vector_search=False)
    processor = EnhancedQueryProcessor(kb, max_user_contexts=3)
    
    # Add contexts in order
    processor._get_user_context("user1")
    processor._get_user_context("user2")
    processor._get_user_context("user3")
    
    # Check initial order (should be user1, user2, user3)
    keys = list(processor.user_contexts.keys())
    assert keys == ["user1", "user2", "user3"]
    
    # Access user2 (should move to end)
    processor._get_user_context("user2")
    keys = list(processor.user_contexts.keys())
    assert keys == ["user1", "user3", "user2"]
    
    # Access user1 (should move to end)
    processor._get_user_context("user1")
    keys = list(processor.user_contexts.keys())
    assert keys == ["user3", "user2", "user1"]


def test_query_context_survives_cleanup():
    """Test that accessing context after eviction creates new one."""
    kb = KnowledgeBase(enable_vector_search=False)
    processor = EnhancedQueryProcessor(kb, max_user_contexts=2)
    
    # Add contexts
    ctx1 = processor._get_user_context("user1")
    ctx2 = processor._get_user_context("user2")
    
    # Add data to first context
    ctx1.add_query("test query", ["doc1"])
    assert len(ctx1.history) == 1
    
    # Add third context - should evict user1
    processor._get_user_context("user3")
    assert "user1" not in processor.user_contexts
    
    # Access user1 again - should create new context (history lost)
    new_ctx1 = processor._get_user_context("user1")
    assert len(new_ctx1.history) == 0
    assert new_ctx1 is not ctx1  # Should be a different object


def test_query_processor_integration_with_lru():
    """Test that process_query method properly uses LRU contexts."""
    kb = KnowledgeBase(enable_vector_search=False)
    kb.add_document(Document(content="test document", source="test"))
    
    processor = EnhancedQueryProcessor(kb, max_user_contexts=2)
    
    # Process queries for different users
    processor.process_query("test query", user_id="user1")
    processor.process_query("test query", user_id="user2")
    
    assert len(processor.user_contexts) == 2
    
    # Add third user - should evict user1
    processor.process_query("test query", user_id="user3")
    
    assert len(processor.user_contexts) == 2
    assert "user1" not in processor.user_contexts
    assert "user2" in processor.user_contexts
    assert "user3" in processor.user_contexts