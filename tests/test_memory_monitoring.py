"""Tests for memory monitoring and metrics collection."""

import pytest
from unittest.mock import Mock, patch
from slack_kb_agent.knowledge_base import KnowledgeBase
from slack_kb_agent.query_processor import EnhancedQueryProcessor
from slack_kb_agent.auth import RateLimiter
from slack_kb_agent.models import Document
from slack_kb_agent.monitoring import MetricsCollector


def test_knowledge_base_memory_stats():
    """Test that KnowledgeBase can report memory statistics."""
    kb = KnowledgeBase(enable_vector_search=False, max_documents=10)
    
    # Add some documents
    for i in range(5):
        kb.add_document(Document(content=f"Document {i} content", source="test"))
    
    stats = kb.get_memory_stats()
    
    assert stats["documents_count"] == 5
    assert stats["sources_count"] == 0
    assert stats["max_documents"] == 10
    assert stats["documents_usage_percent"] == 50.0
    assert stats["estimated_memory_bytes"] > 0
    assert stats["estimated_memory_mb"] > 0


def test_knowledge_base_memory_stats_no_limit():
    """Test memory stats when no document limit is set."""
    kb = KnowledgeBase(enable_vector_search=False)
    
    kb.add_document(Document(content="Test content", source="test"))
    
    stats = kb.get_memory_stats()
    
    assert stats["documents_count"] == 1
    assert stats["max_documents"] is None
    assert "documents_usage_percent" not in stats


def test_query_processor_memory_stats():
    """Test that EnhancedQueryProcessor can report memory statistics."""
    kb = KnowledgeBase(enable_vector_search=False)
    processor = EnhancedQueryProcessor(kb, max_user_contexts=5)
    
    # Create some user contexts
    processor._get_user_context("user1")
    processor._get_user_context("user2")
    
    stats = processor.get_memory_stats()
    
    assert stats["user_contexts_count"] == 2
    assert stats["max_user_contexts"] == 5
    assert stats["user_contexts_usage_percent"] == 40.0
    assert stats["total_history_entries"] == 0


def test_query_processor_memory_stats_with_history():
    """Test memory stats with conversation history."""
    kb = KnowledgeBase(enable_vector_search=False)
    processor = EnhancedQueryProcessor(kb, max_user_contexts=10)
    
    # Create context and add some history
    context = processor._get_user_context("user1")
    context.add_query("test query 1", ["doc1"])
    context.add_query("test query 2", ["doc2"])
    
    stats = processor.get_memory_stats()
    
    assert stats["user_contexts_count"] == 1
    assert stats["total_history_entries"] == 2


def test_rate_limiter_memory_stats():
    """Test that RateLimiter can report memory statistics."""
    limiter = RateLimiter(max_requests=5, window_seconds=60)
    
    # Make some requests
    limiter.is_allowed("user1")
    limiter.is_allowed("user1")
    limiter.is_allowed("user2")
    
    stats = limiter.get_stats()
    
    assert stats["active_identifiers"] == 2
    assert stats["total_requests"] == 3
    assert stats["total_tracked_identifiers"] == 2


@patch('slack_kb_agent.monitoring.PSUTIL_AVAILABLE', True)
@patch('slack_kb_agent.monitoring.psutil')
def test_metrics_collector_memory_collection(mock_psutil):
    """Test that MetricsCollector can collect system memory metrics."""
    # Mock psutil memory info
    mock_memory = Mock()
    mock_memory.used = 1024 * 1024 * 1024  # 1GB
    mock_memory.percent = 75.0
    mock_memory.available = 256 * 1024 * 1024  # 256MB
    mock_psutil.virtual_memory.return_value = mock_memory
    
    collector = MetricsCollector()
    collector.collect_memory_metrics()
    
    # Check that metrics were set
    assert collector.get_metric("system_memory_usage_bytes") == 1024 * 1024 * 1024
    assert collector.get_metric("system_memory_usage_percent") == 75.0
    assert collector.get_metric("system_memory_available_bytes") == 256 * 1024 * 1024


def test_metrics_collector_memory_collection_no_psutil():
    """Test memory collection when psutil is not available."""
    with patch('slack_kb_agent.monitoring.PSUTIL_AVAILABLE', False):
        collector = MetricsCollector()
        collector.collect_memory_metrics()
        
        # Should not crash and should not set any metrics
        assert collector.get_metric("system_memory_usage_bytes") == 0


def test_knowledge_base_metrics_integration():
    """Test integration of KnowledgeBase with metrics system."""
    with patch('slack_kb_agent.knowledge_base.METRICS_AVAILABLE', True):
        mock_metrics = Mock()
        with patch('slack_kb_agent.knowledge_base.get_global_metrics', return_value=mock_metrics):
            kb = KnowledgeBase(enable_vector_search=False, max_documents=5)
            kb.add_document(Document(content="Test", source="test"))
            
            # Verify metrics were updated
            mock_metrics.set_gauge.assert_called()
            
            # Check that document count metric was set
            calls = mock_metrics.set_gauge.call_args_list
            metric_names = [call[0][0] for call in calls]
            assert "kb_documents_count" in metric_names


def test_query_processor_metrics_integration():
    """Test integration of QueryProcessor with metrics system."""
    kb = KnowledgeBase(enable_vector_search=False)
    processor = EnhancedQueryProcessor(kb, max_user_contexts=3)
    
    # Mock the metrics object
    processor.metrics = Mock()
    
    # Create a user context
    processor._get_user_context("user1")
    
    # Verify metrics were updated
    processor.metrics.set_gauge.assert_called()
    
    # Check that context count metric was set
    calls = processor.metrics.set_gauge.call_args_list
    metric_names = [call[0][0] for call in calls]
    assert "query_processor_user_contexts_count" in metric_names


def test_rate_limiter_metrics_integration():
    """Test integration of RateLimiter with metrics system."""
    with patch('slack_kb_agent.auth.get_global_metrics') as mock_get_metrics:
        mock_metrics = Mock()
        mock_get_metrics.return_value = mock_metrics
        
        limiter = RateLimiter(max_requests=3, window_seconds=60)
        limiter.is_allowed("user1")  # Should trigger metrics update for new identifier
        
        # Verify metrics were updated
        mock_metrics.set_gauge.assert_called()
        
        # Check that rate limiter metrics were set
        calls = mock_metrics.set_gauge.call_args_list
        metric_names = [call[0][0] for call in calls]
        assert "rate_limiter_active_identifiers" in metric_names


def test_memory_metrics_error_handling():
    """Test that metrics collection doesn't crash on errors."""
    kb = KnowledgeBase(enable_vector_search=False)
    
    # Mock get_global_metrics to raise an exception
    with patch('slack_kb_agent.knowledge_base.get_global_metrics', side_effect=Exception("Mock error")):
        with patch('slack_kb_agent.knowledge_base.METRICS_AVAILABLE', True):
            # This should not raise an exception
            kb.add_document(Document(content="Test", source="test"))
            
            # Verify the document was still added despite metrics error
            assert len(kb.documents) == 1


def test_metrics_snapshot_includes_memory():
    """Test that metrics snapshot includes memory metrics."""
    with patch('slack_kb_agent.monitoring.PSUTIL_AVAILABLE', True):
        with patch('slack_kb_agent.monitoring.psutil') as mock_psutil:
            # Mock psutil
            mock_memory = Mock()
            mock_memory.used = 1024 * 1024
            mock_memory.percent = 50.0
            mock_memory.available = 1024 * 1024
            mock_psutil.virtual_memory.return_value = mock_memory
            
            collector = MetricsCollector()
            snapshot = collector.get_metrics_snapshot()
            
            # Verify memory metrics are included
            assert "system_memory_usage_bytes" in snapshot["gauges"]
            assert "system_memory_usage_percent" in snapshot["gauges"]
            assert "system_memory_available_bytes" in snapshot["gauges"]


def test_memory_estimation_accuracy():
    """Test that memory estimation is reasonable."""
    kb = KnowledgeBase(enable_vector_search=False)
    
    # Add a document with known content size
    content = "A" * 1000  # 1000 bytes
    source = "test"  # 4 bytes
    kb.add_document(Document(content=content, source=source))
    
    stats = kb.get_memory_stats()
    
    # Should estimate at least the size of content + source
    expected_min_bytes = len(content.encode('utf-8')) + len(source.encode('utf-8'))
    assert stats["estimated_memory_bytes"] >= expected_min_bytes
    
    # Should be reasonable (not wildly off)
    assert stats["estimated_memory_bytes"] < expected_min_bytes * 2