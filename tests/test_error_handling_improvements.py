"""Tests for improved error handling in monitoring module."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from slack_kb_agent.monitoring import MetricsCollector, HealthChecker
from slack_kb_agent.knowledge_base import KnowledgeBase
from slack_kb_agent.models import Document
from slack_kb_agent.exceptions import (
    MetricsCollectionError,
    HealthCheckError,
    SystemResourceError,
    KnowledgeBaseHealthError
)


def test_metrics_collector_memory_error_handling():
    """Test that memory metrics collection handles errors gracefully."""
    collector = MetricsCollector()
    
    with patch('slack_kb_agent.monitoring.PSUTIL_AVAILABLE', True):
        with patch('slack_kb_agent.monitoring.psutil') as mock_psutil:
            # Test OSError handling
            mock_psutil.virtual_memory.side_effect = OSError("Permission denied")
            
            # Should not raise exception
            collector.collect_memory_metrics()
            
            # Should increment error counters
            assert collector.get_metric("metrics_collection_errors_total") > 0
            assert collector.get_metric("system_memory_collection_errors_total") > 0


def test_metrics_collector_unexpected_error_handling():
    """Test handling of unexpected errors in memory metrics collection."""
    collector = MetricsCollector()
    
    with patch('slack_kb_agent.monitoring.PSUTIL_AVAILABLE', True):
        with patch('slack_kb_agent.monitoring.psutil') as mock_psutil:
            # Test unexpected error
            mock_psutil.virtual_memory.side_effect = RuntimeError("Unexpected error")
            
            # Should not raise exception
            collector.collect_memory_metrics()
            
            # Should increment error counters
            assert collector.get_metric("metrics_collection_errors_total") > 0
            assert collector.get_metric("memory_metrics_unexpected_errors_total") > 0


def test_health_checker_memory_check_errors():
    """Test memory health check error handling."""
    checker = HealthChecker()
    
    with patch('slack_kb_agent.monitoring.PSUTIL_AVAILABLE', True):
        with patch('slack_kb_agent.monitoring.psutil') as mock_psutil:
            # Test OSError
            mock_psutil.virtual_memory.side_effect = OSError("System error")
            result = checker.check_memory()
            assert result == "unknown"
            
            # Test AttributeError
            mock_psutil.virtual_memory.side_effect = AttributeError("No such attribute")
            result = checker.check_memory()
            assert result == "unknown"
            
            # Test unexpected error
            mock_psutil.virtual_memory.side_effect = RuntimeError("Unexpected error")
            result = checker.check_memory()
            assert result == "unknown"


def test_health_checker_disk_check_errors():
    """Test disk space health check error handling."""
    checker = HealthChecker()
    
    with patch('slack_kb_agent.monitoring.PSUTIL_AVAILABLE', True):
        with patch('slack_kb_agent.monitoring.psutil') as mock_psutil:
            # Test OSError
            mock_psutil.disk_usage.side_effect = OSError("Access denied")
            result = checker.check_disk_space()
            assert result == "unknown"
            
            # Test ZeroDivisionError (corrupted disk stats)
            mock_disk = Mock()
            mock_disk.used = 100
            mock_disk.total = 0  # Invalid total
            mock_psutil.disk_usage.return_value = mock_disk
            result = checker.check_disk_space()
            assert result == "unknown"
            
            # Test unexpected error
            mock_psutil.disk_usage.side_effect = RuntimeError("Unexpected error")
            result = checker.check_disk_space()
            assert result == "unknown"


def test_health_checker_kb_check_none():
    """Test knowledge base health check with None KB."""
    checker = HealthChecker()
    
    result = checker.check_knowledge_base(None)
    assert result == "critical"


def test_health_checker_kb_check_invalid_structure():
    """Test knowledge base health check with invalid KB structure."""
    checker = HealthChecker()
    
    # Test KB without documents attribute
    invalid_kb = Mock()
    del invalid_kb.documents  # Remove documents attribute
    
    result = checker.check_knowledge_base(invalid_kb)
    assert result == "critical"


def test_health_checker_kb_check_attribute_error():
    """Test knowledge base health check with AttributeError."""
    checker = HealthChecker()
    
    # Mock KB that raises AttributeError on documents access
    mock_kb = Mock()
    mock_kb.documents = property(lambda self: (_ for _ in ()).throw(AttributeError("Test error")))
    
    result = checker.check_knowledge_base(mock_kb)
    assert result == "critical"


def test_health_checker_kb_check_type_error():
    """Test knowledge base health check with TypeError."""
    checker = HealthChecker()
    
    # Mock KB with non-iterable documents
    mock_kb = Mock()
    mock_kb.documents = "not a list"  # Will cause TypeError on len()
    
    with patch('builtins.len', side_effect=TypeError("Test error")):
        result = checker.check_knowledge_base(mock_kb)
        assert result == "critical"


def test_health_checker_kb_check_unexpected_error():
    """Test knowledge base health check with unexpected error."""
    checker = HealthChecker()
    
    mock_kb = Mock()
    mock_kb.documents = property(lambda self: (_ for _ in ()).throw(RuntimeError("Unexpected error")))
    
    result = checker.check_knowledge_base(mock_kb)
    assert result == "critical"


def test_health_checker_overall_status_unknown():
    """Test that overall health status includes 'unknown' status."""
    checker = HealthChecker()
    
    with patch.object(checker, 'check_memory', return_value='unknown'):
        with patch.object(checker, 'check_disk_space', return_value='healthy'):
            status = checker.get_health_status()
            assert status["status"] == "unknown"
            assert status["checks"]["memory"] == "unknown"
            assert status["checks"]["disk_space"] == "healthy"


def test_health_checker_status_priority():
    """Test that health status priority is correct (critical > warning > unknown > healthy)."""
    checker = HealthChecker()
    
    # Test critical takes priority over warning and unknown
    with patch.object(checker, 'check_memory', return_value='critical'):
        with patch.object(checker, 'check_disk_space', return_value='warning'):
            with patch.object(checker, 'check_knowledge_base', return_value='unknown'):
                kb = Mock()
                status = checker.get_health_status(kb)
                assert status["status"] == "critical"
    
    # Test warning takes priority over unknown
    with patch.object(checker, 'check_memory', return_value='warning'):
        with patch.object(checker, 'check_disk_space', return_value='unknown'):
            status = checker.get_health_status()
            assert status["status"] == "warning"
    
    # Test unknown takes priority over healthy
    with patch.object(checker, 'check_memory', return_value='unknown'):
        with patch.object(checker, 'check_disk_space', return_value='healthy'):
            status = checker.get_health_status()
            assert status["status"] == "unknown"


def test_health_checker_valid_kb():
    """Test health checker with valid knowledge base."""
    checker = HealthChecker()
    
    # Create valid KB
    kb = KnowledgeBase(enable_vector_search=False)
    for i in range(15):  # Add enough documents to be healthy
        kb.add_document(Document(content=f"Document {i}", source="test"))
    
    result = checker.check_knowledge_base(kb)
    assert result == "healthy"


def test_health_checker_empty_kb():
    """Test health checker with empty knowledge base."""
    checker = HealthChecker()
    
    kb = KnowledgeBase(enable_vector_search=False)
    result = checker.check_knowledge_base(kb)
    assert result == "warning"


def test_health_checker_small_kb():
    """Test health checker with small knowledge base."""
    checker = HealthChecker()
    
    kb = KnowledgeBase(enable_vector_search=False)
    for i in range(5):  # Add small number of documents
        kb.add_document(Document(content=f"Document {i}", source="test"))
    
    result = checker.check_knowledge_base(kb)
    assert result == "warning"


def test_metrics_no_psutil():
    """Test metrics collection when psutil is not available."""
    collector = MetricsCollector()
    
    with patch('slack_kb_agent.monitoring.PSUTIL_AVAILABLE', False):
        # Should not raise exception
        collector.collect_memory_metrics()
        
        # Should not have system memory metrics
        assert collector.get_metric("system_memory_usage_bytes") == 0


def test_health_checker_no_psutil():
    """Test health checker when psutil is not available."""
    checker = HealthChecker()
    
    with patch('slack_kb_agent.monitoring.PSUTIL_AVAILABLE', False):
        # Should return healthy when can't check
        assert checker.check_memory() == "healthy"
        assert checker.check_disk_space() == "healthy"


def test_error_logging():
    """Test that errors are properly logged."""
    checker = HealthChecker()
    
    with patch('slack_kb_agent.monitoring.PSUTIL_AVAILABLE', True):
        with patch('slack_kb_agent.monitoring.psutil') as mock_psutil:
            with patch('slack_kb_agent.monitoring.logger') as mock_logger:
                # Test that warning is logged for OSError
                mock_psutil.virtual_memory.side_effect = OSError("Test error")
                checker.check_memory()
                mock_logger.warning.assert_called()
                
                # Test that error is logged for unexpected errors
                mock_psutil.virtual_memory.side_effect = RuntimeError("Unexpected")
                checker.check_memory()
                mock_logger.error.assert_called()


def test_knowledge_base_error_logging():
    """Test that KB health check errors are properly logged."""
    checker = HealthChecker()
    
    with patch('slack_kb_agent.monitoring.logger') as mock_logger:
        # Test None KB logging
        checker.check_knowledge_base(None)
        mock_logger.warning.assert_called_with("Knowledge base is None, cannot check health")
        
        # Test missing attributes logging
        mock_kb = Mock(spec=[])  # KB without documents attribute
        checker.check_knowledge_base(mock_kb)
        mock_logger.error.assert_called()


def test_metrics_error_counters():
    """Test that error metrics are properly incremented."""
    collector = MetricsCollector()
    
    # Reset counters
    collector.counters.clear()
    
    with patch('slack_kb_agent.monitoring.PSUTIL_AVAILABLE', True):
        with patch('slack_kb_agent.monitoring.psutil') as mock_psutil:
            # Trigger OSError
            mock_psutil.virtual_memory.side_effect = OSError("Test error")
            collector.collect_memory_metrics()
            
            # Check error counters
            assert collector.get_metric("metrics_collection_errors_total") == 1
            assert collector.get_metric("system_memory_collection_errors_total") == 1
            
            # Trigger unexpected error
            mock_psutil.virtual_memory.side_effect = RuntimeError("Unexpected")
            collector.collect_memory_metrics()
            
            # Check error counters incremented
            assert collector.get_metric("metrics_collection_errors_total") == 2
            assert collector.get_metric("memory_metrics_unexpected_errors_total") == 1