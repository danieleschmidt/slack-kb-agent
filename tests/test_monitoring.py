"""Tests for monitoring and metrics functionality."""

import unittest
import json
import time
from unittest.mock import MagicMock, patch
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from slack_kb_agent.knowledge_base import KnowledgeBase
from slack_kb_agent.models import Document


class TestMetricsCollector(unittest.TestCase):
    """Test metrics collection and reporting."""

    def test_metrics_collector_initialization(self):
        """Test metrics collector initializes with proper configuration."""
        from slack_kb_agent.monitoring import MetricsCollector
        
        collector = MetricsCollector()
        self.assertEqual(len(collector.counters), 0)
        self.assertTrue(hasattr(collector, 'start_time'))

    def test_counter_metrics(self):
        """Test counter metrics increment correctly."""
        from slack_kb_agent.monitoring import MetricsCollector
        
        collector = MetricsCollector()
        
        # Test incrementing counters
        collector.increment_counter("slack_messages_processed")
        collector.increment_counter("slack_messages_processed")
        collector.increment_counter("search_queries_total")
        
        self.assertEqual(collector.get_metric("slack_messages_processed"), 2)
        self.assertEqual(collector.get_metric("search_queries_total"), 1)
        self.assertEqual(collector.get_metric("nonexistent_metric"), 0)

    def test_histogram_metrics(self):
        """Test histogram metrics track duration and statistics."""
        from slack_kb_agent.monitoring import MetricsCollector
        
        collector = MetricsCollector()
        
        # Test recording histogram values
        collector.record_histogram("query_duration_seconds", 0.5)
        collector.record_histogram("query_duration_seconds", 1.2)
        collector.record_histogram("query_duration_seconds", 0.8)
        
        stats = collector.get_histogram_stats("query_duration_seconds")
        self.assertEqual(stats["count"], 3)
        self.assertEqual(stats["sum"], 2.5)
        self.assertLess(abs(stats["avg"] - 0.833), 0.01)

    def test_gauge_metrics(self):
        """Test gauge metrics track current values."""
        from slack_kb_agent.monitoring import MetricsCollector
        
        collector = MetricsCollector()
        
        # Test setting gauge values
        collector.set_gauge("active_connections", 5)
        collector.set_gauge("memory_usage_mb", 256.5)
        collector.set_gauge("active_connections", 3)  # Override
        
        assert collector.get_metric("active_connections") == 3
        assert collector.get_metric("memory_usage_mb") == 256.5

    def test_metrics_export(self):
        """Test metrics can be exported in various formats."""
        from slack_kb_agent.monitoring import MetricsCollector
        
        collector = MetricsCollector()
        collector.increment_counter("test_counter")
        collector.set_gauge("test_gauge", 42)
        collector.record_histogram("test_histogram", 1.5)
        
        # Test Prometheus format export
        prometheus_output = collector.export_prometheus()
        assert "test_counter 1" in prometheus_output
        assert "test_gauge 42" in prometheus_output
        assert "test_histogram_count 1" in prometheus_output
        
        # Test JSON format export
        json_output = collector.export_json()
        data = json.loads(json_output)
        assert data["counters"]["test_counter"] == 1
        assert data["gauges"]["test_gauge"] == 42


class TestHealthChecker(unittest.TestCase):
    """Test health check functionality."""

    def test_health_checker_initialization(self):
        """Test health checker initializes with system checks."""
        from slack_kb_agent.monitoring import HealthChecker
        
        checker = HealthChecker()
        assert hasattr(checker, 'checks')
        assert len(checker.checks) > 0

    def test_basic_health_checks(self):
        """Test basic system health checks."""
        from slack_kb_agent.monitoring import HealthChecker
        
        checker = HealthChecker()
        
        # Test individual checks
        assert checker.check_memory() in ["healthy", "warning", "critical"]
        assert checker.check_disk_space() in ["healthy", "warning", "critical"]
        assert checker.check_knowledge_base(KnowledgeBase()) == "healthy"

    def test_knowledge_base_health_check(self):
        """Test knowledge base specific health checks."""
        from slack_kb_agent.monitoring import HealthChecker
        
        checker = HealthChecker()
        
        # Test empty knowledge base
        empty_kb = KnowledgeBase()
        result = checker.check_knowledge_base(empty_kb)
        assert result == "warning"  # Empty KB should be warning
        
        # Test populated knowledge base
        populated_kb = KnowledgeBase()
        populated_kb.add_document(Document(content="Test content", source="test"))
        result = checker.check_knowledge_base(populated_kb)
        assert result == "healthy"

    def test_overall_health_status(self):
        """Test overall health status aggregation."""
        from slack_kb_agent.monitoring import HealthChecker
        
        checker = HealthChecker()
        kb = KnowledgeBase()
        
        health_report = checker.get_health_status(kb)
        
        assert "status" in health_report
        assert "checks" in health_report
        assert "timestamp" in health_report
        assert health_report["status"] in ["healthy", "warning", "critical"]


class TestPerformanceTracker(unittest.TestCase):
    """Test performance tracking and profiling."""

    def test_performance_tracker_context_manager(self):
        """Test performance tracker as context manager."""
        from slack_kb_agent.monitoring import PerformanceTracker, MetricsCollector
        
        metrics = MetricsCollector()
        tracker = PerformanceTracker(metrics)
        
        # Test context manager usage
        with tracker.track("test_operation"):
            time.sleep(0.1)  # Simulate work
        
        # Check that duration was recorded
        stats = metrics.get_histogram_stats("test_operation_duration_seconds")
        assert stats["count"] == 1
        assert stats["sum"] > 0.09  # Should be at least 0.1 seconds

    def test_performance_tracker_decorator(self):
        """Test performance tracker as decorator."""
        from slack_kb_agent.monitoring import PerformanceTracker, MetricsCollector
        
        metrics = MetricsCollector()
        tracker = PerformanceTracker(metrics)
        
        @tracker.track_function
        def test_function():
            time.sleep(0.05)
            return "result"
        
        result = test_function()
        assert result == "result"
        
        # Check that duration was recorded
        stats = metrics.get_histogram_stats("test_function_duration_seconds")
        assert stats["count"] == 1
        assert stats["sum"] > 0.04

    def test_performance_tracker_manual(self):
        """Test manual performance tracking."""
        from slack_kb_agent.monitoring import PerformanceTracker, MetricsCollector
        
        metrics = MetricsCollector()
        tracker = PerformanceTracker(metrics)
        
        # Test manual timing
        start_time = tracker.start_timer("manual_test")
        time.sleep(0.02)
        tracker.end_timer("manual_test", start_time)
        
        stats = metrics.get_histogram_stats("manual_test_duration_seconds")
        assert stats["count"] == 1


class TestStructuredLogger(unittest.TestCase):
    """Test structured logging functionality."""

    def test_structured_logger_initialization(self):
        """Test structured logger initializes correctly."""
        from slack_kb_agent.monitoring import StructuredLogger
        
        logger = StructuredLogger("test_component")
        assert logger.component == "test_component"
        assert hasattr(logger, 'logger')

    def test_structured_logging_format(self):
        """Test structured log messages are properly formatted."""
        from slack_kb_agent.monitoring import StructuredLogger
        
        with patch('logging.getLogger') as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger
            
            structured_logger = StructuredLogger("test")
            
            # Test info logging
            structured_logger.info("Test message", extra_field="value", count=42)
            
            # Verify logger was called with structured data
            mock_logger.info.assert_called_once()
            call_args = mock_logger.info.call_args[0][0]
            
            # Parse the JSON log message
            log_data = json.loads(call_args)
            assert log_data["level"] == "INFO"
            assert log_data["message"] == "Test message"
            assert log_data["component"] == "test"
            assert log_data["extra_field"] == "value"
            assert log_data["count"] == 42

    def test_structured_logging_levels(self):
        """Test all structured logging levels work correctly."""
        from slack_kb_agent.monitoring import StructuredLogger
        
        with patch('logging.getLogger') as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger
            
            structured_logger = StructuredLogger("test")
            
            # Test all levels
            structured_logger.debug("Debug message")
            structured_logger.info("Info message")
            structured_logger.warning("Warning message")
            structured_logger.error("Error message")
            structured_logger.critical("Critical message")
            
            # Verify all levels were called
            mock_logger.debug.assert_called_once()
            mock_logger.info.assert_called_once()
            mock_logger.warning.assert_called_once()
            mock_logger.error.assert_called_once()
            mock_logger.critical.assert_called_once()


class TestIntegratedMonitoring(unittest.TestCase):
    """Test integrated monitoring with knowledge base operations."""

    def test_knowledge_base_metrics_integration(self):
        """Test knowledge base operations generate metrics."""
        from slack_kb_agent.monitoring import MetricsCollector, MonitoredKnowledgeBase
        
        metrics = MetricsCollector()
        kb = MonitoredKnowledgeBase(metrics=metrics)
        
        # Add documents and perform searches
        kb.add_document(Document(content="Test document", source="test"))
        results = kb.search("test")
        
        # Verify metrics were recorded
        assert metrics.get_metric("kb_documents_added") >= 1
        assert metrics.get_metric("kb_search_queries") >= 1

    def test_slack_bot_metrics_integration(self):
        """Test Slack bot operations generate metrics."""
        from slack_kb_agent.monitoring import MetricsCollector
        from slack_kb_agent.slack_bot import SlackBotServer
        
        # This test would require mocking Slack dependencies
        # Testing the metrics interface without actual Slack connection
        metrics = MetricsCollector()
        
        # Simulate bot metrics
        metrics.increment_counter("slack_messages_received")
        metrics.increment_counter("slack_responses_sent")
        metrics.record_histogram("slack_response_time_seconds", 0.5)
        
        assert metrics.get_metric("slack_messages_received") == 1
        assert metrics.get_metric("slack_responses_sent") == 1

    def test_monitoring_configuration(self):
        """Test monitoring can be configured via environment variables."""
        from slack_kb_agent.monitoring import MonitoringConfig
        
        with patch.dict('os.environ', {
            'MONITORING_ENABLED': 'true',
            'METRICS_PORT': '9090',
            'HEALTH_CHECK_INTERVAL': '30'
        }):
            config = MonitoringConfig.from_env()
            assert config.enabled is True
            assert config.metrics_port == 9090
            assert config.health_check_interval == 30

    def test_monitoring_disabled_gracefully(self):
        """Test monitoring can be disabled without breaking functionality."""
        from slack_kb_agent.monitoring import MetricsCollector
        
        # Test with disabled metrics
        collector = MetricsCollector(enabled=False)
        collector.increment_counter("test_metric")
        
        # Should not raise errors but also not record metrics
        assert collector.get_metric("test_metric") == 0

    def test_metrics_persistence(self):
        """Test metrics can be persisted and loaded."""
        from slack_kb_agent.monitoring import MetricsCollector
        
        collector = MetricsCollector()
        collector.increment_counter("persistent_metric")
        collector.set_gauge("gauge_metric", 100)
        
        # Test saving metrics
        metrics_data = collector.get_metrics_snapshot()
        assert "persistent_metric" in metrics_data["counters"]
        assert metrics_data["counters"]["persistent_metric"] == 1