"""Comprehensive monitoring and metrics system for production observability."""

from __future__ import annotations

import json
import logging
import os
import time
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import wraps
from typing import Dict, List, Any, Optional, Union
from collections import defaultdict, deque
import statistics

from .exceptions import (
    MetricsCollectionError,
    HealthCheckError,
    SystemResourceError,
    KnowledgeBaseHealthError
)
from .constants import MonitoringDefaults, NetworkDefaults, EnvironmentConfig

# Optional system monitoring dependency
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None

logger = logging.getLogger(__name__)


@dataclass
class MonitoringConfig:
    """Configuration for monitoring system."""
    
    enabled: bool = True
    metrics_port: int = NetworkDefaults.DEFAULT_MONITORING_PORT
    health_check_interval: int = MonitoringDefaults.HEALTH_CHECK_INTERVAL_SECONDS
    metrics_retention_hours: int = MonitoringDefaults.METRICS_RETENTION_HOURS
    log_level: str = "INFO"
    
    @classmethod
    def from_env(cls) -> MonitoringConfig:
        """Create configuration from environment variables."""
        return cls(
            enabled=os.getenv("MONITORING_ENABLED", "true").lower() == "true",
            metrics_port=EnvironmentConfig.get_monitoring_port(),
            health_check_interval=EnvironmentConfig.get_health_check_interval(),
            metrics_retention_hours=int(os.getenv("METRICS_RETENTION_HOURS", str(MonitoringDefaults.METRICS_RETENTION_HOURS))),
            log_level=os.getenv("LOG_LEVEL", "INFO").upper()
        )


class MetricsCollector:
    """Thread-safe metrics collection system."""
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.start_time = time.time()
        self._lock = threading.Lock()
        
        # Metrics storage
        self.counters: Dict[str, int] = defaultdict(int)
        self.gauges: Dict[str, float] = {}
        self.histograms: Dict[str, List[float]] = defaultdict(list)
        self.histogram_stats: Dict[str, Dict[str, float]] = defaultdict(dict)
        
        # For legacy compatibility
        self.metrics: Dict[str, Any] = {}
    
    def increment_counter(self, name: str, value: int = 1) -> None:
        """Increment a counter metric."""
        if not self.enabled:
            return
        
        with self._lock:
            self.counters[name] += value
    
    def set_gauge(self, name: str, value: float) -> None:
        """Set a gauge metric value."""
        if not self.enabled:
            return
        
        with self._lock:
            self.gauges[name] = value
    
    def record_histogram(self, name: str, value: float) -> None:
        """Record a histogram value and update statistics."""
        if not self.enabled:
            return
        
        with self._lock:
            self.histograms[name].append(value)
            
            # Update running statistics
            values = self.histograms[name]
            self.histogram_stats[name] = {
                "count": len(values),
                "sum": sum(values),
                "avg": statistics.mean(values),
                "min": min(values),
                "max": max(values),
                "p50": statistics.median(values),
                "p95": self._percentile(values, 95),
                "p99": self._percentile(values, 99)
            }
    
    def _percentile(self, values: List[float], percentile: float) -> float:
        """Calculate percentile of values."""
        if not values:
            return 0.0
        sorted_values = sorted(values)
        index = int((percentile / 100) * len(sorted_values))
        return sorted_values[min(index, len(sorted_values) - 1)]
    
    def get_metric(self, name: str) -> Union[int, float]:
        """Get a metric value (counters and gauges)."""
        with self._lock:
            if name in self.counters:
                return self.counters[name]
            elif name in self.gauges:
                return self.gauges[name]
            return 0
    
    def get_histogram_stats(self, name: str) -> Dict[str, float]:
        """Get histogram statistics."""
        with self._lock:
            return self.histogram_stats.get(name, {
                "count": 0, "sum": 0, "avg": 0, "min": 0, "max": 0,
                "p50": 0, "p95": 0, "p99": 0
            })
    
    def export_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []
        
        with self._lock:
            # Export counters
            for name, value in self.counters.items():
                lines.append(f"# TYPE {name} counter")
                lines.append(f"{name} {value}")
                lines.append("")
            
            # Export gauges
            for name, value in self.gauges.items():
                lines.append(f"# TYPE {name} gauge")
                lines.append(f"{name} {value}")
                lines.append("")
            
            # Export histograms
            for name, stats in self.histogram_stats.items():
                lines.append(f"# TYPE {name} histogram")
                lines.append(f"{name}_count {stats['count']}")
                lines.append(f"{name}_sum {stats['sum']}")
                lines.append(f"{name}_bucket{{le=\"+Inf\"}} {stats['count']}")
                lines.append("")
        
        return "\n".join(lines)
    
    def export_json(self) -> str:
        """Export metrics in JSON format."""
        with self._lock:
            data = {
                "timestamp": time.time(),
                "uptime_seconds": time.time() - self.start_time,
                "counters": dict(self.counters),
                "gauges": dict(self.gauges),
                "histograms": dict(self.histogram_stats)
            }
        
        return json.dumps(data, indent=2)
    
    def collect_memory_metrics(self) -> None:
        """Collect memory usage metrics from key components."""
        try:
            # System memory metrics
            if PSUTIL_AVAILABLE:
                try:
                    memory = psutil.virtual_memory()
                    self.set_gauge("system_memory_usage_bytes", memory.used)
                    self.set_gauge("system_memory_usage_percent", memory.percent)
                    self.set_gauge("system_memory_available_bytes", memory.available)
                except (AttributeError, OSError) as e:
                    # Log specific system resource errors but don't crash
                    logger.warning(f"Failed to collect system memory metrics: {e}")
                    self.increment_counter("metrics_collection_errors_total")
                    self.increment_counter("system_memory_collection_errors_total")
            
            # Cache metrics (if Redis caching is available)
            try:
                from .cache import get_cache_manager
                cache_manager = get_cache_manager()
                cache_stats = cache_manager.get_cache_stats()
                
                if cache_stats.get("status") == "connected":
                    self.set_gauge("cache_available", 1)
                    self.set_gauge("cache_hit_rate", cache_stats.get("hit_rate", 0))
                    self.set_gauge("cache_connected_clients", cache_stats.get("connected_clients", 0))
                    
                    key_counts = cache_stats.get("key_counts", {})
                    for namespace, count in key_counts.items():
                        self.set_gauge(f"cache_keys_{namespace}", count)
                else:
                    self.set_gauge("cache_available", 0)
                    
            except (ImportError, AttributeError) as e:
                # Cache module may not be available or properly configured
                self.set_gauge("cache_available", 0)
                logger.debug(f"Cache metrics unavailable: {type(e).__name__}: {e}")
            except MetricsCollectionError as e:
                # Expected metrics collection error
                self.set_gauge("cache_available", 0)
                logger.warning(f"Cache metrics collection failed: {e}")
            except Exception as e:
                # Unexpected error - log for investigation
                self.set_gauge("cache_available", 0)
                logger.error(f"Unexpected error in cache metrics collection: {type(e).__name__}: {e}")
            
            # Application-specific memory metrics would be collected by the components themselves
            # This is called periodically to update memory-related gauges
            
        except (OSError, AttributeError) as e:
            # System resource access errors
            logger.warning(f"System resource error in memory metrics collection: {type(e).__name__}: {e}")
            self.increment_counter("metrics_collection_errors_total")
            self.increment_counter("memory_metrics_system_errors_total")
        except MetricsCollectionError as e:
            # Expected metrics collection error
            logger.warning(f"Memory metrics collection failed: {e}")
            self.increment_counter("metrics_collection_errors_total")
        except Exception as e:
            # Unexpected error - log for investigation
            logger.error(f"Unexpected error in memory metrics collection: {type(e).__name__}: {e}")
            self.increment_counter("metrics_collection_errors_total")
            self.increment_counter("memory_metrics_unexpected_errors_total")
            # Don't let metrics collection crash the application
    
    def get_metrics_snapshot(self) -> Dict[str, Any]:
        """Get a snapshot of all metrics."""
        # Update memory metrics before returning
        self.collect_memory_metrics()
        
        with self._lock:
            return {
                "timestamp": time.time(),
                "uptime_seconds": time.time() - self.start_time,
                "counters": dict(self.counters),
                "gauges": dict(self.gauges),
                "histograms": dict(self.histogram_stats)
            }


class HealthChecker:
    """System health monitoring and checks."""
    
    def __init__(self):
        self.checks = [
            "memory",
            "disk_space", 
            "knowledge_base"
        ]
    
    def check_memory(self) -> str:
        """Check system memory usage."""
        if not PSUTIL_AVAILABLE:
            return "healthy"  # Can't check without psutil
        
        try:
            memory = psutil.virtual_memory()
            usage_percent = memory.percent
            
            if usage_percent > 90:
                return "critical"
            elif usage_percent > 75:
                return "warning"
            else:
                return "healthy"
        except (AttributeError, OSError) as e:
            # Log specific system resource errors
            logger.warning(f"Failed to check memory usage: {e}")
            return "unknown"  # Indicate we couldn't check
        except SystemResourceError as e:
            # Expected system resource error
            logger.warning(f"Memory health check failed: {e}")
            return "unknown"
        except Exception as e:
            # Unexpected error - log for investigation
            logger.error(f"Unexpected error checking memory health: {type(e).__name__}: {e}")
            return "unknown"
    
    def check_disk_space(self) -> str:
        """Check disk space usage."""
        if not PSUTIL_AVAILABLE:
            return "healthy"
        
        try:
            disk = psutil.disk_usage('/')
            usage_percent = (disk.used / disk.total) * 100
            
            if usage_percent > 95:
                return "critical"
            elif usage_percent > 85:
                return "warning"
            else:
                return "healthy"
        except (AttributeError, OSError, ZeroDivisionError) as e:
            # Log specific system resource errors
            logger.warning(f"Failed to check disk space: {e}")
            return "unknown"
        except SystemResourceError as e:
            # Expected system resource error
            logger.warning(f"Disk space health check failed: {e}")
            return "unknown"
        except Exception as e:
            # Unexpected error - log for investigation
            logger.error(f"Unexpected error checking disk space: {type(e).__name__}: {e}")
            return "unknown"
    
    def check_knowledge_base(self, kb) -> str:
        """Check knowledge base health."""
        if kb is None:
            logger.warning("Knowledge base is None, cannot check health")
            return "critical"
        
        try:
            if not hasattr(kb, 'documents'):
                logger.error("Knowledge base missing 'documents' attribute")
                return "critical"
            
            doc_count = len(kb.documents)
            if doc_count == 0:
                logger.warning("Knowledge base is empty")
                return "warning"  # Empty KB is concerning but not critical
            elif doc_count < 10:
                logger.info(f"Knowledge base has only {doc_count} documents, might indicate ingestion issues")
                return "warning"  # Small KB might indicate ingestion issues
            else:
                return "healthy"
        except (AttributeError, TypeError) as e:
            # Log specific knowledge base errors
            logger.error(f"Knowledge base health check failed - invalid KB structure: {e}")
            return "critical"
        except KnowledgeBaseHealthError as e:
            # Expected knowledge base health error
            logger.warning(f"Knowledge base health check failed: {e}")
            return "critical"
        except Exception as e:
            # Unexpected error - log for investigation
            logger.error(f"Unexpected error checking knowledge base health: {type(e).__name__}: {e}")
            return "critical"
    
    def get_health_status(self, kb=None) -> Dict[str, Any]:
        """Get overall health status."""
        check_results = {}
        
        # Run all health checks
        check_results["memory"] = self.check_memory()
        check_results["disk_space"] = self.check_disk_space()
        
        if kb is not None:
            check_results["knowledge_base"] = self.check_knowledge_base(kb)
        
        # Determine overall status
        if any(status == "critical" for status in check_results.values()):
            overall_status = "critical"
        elif any(status == "warning" for status in check_results.values()):
            overall_status = "warning"
        elif any(status == "unknown" for status in check_results.values()):
            overall_status = "unknown"
        else:
            overall_status = "healthy"
        
        return {
            "status": overall_status,
            "timestamp": time.time(),
            "checks": check_results
        }


class PerformanceTracker:
    """Track performance metrics and profiling."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
    
    @contextmanager
    def track(self, operation_name: str):
        """Context manager for tracking operation duration."""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.metrics.record_histogram(f"{operation_name}_duration_seconds", duration)
    
    def track_function(self, func):
        """Decorator for tracking function performance."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            with self.track(func.__name__):
                return func(*args, **kwargs)
        return wrapper
    
    def start_timer(self, operation_name: str) -> float:
        """Start a manual timer."""
        return time.time()
    
    def end_timer(self, operation_name: str, start_time: float) -> None:
        """End a manual timer and record duration."""
        duration = time.time() - start_time
        self.metrics.record_histogram(f"{operation_name}_duration_seconds", duration)


class StructuredLogger:
    """Structured JSON logging with context."""
    
    def __init__(self, component: str):
        self.component = component
        self.logger = logging.getLogger(f"slack_kb_agent.{component}")
    
    def _log(self, level: str, message: str, **extra_fields) -> None:
        """Log a structured message."""
        log_data = {
            "timestamp": time.time(),
            "level": level,
            "component": self.component,
            "message": message,
            **extra_fields
        }
        
        # Use the appropriate logging level
        log_method = getattr(self.logger, level.lower())
        log_method(json.dumps(log_data))
    
    def debug(self, message: str, **extra_fields) -> None:
        """Log debug message."""
        self._log("DEBUG", message, **extra_fields)
    
    def info(self, message: str, **extra_fields) -> None:
        """Log info message."""
        self._log("INFO", message, **extra_fields)
    
    def warning(self, message: str, **extra_fields) -> None:
        """Log warning message."""
        self._log("WARNING", message, **extra_fields)
    
    def error(self, message: str, **extra_fields) -> None:
        """Log error message."""
        self._log("ERROR", message, **extra_fields)
    
    def critical(self, message: str, **extra_fields) -> None:
        """Log critical message."""
        self._log("CRITICAL", message, **extra_fields)


class MonitoredKnowledgeBase:
    """Knowledge base wrapper that adds monitoring."""
    
    def __init__(self, kb=None, metrics: Optional[MetricsCollector] = None):
        from .knowledge_base import KnowledgeBase
        
        self.kb = kb or KnowledgeBase()
        self.metrics = metrics or MetricsCollector()
        self.performance = PerformanceTracker(self.metrics)
        self.logger = StructuredLogger("knowledge_base")
    
    def add_document(self, document) -> None:
        """Add document with metrics tracking."""
        with self.performance.track("kb_add_document"):
            self.kb.add_document(document)
            self.metrics.increment_counter("kb_documents_added")
            self.metrics.set_gauge("kb_total_documents", len(self.kb.documents))
    
    def search(self, query: str):
        """Search with metrics tracking."""
        with self.performance.track("kb_search"):
            self.metrics.increment_counter("kb_search_queries")
            results = self.kb.search(query)
            self.metrics.record_histogram("kb_search_results_count", len(results))
            
            self.logger.info(
                "Search performed",
                query_length=len(query),
                result_count=len(results)
            )
            
            return results
    
    def search_semantic(self, query: str, **kwargs):
        """Semantic search with metrics tracking."""
        if not hasattr(self.kb, 'search_semantic'):
            return self.search(query)
        
        with self.performance.track("kb_semantic_search"):
            self.metrics.increment_counter("kb_semantic_search_queries")
            results = self.kb.search_semantic(query, **kwargs)
            self.metrics.record_histogram("kb_semantic_search_results_count", len(results))
            
            return results
    
    def __getattr__(self, name):
        """Delegate all other attributes to the wrapped knowledge base."""
        return getattr(self.kb, name)


# Global metrics collector instance
_global_metrics = None

def get_global_metrics() -> MetricsCollector:
    """Get or create the global metrics collector."""
    global _global_metrics
    if _global_metrics is None:
        config = MonitoringConfig.from_env()
        _global_metrics = MetricsCollector(enabled=config.enabled)
    return _global_metrics


def setup_monitoring(config: Optional[MonitoringConfig] = None) -> Dict[str, Any]:
    """Set up comprehensive monitoring system."""
    if config is None:
        config = MonitoringConfig.from_env()
    
    if not config.enabled:
        return {"status": "disabled"}
    
    # Initialize global metrics
    metrics = get_global_metrics()
    
    # Set up structured logging
    root_logger = logging.getLogger("slack_kb_agent")
    root_logger.setLevel(getattr(logging, config.log_level))
    
    # Initialize health checker
    health_checker = HealthChecker()
    
    # Initialize performance tracker
    performance_tracker = PerformanceTracker(metrics)
    
    return {
        "status": "enabled",
        "config": config,
        "metrics": metrics,
        "health_checker": health_checker,
        "performance_tracker": performance_tracker
    }


def create_monitoring_middleware():
    """Create monitoring middleware for web frameworks."""
    metrics = get_global_metrics()
    
    def middleware(request, response, start_time):
        # Record request metrics
        metrics.increment_counter("http_requests_total")
        metrics.record_histogram("http_request_duration_seconds", time.time() - start_time)
        
        # Record status code metrics
        status_code = getattr(response, 'status_code', 200)
        metrics.increment_counter(f"http_responses_{status_code}")
    
    return middleware


def get_monitoring_endpoints():
    """Get monitoring endpoints for health checks and metrics."""
    metrics = get_global_metrics()
    health_checker = HealthChecker()
    
    def metrics_endpoint():
        """Prometheus metrics endpoint."""
        return metrics.export_prometheus(), 200, {"Content-Type": "text/plain"}
    
    def health_endpoint(kb=None):
        """Health check endpoint."""
        health_status = health_checker.get_health_status(kb)
        status_code = 200 if health_status["status"] == "healthy" else 503
        return json.dumps(health_status), status_code, {"Content-Type": "application/json"}
    
    def metrics_json_endpoint():
        """JSON metrics endpoint."""
        return metrics.export_json(), 200, {"Content-Type": "application/json"}
    
    return {
        "/metrics": metrics_endpoint,
        "/health": health_endpoint,
        "/metrics.json": metrics_json_endpoint
    }


def start_monitoring_server(port: int = 9090, kb=None):
    """Start HTTP server for monitoring endpoints."""
    from http.server import HTTPServer, BaseHTTPRequestHandler
    import threading
    
    endpoints = get_monitoring_endpoints()
    
    class MonitoringHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            """Handle GET requests to monitoring endpoints."""
            if self.path == "/metrics":
                content, status, headers = endpoints["/metrics"]()
            elif self.path == "/health":
                content, status, headers = endpoints["/health"](kb)
            elif self.path == "/metrics.json":
                content, status, headers = endpoints["/metrics.json"]()
            else:
                content = "Not Found"
                status = 404
                headers = {"Content-Type": "text/plain"}
            
            self.send_response(status)
            for key, value in headers.items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(content.encode('utf-8'))
        
        def log_message(self, format, *args):
            """Suppress default HTTP server logging."""
            pass
    
    def run_server():
        server = HTTPServer(('0.0.0.0', port), MonitoringHandler)
        logger = StructuredLogger("monitoring_server")
        logger.info(f"Monitoring server started", port=port)
        server.serve_forever()
    
    # Start server in background thread
    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()
    
    return thread


# Convenience functions for easy integration
def track_query_performance(func):
    """Decorator to track query performance."""
    performance = PerformanceTracker(get_global_metrics())
    return performance.track_function(func)


def log_structured(component: str, level: str, message: str, **extra_fields):
    """Log a structured message."""
    logger = StructuredLogger(component)
    getattr(logger, level.lower())(message, **extra_fields)


def increment_metric(name: str, value: int = 1):
    """Increment a global metric."""
    get_global_metrics().increment_counter(name, value)


def set_metric(name: str, value: float):
    """Set a global gauge metric."""
    get_global_metrics().set_gauge(name, value)


def record_timing(name: str, duration: float):
    """Record a timing metric."""
    get_global_metrics().record_histogram(name, duration)