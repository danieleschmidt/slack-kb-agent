"""Comprehensive Monitoring and Observability System.

This module provides enterprise-grade monitoring, alerting, and observability
for the Slack KB Agent with real-time metrics, health checks, and performance tracking.
"""

import logging
import statistics
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import psutil

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class HealthCheck:
    """Health check definition."""
    name: str
    check_function: Callable[[], bool]
    timeout: float = 5.0
    interval: float = 30.0
    failure_threshold: int = 3
    recovery_threshold: int = 2
    description: str = ""


@dataclass
class Metric:
    """Metric data point."""
    name: str
    value: float
    timestamp: datetime = field(default_factory=datetime.now)
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Alert:
    """Alert definition."""
    id: str
    name: str
    severity: AlertSeverity
    message: str
    component: str
    timestamp: datetime = field(default_factory=datetime.now)
    resolved: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class MetricsCollector:
    """Collects and manages system and application metrics."""

    def __init__(self, max_data_points: int = 10000):
        self.metrics_data = defaultdict(lambda: deque(maxlen=max_data_points))
        self.custom_metrics = {}
        self.collection_interval = 10.0  # seconds
        self.collection_thread = None
        self.running = False
        self._lock = threading.Lock()

    def start_collection(self):
        """Start automatic metrics collection."""
        if self.running:
            return

        self.running = True
        self.collection_thread = threading.Thread(target=self._collection_loop, daemon=True)
        self.collection_thread.start()
        logger.info("Metrics collection started")

    def stop_collection(self):
        """Stop automatic metrics collection."""
        self.running = False
        if self.collection_thread and self.collection_thread.is_alive():
            self.collection_thread.join(timeout=5)
        logger.info("Metrics collection stopped")

    def record_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Record a custom metric."""
        with self._lock:
            metric = Metric(
                name=name,
                value=value,
                tags=tags or {},
                timestamp=datetime.now()
            )
            self.metrics_data[name].append(metric)

    def record_counter(self, name: str, increment: float = 1.0, tags: Optional[Dict[str, str]] = None):
        """Record a counter metric."""
        self.record_metric(f"{name}_total", increment, tags)

    def record_timer(self, name: str, duration: float, tags: Optional[Dict[str, str]] = None):
        """Record a timer metric."""
        self.record_metric(f"{name}_duration_seconds", duration, tags)

    def record_gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Record a gauge metric."""
        self.record_metric(f"{name}_value", value, tags)

    def get_metric_statistics(self, name: str, time_window: Optional[timedelta] = None) -> Dict[str, float]:
        """Get statistics for a metric over a time window."""
        with self._lock:
            if name not in self.metrics_data:
                return {}

            metrics = list(self.metrics_data[name])

            # Filter by time window if specified
            if time_window:
                cutoff_time = datetime.now() - time_window
                metrics = [m for m in metrics if m.timestamp >= cutoff_time]

            if not metrics:
                return {}

            values = [m.value for m in metrics]

            try:
                return {
                    "count": len(values),
                    "min": min(values),
                    "max": max(values),
                    "mean": statistics.mean(values),
                    "median": statistics.median(values),
                    "std_dev": statistics.stdev(values) if len(values) > 1 else 0.0,
                    "p95": self._percentile(values, 95),
                    "p99": self._percentile(values, 99),
                    "latest": values[-1] if values else 0.0
                }
            except Exception as e:
                logger.error(f"Error calculating statistics for {name}: {e}")
                return {"error": str(e)}

    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all collected metrics."""
        with self._lock:
            result = {}
            for name, metrics in self.metrics_data.items():
                if metrics:
                    latest_metric = metrics[-1]
                    result[name] = {
                        "value": latest_metric.value,
                        "timestamp": latest_metric.timestamp.isoformat(),
                        "tags": latest_metric.tags
                    }
            return result

    def _collection_loop(self):
        """Main metrics collection loop."""
        while self.running:
            try:
                self._collect_system_metrics()
                self._collect_application_metrics()
                time.sleep(self.collection_interval)
            except Exception as e:
                logger.error(f"Error in metrics collection: {e}")
                time.sleep(self.collection_interval)

    def _collect_system_metrics(self):
        """Collect system-level metrics."""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=None)
            self.record_gauge("system_cpu_usage_percent", cpu_percent)

            # Memory metrics
            memory = psutil.virtual_memory()
            self.record_gauge("system_memory_usage_percent", memory.percent)
            self.record_gauge("system_memory_available_bytes", memory.available)
            self.record_gauge("system_memory_used_bytes", memory.used)

            # Disk metrics
            disk = psutil.disk_usage('/')
            self.record_gauge("system_disk_usage_percent", disk.percent)
            self.record_gauge("system_disk_free_bytes", disk.free)
            self.record_gauge("system_disk_used_bytes", disk.used)

            # Network metrics
            network = psutil.net_io_counters()
            self.record_counter("system_network_bytes_sent", network.bytes_sent)
            self.record_counter("system_network_bytes_recv", network.bytes_recv)

            # Process metrics
            process = psutil.Process()
            self.record_gauge("process_memory_rss_bytes", process.memory_info().rss)
            self.record_gauge("process_memory_vms_bytes", process.memory_info().vms)
            self.record_gauge("process_cpu_percent", process.cpu_percent())
            self.record_gauge("process_num_threads", process.num_threads())

        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")

    def _collect_application_metrics(self):
        """Collect application-specific metrics."""
        try:
            # Python-specific metrics
            import gc
            self.record_gauge("python_gc_objects", len(gc.get_objects()))
            self.record_gauge("python_gc_collections", sum(gc.get_stats()[i]['collections'] for i in range(3)))

            # Thread metrics
            self.record_gauge("python_threads_active", threading.active_count())

        except Exception as e:
            logger.error(f"Error collecting application metrics: {e}")

    def _percentile(self, values: List[float], percentile: float) -> float:
        """Calculate percentile of values."""
        if not values:
            return 0.0

        sorted_values = sorted(values)
        index = (percentile / 100.0) * (len(sorted_values) - 1)

        if index.is_integer():
            return sorted_values[int(index)]
        else:
            lower_index = int(index)
            upper_index = min(lower_index + 1, len(sorted_values) - 1)
            weight = index - lower_index
            return sorted_values[lower_index] * (1 - weight) + sorted_values[upper_index] * weight


class HealthChecker:
    """Monitors system health with configurable checks."""

    def __init__(self):
        self.health_checks = {}
        self.check_results = {}
        self.check_history = defaultdict(lambda: deque(maxlen=100))
        self.monitoring_thread = None
        self.running = False
        self._lock = threading.Lock()

    def register_health_check(self, health_check: HealthCheck):
        """Register a health check."""
        with self._lock:
            self.health_checks[health_check.name] = health_check
            self.check_results[health_check.name] = {
                "status": HealthStatus.HEALTHY,
                "consecutive_failures": 0,
                "consecutive_successes": 0,
                "last_check": None,
                "last_error": None
            }
        logger.info(f"Registered health check: {health_check.name}")

    def start_monitoring(self):
        """Start health monitoring."""
        if self.running:
            return

        self.running = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info("Health monitoring started")

    def stop_monitoring(self):
        """Stop health monitoring."""
        self.running = False
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5)
        logger.info("Health monitoring stopped")

    def run_health_check(self, name: str) -> Dict[str, Any]:
        """Run a specific health check manually."""
        if name not in self.health_checks:
            return {"error": f"Health check {name} not found"}

        return self._execute_health_check(name)

    def get_health_status(self) -> Dict[str, Any]:
        """Get overall health status."""
        with self._lock:
            overall_status = HealthStatus.HEALTHY
            checks_summary = {}

            for name, result in self.check_results.items():
                checks_summary[name] = {
                    "status": result["status"].value,
                    "last_check": result["last_check"].isoformat() if result["last_check"] else None,
                    "consecutive_failures": result["consecutive_failures"],
                    "last_error": result["last_error"]
                }

                # Determine overall status
                if result["status"] == HealthStatus.CRITICAL:
                    overall_status = HealthStatus.CRITICAL
                elif result["status"] == HealthStatus.UNHEALTHY and overall_status != HealthStatus.CRITICAL:
                    overall_status = HealthStatus.UNHEALTHY
                elif result["status"] == HealthStatus.DEGRADED and overall_status == HealthStatus.HEALTHY:
                    overall_status = HealthStatus.DEGRADED

            return {
                "overall_status": overall_status.value,
                "checks": checks_summary,
                "timestamp": datetime.now().isoformat()
            }

    def _monitoring_loop(self):
        """Main health monitoring loop."""
        while self.running:
            try:
                self._run_all_checks()
                time.sleep(5)  # Check every 5 seconds
            except Exception as e:
                logger.error(f"Error in health monitoring loop: {e}")
                time.sleep(5)

    def _run_all_checks(self):
        """Run all registered health checks."""
        current_time = datetime.now()

        for name, health_check in self.health_checks.items():
            with self._lock:
                result = self.check_results[name]

                # Check if it's time to run this check
                if (result["last_check"] is None or
                    current_time - result["last_check"] >= timedelta(seconds=health_check.interval)):

                    self._execute_health_check(name)

    def _execute_health_check(self, name: str) -> Dict[str, Any]:
        """Execute a single health check."""
        health_check = self.health_checks[name]
        result = self.check_results[name]

        start_time = time.time()
        check_passed = False
        error_message = None

        try:
            # Run the health check with timeout
            check_passed = self._run_with_timeout(health_check.check_function, health_check.timeout)
        except Exception as e:
            error_message = str(e)
            logger.error(f"Health check {name} failed: {e}")

        execution_time = time.time() - start_time

        # Update result
        with self._lock:
            result["last_check"] = datetime.now()
            result["last_error"] = error_message

            if check_passed:
                result["consecutive_failures"] = 0
                result["consecutive_successes"] += 1

                # Recovery check
                if (result["status"] != HealthStatus.HEALTHY and
                    result["consecutive_successes"] >= health_check.recovery_threshold):
                    result["status"] = HealthStatus.HEALTHY
                    logger.info(f"Health check {name} recovered")
            else:
                result["consecutive_successes"] = 0
                result["consecutive_failures"] += 1

                # Failure threshold check
                if result["consecutive_failures"] >= health_check.failure_threshold:
                    if result["consecutive_failures"] >= health_check.failure_threshold * 2:
                        result["status"] = HealthStatus.CRITICAL
                    elif result["consecutive_failures"] >= health_check.failure_threshold:
                        result["status"] = HealthStatus.UNHEALTHY
                    else:
                        result["status"] = HealthStatus.DEGRADED

                    logger.warning(f"Health check {name} failing: {result['consecutive_failures']} consecutive failures")

            # Record history
            self.check_history[name].append({
                "timestamp": result["last_check"],
                "passed": check_passed,
                "execution_time": execution_time,
                "error": error_message
            })

        return {
            "name": name,
            "passed": check_passed,
            "execution_time": execution_time,
            "error": error_message,
            "status": result["status"].value
        }

    def _run_with_timeout(self, func: Callable[[], bool], timeout: float) -> bool:
        """Run function with timeout."""
        import signal

        def timeout_handler(signum, frame):
            raise TimeoutError("Health check timed out")

        # Set timeout
        if hasattr(signal, 'SIGALRM'):  # Unix systems
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(timeout))

        try:
            result = func()
            return bool(result)
        finally:
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)


class AlertManager:
    """Manages alerts and notifications."""

    def __init__(self, max_alerts: int = 1000):
        self.alerts = deque(maxlen=max_alerts)
        self.alert_rules = {}
        self.notification_handlers = {}
        self.alert_history = defaultdict(lambda: deque(maxlen=100))
        self._lock = threading.Lock()

    def create_alert(self, alert: Alert):
        """Create a new alert."""
        with self._lock:
            self.alerts.append(alert)
            self.alert_history[alert.component].append(alert)

            # Trigger notifications
            self._trigger_notifications(alert)

            logger.warning(f"Alert created: {alert.name} ({alert.severity.value}) - {alert.message}")

    def resolve_alert(self, alert_id: str):
        """Resolve an alert by ID."""
        with self._lock:
            for alert in self.alerts:
                if alert.id == alert_id and not alert.resolved:
                    alert.resolved = True
                    alert.metadata["resolved_at"] = datetime.now().isoformat()
                    logger.info(f"Alert resolved: {alert.name}")
                    return True
            return False

    def register_alert_rule(self, name: str, condition: Callable[[Dict[str, Any]], bool],
                          alert_template: Dict[str, Any]):
        """Register an alert rule."""
        self.alert_rules[name] = {
            "condition": condition,
            "template": alert_template,
            "last_triggered": None
        }

    def register_notification_handler(self, severity: AlertSeverity, handler: Callable[[Alert], None]):
        """Register notification handler for alert severity."""
        if severity not in self.notification_handlers:
            self.notification_handlers[severity] = []
        self.notification_handlers[severity].append(handler)

    def evaluate_alert_rules(self, metrics: Dict[str, Any]):
        """Evaluate all alert rules against current metrics."""
        for rule_name, rule in self.alert_rules.items():
            try:
                if rule["condition"](metrics):
                    # Check cooldown period
                    if (rule["last_triggered"] is None or
                        datetime.now() - rule["last_triggered"] > timedelta(minutes=5)):

                        # Create alert from template
                        template = rule["template"]
                        alert = Alert(
                            id=f"{rule_name}_{int(time.time())}",
                            name=template["name"],
                            severity=AlertSeverity(template["severity"]),
                            message=template["message"].format(**metrics),
                            component=template["component"],
                            metadata={"rule": rule_name, "metrics": metrics}
                        )

                        self.create_alert(alert)
                        rule["last_triggered"] = datetime.now()

            except Exception as e:
                logger.error(f"Error evaluating alert rule {rule_name}: {e}")

    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get all active (unresolved) alerts."""
        with self._lock:
            return [
                {
                    "id": alert.id,
                    "name": alert.name,
                    "severity": alert.severity.value,
                    "message": alert.message,
                    "component": alert.component,
                    "timestamp": alert.timestamp.isoformat(),
                    "metadata": alert.metadata
                }
                for alert in self.alerts if not alert.resolved
            ]

    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alert statistics."""
        with self._lock:
            active_alerts = [a for a in self.alerts if not a.resolved]

            severity_counts = defaultdict(int)
            component_counts = defaultdict(int)

            for alert in active_alerts:
                severity_counts[alert.severity.value] += 1
                component_counts[alert.component] += 1

            return {
                "total_alerts": len(self.alerts),
                "active_alerts": len(active_alerts),
                "resolved_alerts": len(self.alerts) - len(active_alerts),
                "alerts_by_severity": dict(severity_counts),
                "alerts_by_component": dict(component_counts),
                "alert_rules": len(self.alert_rules)
            }

    def _trigger_notifications(self, alert: Alert):
        """Trigger notifications for an alert."""
        handlers = self.notification_handlers.get(alert.severity, [])
        for handler in handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Notification handler failed: {e}")


class ComprehensiveMonitor:
    """Main monitoring orchestrator."""

    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.health_checker = HealthChecker()
        self.alert_manager = AlertManager()
        self.running = False
        self._setup_default_checks()
        self._setup_default_alerts()

    def start(self):
        """Start all monitoring components."""
        if self.running:
            return

        self.running = True
        self.metrics_collector.start_collection()
        self.health_checker.start_monitoring()

        # Start alert evaluation loop
        self._start_alert_evaluation()

        logger.info("Comprehensive monitoring started")

    def stop(self):
        """Stop all monitoring components."""
        self.running = False
        self.metrics_collector.stop_collection()
        self.health_checker.stop_monitoring()
        logger.info("Comprehensive monitoring stopped")

    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get comprehensive monitoring status."""
        return {
            "monitoring_active": self.running,
            "health_status": self.health_checker.get_health_status(),
            "active_alerts": self.alert_manager.get_active_alerts(),
            "alert_statistics": self.alert_manager.get_alert_statistics(),
            "system_metrics": self.metrics_collector.get_all_metrics(),
            "timestamp": datetime.now().isoformat()
        }

    def record_operation_metric(self, operation: str, duration: float, success: bool):
        """Record operation metrics."""
        self.metrics_collector.record_timer(f"operation_{operation}", duration)
        self.metrics_collector.record_counter(
            f"operation_{operation}_total",
            tags={"status": "success" if success else "error"}
        )

    def _setup_default_checks(self):
        """Setup default health checks."""
        # Database connectivity check
        def check_database():
            try:
                # Mock database check
                return True
            except Exception:
                return False

        self.health_checker.register_health_check(HealthCheck(
            name="database_connectivity",
            check_function=check_database,
            timeout=5.0,
            interval=30.0,
            failure_threshold=3,
            description="Database connectivity check"
        ))

        # Memory usage check
        def check_memory():
            memory = psutil.virtual_memory()
            return memory.percent < 90  # Alert if memory usage > 90%

        self.health_checker.register_health_check(HealthCheck(
            name="memory_usage",
            check_function=check_memory,
            timeout=2.0,
            interval=15.0,
            failure_threshold=2,
            description="Memory usage check"
        ))

        # Disk space check
        def check_disk_space():
            disk = psutil.disk_usage('/')
            return disk.percent < 85  # Alert if disk usage > 85%

        self.health_checker.register_health_check(HealthCheck(
            name="disk_space",
            check_function=check_disk_space,
            timeout=2.0,
            interval=60.0,
            failure_threshold=1,
            description="Disk space check"
        ))

    def _setup_default_alerts(self):
        """Setup default alert rules."""
        # High memory usage alert
        def high_memory_condition(metrics):
            memory_metric = metrics.get("system_memory_usage_percent")
            return memory_metric and memory_metric.get("value", 0) > 85

        self.alert_manager.register_alert_rule(
            "high_memory_usage",
            high_memory_condition,
            {
                "name": "High Memory Usage",
                "severity": "warning",
                "message": "Memory usage is {system_memory_usage_percent[value]:.1f}%",
                "component": "system"
            }
        )

        # High CPU usage alert
        def high_cpu_condition(metrics):
            cpu_metric = metrics.get("system_cpu_usage_percent")
            return cpu_metric and cpu_metric.get("value", 0) > 80

        self.alert_manager.register_alert_rule(
            "high_cpu_usage",
            high_cpu_condition,
            {
                "name": "High CPU Usage",
                "severity": "warning",
                "message": "CPU usage is {system_cpu_usage_percent[value]:.1f}%",
                "component": "system"
            }
        )

        # Low disk space alert
        def low_disk_condition(metrics):
            disk_metric = metrics.get("system_disk_usage_percent")
            return disk_metric and disk_metric.get("value", 0) > 85

        self.alert_manager.register_alert_rule(
            "low_disk_space",
            low_disk_condition,
            {
                "name": "Low Disk Space",
                "severity": "error",
                "message": "Disk usage is {system_disk_usage_percent[value]:.1f}%",
                "component": "system"
            }
        )

    def _start_alert_evaluation(self):
        """Start alert evaluation loop."""
        def alert_evaluation_loop():
            while self.running:
                try:
                    metrics = self.metrics_collector.get_all_metrics()
                    self.alert_manager.evaluate_alert_rules(metrics)
                    time.sleep(30)  # Evaluate every 30 seconds
                except Exception as e:
                    logger.error(f"Error in alert evaluation: {e}")
                    time.sleep(30)

        threading.Thread(target=alert_evaluation_loop, daemon=True).start()


# Global monitoring instance
_comprehensive_monitor = None

def get_comprehensive_monitor() -> ComprehensiveMonitor:
    """Get global comprehensive monitor instance."""
    global _comprehensive_monitor
    if _comprehensive_monitor is None:
        _comprehensive_monitor = ComprehensiveMonitor()
    return _comprehensive_monitor

def start_monitoring():
    """Start comprehensive monitoring."""
    monitor = get_comprehensive_monitor()
    monitor.start()

def stop_monitoring():
    """Stop comprehensive monitoring."""
    monitor = get_comprehensive_monitor()
    monitor.stop()

def get_monitoring_status() -> Dict[str, Any]:
    """Get current monitoring status."""
    monitor = get_comprehensive_monitor()
    return monitor.get_monitoring_status()

def record_operation(operation: str, duration: float, success: bool = True):
    """Record operation metrics."""
    monitor = get_comprehensive_monitor()
    monitor.record_operation_metric(operation, duration, success)
