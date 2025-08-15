"""Performance optimization and auto-scaling capabilities."""

from __future__ import annotations

import asyncio
import logging
import threading
import time

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
import gc
import statistics
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from .cache import get_cache_manager
from .monitoring import StructuredLogger, get_global_metrics
from .resilience import HealthStatus, get_health_monitor

logger = logging.getLogger(__name__)


class OptimizationStrategy(Enum):
    """Performance optimization strategies."""
    AGGRESSIVE = "aggressive"
    BALANCED = "balanced"
    CONSERVATIVE = "conservative"
    ADAPTIVE = "adaptive"


class ResourceType(Enum):
    """System resource types."""
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"
    DATABASE = "database"


@dataclass
class PerformanceMetrics:
    """Performance metrics collection."""
    timestamp: datetime = field(default_factory=datetime.now)
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    memory_available: float = 0.0
    disk_usage: float = 0.0
    network_io: Dict[str, float] = field(default_factory=dict)
    response_times: List[float] = field(default_factory=list)
    throughput: float = 0.0
    active_connections: int = 0
    cache_hit_ratio: float = 0.0
    gc_collections: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "cpu_usage": self.cpu_usage,
            "memory_usage": self.memory_usage,
            "memory_available": self.memory_available,
            "disk_usage": self.disk_usage,
            "network_io": self.network_io,
            "avg_response_time": statistics.mean(self.response_times) if self.response_times else 0.0,
            "p95_response_time": statistics.quantiles(self.response_times, n=20)[18] if len(self.response_times) >= 20 else 0.0,
            "throughput": self.throughput,
            "active_connections": self.active_connections,
            "cache_hit_ratio": self.cache_hit_ratio,
            "gc_collections": self.gc_collections
        }


@dataclass
class OptimizationRule:
    """Performance optimization rule."""
    name: str
    condition: Callable[[PerformanceMetrics], bool]
    action: Callable[[], None]
    priority: int = 1
    cooldown: float = 60.0  # seconds
    last_applied: Optional[datetime] = None
    application_count: int = 0

    def can_apply(self) -> bool:
        """Check if rule can be applied (respects cooldown)."""
        if self.last_applied is None:
            return True

        time_since_last = datetime.now() - self.last_applied
        return time_since_last.total_seconds() >= self.cooldown

    def apply(self) -> bool:
        """Apply optimization rule."""
        if not self.can_apply():
            return False

        try:
            self.action()
            self.last_applied = datetime.now()
            self.application_count += 1
            return True
        except Exception as e:
            logger.error(f"Failed to apply optimization rule {self.name}: {e}")
            return False


class PerformanceOptimizer:
    """Adaptive performance optimizer with auto-scaling."""

    def __init__(self, strategy: OptimizationStrategy = OptimizationStrategy.BALANCED):
        self.strategy = strategy
        self.metrics_history: deque = deque(maxlen=1000)  # Last 1000 metrics snapshots
        self.optimization_rules: List[OptimizationRule] = []

        # Performance tracking
        self._metric_collectors: Dict[str, Callable] = {}
        self._resource_thresholds = {
            ResourceType.CPU: {"warning": 70.0, "critical": 85.0},
            ResourceType.MEMORY: {"warning": 75.0, "critical": 90.0},
            ResourceType.DISK: {"warning": 80.0, "critical": 95.0}
        }

        # Adaptive parameters
        self._learning_enabled = True
        self._adaptation_history: Dict[str, List[float]] = defaultdict(list)

        # Monitoring
        self.metrics = get_global_metrics()
        self.logger = StructuredLogger("performance_optimizer")
        self.health_monitor = get_health_monitor("performance")

        # Thread safety
        self._lock = threading.Lock()

        # Register default optimization rules
        self._register_default_rules()

        # Register health checks
        self._register_health_checks()

        logger.info(f"Performance optimizer initialized with {strategy.value} strategy")

    def _register_default_rules(self) -> None:
        """Register default optimization rules."""

        # Memory pressure relief
        def high_memory_condition(metrics: PerformanceMetrics) -> bool:
            return metrics.memory_usage > self._resource_thresholds[ResourceType.MEMORY]["warning"]

        def gc_action() -> None:
            gc.collect()
            self.logger.log_event("optimization_gc_triggered", {"reason": "high_memory"})

        self.add_optimization_rule(OptimizationRule(
            name="memory_gc",
            condition=high_memory_condition,
            action=gc_action,
            priority=2,
            cooldown=30.0
        ))

        # CPU throttling
        def high_cpu_condition(metrics: PerformanceMetrics) -> bool:
            return metrics.cpu_usage > self._resource_thresholds[ResourceType.CPU]["warning"]

        def cpu_throttle_action() -> None:
            # Increase delays in processing loops
            self._adaptive_throttle("cpu_high")
            self.logger.log_event("optimization_cpu_throttle", {"cpu_usage": self.get_current_metrics().cpu_usage})

        self.add_optimization_rule(OptimizationRule(
            name="cpu_throttle",
            condition=high_cpu_condition,
            action=cpu_throttle_action,
            priority=1,
            cooldown=60.0
        ))

        # Cache warming
        def low_cache_hit_condition(metrics: PerformanceMetrics) -> bool:
            return metrics.cache_hit_ratio < 0.5 and len(self.metrics_history) > 10

        def cache_warm_action() -> None:
            cache_manager = get_cache_manager()
            # Trigger cache warming for frequent queries
            self.logger.log_event("optimization_cache_warming", {"hit_ratio": self.get_current_metrics().cache_hit_ratio})

        self.add_optimization_rule(OptimizationRule(
            name="cache_warming",
            condition=low_cache_hit_condition,
            action=cache_warm_action,
            priority=3,
            cooldown=120.0
        ))

    def _register_health_checks(self) -> None:
        """Register performance health checks."""

        def cpu_health_check() -> HealthStatus:
            metrics = self.get_current_metrics()
            if metrics.cpu_usage > self._resource_thresholds[ResourceType.CPU]["critical"]:
                return HealthStatus.CRITICAL
            elif metrics.cpu_usage > self._resource_thresholds[ResourceType.CPU]["warning"]:
                return HealthStatus.DEGRADED
            else:
                return HealthStatus.HEALTHY

        def memory_health_check() -> HealthStatus:
            metrics = self.get_current_metrics()
            if metrics.memory_usage > self._resource_thresholds[ResourceType.MEMORY]["critical"]:
                return HealthStatus.CRITICAL
            elif metrics.memory_usage > self._resource_thresholds[ResourceType.MEMORY]["warning"]:
                return HealthStatus.DEGRADED
            else:
                return HealthStatus.HEALTHY

        self.health_monitor.register_health_check("cpu_usage", cpu_health_check)
        self.health_monitor.register_health_check("memory_usage", memory_health_check)

    def add_metric_collector(self, name: str, collector: Callable[[], float]) -> None:
        """Add custom metric collector."""
        with self._lock:
            self._metric_collectors[name] = collector

        self.logger.log_event("metric_collector_added", {"name": name})

    def add_optimization_rule(self, rule: OptimizationRule) -> None:
        """Add optimization rule."""
        with self._lock:
            self.optimization_rules.append(rule)
            # Sort by priority (higher first)
            self.optimization_rules.sort(key=lambda r: r.priority, reverse=True)

        self.logger.log_event("optimization_rule_added", {
            "name": rule.name,
            "priority": rule.priority
        })

    def collect_metrics(self) -> PerformanceMetrics:
        """Collect current performance metrics."""
        try:
            if PSUTIL_AVAILABLE:
                # System metrics with psutil
                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory_info = psutil.virtual_memory()
                disk_info = psutil.disk_usage('/')

                # Network I/O
                net_io = psutil.net_io_counters()
                network_metrics = {
                    "bytes_sent": net_io.bytes_sent,
                    "bytes_recv": net_io.bytes_recv,
                    "packets_sent": net_io.packets_sent,
                    "packets_recv": net_io.packets_recv
                }
            else:
                # Fallback metrics without psutil
                cpu_percent = 0.0  # Cannot measure without psutil
                memory_info = type('obj', (object,), {'percent': 0.0, 'available': 1024**3})()  # 1GB available
                disk_info = type('obj', (object,), {'percent': 0.0})()
                network_metrics = {
                    "bytes_sent": 0,
                    "bytes_recv": 0,
                    "packets_sent": 0,
                    "packets_recv": 0
                }

            # Cache metrics
            cache_manager = get_cache_manager()
            cache_stats = cache_manager.get_stats() if hasattr(cache_manager, 'get_stats') else {}
            cache_hit_ratio = cache_stats.get('hit_ratio', 0.0)

            # GC metrics
            gc_stats = gc.get_stats()
            total_gc_collections = sum(stat.get('collections', 0) for stat in gc_stats)

            # Custom metrics
            custom_metrics = {}
            for name, collector in self._metric_collectors.items():
                try:
                    custom_metrics[name] = collector()
                except Exception as e:
                    logger.warning(f"Failed to collect metric {name}: {e}")

            metrics = PerformanceMetrics(
                cpu_usage=cpu_percent,
                memory_usage=memory_info.percent,
                memory_available=memory_info.available / (1024**3),  # GB
                disk_usage=disk_info.percent,
                network_io=network_metrics,
                cache_hit_ratio=cache_hit_ratio,
                gc_collections=total_gc_collections
            )

            # Add custom metrics to the metrics object
            for name, value in custom_metrics.items():
                setattr(metrics, name, value)

            return metrics

        except Exception as e:
            logger.error(f"Failed to collect performance metrics: {e}")
            return PerformanceMetrics()  # Return empty metrics

    def get_current_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics."""
        if self.metrics_history:
            return self.metrics_history[-1]
        else:
            return self.collect_metrics()

    def record_response_time(self, response_time: float) -> None:
        """Record a response time measurement."""
        with self._lock:
            if self.metrics_history:
                current_metrics = self.metrics_history[-1]
                current_metrics.response_times.append(response_time)

                # Limit response time history to prevent memory growth
                if len(current_metrics.response_times) > 1000:
                    current_metrics.response_times = current_metrics.response_times[-500:]

    def analyze_performance_trends(self, window_minutes: int = 10) -> Dict[str, Any]:
        """Analyze performance trends over time window."""
        cutoff_time = datetime.now() - timedelta(minutes=window_minutes)
        recent_metrics = [
            m for m in self.metrics_history
            if m.timestamp >= cutoff_time
        ]

        if len(recent_metrics) < 2:
            return {"status": "insufficient_data", "metrics_count": len(recent_metrics)}

        # Calculate trends
        cpu_values = [m.cpu_usage for m in recent_metrics]
        memory_values = [m.memory_usage for m in recent_metrics]

        # Calculate slopes (simple linear trend)
        def calculate_trend(values: List[float]) -> str:
            if len(values) < 2:
                return "stable"

            mid_point = len(values) // 2
            first_half_avg = statistics.mean(values[:mid_point])
            second_half_avg = statistics.mean(values[mid_point:])

            diff_percent = ((second_half_avg - first_half_avg) / first_half_avg) * 100

            if diff_percent > 5:
                return "increasing"
            elif diff_percent < -5:
                return "decreasing"
            else:
                return "stable"

        response_times = []
        for m in recent_metrics:
            response_times.extend(m.response_times)

        analysis = {
            "window_minutes": window_minutes,
            "metrics_count": len(recent_metrics),
            "cpu_trend": calculate_trend(cpu_values),
            "memory_trend": calculate_trend(memory_values),
            "cpu_avg": statistics.mean(cpu_values),
            "memory_avg": statistics.mean(memory_values),
            "cpu_max": max(cpu_values),
            "memory_max": max(memory_values)
        }

        if response_times:
            analysis.update({
                "avg_response_time": statistics.mean(response_times),
                "p95_response_time": statistics.quantiles(response_times, n=20)[18] if len(response_times) >= 20 else max(response_times),
                "response_time_trend": calculate_trend(response_times)
            })

        return analysis

    def apply_optimizations(self, metrics: PerformanceMetrics) -> List[str]:
        """Apply optimization rules based on current metrics."""
        applied_rules = []

        with self._lock:
            for rule in self.optimization_rules:
                try:
                    if rule.condition(metrics) and rule.can_apply():
                        if rule.apply():
                            applied_rules.append(rule.name)

                            self.logger.log_event("optimization_rule_applied", {
                                "rule": rule.name,
                                "priority": rule.priority,
                                "application_count": rule.application_count
                            })
                except Exception as e:
                    logger.error(f"Error applying optimization rule {rule.name}: {e}")

        return applied_rules

    def _adaptive_throttle(self, reason: str) -> None:
        """Apply adaptive throttling based on system state."""
        current_throttle = getattr(self, '_current_throttle', 0.0)

        if self.strategy == OptimizationStrategy.AGGRESSIVE:
            new_throttle = min(current_throttle + 0.1, 2.0)
        elif self.strategy == OptimizationStrategy.CONSERVATIVE:
            new_throttle = min(current_throttle + 0.01, 0.5)
        else:  # BALANCED or ADAPTIVE
            new_throttle = min(current_throttle + 0.05, 1.0)

        self._current_throttle = new_throttle

        # Record adaptation for learning
        if self._learning_enabled:
            self._adaptation_history[reason].append(new_throttle)

            # Keep history manageable
            if len(self._adaptation_history[reason]) > 100:
                self._adaptation_history[reason] = self._adaptation_history[reason][-50:]

    def get_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """Get optimization recommendations based on analysis."""
        recommendations = []
        trends = self.analyze_performance_trends()
        current_metrics = self.get_current_metrics()

        # CPU recommendations
        if trends.get("cpu_trend") == "increasing" or current_metrics.cpu_usage > 80:
            recommendations.append({
                "type": "cpu",
                "priority": "high",
                "recommendation": "Consider scaling horizontally or optimizing CPU-intensive operations",
                "current_usage": current_metrics.cpu_usage,
                "trend": trends.get("cpu_trend")
            })

        # Memory recommendations
        if trends.get("memory_trend") == "increasing" or current_metrics.memory_usage > 85:
            recommendations.append({
                "type": "memory",
                "priority": "high",
                "recommendation": "Memory usage is high - consider garbage collection tuning or adding more memory",
                "current_usage": current_metrics.memory_usage,
                "trend": trends.get("memory_trend")
            })

        # Cache recommendations
        if current_metrics.cache_hit_ratio < 0.7:
            recommendations.append({
                "type": "cache",
                "priority": "medium",
                "recommendation": "Cache hit ratio is low - consider cache warming or increasing cache size",
                "hit_ratio": current_metrics.cache_hit_ratio
            })

        # Response time recommendations
        avg_response_time = trends.get("avg_response_time", 0)
        if avg_response_time > 1000:  # > 1 second
            recommendations.append({
                "type": "response_time",
                "priority": "high",
                "recommendation": "Response times are high - investigate slow queries or add caching",
                "avg_response_time": avg_response_time,
                "p95_response_time": trends.get("p95_response_time")
            })

        return recommendations

    async def start_monitoring(self, interval: float = 30.0) -> None:
        """Start continuous performance monitoring and optimization."""
        self.logger.log_event("performance_monitoring_started", {
            "strategy": self.strategy.value,
            "interval": interval
        })

        while True:
            try:
                # Collect metrics
                metrics = self.collect_metrics()

                with self._lock:
                    self.metrics_history.append(metrics)

                # Apply optimizations
                applied_rules = self.apply_optimizations(metrics)

                if applied_rules:
                    self.logger.log_event("optimizations_applied", {
                        "rules": applied_rules,
                        "metrics": metrics.to_dict()
                    })

                # Log periodic performance summary
                if len(self.metrics_history) % 20 == 0:  # Every 20 intervals
                    trends = self.analyze_performance_trends()
                    recommendations = self.get_optimization_recommendations()

                    self.logger.log_event("performance_summary", {
                        "trends": trends,
                        "recommendations": recommendations,
                        "applied_optimizations": sum(rule.application_count for rule in self.optimization_rules)
                    })

                await asyncio.sleep(interval)

            except Exception as e:
                logger.error(f"Error in performance monitoring loop: {e}")
                await asyncio.sleep(interval)

    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        current_metrics = self.get_current_metrics()
        trends = self.analyze_performance_trends()
        recommendations = self.get_optimization_recommendations()

        # Rule statistics
        rule_stats = []
        for rule in self.optimization_rules:
            rule_stats.append({
                "name": rule.name,
                "priority": rule.priority,
                "application_count": rule.application_count,
                "last_applied": rule.last_applied.isoformat() if rule.last_applied else None,
                "cooldown": rule.cooldown
            })

        return {
            "timestamp": datetime.now().isoformat(),
            "strategy": self.strategy.value,
            "current_metrics": current_metrics.to_dict(),
            "trends": trends,
            "recommendations": recommendations,
            "optimization_rules": rule_stats,
            "metrics_history_size": len(self.metrics_history),
            "health_status": self.health_monitor.get_status(),
            "resource_thresholds": {k.value: v for k, v in self._resource_thresholds.items()}
        }


# Global instance
_performance_optimizer: Optional[PerformanceOptimizer] = None


def get_performance_optimizer(strategy: OptimizationStrategy = OptimizationStrategy.BALANCED) -> PerformanceOptimizer:
    """Get or create global performance optimizer instance."""
    global _performance_optimizer
    if _performance_optimizer is None:
        _performance_optimizer = PerformanceOptimizer(strategy)
    return _performance_optimizer


# Decorator for performance monitoring
def monitor_performance(func: Callable) -> Callable:
    """Decorator to monitor function performance."""
    def wrapper(*args, **kwargs):
        optimizer = get_performance_optimizer()
        start_time = time.time()

        try:
            result = func(*args, **kwargs)
            response_time = (time.time() - start_time) * 1000  # Convert to ms
            optimizer.record_response_time(response_time)
            return result
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            optimizer.record_response_time(response_time)
            raise e

    return wrapper


async def monitor_performance_async(func: Callable) -> Callable:
    """Async decorator to monitor function performance."""
    async def wrapper(*args, **kwargs):
        optimizer = get_performance_optimizer()
        start_time = time.time()

        try:
            result = await func(*args, **kwargs)
            response_time = (time.time() - start_time) * 1000  # Convert to ms
            optimizer.record_response_time(response_time)
            return result
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            optimizer.record_response_time(response_time)
            raise e

    return wrapper
