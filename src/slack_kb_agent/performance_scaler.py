"""
Advanced performance optimization and auto-scaling for Slack KB Agent.
Implements intelligent resource management, caching strategies, and load balancing.
"""

import asyncio
import logging
import threading
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import psutil

logger = logging.getLogger(__name__)


class ScalingStrategy(Enum):
    """Auto-scaling strategies."""
    REACTIVE = "reactive"  # Scale based on current load
    PREDICTIVE = "predictive"  # Scale based on predicted load
    SCHEDULED = "scheduled"  # Scale based on schedule
    HYBRID = "hybrid"  # Combination of strategies


@dataclass
class PerformanceMetrics:
    """Performance metrics for scaling decisions."""
    cpu_usage: float
    memory_usage: float
    response_time: float
    request_rate: float
    error_rate: float
    concurrent_users: int
    timestamp: float = field(default_factory=time.time)


@dataclass
class ScalingEvent:
    """Represents a scaling event."""
    event_type: str  # "scale_up", "scale_down"
    trigger: str  # What triggered the scaling
    timestamp: float
    old_capacity: int
    new_capacity: int
    metrics: PerformanceMetrics


class PerformanceMonitor:
    """Monitors system performance and resource usage."""

    def __init__(self, history_size: int = 1000):
        self.metrics_history: deque = deque(maxlen=history_size)
        self.current_metrics = PerformanceMetrics(0, 0, 0, 0, 0, 0)
        self.request_times: deque = deque(maxlen=100)
        self.error_count = 0
        self.request_count = 0
        self.start_time = time.time()

    def record_request(self, response_time: float, success: bool = True):
        """Record a request for performance tracking."""
        self.request_times.append(response_time)
        self.request_count += 1

        if not success:
            self.error_count += 1

    def get_current_metrics(self) -> PerformanceMetrics:
        """Get current system performance metrics."""
        # System resource usage
        cpu_usage = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        memory_usage = memory.percent

        # Calculate response time (average of recent requests)
        avg_response_time = sum(self.request_times) / len(self.request_times) if self.request_times else 0

        # Calculate request rate (requests per second)
        elapsed_time = max(1, time.time() - self.start_time)
        request_rate = self.request_count / elapsed_time

        # Calculate error rate
        error_rate = (self.error_count / max(1, self.request_count)) * 100

        # Estimate concurrent users (simplified)
        concurrent_users = min(len(self.request_times), int(request_rate * avg_response_time))

        metrics = PerformanceMetrics(
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            response_time=avg_response_time,
            request_rate=request_rate,
            error_rate=error_rate,
            concurrent_users=concurrent_users
        )

        self.metrics_history.append(metrics)
        self.current_metrics = metrics
        return metrics

    def get_trend_analysis(self, window_size: int = 60) -> Dict[str, Any]:
        """Analyze performance trends over time."""
        if len(self.metrics_history) < 2:
            return {"trend": "insufficient_data"}

        recent_metrics = list(self.metrics_history)[-window_size:]

        # Calculate trends
        cpu_trend = self._calculate_trend([m.cpu_usage for m in recent_metrics])
        memory_trend = self._calculate_trend([m.memory_usage for m in recent_metrics])
        response_time_trend = self._calculate_trend([m.response_time for m in recent_metrics])
        request_rate_trend = self._calculate_trend([m.request_rate for m in recent_metrics])

        return {
            "cpu_trend": cpu_trend,
            "memory_trend": memory_trend,
            "response_time_trend": response_time_trend,
            "request_rate_trend": request_rate_trend,
            "overall_trend": self._determine_overall_trend({
                "cpu": cpu_trend,
                "memory": memory_trend,
                "response_time": response_time_trend,
                "request_rate": request_rate_trend
            })
        }

    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction for a list of values."""
        if len(values) < 2:
            return "stable"

        # Simple linear trend calculation
        n = len(values)
        x_avg = (n - 1) / 2
        y_avg = sum(values) / n

        numerator = sum((i - x_avg) * (values[i] - y_avg) for i in range(n))
        denominator = sum((i - x_avg) ** 2 for i in range(n))

        if denominator == 0:
            return "stable"

        slope = numerator / denominator

        if abs(slope) < 0.1:
            return "stable"
        elif slope > 0:
            return "increasing"
        else:
            return "decreasing"

    def _determine_overall_trend(self, trends: Dict[str, str]) -> str:
        """Determine overall system trend."""
        increasing_count = sum(1 for trend in trends.values() if trend == "increasing")
        decreasing_count = sum(1 for trend in trends.values() if trend == "decreasing")

        if increasing_count >= 3:
            return "degrading"
        elif decreasing_count >= 3:
            return "improving"
        else:
            return "stable"


class AutoScaler:
    """Implements intelligent auto-scaling based on performance metrics."""

    def __init__(self, strategy: ScalingStrategy = ScalingStrategy.HYBRID,
                 min_capacity: int = 1, max_capacity: int = 10):
        self.strategy = strategy
        self.min_capacity = min_capacity
        self.max_capacity = max_capacity
        self.current_capacity = min_capacity
        self.scaling_events: List[ScalingEvent] = []
        self.scaling_cooldown = 300  # 5 minutes
        self.last_scaling_time = 0

        # Scaling thresholds
        self.scale_up_thresholds = {
            "cpu_usage": 80,
            "memory_usage": 85,
            "response_time": 2.0,
            "error_rate": 5.0
        }

        self.scale_down_thresholds = {
            "cpu_usage": 30,
            "memory_usage": 40,
            "response_time": 0.5,
            "error_rate": 1.0
        }

    def should_scale_up(self, metrics: PerformanceMetrics) -> bool:
        """Determine if scaling up is needed."""
        if time.time() - self.last_scaling_time < self.scaling_cooldown:
            return False

        if self.current_capacity >= self.max_capacity:
            return False

        # Check if any threshold is exceeded
        triggers = []
        if metrics.cpu_usage > self.scale_up_thresholds["cpu_usage"]:
            triggers.append("cpu")
        if metrics.memory_usage > self.scale_up_thresholds["memory_usage"]:
            triggers.append("memory")
        if metrics.response_time > self.scale_up_thresholds["response_time"]:
            triggers.append("response_time")
        if metrics.error_rate > self.scale_up_thresholds["error_rate"]:
            triggers.append("error_rate")

        return len(triggers) >= 2  # Require multiple triggers

    def should_scale_down(self, metrics: PerformanceMetrics) -> bool:
        """Determine if scaling down is possible."""
        if time.time() - self.last_scaling_time < self.scaling_cooldown:
            return False

        if self.current_capacity <= self.min_capacity:
            return False

        # Check if all metrics are below scale-down thresholds
        return (metrics.cpu_usage < self.scale_down_thresholds["cpu_usage"] and
                metrics.memory_usage < self.scale_down_thresholds["memory_usage"] and
                metrics.response_time < self.scale_down_thresholds["response_time"] and
                metrics.error_rate < self.scale_down_thresholds["error_rate"])

    def scale_up(self, metrics: PerformanceMetrics, trigger: str) -> bool:
        """Scale up the system capacity."""
        old_capacity = self.current_capacity
        new_capacity = min(self.max_capacity, old_capacity + 1)

        if new_capacity == old_capacity:
            return False

        self.current_capacity = new_capacity
        self.last_scaling_time = time.time()

        event = ScalingEvent(
            event_type="scale_up",
            trigger=trigger,
            timestamp=time.time(),
            old_capacity=old_capacity,
            new_capacity=new_capacity,
            metrics=metrics
        )

        self.scaling_events.append(event)
        logger.info(f"Scaled up from {old_capacity} to {new_capacity} (trigger: {trigger})")
        return True

    def scale_down(self, metrics: PerformanceMetrics, trigger: str) -> bool:
        """Scale down the system capacity."""
        old_capacity = self.current_capacity
        new_capacity = max(self.min_capacity, old_capacity - 1)

        if new_capacity == old_capacity:
            return False

        self.current_capacity = new_capacity
        self.last_scaling_time = time.time()

        event = ScalingEvent(
            event_type="scale_down",
            trigger=trigger,
            timestamp=time.time(),
            old_capacity=old_capacity,
            new_capacity=new_capacity,
            metrics=metrics
        )

        self.scaling_events.append(event)
        logger.info(f"Scaled down from {old_capacity} to {new_capacity} (trigger: {trigger})")
        return True


class IntelligentLoadBalancer:
    """Distributes requests across available resources intelligently."""

    def __init__(self, initial_workers: int = 4):
        self.workers: List[Dict[str, Any]] = []
        self.worker_metrics: Dict[int, List[float]] = defaultdict(list)
        self.request_queue = asyncio.Queue()
        self.executor = ThreadPoolExecutor(max_workers=initial_workers)
        self.worker_id_counter = 0

        # Initialize workers
        for _ in range(initial_workers):
            self._add_worker()

    def _add_worker(self) -> int:
        """Add a new worker to the pool."""
        worker_id = self.worker_id_counter
        self.worker_id_counter += 1

        worker = {
            "id": worker_id,
            "status": "idle",
            "current_load": 0,
            "total_requests": 0,
            "avg_response_time": 0,
            "last_request_time": 0
        }

        self.workers.append(worker)
        logger.info(f"Added worker {worker_id}")
        return worker_id

    def _remove_worker(self) -> bool:
        """Remove a worker from the pool."""
        if len(self.workers) <= 1:
            return False

        # Find least busy worker
        idle_workers = [w for w in self.workers if w["status"] == "idle"]
        if idle_workers:
            worker = idle_workers[0]
            self.workers.remove(worker)
            logger.info(f"Removed worker {worker['id']}")
            return True

        return False

    def get_best_worker(self) -> Optional[Dict[str, Any]]:
        """Select the best worker for the next request."""
        if not self.workers:
            return None

        # Prefer idle workers
        idle_workers = [w for w in self.workers if w["status"] == "idle"]
        if idle_workers:
            # Among idle workers, choose the one with best performance
            return min(idle_workers, key=lambda w: w["avg_response_time"])

        # If no idle workers, choose the least loaded
        return min(self.workers, key=lambda w: w["current_load"])

    def assign_request(self, request_handler: Callable, *args, **kwargs) -> asyncio.Future:
        """Assign a request to the best available worker."""
        worker = self.get_best_worker()
        if not worker:
            raise RuntimeError("No workers available")

        worker["status"] = "busy"
        worker["current_load"] += 1
        worker["total_requests"] += 1
        worker["last_request_time"] = time.time()

        # Submit task to thread pool
        future = self.executor.submit(self._execute_request, worker, request_handler, *args, **kwargs)
        return future

    def _execute_request(self, worker: Dict[str, Any], handler: Callable, *args, **kwargs) -> Any:
        """Execute a request and update worker metrics."""
        start_time = time.time()

        try:
            result = handler(*args, **kwargs)
            success = True
        except Exception as e:
            logger.error(f"Request failed on worker {worker['id']}: {e}")
            result = None
            success = False
        finally:
            # Update worker metrics
            response_time = time.time() - start_time
            worker["current_load"] -= 1
            worker["status"] = "idle"

            # Update average response time
            old_avg = worker["avg_response_time"]
            total_requests = worker["total_requests"]
            worker["avg_response_time"] = ((old_avg * (total_requests - 1)) + response_time) / total_requests

            # Record metrics
            self.worker_metrics[worker["id"]].append(response_time)
            if len(self.worker_metrics[worker["id"]]) > 100:
                self.worker_metrics[worker["id"]].pop(0)

        return result

    def scale_workers(self, new_count: int):
        """Scale the number of workers."""
        current_count = len(self.workers)

        if new_count > current_count:
            # Scale up
            for _ in range(new_count - current_count):
                self._add_worker()
        elif new_count < current_count:
            # Scale down
            for _ in range(current_count - new_count):
                if not self._remove_worker():
                    break

        # Update thread pool
        self.executor._max_workers = new_count


class AdvancedCacheManager:
    """Intelligent multi-level caching system."""

    def __init__(self, max_memory_cache: int = 1000, max_disk_cache: int = 10000):
        self.l1_cache = {}  # Memory cache (fastest)
        self.l2_cache = {}  # Compressed memory cache
        self.l3_cache = {}  # Disk cache (simulated)

        self.max_l1_size = max_memory_cache
        self.max_l2_size = max_memory_cache * 2
        self.max_l3_size = max_disk_cache

        # Cache statistics
        self.cache_stats = {
            "l1_hits": 0, "l1_misses": 0,
            "l2_hits": 0, "l2_misses": 0,
            "l3_hits": 0, "l3_misses": 0,
            "evictions": 0
        }

        # Access tracking for intelligent eviction
        self.access_times: Dict[str, float] = {}
        self.access_frequency: Dict[str, int] = defaultdict(int)

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache with intelligent promotion."""
        current_time = time.time()

        # Check L1 cache (fastest)
        if key in self.l1_cache:
            self.cache_stats["l1_hits"] += 1
            self.access_times[key] = current_time
            self.access_frequency[key] += 1
            return self.l1_cache[key]

        self.cache_stats["l1_misses"] += 1

        # Check L2 cache
        if key in self.l2_cache:
            self.cache_stats["l2_hits"] += 1
            value = self.l2_cache[key]
            # Promote to L1 if frequently accessed
            self.access_frequency[key] += 1
            if self.access_frequency[key] > 3:
                self._promote_to_l1(key, value)
            return value

        self.cache_stats["l2_misses"] += 1

        # Check L3 cache
        if key in self.l3_cache:
            self.cache_stats["l3_hits"] += 1
            value = self.l3_cache[key]
            # Promote to L2
            self._promote_to_l2(key, value)
            self.access_frequency[key] += 1
            return value

        self.cache_stats["l3_misses"] += 1
        return None

    def set(self, key: str, value: Any, ttl: Optional[float] = None):
        """Set value in cache with intelligent placement."""
        current_time = time.time()
        self.access_times[key] = current_time
        self.access_frequency[key] = 1

        # Store in L1 cache
        self._set_l1(key, value)

    def _set_l1(self, key: str, value: Any):
        """Set value in L1 cache with eviction if needed."""
        if len(self.l1_cache) >= self.max_l1_size and key not in self.l1_cache:
            self._evict_from_l1()

        self.l1_cache[key] = value

    def _promote_to_l1(self, key: str, value: Any):
        """Promote value from lower cache to L1."""
        self._set_l1(key, value)
        # Remove from L2 if present
        self.l2_cache.pop(key, None)

    def _promote_to_l2(self, key: str, value: Any):
        """Promote value from L3 to L2."""
        if len(self.l2_cache) >= self.max_l2_size and key not in self.l2_cache:
            self._evict_from_l2()

        self.l2_cache[key] = value
        self.l3_cache.pop(key, None)

    def _evict_from_l1(self):
        """Evict least valuable item from L1 cache."""
        if not self.l1_cache:
            return

        # Find item with lowest score (LRU + frequency)
        current_time = time.time()
        scores = {}

        for key in self.l1_cache:
            last_access = self.access_times.get(key, 0)
            frequency = self.access_frequency.get(key, 1)
            # Score combines recency and frequency
            scores[key] = frequency / (current_time - last_access + 1)

        # Evict item with lowest score
        evict_key = min(scores.keys(), key=lambda k: scores[k])
        value = self.l1_cache.pop(evict_key)

        # Demote to L2
        if len(self.l2_cache) < self.max_l2_size:
            self.l2_cache[evict_key] = value
        elif len(self.l3_cache) < self.max_l3_size:
            self.l3_cache[evict_key] = value

        self.cache_stats["evictions"] += 1

    def _evict_from_l2(self):
        """Evict item from L2 to L3."""
        if not self.l2_cache:
            return

        # Simple FIFO for L2 -> L3
        key = next(iter(self.l2_cache))
        value = self.l2_cache.pop(key)

        if len(self.l3_cache) < self.max_l3_size:
            self.l3_cache[key] = value

        self.cache_stats["evictions"] += 1

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        total_requests = sum([
            self.cache_stats["l1_hits"], self.cache_stats["l1_misses"],
            self.cache_stats["l2_hits"], self.cache_stats["l2_misses"],
            self.cache_stats["l3_hits"], self.cache_stats["l3_misses"]
        ])

        if total_requests == 0:
            hit_rate = 0
        else:
            total_hits = (self.cache_stats["l1_hits"] +
                         self.cache_stats["l2_hits"] +
                         self.cache_stats["l3_hits"])
            hit_rate = (total_hits / total_requests) * 100

        return {
            "hit_rate_percent": hit_rate,
            "l1_size": len(self.l1_cache),
            "l2_size": len(self.l2_cache),
            "l3_size": len(self.l3_cache),
            "total_evictions": self.cache_stats["evictions"],
            "cache_stats": self.cache_stats
        }


class ResourceOptimizer:
    """Optimizes resource usage based on performance patterns."""

    def __init__(self):
        self.performance_monitor = PerformanceMonitor()
        self.auto_scaler = AutoScaler()
        self.load_balancer = IntelligentLoadBalancer()
        self.cache_manager = AdvancedCacheManager()
        self.optimization_thread = None
        self.running = False

    def start_optimization(self):
        """Start the optimization loop."""
        if self.running:
            return

        self.running = True
        self.optimization_thread = threading.Thread(target=self._optimization_loop, daemon=True)
        self.optimization_thread.start()
        logger.info("Resource optimization started")

    def stop_optimization(self):
        """Stop the optimization loop."""
        self.running = False
        if self.optimization_thread and self.optimization_thread.is_alive():
            self.optimization_thread.join(timeout=5)
        logger.info("Resource optimization stopped")

    def _optimization_loop(self):
        """Main optimization loop."""
        while self.running:
            try:
                # Get current metrics
                metrics = self.performance_monitor.get_current_metrics()

                # Check for scaling needs
                if self.auto_scaler.should_scale_up(metrics):
                    self.auto_scaler.scale_up(metrics, "performance_threshold")
                    # Update load balancer workers
                    self.load_balancer.scale_workers(self.auto_scaler.current_capacity)

                elif self.auto_scaler.should_scale_down(metrics):
                    self.auto_scaler.scale_down(metrics, "resource_optimization")
                    # Update load balancer workers
                    self.load_balancer.scale_workers(self.auto_scaler.current_capacity)

                # Optimize cache based on hit rate
                cache_stats = self.cache_manager.get_cache_stats()
                if cache_stats["hit_rate_percent"] < 70:
                    logger.info("Low cache hit rate detected, consider increasing cache size")

                # Sleep before next optimization cycle
                time.sleep(30)  # Check every 30 seconds

            except Exception as e:
                logger.error(f"Error in optimization loop: {e}")
                time.sleep(60)  # Wait longer on error

    def get_optimization_status(self) -> Dict[str, Any]:
        """Get current optimization status."""
        metrics = self.performance_monitor.get_current_metrics()
        cache_stats = self.cache_manager.get_cache_stats()
        trend_analysis = self.performance_monitor.get_trend_analysis()

        return {
            "running": self.running,
            "current_capacity": self.auto_scaler.current_capacity,
            "performance_metrics": {
                "cpu_usage": metrics.cpu_usage,
                "memory_usage": metrics.memory_usage,
                "response_time": metrics.response_time,
                "request_rate": metrics.request_rate,
                "error_rate": metrics.error_rate,
                "concurrent_users": metrics.concurrent_users
            },
            "cache_performance": cache_stats,
            "trend_analysis": trend_analysis,
            "scaling_events": len(self.auto_scaler.scaling_events),
            "workers": len(self.load_balancer.workers)
        }


# Global resource optimizer instance
resource_optimizer = ResourceOptimizer()


def optimize_function(cache_key: Optional[str] = None, timeout: float = 30.0):
    """
    Decorator to optimize function execution with caching and load balancing.
    
    Args:
        cache_key: Optional cache key, if None will generate from function args
        timeout: Execution timeout in seconds
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Generate cache key if not provided
            if cache_key:
                key = cache_key
            else:
                key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"

            # Check cache first
            cached_result = resource_optimizer.cache_manager.get(key)
            if cached_result is not None:
                return cached_result

            # Execute with load balancing
            start_time = time.time()
            try:
                future = resource_optimizer.load_balancer.assign_request(func, *args, **kwargs)
                result = future.result(timeout=timeout)
                success = True
            except Exception as e:
                logger.error(f"Function execution failed: {e}")
                result = None
                success = False

            # Record performance metrics
            response_time = time.time() - start_time
            resource_optimizer.performance_monitor.record_request(response_time, success)

            # Cache successful results
            if success and result is not None:
                resource_optimizer.cache_manager.set(key, result)

            return result
        return wrapper
    return decorator


def get_scaling_status() -> Dict[str, Any]:
    """Get current scaling and performance status."""
    return resource_optimizer.get_optimization_status()
