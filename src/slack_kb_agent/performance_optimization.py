"""Advanced performance optimization and scaling system.

This module provides comprehensive performance monitoring and optimization features:
- Intelligent caching with eviction strategies
- Query optimization and response caching
- Resource usage monitoring and optimization
- Database connection pooling and optimization
- Asynchronous processing capabilities
- Load balancing and scaling strategies
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import statistics
import threading
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.warning("psutil not available, system monitoring limited")


class CacheStrategy(Enum):
    """Cache eviction strategies."""
    LRU = "lru"           # Least Recently Used
    LFU = "lfu"           # Least Frequently Used
    FIFO = "fifo"         # First In, First Out
    TTL = "ttl"           # Time To Live
    ADAPTIVE = "adaptive"  # Adaptive based on usage patterns


class OptimizationLevel(Enum):
    """Performance optimization levels."""
    CONSERVATIVE = "conservative"  # Safe optimizations
    BALANCED = "balanced"          # Balance of performance and safety
    AGGRESSIVE = "aggressive"      # Maximum performance optimizations


@dataclass
class PerformanceMetrics:
    """Performance monitoring metrics."""
    query_count: int = 0
    total_response_time: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    database_queries: int = 0
    concurrent_requests: int = 0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    last_updated: float = 0.0

    def average_response_time(self) -> float:
        """Calculate average response time."""
        return self.total_response_time / self.query_count if self.query_count > 0 else 0.0

    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0


@dataclass
class ResourceUsage:
    """System resource usage information."""
    cpu_percent: float
    memory_mb: float
    memory_percent: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    network_sent_mb: float
    network_recv_mb: float
    open_files: int
    active_threads: int
    timestamp: float


class IntelligentCache:
    """Advanced caching system with multiple eviction strategies."""

    def __init__(self, max_size: int = 10000, strategy: CacheStrategy = CacheStrategy.ADAPTIVE):
        self.max_size = max_size
        self.strategy = strategy
        self.cache: Dict[str, Any] = {}
        self.access_times: Dict[str, float] = {}
        self.access_counts: Dict[str, int] = defaultdict(int)
        self.creation_times: Dict[str, float] = {}
        self.ttl_values: Dict[str, float] = {}
        self.size_tracking: Dict[str, int] = {}

        # Performance metrics
        self.hits = 0
        self.misses = 0
        self.evictions = 0

        # Lock for thread safety
        self.lock = threading.RLock()

        # Background cleanup
        self.cleanup_interval = 60.0  # seconds
        self.last_cleanup = time.time()

    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        with self.lock:
            current_time = time.time()

            # Check if key exists and is not expired
            if key in self.cache:
                # Check TTL expiration
                if key in self.ttl_values and current_time > self.ttl_values[key]:
                    self._remove_key(key)
                    self.misses += 1
                    return None

                # Update access tracking
                self.access_times[key] = current_time
                self.access_counts[key] += 1
                self.hits += 1

                return self.cache[key]
            else:
                self.misses += 1
                return None

    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Put item in cache with optional TTL."""
        with self.lock:
            current_time = time.time()

            # Calculate value size (approximate)
            value_size = self._estimate_size(value)

            # Check if we need to evict items
            while len(self.cache) >= self.max_size and key not in self.cache:
                self._evict_item()

            # Store item
            self.cache[key] = value
            self.access_times[key] = current_time
            self.access_counts[key] += 1
            self.creation_times[key] = current_time
            self.size_tracking[key] = value_size

            # Set TTL if provided
            if ttl is not None:
                self.ttl_values[key] = current_time + ttl

            # Periodic cleanup
            if current_time - self.last_cleanup > self.cleanup_interval:
                self._cleanup_expired()
                self.last_cleanup = current_time

    def invalidate(self, key: str) -> bool:
        """Invalidate specific cache entry."""
        with self.lock:
            if key in self.cache:
                self._remove_key(key)
                return True
            return False

    def clear(self) -> None:
        """Clear all cache entries."""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()
            self.access_counts.clear()
            self.creation_times.clear()
            self.ttl_values.clear()
            self.size_tracking.clear()

    def get_metrics(self) -> Dict[str, Any]:
        """Get cache performance metrics."""
        with self.lock:
            total_requests = self.hits + self.misses
            hit_rate = self.hits / total_requests if total_requests > 0 else 0.0

            total_size = sum(self.size_tracking.values())

            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'hits': self.hits,
                'misses': self.misses,
                'hit_rate': hit_rate,
                'evictions': self.evictions,
                'total_size_bytes': total_size,
                'strategy': self.strategy.value,
                'utilization': len(self.cache) / self.max_size
            }

    def _evict_item(self) -> None:
        """Evict item based on selected strategy."""
        if not self.cache:
            return

        if self.strategy == CacheStrategy.LRU:
            # Remove least recently used
            oldest_key = min(self.access_times, key=self.access_times.get)
        elif self.strategy == CacheStrategy.LFU:
            # Remove least frequently used
            oldest_key = min(self.access_counts, key=self.access_counts.get)
        elif self.strategy == CacheStrategy.FIFO:
            # Remove oldest created
            oldest_key = min(self.creation_times, key=self.creation_times.get)
        elif self.strategy == CacheStrategy.ADAPTIVE:
            # Use adaptive strategy based on access patterns
            oldest_key = self._adaptive_eviction()
        else:
            # Default to LRU
            oldest_key = min(self.access_times, key=self.access_times.get)

        self._remove_key(oldest_key)
        self.evictions += 1

    def _adaptive_eviction(self) -> str:
        """Adaptive eviction strategy based on usage patterns."""
        current_time = time.time()
        scores = {}

        for key in self.cache.keys():
            # Calculate composite score
            recency_score = 1.0 / (current_time - self.access_times.get(key, current_time))
            frequency_score = self.access_counts.get(key, 1)
            age_penalty = (current_time - self.creation_times.get(key, current_time)) / 3600  # Hours

            # Combine scores (lower is worse)
            composite_score = recency_score * frequency_score - age_penalty
            scores[key] = composite_score

        # Return key with lowest score
        return min(scores, key=scores.get)

    def _cleanup_expired(self) -> None:
        """Remove expired TTL entries."""
        current_time = time.time()
        expired_keys = [
            key for key, expiry_time in self.ttl_values.items()
            if current_time > expiry_time
        ]

        for key in expired_keys:
            self._remove_key(key)

    def _remove_key(self, key: str) -> None:
        """Remove key and all associated tracking data."""
        self.cache.pop(key, None)
        self.access_times.pop(key, None)
        self.access_counts.pop(key, None)
        self.creation_times.pop(key, None)
        self.ttl_values.pop(key, None)
        self.size_tracking.pop(key, None)

    def _estimate_size(self, value: Any) -> int:
        """Estimate memory size of value."""
        try:
            if isinstance(value, str):
                return len(value.encode('utf-8'))
            elif isinstance(value, (list, tuple)):
                return sum(self._estimate_size(item) for item in value)
            elif isinstance(value, dict):
                return sum(
                    self._estimate_size(k) + self._estimate_size(v)
                    for k, v in value.items()
                )
            else:
                # Rough estimate
                return len(str(value)) * 4
        except:
            return 1000  # Default estimate


class QueryOptimizer:
    """Optimize queries for better performance."""

    def __init__(self):
        self.query_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            'count': 0,
            'total_time': 0.0,
            'avg_time': 0.0,
            'last_seen': 0.0,
            'optimization_level': OptimizationLevel.CONSERVATIVE
        })
        self.optimization_cache = IntelligentCache(max_size=5000, strategy=CacheStrategy.ADAPTIVE)

    def optimize_query(self, query: str) -> str:
        """Optimize query for better performance."""
        query_hash = self._hash_query(query)

        # Check if we have cached optimization
        cached_result = self.optimization_cache.get(query_hash)
        if cached_result:
            return cached_result

        # Apply optimization strategies
        optimized = self._apply_optimizations(query)

        # Cache the result
        self.optimization_cache.put(query_hash, optimized, ttl=3600)  # Cache for 1 hour

        return optimized

    def record_query_performance(self, query: str, execution_time: float):
        """Record query performance for optimization insights."""
        query_hash = self._hash_query(query)
        stats = self.query_stats[query_hash]

        stats['count'] += 1
        stats['total_time'] += execution_time
        stats['avg_time'] = stats['total_time'] / stats['count']
        stats['last_seen'] = time.time()

        # Adjust optimization level based on performance
        if stats['avg_time'] > 2.0:  # Slow queries
            stats['optimization_level'] = OptimizationLevel.AGGRESSIVE
        elif stats['avg_time'] > 0.5:
            stats['optimization_level'] = OptimizationLevel.BALANCED
        else:
            stats['optimization_level'] = OptimizationLevel.CONSERVATIVE

    def get_slow_queries(self, threshold: float = 1.0) -> List[Tuple[str, Dict[str, Any]]]:
        """Get queries that are performing slowly."""
        slow_queries = [
            (query_hash, stats) for query_hash, stats in self.query_stats.items()
            if stats['avg_time'] > threshold and stats['count'] >= 5
        ]

        # Sort by average time descending
        return sorted(slow_queries, key=lambda x: x[1]['avg_time'], reverse=True)

    def _hash_query(self, query: str) -> str:
        """Generate hash for query caching."""
        # Normalize query for consistent hashing
        normalized = ' '.join(query.lower().split())
        return hashlib.md5(normalized.encode()).hexdigest()[:16]

    def _apply_optimizations(self, query: str) -> str:
        """Apply various query optimizations."""
        optimized = query

        # Remove redundant whitespace
        optimized = ' '.join(optimized.split())

        # Expand common abbreviations for better search
        abbreviations = {
            'auth': 'authentication',
            'config': 'configuration',
            'db': 'database',
            'api': 'application programming interface',
            'ui': 'user interface',
            'ux': 'user experience'
        }

        for abbrev, expansion in abbreviations.items():
            # Add expanded form alongside abbreviation
            if abbrev in optimized.lower():
                optimized += f" {expansion}"

        return optimized


class ResourceMonitor:
    """Monitor system resource usage and performance."""

    def __init__(self, monitoring_interval: float = 10.0):
        self.monitoring_interval = monitoring_interval
        self.metrics_history: deque = deque(maxlen=1000)  # Keep last 1000 measurements
        self.current_metrics = PerformanceMetrics()

        # Resource usage tracking
        self.resource_history: deque = deque(maxlen=100)

        # Thresholds for alerting
        self.cpu_threshold = 80.0
        self.memory_threshold = 85.0
        self.response_time_threshold = 2.0

        # Start monitoring
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

    def record_query_metrics(self, response_time: float, cache_hit: bool = False,
                           database_query: bool = False):
        """Record metrics for a single query."""
        self.current_metrics.query_count += 1
        self.current_metrics.total_response_time += response_time

        if cache_hit:
            self.current_metrics.cache_hits += 1
        else:
            self.current_metrics.cache_misses += 1

        if database_query:
            self.current_metrics.database_queries += 1

        self.current_metrics.last_updated = time.time()

    def get_current_performance(self) -> PerformanceMetrics:
        """Get current performance metrics."""
        # Update system resource info
        if PSUTIL_AVAILABLE:
            try:
                process = psutil.Process()
                self.current_metrics.memory_usage_mb = process.memory_info().rss / 1024 / 1024
                self.current_metrics.cpu_usage_percent = process.cpu_percent()
            except:
                pass

        return self.current_metrics

    def get_resource_usage(self) -> Optional[ResourceUsage]:
        """Get detailed system resource usage."""
        if not PSUTIL_AVAILABLE:
            return None

        try:
            # System-wide metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk_io = psutil.disk_io_counters()
            network_io = psutil.net_io_counters()

            # Process-specific metrics
            process = psutil.Process()
            process_memory = process.memory_info().rss / 1024 / 1024  # MB
            open_files = len(process.open_files()) if hasattr(process, 'open_files') else 0
            active_threads = process.num_threads()

            return ResourceUsage(
                cpu_percent=cpu_percent,
                memory_mb=process_memory,
                memory_percent=memory.percent,
                disk_io_read_mb=(disk_io.read_bytes if disk_io else 0) / 1024 / 1024,
                disk_io_write_mb=(disk_io.write_bytes if disk_io else 0) / 1024 / 1024,
                network_sent_mb=(network_io.bytes_sent if network_io else 0) / 1024 / 1024,
                network_recv_mb=(network_io.bytes_recv if network_io else 0) / 1024 / 1024,
                open_files=open_files,
                active_threads=active_threads,
                timestamp=time.time()
            )
        except Exception as e:
            logger.error(f"Error collecting resource usage: {e}")
            return None

    def get_performance_summary(self, window_minutes: int = 60) -> Dict[str, Any]:
        """Get performance summary for specified time window."""
        cutoff_time = time.time() - (window_minutes * 60)
        recent_metrics = [m for m in self.metrics_history if m.last_updated > cutoff_time]

        if not recent_metrics:
            return {'error': 'No metrics available for specified time window'}

        # Calculate aggregated metrics
        total_queries = sum(m.query_count for m in recent_metrics)
        total_response_time = sum(m.total_response_time for m in recent_metrics)
        total_cache_hits = sum(m.cache_hits for m in recent_metrics)
        total_cache_misses = sum(m.cache_misses for m in recent_metrics)
        total_db_queries = sum(m.database_queries for m in recent_metrics)

        avg_response_time = total_response_time / total_queries if total_queries > 0 else 0.0
        cache_hit_rate = total_cache_hits / (total_cache_hits + total_cache_misses) if (total_cache_hits + total_cache_misses) > 0 else 0.0

        # Resource usage statistics
        resource_stats = {}
        if self.resource_history:
            recent_resources = [r for r in self.resource_history if r.timestamp > cutoff_time]
            if recent_resources:
                resource_stats = {
                    'avg_cpu_percent': statistics.mean(r.cpu_percent for r in recent_resources),
                    'max_cpu_percent': max(r.cpu_percent for r in recent_resources),
                    'avg_memory_mb': statistics.mean(r.memory_mb for r in recent_resources),
                    'max_memory_mb': max(r.memory_mb for r in recent_resources),
                    'avg_active_threads': statistics.mean(r.active_threads for r in recent_resources)
                }

        return {
            'time_window_minutes': window_minutes,
            'total_queries': total_queries,
            'queries_per_minute': total_queries / window_minutes,
            'avg_response_time': avg_response_time,
            'cache_hit_rate': cache_hit_rate,
            'database_queries': total_db_queries,
            'database_query_rate': total_db_queries / total_queries if total_queries > 0 else 0.0,
            'resource_usage': resource_stats,
            'performance_alerts': self._check_performance_alerts(avg_response_time, resource_stats)
        }

    def _monitor_loop(self):
        """Background monitoring loop."""
        while self.monitoring_active:
            try:
                # Record current metrics
                current_snapshot = PerformanceMetrics(
                    query_count=self.current_metrics.query_count,
                    total_response_time=self.current_metrics.total_response_time,
                    cache_hits=self.current_metrics.cache_hits,
                    cache_misses=self.current_metrics.cache_misses,
                    database_queries=self.current_metrics.database_queries,
                    memory_usage_mb=self.current_metrics.memory_usage_mb,
                    cpu_usage_percent=self.current_metrics.cpu_usage_percent,
                    last_updated=time.time()
                )
                self.metrics_history.append(current_snapshot)

                # Record resource usage
                resource_usage = self.get_resource_usage()
                if resource_usage:
                    self.resource_history.append(resource_usage)

                time.sleep(self.monitoring_interval)

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.monitoring_interval)

    def _check_performance_alerts(self, avg_response_time: float, resource_stats: Dict) -> List[str]:
        """Check for performance issues that need attention."""
        alerts = []

        if avg_response_time > self.response_time_threshold:
            alerts.append(f"High response time: {avg_response_time:.2f}s (threshold: {self.response_time_threshold}s)")

        if resource_stats:
            if resource_stats.get('max_cpu_percent', 0) > self.cpu_threshold:
                alerts.append(f"High CPU usage: {resource_stats['max_cpu_percent']:.1f}% (threshold: {self.cpu_threshold}%)")

            if resource_stats.get('max_memory_mb', 0) > 1000:  # > 1GB
                alerts.append(f"High memory usage: {resource_stats['max_memory_mb']:.1f}MB")

        return alerts

    def shutdown(self):
        """Shutdown the monitoring system."""
        self.monitoring_active = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5.0)


class AsyncQueryProcessor:
    """Asynchronous query processing for better concurrency."""

    def __init__(self, max_workers: int = 10):
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.active_tasks: Dict[str, Any] = {}

    async def process_query_async(self, query: str, processor_func: Callable,
                                 *args, **kwargs) -> Any:
        """Process query asynchronously."""
        loop = asyncio.get_event_loop()

        # Generate task ID
        task_id = hashlib.md5(f"{query}_{time.time()}".encode()).hexdigest()[:8]

        # Submit to thread pool
        future = self.executor.submit(processor_func, query, *args, **kwargs)
        self.active_tasks[task_id] = {
            'future': future,
            'query': query,
            'start_time': time.time()
        }

        try:
            # Wait for result
            result = await loop.run_in_executor(None, future.result)
            return result
        finally:
            # Clean up
            self.active_tasks.pop(task_id, None)

    def get_active_tasks(self) -> Dict[str, Any]:
        """Get information about currently active tasks."""
        current_time = time.time()

        return {
            task_id: {
                'query': task_info['query'],
                'duration': current_time - task_info['start_time'],
                'done': task_info['future'].done()
            }
            for task_id, task_info in self.active_tasks.items()
        }

    def shutdown(self):
        """Shutdown the async processor."""
        self.executor.shutdown(wait=True)


class PerformanceOptimizationSystem:
    """Comprehensive performance optimization and monitoring system."""

    def __init__(self, optimization_level: OptimizationLevel = OptimizationLevel.BALANCED):
        self.optimization_level = optimization_level

        # Core components
        self.cache = IntelligentCache(
            max_size=20000 if optimization_level == OptimizationLevel.AGGRESSIVE else 10000,
            strategy=CacheStrategy.ADAPTIVE
        )
        self.query_optimizer = QueryOptimizer()
        self.resource_monitor = ResourceMonitor()
        self.async_processor = AsyncQueryProcessor(
            max_workers=20 if optimization_level == OptimizationLevel.AGGRESSIVE else 10
        )

        # Performance tracking
        self.optimization_stats = defaultdict(int)
        self.last_optimization_time = time.time()

    def optimize_and_cache_query(self, query: str, processor_func: Callable,
                                *args, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        """Optimize query and use caching for better performance."""
        start_time = time.time()

        # Optimize query
        optimized_query = self.query_optimizer.optimize_query(query)
        self.optimization_stats['queries_optimized'] += 1

        # Check cache first
        cache_key = self._generate_cache_key(optimized_query, args, kwargs)
        cached_result = self.cache.get(cache_key)

        if cached_result is not None:
            self.optimization_stats['cache_hits'] += 1
            self.resource_monitor.record_query_metrics(
                response_time=time.time() - start_time,
                cache_hit=True
            )

            return cached_result['result'], {
                'optimized_query': optimized_query,
                'cache_hit': True,
                'response_time': time.time() - start_time
            }

        # Process query
        result = processor_func(optimized_query, *args, **kwargs)
        response_time = time.time() - start_time

        # Cache result (with TTL based on query type)
        ttl = self._calculate_cache_ttl(optimized_query)
        self.cache.put(cache_key, {'result': result, 'timestamp': time.time()}, ttl=ttl)

        # Record metrics
        self.optimization_stats['cache_misses'] += 1
        self.query_optimizer.record_query_performance(optimized_query, response_time)
        self.resource_monitor.record_query_metrics(
            response_time=response_time,
            cache_hit=False,
            database_query=True  # Assume database query if not cached
        )

        return result, {
            'optimized_query': optimized_query,
            'cache_hit': False,
            'response_time': response_time,
            'ttl': ttl
        }

    async def process_query_async(self, query: str, processor_func: Callable,
                                 *args, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        """Process query asynchronously with optimization."""
        # First try synchronous optimized version for cache check
        cache_key = self._generate_cache_key(query, args, kwargs)
        cached_result = self.cache.get(cache_key)

        if cached_result is not None:
            return cached_result['result'], {'cache_hit': True, 'async': True}

        # Process asynchronously
        result = await self.async_processor.process_query_async(
            query, self.optimize_and_cache_query, processor_func, *args, **kwargs
        )

        return result[0], {**result[1], 'async': True}

    def get_performance_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive performance dashboard."""
        dashboard = {
            'timestamp': datetime.now().isoformat(),
            'optimization_level': self.optimization_level.value,
            'cache_metrics': self.cache.get_metrics(),
            'performance_metrics': asdict(self.resource_monitor.get_current_performance()),
            'optimization_stats': dict(self.optimization_stats),
            'slow_queries': self.query_optimizer.get_slow_queries(),
            'active_async_tasks': self.async_processor.get_active_tasks(),
            'performance_summary': self.resource_monitor.get_performance_summary(60),
            'recommendations': self._generate_recommendations()
        }

        # Add resource usage if available
        resource_usage = self.resource_monitor.get_resource_usage()
        if resource_usage:
            dashboard['resource_usage'] = asdict(resource_usage)

        return dashboard

    def _generate_cache_key(self, query: str, args: Tuple, kwargs: Dict) -> str:
        """Generate cache key for query and parameters."""
        # Include query and serializable parameters
        key_data = {
            'query': query,
            'args': str(args),
            'kwargs': str(sorted(kwargs.items()))
        }
        key_string = str(key_data)
        return hashlib.md5(key_string.encode()).hexdigest()

    def _calculate_cache_ttl(self, query: str) -> float:
        """Calculate appropriate TTL for query results."""
        query_lower = query.lower()

        # Dynamic content - shorter TTL
        if any(word in query_lower for word in ['status', 'current', 'recent', 'latest']):
            return 300  # 5 minutes

        # Documentation content - longer TTL
        if any(word in query_lower for word in ['documentation', 'guide', 'tutorial', 'reference']):
            return 3600  # 1 hour

        # Default TTL
        return 1800  # 30 minutes

    def _generate_recommendations(self) -> List[str]:
        """Generate performance optimization recommendations."""
        recommendations = []

        cache_metrics = self.cache.get_metrics()
        performance_summary = self.resource_monitor.get_performance_summary(60)

        # Cache recommendations
        if cache_metrics['hit_rate'] < 0.5:
            recommendations.append(
                f"Low cache hit rate ({cache_metrics['hit_rate']:.1%}). "
                "Consider increasing cache size or adjusting TTL values."
            )

        if cache_metrics['utilization'] > 0.9:
            recommendations.append(
                "Cache utilization is high (>90%). Consider increasing cache size."
            )

        # Performance recommendations
        if 'avg_response_time' in performance_summary and performance_summary['avg_response_time'] > 1.0:
            recommendations.append(
                f"Average response time is high ({performance_summary['avg_response_time']:.2f}s). "
                "Consider query optimization or increased caching."
            )

        # Resource recommendations
        resource_usage = performance_summary.get('resource_usage', {})
        if resource_usage.get('max_cpu_percent', 0) > 80:
            recommendations.append(
                "High CPU usage detected. Consider scaling or optimizing heavy operations."
            )

        if resource_usage.get('max_memory_mb', 0) > 1000:
            recommendations.append(
                "High memory usage detected. Consider reducing cache size or fixing memory leaks."
            )

        # Query optimization recommendations
        slow_queries = self.query_optimizer.get_slow_queries(threshold=0.5)
        if slow_queries:
            recommendations.append(
                f"Found {len(slow_queries)} slow queries. Review query optimization strategies."
            )

        if not recommendations:
            recommendations.append("System performance is optimal.")

        return recommendations

    def shutdown(self):
        """Shutdown all performance monitoring and optimization components."""
        self.resource_monitor.shutdown()
        self.async_processor.shutdown()
        logger.info("Performance optimization system shutdown complete")


# Global optimization system
_global_optimization_system: Optional[PerformanceOptimizationSystem] = None


def get_optimization_system(optimization_level: OptimizationLevel = OptimizationLevel.BALANCED) -> PerformanceOptimizationSystem:
    """Get or create global optimization system."""
    global _global_optimization_system

    if _global_optimization_system is None:
        _global_optimization_system = PerformanceOptimizationSystem(optimization_level)

    return _global_optimization_system


def optimized_query_execution(func: Callable) -> Callable:
    """Decorator for optimized query execution."""
    def wrapper(query: str, *args, **kwargs):
        optimization_system = get_optimization_system()
        return optimization_system.optimize_and_cache_query(query, func, *args, **kwargs)

    return wrapper


# Example usage function
def create_high_performance_cache() -> IntelligentCache:
    """Create a high-performance cache with optimized settings."""
    return IntelligentCache(
        max_size=50000,
        strategy=CacheStrategy.ADAPTIVE
    )
