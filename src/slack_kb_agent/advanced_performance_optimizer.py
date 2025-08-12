"""Advanced Performance Optimizer with Auto-Scaling and Intelligent Caching.

This module provides enterprise-grade performance optimization including
adaptive caching, concurrent processing, resource pooling, and auto-scaling.
"""

import asyncio
import json
import time
import logging
import threading
import hashlib
import psutil
from collections import defaultdict, deque, OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any, Callable, Union
from datetime import datetime, timedelta
from pathlib import Path
import statistics
import sys
import gc
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import queue
import weakref

logger = logging.getLogger(__name__)


class CacheStrategy(Enum):
    """Cache eviction strategies."""
    LRU = "lru"
    LFU = "lfu"
    TTL = "ttl"
    ADAPTIVE = "adaptive"


class ScalingMetric(Enum):
    """Metrics for auto-scaling decisions."""
    CPU_USAGE = "cpu_usage"
    MEMORY_USAGE = "memory_usage"
    QUEUE_LENGTH = "queue_length"
    RESPONSE_TIME = "response_time"
    THROUGHPUT = "throughput"


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    size_bytes: int = 0
    ttl: Optional[float] = None
    
    def is_expired(self) -> bool:
        """Check if entry is expired."""
        if self.ttl is None:
            return False
        return (datetime.now() - self.created_at).total_seconds() > self.ttl
    
    def update_access(self):
        """Update access metadata."""
        self.last_accessed = datetime.now()
        self.access_count += 1


@dataclass
class PerformanceMetrics:
    """Performance metrics tracking."""
    operation_count: int = 0
    total_duration: float = 0.0
    min_duration: float = float('inf')
    max_duration: float = 0.0
    error_count: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    last_updated: datetime = field(default_factory=datetime.now)
    
    def add_operation(self, duration: float, success: bool = True):
        """Add operation metrics."""
        self.operation_count += 1
        self.total_duration += duration
        self.min_duration = min(self.min_duration, duration)
        self.max_duration = max(self.max_duration, duration)
        if not success:
            self.error_count += 1
        self.last_updated = datetime.now()
    
    def get_average_duration(self) -> float:
        """Get average operation duration."""
        return self.total_duration / self.operation_count if self.operation_count > 0 else 0.0
    
    def get_cache_hit_rate(self) -> float:
        """Get cache hit rate."""
        total_requests = self.cache_hits + self.cache_misses
        return self.cache_hits / total_requests if total_requests > 0 else 0.0


class AdaptiveCache:
    """Intelligent cache with multiple strategies and auto-tuning."""
    
    def __init__(self, max_size: int = 10000, default_ttl: Optional[float] = None, 
                 strategy: CacheStrategy = CacheStrategy.ADAPTIVE):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.strategy = strategy
        self.cache = OrderedDict()
        self.metadata = {}
        self.stats = PerformanceMetrics()
        self._lock = threading.RLock()
        
        # Adaptive strategy parameters
        self.hit_rate_window = deque(maxlen=1000)
        self.access_patterns = defaultdict(int)
        self.strategy_performance = defaultdict(lambda: {"hits": 0, "total": 0})
        
        # Background cleanup
        self.cleanup_thread = None
        self.running = True
        self._start_cleanup_thread()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._lock:
            if key in self.cache:
                entry = self.cache[key]
                
                # Check expiration
                if entry.is_expired():
                    self._remove_entry(key)
                    self.stats.cache_misses += 1
                    return None
                
                # Update access metadata
                entry.update_access()
                
                # Move to end for LRU
                if self.strategy in [CacheStrategy.LRU, CacheStrategy.ADAPTIVE]:
                    self.cache.move_to_end(key)
                
                self.stats.cache_hits += 1
                self._track_access_pattern(key)
                return entry.value
            
            self.stats.cache_misses += 1
            return None
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """Put value in cache."""
        with self._lock:
            # Calculate size
            size_bytes = self._estimate_size(value)
            
            # Check if we need to evict
            if len(self.cache) >= self.max_size:
                self._evict_entries()
            
            # Create cache entry
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=datetime.now(),
                last_accessed=datetime.now(),
                access_count=1,
                size_bytes=size_bytes,
                ttl=ttl or self.default_ttl
            )
            
            # Store entry
            self.cache[key] = entry
            self.metadata[key] = {
                "strategy_used": self.strategy.value,
                "created": entry.created_at
            }
            
            return True
    
    def delete(self, key: str) -> bool:
        """Delete entry from cache."""
        with self._lock:
            if key in self.cache:
                self._remove_entry(key)
                return True
            return False
    
    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            self.cache.clear()
            self.metadata.clear()
            gc.collect()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_size = sum(entry.size_bytes for entry in self.cache.values())
            
            return {
                "entries": len(self.cache),
                "max_size": self.max_size,
                "total_size_bytes": total_size,
                "hit_rate": self.stats.get_cache_hit_rate(),
                "hits": self.stats.cache_hits,
                "misses": self.stats.cache_misses,
                "strategy": self.strategy.value,
                "average_access_count": statistics.mean([e.access_count for e in self.cache.values()]) if self.cache else 0,
                "oldest_entry": min(e.created_at for e in self.cache.values()).isoformat() if self.cache else None
            }
    
    def optimize_strategy(self):
        """Optimize cache strategy based on performance."""
        if self.strategy != CacheStrategy.ADAPTIVE:
            return
        
        with self._lock:
            # Analyze access patterns
            if len(self.hit_rate_window) < 100:
                return
            
            recent_hit_rate = statistics.mean(list(self.hit_rate_window)[-50:])
            
            # Switch strategy based on patterns
            if recent_hit_rate < 0.3:
                # Low hit rate, try TTL strategy
                self._switch_strategy(CacheStrategy.TTL)
            elif recent_hit_rate > 0.8:
                # High hit rate, stick with LRU
                self._switch_strategy(CacheStrategy.LRU)
            else:
                # Medium hit rate, try LFU
                self._switch_strategy(CacheStrategy.LFU)
    
    def _evict_entries(self):
        """Evict entries based on strategy."""
        if not self.cache:
            return
        
        if self.strategy == CacheStrategy.LRU:
            # Remove least recently used
            key = next(iter(self.cache))
            self._remove_entry(key)
        
        elif self.strategy == CacheStrategy.LFU:
            # Remove least frequently used
            min_access = min(entry.access_count for entry in self.cache.values())
            for key, entry in self.cache.items():
                if entry.access_count == min_access:
                    self._remove_entry(key)
                    break
        
        elif self.strategy == CacheStrategy.TTL:
            # Remove expired entries first, then oldest
            expired_keys = [key for key, entry in self.cache.items() if entry.is_expired()]
            if expired_keys:
                for key in expired_keys:
                    self._remove_entry(key)
            else:
                # Remove oldest
                oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k].created_at)
                self._remove_entry(oldest_key)
        
        elif self.strategy == CacheStrategy.ADAPTIVE:
            # Use hybrid approach
            self._adaptive_eviction()
    
    def _adaptive_eviction(self):
        """Adaptive eviction strategy."""
        now = datetime.now()
        
        # Score entries based on multiple factors
        scores = {}
        for key, entry in self.cache.items():
            age_score = (now - entry.created_at).total_seconds() / 3600  # Hours
            recency_score = (now - entry.last_accessed).total_seconds() / 3600  # Hours
            frequency_score = 1.0 / max(1, entry.access_count)
            
            # Combined score (lower is better for eviction)
            scores[key] = age_score * 0.3 + recency_score * 0.5 + frequency_score * 0.2
        
        # Remove entry with highest score
        worst_key = max(scores.keys(), key=lambda k: scores[k])
        self._remove_entry(worst_key)
    
    def _remove_entry(self, key: str):
        """Remove entry and cleanup metadata."""
        if key in self.cache:
            del self.cache[key]
        if key in self.metadata:
            del self.metadata[key]
    
    def _estimate_size(self, value: Any) -> int:
        """Estimate size of value in bytes."""
        try:
            return sys.getsizeof(value)
        except Exception:
            return 1024  # Default estimate
    
    def _track_access_pattern(self, key: str):
        """Track access patterns for optimization."""
        self.access_patterns[key] += 1
        
        # Track hit rate over time
        current_hit_rate = self.stats.get_cache_hit_rate()
        self.hit_rate_window.append(current_hit_rate)
    
    def _switch_strategy(self, new_strategy: CacheStrategy):
        """Switch cache strategy."""
        if self.strategy != new_strategy:
            logger.info(f"Switching cache strategy from {self.strategy.value} to {new_strategy.value}")
            self.strategy = new_strategy
    
    def _start_cleanup_thread(self):
        """Start background cleanup thread."""
        def cleanup_loop():
            while self.running:
                try:
                    with self._lock:
                        # Remove expired entries
                        expired_keys = [key for key, entry in self.cache.items() if entry.is_expired()]
                        for key in expired_keys:
                            self._remove_entry(key)
                        
                        # Optimize strategy periodically
                        self.optimize_strategy()
                    
                    time.sleep(60)  # Cleanup every minute
                except Exception as e:
                    logger.error(f"Cache cleanup error: {e}")
                    time.sleep(60)
        
        self.cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
        self.cleanup_thread.start()
    
    def shutdown(self):
        """Shutdown cache and cleanup resources."""
        self.running = False
        if self.cleanup_thread and self.cleanup_thread.is_alive():
            self.cleanup_thread.join(timeout=5)


class ConcurrentProcessor:
    """High-performance concurrent processing with resource pooling."""
    
    def __init__(self, max_workers: int = None, enable_process_pool: bool = False):
        self.max_workers = max_workers or min(32, (psutil.cpu_count() or 1) + 4)
        self.enable_process_pool = enable_process_pool
        
        # Thread pool for I/O bound tasks
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)
        
        # Process pool for CPU bound tasks
        self.process_pool = ProcessPoolExecutor(max_workers=psutil.cpu_count()) if enable_process_pool else None
        
        # Task queues
        self.high_priority_queue = queue.PriorityQueue()
        self.normal_priority_queue = queue.Queue()
        self.low_priority_queue = queue.Queue()
        
        # Processing metrics
        self.metrics = PerformanceMetrics()
        self.active_tasks = {}
        self._lock = threading.Lock()
        
        # Start worker threads
        self.running = True
        self.worker_threads = []
        self._start_worker_threads()
    
    def submit_task(self, func: Callable, *args, priority: int = 1, **kwargs) -> asyncio.Future:
        """Submit task for concurrent processing."""
        task_id = f"task_{int(time.time() * 1000000)}"
        
        task_info = {
            "id": task_id,
            "func": func,
            "args": args,
            "kwargs": kwargs,
            "submitted_at": datetime.now(),
            "priority": priority
        }
        
        # Select appropriate queue based on priority
        if priority >= 3:
            self.high_priority_queue.put((priority, task_info))
        elif priority >= 1:
            self.normal_priority_queue.put(task_info)
        else:
            self.low_priority_queue.put(task_info)
        
        # Create future for result
        future = asyncio.Future()
        
        with self._lock:
            self.active_tasks[task_id] = {
                "future": future,
                "info": task_info
            }
        
        return future
    
    def submit_batch(self, tasks: List[Tuple[Callable, tuple, dict]], priority: int = 1) -> List[asyncio.Future]:
        """Submit batch of tasks for processing."""
        futures = []
        for func, args, kwargs in tasks:
            future = self.submit_task(func, *args, priority=priority, **kwargs)
            futures.append(future)
        return futures
    
    def process_cpu_intensive(self, func: Callable, *args, **kwargs) -> asyncio.Future:
        """Process CPU-intensive task using process pool."""
        if not self.process_pool:
            return self.submit_task(func, *args, **kwargs)
        
        future = asyncio.Future()
        
        def process_callback(process_future):
            try:
                result = process_future.result()
                future.set_result(result)
            except Exception as e:
                future.set_exception(e)
        
        process_future = self.process_pool.submit(func, *args, **kwargs)
        process_future.add_done_callback(process_callback)
        
        return future
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        with self._lock:
            return {
                "max_workers": self.max_workers,
                "active_tasks": len(self.active_tasks),
                "high_priority_queue_size": self.high_priority_queue.qsize(),
                "normal_priority_queue_size": self.normal_priority_queue.qsize(),
                "low_priority_queue_size": self.low_priority_queue.qsize(),
                "total_operations": self.metrics.operation_count,
                "average_duration": self.metrics.get_average_duration(),
                "error_count": self.metrics.error_count,
                "process_pool_enabled": self.enable_process_pool
            }
    
    def _start_worker_threads(self):
        """Start worker threads for task processing."""
        for i in range(min(8, self.max_workers)):
            thread = threading.Thread(target=self._worker_loop, daemon=True)
            thread.start()
            self.worker_threads.append(thread)
    
    def _worker_loop(self):
        """Main worker loop for processing tasks."""
        while self.running:
            task_info = None
            
            try:
                # Check high priority queue first
                try:
                    _, task_info = self.high_priority_queue.get_nowait()
                except queue.Empty:
                    pass
                
                # Check normal priority queue
                if task_info is None:
                    try:
                        task_info = self.normal_priority_queue.get_nowait()
                    except queue.Empty:
                        pass
                
                # Check low priority queue
                if task_info is None:
                    try:
                        task_info = self.low_priority_queue.get(timeout=1.0)
                    except queue.Empty:
                        continue
                
                # Process task
                start_time = time.time()
                task_id = task_info["id"]
                
                try:
                    result = task_info["func"](*task_info["args"], **task_info["kwargs"])
                    
                    # Set result in future
                    with self._lock:
                        if task_id in self.active_tasks:
                            self.active_tasks[task_id]["future"].set_result(result)
                            del self.active_tasks[task_id]
                    
                    # Record metrics
                    duration = time.time() - start_time
                    self.metrics.add_operation(duration, True)
                    
                except Exception as e:
                    # Set exception in future
                    with self._lock:
                        if task_id in self.active_tasks:
                            self.active_tasks[task_id]["future"].set_exception(e)
                            del self.active_tasks[task_id]
                    
                    # Record error metrics
                    duration = time.time() - start_time
                    self.metrics.add_operation(duration, False)
                    
                    logger.error(f"Task {task_id} failed: {e}")
                
            except Exception as e:
                logger.error(f"Worker thread error: {e}")
                time.sleep(1)
    
    def shutdown(self):
        """Shutdown processor and cleanup resources."""
        self.running = False
        
        # Shutdown thread pool
        self.thread_pool.shutdown(wait=True)
        
        # Shutdown process pool
        if self.process_pool:
            self.process_pool.shutdown(wait=True)
        
        # Wait for worker threads
        for thread in self.worker_threads:
            thread.join(timeout=5)


class AutoScaler:
    """Intelligent auto-scaling system based on performance metrics."""
    
    def __init__(self, min_workers: int = 2, max_workers: int = 32):
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.current_workers = min_workers
        
        # Scaling thresholds
        self.scale_up_threshold = 0.75
        self.scale_down_threshold = 0.25
        self.scale_up_cpu_threshold = 70.0
        self.scale_down_cpu_threshold = 30.0
        
        # Metrics tracking
        self.metrics_history = deque(maxlen=100)
        self.scaling_decisions = deque(maxlen=50)
        
        # Cooldown periods
        self.scale_up_cooldown = timedelta(minutes=2)
        self.scale_down_cooldown = timedelta(minutes=5)
        self.last_scale_action = None
        
        self._lock = threading.Lock()
    
    def evaluate_scaling(self, metrics: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Evaluate if scaling action is needed."""
        with self._lock:
            # Record metrics
            current_time = datetime.now()
            self.metrics_history.append({
                "timestamp": current_time,
                "metrics": metrics.copy()
            })
            
            # Check cooldown
            if self.last_scale_action:
                time_since_action = current_time - self.last_scale_action
                if time_since_action < self.scale_up_cooldown:
                    return None
            
            # Calculate scaling factors
            scaling_factors = self._calculate_scaling_factors(metrics)
            
            # Make scaling decision
            decision = self._make_scaling_decision(scaling_factors)
            
            if decision:
                self.last_scale_action = current_time
                self.scaling_decisions.append({
                    "timestamp": current_time,
                    "decision": decision,
                    "factors": scaling_factors
                })
                
                # Update current workers
                if decision["action"] == "scale_up":
                    self.current_workers = min(self.max_workers, self.current_workers + decision["count"])
                elif decision["action"] == "scale_down":
                    self.current_workers = max(self.min_workers, self.current_workers - decision["count"])
                
                logger.info(f"Auto-scaling decision: {decision['action']} by {decision['count']} workers. Total: {self.current_workers}")
            
            return decision
    
    def get_scaling_stats(self) -> Dict[str, Any]:
        """Get auto-scaling statistics."""
        with self._lock:
            recent_decisions = list(self.scaling_decisions)[-10:]
            
            return {
                "current_workers": self.current_workers,
                "min_workers": self.min_workers,
                "max_workers": self.max_workers,
                "recent_decisions": [
                    {
                        "timestamp": d["timestamp"].isoformat(),
                        "action": d["decision"]["action"],
                        "count": d["decision"]["count"],
                        "reason": d["decision"]["reason"]
                    }
                    for d in recent_decisions
                ],
                "total_scaling_actions": len(self.scaling_decisions),
                "last_scale_action": self.last_scale_action.isoformat() if self.last_scale_action else None
            }
    
    def _calculate_scaling_factors(self, metrics: Dict[str, Any]) -> Dict[str, float]:
        """Calculate scaling factors from metrics."""
        factors = {}
        
        # CPU utilization factor
        cpu_usage = metrics.get("cpu_usage", 0)
        if cpu_usage > self.scale_up_cpu_threshold:
            factors["cpu"] = min(2.0, cpu_usage / self.scale_up_cpu_threshold)
        elif cpu_usage < self.scale_down_cpu_threshold:
            factors["cpu"] = max(0.5, cpu_usage / self.scale_down_cpu_threshold)
        else:
            factors["cpu"] = 1.0
        
        # Memory utilization factor
        memory_usage = metrics.get("memory_usage", 0)
        if memory_usage > 80:
            factors["memory"] = min(2.0, memory_usage / 80)
        else:
            factors["memory"] = 1.0
        
        # Queue length factor
        queue_length = metrics.get("queue_length", 0)
        if queue_length > self.current_workers * 2:
            factors["queue"] = min(3.0, queue_length / (self.current_workers * 2))
        else:
            factors["queue"] = max(0.5, queue_length / max(1, self.current_workers))
        
        # Response time factor
        response_time = metrics.get("avg_response_time", 0)
        if response_time > 2.0:  # 2 second threshold
            factors["response_time"] = min(2.0, response_time / 2.0)
        else:
            factors["response_time"] = 1.0
        
        # Throughput factor
        throughput = metrics.get("throughput", 0)
        expected_throughput = self.current_workers * 10  # 10 requests per worker
        if throughput < expected_throughput * 0.7:
            factors["throughput"] = 0.7
        else:
            factors["throughput"] = 1.0
        
        return factors
    
    def _make_scaling_decision(self, factors: Dict[str, float]) -> Optional[Dict[str, Any]]:
        """Make scaling decision based on factors."""
        # Calculate overall scaling need
        scale_factors = [factors.get(key, 1.0) for key in ["cpu", "memory", "queue", "response_time"]]
        avg_factor = statistics.mean(scale_factors)
        max_factor = max(scale_factors)
        
        # Scale up conditions
        if (avg_factor > 1.5 or max_factor > 2.0) and self.current_workers < self.max_workers:
            scale_count = min(
                self.max_workers - self.current_workers,
                max(1, int((avg_factor - 1.0) * self.current_workers))
            )
            
            return {
                "action": "scale_up",
                "count": scale_count,
                "reason": f"High load detected (avg_factor: {avg_factor:.2f})",
                "factors": factors
            }
        
        # Scale down conditions
        elif avg_factor < 0.6 and max_factor < 0.8 and self.current_workers > self.min_workers:
            # Be more conservative with scale down
            if len(self.metrics_history) >= 5:
                # Check if consistently low for last 5 measurements
                recent_factors = []
                for metric_entry in list(self.metrics_history)[-5:]:
                    recent_calc = self._calculate_scaling_factors(metric_entry["metrics"])
                    recent_avg = statistics.mean([recent_calc.get(key, 1.0) for key in ["cpu", "memory", "queue"]])
                    recent_factors.append(recent_avg)
                
                if all(f < 0.7 for f in recent_factors):
                    scale_count = min(
                        self.current_workers - self.min_workers,
                        max(1, int((0.8 - avg_factor) * self.current_workers / 2))
                    )
                    
                    return {
                        "action": "scale_down",
                        "count": scale_count,
                        "reason": f"Consistently low load (avg_factor: {avg_factor:.2f})",
                        "factors": factors
                    }
        
        return None


class AdvancedPerformanceOptimizer:
    """Main performance optimization orchestrator."""
    
    def __init__(self, enable_auto_scaling: bool = True):
        self.cache = AdaptiveCache(max_size=50000, default_ttl=3600)
        self.processor = ConcurrentProcessor(enable_process_pool=True)
        self.auto_scaler = AutoScaler() if enable_auto_scaling else None
        
        # Performance tracking
        self.operation_metrics = defaultdict(PerformanceMetrics)
        self.optimization_history = deque(maxlen=1000)
        
        # Resource monitoring
        self.resource_monitor_thread = None
        self.running = True
        self._lock = threading.Lock()
        
        # Start monitoring
        self._start_resource_monitoring()
    
    def optimize_operation(self, operation_name: str, func: Callable, *args, cache_key: Optional[str] = None, 
                         cache_ttl: Optional[float] = None, priority: int = 1, **kwargs) -> Any:
        """Optimize operation with caching and concurrent processing."""
        start_time = time.time()
        
        try:
            # Try cache first
            if cache_key:
                cached_result = self.cache.get(cache_key)
                if cached_result is not None:
                    duration = time.time() - start_time
                    self.operation_metrics[operation_name].add_operation(duration, True)
                    self.operation_metrics[operation_name].cache_hits += 1
                    return cached_result
                else:
                    self.operation_metrics[operation_name].cache_misses += 1
            
            # Execute operation
            if priority > 2 or self._is_cpu_intensive(func):
                # Use concurrent processing for high priority or CPU intensive tasks
                future = self.processor.submit_task(func, *args, priority=priority, **kwargs)
                result = future.result()  # This would be awaited in async context
            else:
                # Execute directly for simple tasks
                result = func(*args, **kwargs)
            
            # Cache result if applicable
            if cache_key and result is not None:
                self.cache.put(cache_key, result, cache_ttl)
            
            # Record metrics
            duration = time.time() - start_time
            self.operation_metrics[operation_name].add_operation(duration, True)
            
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            self.operation_metrics[operation_name].add_operation(duration, False)
            raise
    
    def batch_optimize(self, operations: List[Dict[str, Any]]) -> List[Any]:
        """Optimize batch of operations with intelligent scheduling."""
        start_time = time.time()
        
        # Separate cached and non-cached operations
        cached_results = {}
        pending_operations = []
        
        for i, op in enumerate(operations):
            cache_key = op.get("cache_key")
            if cache_key:
                cached_result = self.cache.get(cache_key)
                if cached_result is not None:
                    cached_results[i] = cached_result
                    continue
            
            pending_operations.append((i, op))
        
        # Submit remaining operations for concurrent processing
        futures = {}
        for i, op in pending_operations:
            func = op["func"]
            args = op.get("args", ())
            kwargs = op.get("kwargs", {})
            priority = op.get("priority", 1)
            
            future = self.processor.submit_task(func, *args, priority=priority, **kwargs)
            futures[i] = future
        
        # Collect results
        results = [None] * len(operations)
        
        # Add cached results
        for i, result in cached_results.items():
            results[i] = result
        
        # Collect processed results
        for i, future in futures.items():
            try:
                result = future.result()
                results[i] = result
                
                # Cache if applicable
                op = operations[i]
                cache_key = op.get("cache_key")
                cache_ttl = op.get("cache_ttl")
                if cache_key and result is not None:
                    self.cache.put(cache_key, result, cache_ttl)
                    
            except Exception as e:
                results[i] = e
        
        # Record batch metrics
        duration = time.time() - start_time
        self.operation_metrics["batch_operation"].add_operation(duration, True)
        
        return results
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get comprehensive optimization statistics."""
        with self._lock:
            cache_stats = self.cache.get_stats()
            processor_stats = self.processor.get_processing_stats()
            auto_scaler_stats = self.auto_scaler.get_scaling_stats() if self.auto_scaler else {}
            
            # Operation metrics
            operation_stats = {}
            for op_name, metrics in self.operation_metrics.items():
                operation_stats[op_name] = {
                    "total_operations": metrics.operation_count,
                    "average_duration": metrics.get_average_duration(),
                    "min_duration": metrics.min_duration if metrics.min_duration != float('inf') else 0,
                    "max_duration": metrics.max_duration,
                    "error_rate": metrics.error_count / max(1, metrics.operation_count),
                    "cache_hit_rate": metrics.get_cache_hit_rate()
                }
            
            return {
                "cache": cache_stats,
                "processor": processor_stats,
                "auto_scaler": auto_scaler_stats,
                "operations": operation_stats,
                "system_resources": self._get_system_resources(),
                "optimization_recommendations": self._get_optimization_recommendations()
            }
    
    def _is_cpu_intensive(self, func: Callable) -> bool:
        """Determine if function is CPU intensive."""
        # Simple heuristic based on function name and module
        func_name = getattr(func, '__name__', str(func)).lower()
        cpu_intensive_keywords = [
            'process', 'calculate', 'compute', 'analyze', 'transform',
            'encode', 'decode', 'hash', 'search', 'sort', 'algorithm'
        ]
        
        return any(keyword in func_name for keyword in cpu_intensive_keywords)
    
    def _start_resource_monitoring(self):
        """Start resource monitoring for auto-scaling."""
        if not self.auto_scaler:
            return
        
        def monitor_loop():
            while self.running:
                try:
                    # Collect current metrics
                    metrics = {
                        "cpu_usage": psutil.cpu_percent(interval=1),
                        "memory_usage": psutil.virtual_memory().percent,
                        "queue_length": (
                            self.processor.high_priority_queue.qsize() +
                            self.processor.normal_priority_queue.qsize() +
                            self.processor.low_priority_queue.qsize()
                        ),
                        "active_tasks": len(self.processor.active_tasks),
                        "avg_response_time": self._calculate_avg_response_time(),
                        "throughput": self._calculate_throughput()
                    }
                    
                    # Evaluate scaling
                    scaling_decision = self.auto_scaler.evaluate_scaling(metrics)
                    
                    if scaling_decision:
                        self._apply_scaling_decision(scaling_decision)
                    
                    time.sleep(30)  # Monitor every 30 seconds
                    
                except Exception as e:
                    logger.error(f"Resource monitoring error: {e}")
                    time.sleep(30)
        
        self.resource_monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.resource_monitor_thread.start()
    
    def _calculate_avg_response_time(self) -> float:
        """Calculate average response time across all operations."""
        total_duration = 0
        total_operations = 0
        
        for metrics in self.operation_metrics.values():
            total_duration += metrics.total_duration
            total_operations += metrics.operation_count
        
        return total_duration / max(1, total_operations)
    
    def _calculate_throughput(self) -> float:
        """Calculate current throughput (operations per second)."""
        # Calculate throughput over last minute
        now = datetime.now()
        cutoff = now - timedelta(minutes=1)
        
        recent_operations = 0
        for metrics in self.operation_metrics.values():
            if metrics.last_updated >= cutoff:
                recent_operations += metrics.operation_count
        
        return recent_operations / 60.0  # Operations per second
    
    def _apply_scaling_decision(self, decision: Dict[str, Any]):
        """Apply auto-scaling decision."""
        if decision["action"] == "scale_up":
            # Increase worker capacity
            new_workers = self.processor.max_workers + decision["count"]
            self.processor.max_workers = min(64, new_workers)
        elif decision["action"] == "scale_down":
            # Decrease worker capacity
            new_workers = self.processor.max_workers - decision["count"]
            self.processor.max_workers = max(2, new_workers)
    
    def _get_system_resources(self) -> Dict[str, Any]:
        """Get current system resource usage."""
        return {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent,
            "load_average": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None,
            "process_count": len(psutil.pids())
        }
    
    def _get_optimization_recommendations(self) -> List[str]:
        """Get optimization recommendations based on current metrics."""
        recommendations = []
        
        cache_stats = self.cache.get_stats()
        if cache_stats["hit_rate"] < 0.5:
            recommendations.append("Consider increasing cache size or TTL")
        
        processor_stats = self.processor.get_processing_stats()
        if processor_stats["active_tasks"] > processor_stats["max_workers"] * 2:
            recommendations.append("Consider increasing worker pool size")
        
        system_resources = self._get_system_resources()
        if system_resources["memory_percent"] > 85:
            recommendations.append("High memory usage detected - consider memory optimization")
        
        if system_resources["cpu_percent"] > 80:
            recommendations.append("High CPU usage detected - consider CPU optimization")
        
        return recommendations
    
    def shutdown(self):
        """Shutdown optimizer and cleanup resources."""
        self.running = False
        self.cache.shutdown()
        self.processor.shutdown()
        
        if self.resource_monitor_thread and self.resource_monitor_thread.is_alive():
            self.resource_monitor_thread.join(timeout=5)


# Global performance optimizer instance
_performance_optimizer = None

def get_performance_optimizer() -> AdvancedPerformanceOptimizer:
    """Get global performance optimizer instance."""
    global _performance_optimizer
    if _performance_optimizer is None:
        _performance_optimizer = AdvancedPerformanceOptimizer()
    return _performance_optimizer

def optimize_operation(operation_name: str, func: Callable, *args, cache_key: Optional[str] = None, 
                      cache_ttl: Optional[float] = None, priority: int = 1, **kwargs) -> Any:
    """Optimize operation with caching and concurrent processing."""
    optimizer = get_performance_optimizer()
    return optimizer.optimize_operation(operation_name, func, *args, cache_key=cache_key, 
                                      cache_ttl=cache_ttl, priority=priority, **kwargs)

def get_optimization_stats() -> Dict[str, Any]:
    """Get optimization statistics."""
    optimizer = get_performance_optimizer()
    return optimizer.get_optimization_stats()