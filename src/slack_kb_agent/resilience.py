"""Resilience and reliability patterns for robust system operation."""

from __future__ import annotations

import asyncio
import logging
import time
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Callable, Any, Union
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
import random

from .monitoring import get_global_metrics, StructuredLogger
from .cache import get_cache_manager
from .exceptions import SlackKBAgentError

logger = logging.getLogger(__name__)


class BackoffStrategy(Enum):
    """Backoff strategies for retry mechanisms."""
    FIXED = "fixed"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    JITTERED = "jittered"


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


@dataclass
class RetryConfig:
    """Configuration for retry mechanisms."""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    backoff_strategy: BackoffStrategy = BackoffStrategy.EXPONENTIAL
    jitter_factor: float = 0.1
    retryable_exceptions: List[type] = field(default_factory=lambda: [Exception])
    
    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt number."""
        if self.backoff_strategy == BackoffStrategy.FIXED:
            delay = self.base_delay
        elif self.backoff_strategy == BackoffStrategy.LINEAR:
            delay = self.base_delay * attempt
        elif self.backoff_strategy == BackoffStrategy.EXPONENTIAL:
            delay = self.base_delay * (2 ** (attempt - 1))
        elif self.backoff_strategy == BackoffStrategy.JITTERED:
            delay = self.base_delay * (2 ** (attempt - 1))
            jitter = delay * self.jitter_factor * random.random()
            delay += jitter
        else:
            delay = self.base_delay
        
        return min(delay, self.max_delay)


@dataclass
class BulkheadConfig:
    """Configuration for bulkhead isolation."""
    name: str
    max_concurrent: int = 10
    queue_size: int = 100
    timeout: float = 30.0
    rejection_handler: Optional[Callable] = None


class ResilientExecutor:
    """Resilient task executor with retry, circuit breaker, and bulkhead patterns."""
    
    def __init__(self, retry_config: Optional[RetryConfig] = None):
        self.retry_config = retry_config or RetryConfig()
        self.metrics = get_global_metrics()
        self.logger = StructuredLogger("resilient_executor")
        
        # Execution statistics
        self.execution_stats = {
            "total_attempts": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "retries": 0,
            "average_duration": 0.0
        }
        
        self._lock = threading.Lock()
    
    async def execute_with_retry(
        self,
        func: Callable,
        *args,
        retry_config: Optional[RetryConfig] = None,
        **kwargs
    ) -> Any:
        """Execute function with retry logic."""
        config = retry_config or self.retry_config
        last_exception = None
        
        for attempt in range(1, config.max_attempts + 1):
            start_time = time.time()
            
            try:
                with self._lock:
                    self.execution_stats["total_attempts"] += 1
                
                self.logger.log_event("execution_attempt", {
                    "attempt": attempt,
                    "max_attempts": config.max_attempts,
                    "function": func.__name__ if hasattr(func, '__name__') else str(func)
                })
                
                # Execute function
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                # Success - update stats
                duration = time.time() - start_time
                with self._lock:
                    self.execution_stats["successful_executions"] += 1
                    self._update_average_duration(duration)
                
                self.logger.log_event("execution_success", {
                    "attempt": attempt,
                    "duration": duration,
                    "function": func.__name__ if hasattr(func, '__name__') else str(func)
                })
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                last_exception = e
                
                # Check if exception is retryable
                if not any(isinstance(e, exc_type) for exc_type in config.retryable_exceptions):
                    self.logger.log_event("execution_non_retryable_error", {
                        "attempt": attempt,
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "duration": duration
                    })
                    
                    with self._lock:
                        self.execution_stats["failed_executions"] += 1
                    
                    raise e
                
                # Log retry attempt
                self.logger.log_event("execution_retry", {
                    "attempt": attempt,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "duration": duration
                })
                
                # If this was the last attempt, don't wait
                if attempt == config.max_attempts:
                    break
                
                # Calculate and apply backoff delay
                delay = config.calculate_delay(attempt)
                await asyncio.sleep(delay)
                
                with self._lock:
                    self.execution_stats["retries"] += 1
        
        # All attempts failed
        with self._lock:
            self.execution_stats["failed_executions"] += 1
        
        self.logger.log_event("execution_exhausted", {
            "max_attempts": config.max_attempts,
            "final_error": str(last_exception),
            "function": func.__name__ if hasattr(func, '__name__') else str(func)
        })
        
        raise last_exception
    
    def _update_average_duration(self, duration: float) -> None:
        """Update average execution duration."""
        current_avg = self.execution_stats["average_duration"]
        successful_count = self.execution_stats["successful_executions"]
        
        # Calculate new average
        new_avg = ((current_avg * (successful_count - 1)) + duration) / successful_count
        self.execution_stats["average_duration"] = new_avg
    
    def get_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        with self._lock:
            return self.execution_stats.copy()


class CircuitBreaker:
    """Circuit breaker pattern implementation."""
    
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: type = Exception
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        # State tracking
        self._state = "closed"  # closed, open, half_open
        self._failure_count = 0
        self._last_failure_time: Optional[datetime] = None
        self._lock = threading.Lock()
        
        # Metrics
        self.metrics = get_global_metrics()
        self.logger = StructuredLogger(f"circuit_breaker_{name}")
        
        self._stats = {
            "calls": 0,
            "successes": 0,
            "failures": 0,
            "circuit_open_count": 0,
            "state_changes": []
        }
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function through circuit breaker."""
        with self._lock:
            self._stats["calls"] += 1
            
            # Check if circuit should transition from open to half-open
            if self._state == "open" and self._should_attempt_reset():
                self._state = "half_open"
                self._log_state_change("half_open")
            
            # If circuit is open, fail fast
            if self._state == "open":
                self._stats["failures"] += 1
                raise SlackKBAgentError(f"Circuit breaker {self.name} is OPEN")
        
        try:
            # Execute function
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            # Success - handle state transitions
            with self._lock:
                self._stats["successes"] += 1
                
                if self._state == "half_open":
                    # Reset circuit breaker
                    self._state = "closed"
                    self._failure_count = 0
                    self._last_failure_time = None
                    self._log_state_change("closed")
            
            return result
            
        except self.expected_exception as e:
            # Handle failure
            with self._lock:
                self._stats["failures"] += 1
                self._failure_count += 1
                self._last_failure_time = datetime.now()
                
                # Check if threshold is exceeded
                if self._failure_count >= self.failure_threshold:
                    if self._state != "open":
                        self._state = "open"
                        self._stats["circuit_open_count"] += 1
                        self._log_state_change("open")
            
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit should attempt reset."""
        if self._last_failure_time is None:
            return False
        
        time_since_failure = datetime.now() - self._last_failure_time
        return time_since_failure.total_seconds() >= self.recovery_timeout
    
    def _log_state_change(self, new_state: str) -> None:
        """Log circuit breaker state change."""
        self._stats["state_changes"].append({
            "timestamp": datetime.now().isoformat(),
            "state": new_state,
            "failure_count": self._failure_count
        })
        
        self.logger.log_event("circuit_breaker_state_change", {
            "name": self.name,
            "new_state": new_state,
            "failure_count": self._failure_count
        })
    
    @property
    def state(self) -> str:
        """Get current circuit breaker state."""
        with self._lock:
            return self._state
    
    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        with self._lock:
            return {
                **self._stats.copy(),
                "current_state": self._state,
                "failure_count": self._failure_count,
                "last_failure_time": self._last_failure_time.isoformat() if self._last_failure_time else None
            }
    
    def reset(self) -> None:
        """Manually reset circuit breaker."""
        with self._lock:
            self._state = "closed"
            self._failure_count = 0
            self._last_failure_time = None
            self._log_state_change("closed_manual_reset")


class BulkheadIsolation:
    """Bulkhead isolation pattern for resource management."""
    
    def __init__(self, config: BulkheadConfig):
        self.config = config
        self.executor = ThreadPoolExecutor(
            max_workers=config.max_concurrent,
            thread_name_prefix=f"bulkhead_{config.name}"
        )
        
        self.metrics = get_global_metrics()
        self.logger = StructuredLogger(f"bulkhead_{config.name}")
        
        # Resource tracking
        self._active_tasks = 0
        self._queued_tasks = 0
        self._rejected_tasks = 0
        self._lock = threading.Lock()
        
        self._stats = {
            "submitted_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "rejected_tasks": 0,
            "average_execution_time": 0.0
        }
    
    async def submit(self, func: Callable, *args, **kwargs) -> Any:
        """Submit task to bulkhead with isolation."""
        with self._lock:
            # Check if we should reject due to queue size
            if self._queued_tasks >= self.config.queue_size:
                self._rejected_tasks += 1
                self._stats["rejected_tasks"] += 1
                
                if self.config.rejection_handler:
                    return self.config.rejection_handler(func, *args, **kwargs)
                else:
                    raise SlackKBAgentError(
                        f"Bulkhead {self.config.name} queue is full"
                    )
            
            self._queued_tasks += 1
            self._stats["submitted_tasks"] += 1
        
        start_time = time.time()
        
        try:
            # Submit to thread pool
            if asyncio.iscoroutinefunction(func):
                # For async functions, we need to run them in the current event loop
                future = asyncio.create_task(func(*args, **kwargs))
            else:
                # For sync functions, use thread pool
                loop = asyncio.get_event_loop()
                future = loop.run_in_executor(self.executor, func, *args, **kwargs)
            
            with self._lock:
                self._active_tasks += 1
                self._queued_tasks -= 1
            
            # Wait for completion with timeout
            result = await asyncio.wait_for(future, timeout=self.config.timeout)
            
            # Success
            execution_time = time.time() - start_time
            with self._lock:
                self._active_tasks -= 1
                self._stats["completed_tasks"] += 1
                self._update_average_execution_time(execution_time)
            
            self.logger.log_event("bulkhead_task_completed", {
                "bulkhead": self.config.name,
                "execution_time": execution_time
            })
            
            return result
            
        except asyncio.TimeoutError:
            with self._lock:
                self._active_tasks -= 1
                self._stats["failed_tasks"] += 1
            
            self.logger.log_event("bulkhead_task_timeout", {
                "bulkhead": self.config.name,
                "timeout": self.config.timeout
            })
            
            raise SlackKBAgentError(
                f"Task timed out in bulkhead {self.config.name}"
            )
            
        except Exception as e:
            with self._lock:
                self._active_tasks -= 1
                self._stats["failed_tasks"] += 1
            
            self.logger.log_event("bulkhead_task_failed", {
                "bulkhead": self.config.name,
                "error": str(e),
                "error_type": type(e).__name__
            })
            
            raise e
    
    def _update_average_execution_time(self, execution_time: float) -> None:
        """Update average execution time."""
        current_avg = self._stats["average_execution_time"]
        completed_count = self._stats["completed_tasks"]
        
        new_avg = ((current_avg * (completed_count - 1)) + execution_time) / completed_count
        self._stats["average_execution_time"] = new_avg
    
    def get_stats(self) -> Dict[str, Any]:
        """Get bulkhead statistics."""
        with self._lock:
            return {
                **self._stats.copy(),
                "active_tasks": self._active_tasks,
                "queued_tasks": self._queued_tasks,
                "rejected_tasks": self._rejected_tasks,
                "max_concurrent": self.config.max_concurrent,
                "queue_size": self.config.queue_size
            }
    
    def shutdown(self, wait: bool = True) -> None:
        """Shutdown bulkhead executor."""
        self.executor.shutdown(wait=wait)


class HealthMonitor:
    """Health monitoring and status tracking."""
    
    def __init__(self, name: str):
        self.name = name
        self.status = HealthStatus.HEALTHY
        self.health_checks: Dict[str, Callable] = {}
        self.last_check_time: Optional[datetime] = None
        self.check_interval = 30.0  # seconds
        
        self.metrics = get_global_metrics()
        self.logger = StructuredLogger(f"health_monitor_{name}")
        
        self._lock = threading.Lock()
        self._health_history: List[Dict[str, Any]] = []
    
    def register_health_check(self, name: str, check_func: Callable) -> None:
        """Register a health check function."""
        with self._lock:
            self.health_checks[name] = check_func
        
        self.logger.log_event("health_check_registered", {
            "monitor": self.name,
            "check_name": name
        })
    
    async def check_health(self) -> HealthStatus:
        """Perform health checks and update status."""
        check_results = {}
        overall_healthy = True
        degraded_checks = []
        failed_checks = []
        
        for check_name, check_func in self.health_checks.items():
            try:
                start_time = time.time()
                
                if asyncio.iscoroutinefunction(check_func):
                    result = await check_func()
                else:
                    result = check_func()
                
                check_time = time.time() - start_time
                
                # Interpret result
                if isinstance(result, bool):
                    check_status = HealthStatus.HEALTHY if result else HealthStatus.UNHEALTHY
                elif isinstance(result, HealthStatus):
                    check_status = result
                else:
                    # Assume healthy if we got a non-exception result
                    check_status = HealthStatus.HEALTHY
                
                check_results[check_name] = {
                    "status": check_status,
                    "duration": check_time,
                    "result": result
                }
                
                if check_status == HealthStatus.UNHEALTHY:
                    overall_healthy = False
                    failed_checks.append(check_name)
                elif check_status == HealthStatus.DEGRADED:
                    degraded_checks.append(check_name)
                
            except Exception as e:
                check_results[check_name] = {
                    "status": HealthStatus.UNHEALTHY,
                    "error": str(e),
                    "error_type": type(e).__name__
                }
                
                overall_healthy = False
                failed_checks.append(check_name)
                
                self.logger.log_event("health_check_failed", {
                    "monitor": self.name,
                    "check_name": check_name,
                    "error": str(e)
                })
        
        # Determine overall status
        if not overall_healthy:
            new_status = HealthStatus.CRITICAL if len(failed_checks) > len(self.health_checks) / 2 else HealthStatus.UNHEALTHY
        elif degraded_checks:
            new_status = HealthStatus.DEGRADED
        else:
            new_status = HealthStatus.HEALTHY
        
        # Update status and history
        with self._lock:
            old_status = self.status
            self.status = new_status
            self.last_check_time = datetime.now()
            
            self._health_history.append({
                "timestamp": self.last_check_time.isoformat(),
                "status": new_status.value,
                "check_results": check_results,
                "failed_checks": failed_checks,
                "degraded_checks": degraded_checks
            })
            
            # Keep only last 100 health checks
            if len(self._health_history) > 100:
                self._health_history = self._health_history[-100:]
        
        # Log status change
        if old_status != new_status:
            self.logger.log_event("health_status_changed", {
                "monitor": self.name,
                "old_status": old_status.value,
                "new_status": new_status.value,
                "failed_checks": failed_checks,
                "degraded_checks": degraded_checks
            })
        
        return new_status
    
    async def start_monitoring(self, interval: Optional[float] = None) -> None:
        """Start continuous health monitoring."""
        if interval:
            self.check_interval = interval
        
        self.logger.log_event("health_monitoring_started", {
            "monitor": self.name,
            "interval": self.check_interval
        })
        
        while True:
            try:
                await self.check_health()
                await asyncio.sleep(self.check_interval)
            except Exception as e:
                self.logger.log_event("health_monitoring_error", {
                    "monitor": self.name,
                    "error": str(e)
                })
                await asyncio.sleep(self.check_interval)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current health status."""
        with self._lock:
            return {
                "name": self.name,
                "status": self.status.value,
                "last_check_time": self.last_check_time.isoformat() if self.last_check_time else None,
                "check_count": len(self.health_checks),
                "history_length": len(self._health_history)
            }
    
    def get_health_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get health check history."""
        with self._lock:
            history = self._health_history.copy()
            if limit:
                history = history[-limit:]
            return history


# Global instances registry
_resilience_components: Dict[str, Any] = {}


def get_resilient_executor(name: str = "default", retry_config: Optional[RetryConfig] = None) -> ResilientExecutor:
    """Get or create resilient executor instance."""
    if name not in _resilience_components:
        _resilience_components[name] = ResilientExecutor(retry_config)
    return _resilience_components[name]


def get_circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0,
    expected_exception: type = Exception
) -> CircuitBreaker:
    """Get or create circuit breaker instance."""
    if name not in _resilience_components:
        _resilience_components[name] = CircuitBreaker(
            name, failure_threshold, recovery_timeout, expected_exception
        )
    return _resilience_components[name]


def get_bulkhead(config: BulkheadConfig) -> BulkheadIsolation:
    """Get or create bulkhead isolation instance."""
    if config.name not in _resilience_components:
        _resilience_components[config.name] = BulkheadIsolation(config)
    return _resilience_components[config.name]


def get_health_monitor(name: str) -> HealthMonitor:
    """Get or create health monitor instance."""
    if name not in _resilience_components:
        _resilience_components[name] = HealthMonitor(name)
    return _resilience_components[name]


def get_resilience_stats() -> Dict[str, Any]:
    """Get statistics from all resilience components."""
    stats = {}
    
    for name, component in _resilience_components.items():
        if hasattr(component, 'get_stats'):
            stats[name] = component.get_stats()
        elif hasattr(component, 'get_status'):
            stats[name] = component.get_status()
    
    return stats


def cleanup_resilience_components() -> None:
    """Cleanup all resilience components."""
    for name, component in _resilience_components.items():
        if hasattr(component, 'shutdown'):
            try:
                component.shutdown()
            except Exception as e:
                logger.warning(f"Error shutting down {name}: {e}")
    
    _resilience_components.clear()