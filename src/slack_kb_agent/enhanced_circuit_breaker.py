"""Enhanced circuit breaker system with advanced patterns and monitoring.

This module extends the basic circuit breaker with additional features:
- Bulkhead pattern for resource isolation
- Adaptive thresholds based on historical data
- Health scoring with multiple metrics
- Cascading circuit breaker protection
- Integration with security and monitoring systems
"""

from __future__ import annotations

import time
import logging
import threading
import statistics
from collections import defaultdict, deque
from typing import Callable, Any, Optional, Dict, List, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from contextlib import contextmanager
from datetime import datetime, timedelta

from .circuit_breaker import CircuitBreaker, CircuitState, CircuitBreakerConfig

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status levels for services."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


class FailureType(Enum):
    """Types of failures that can be tracked."""
    TIMEOUT = "timeout"
    CONNECTION_ERROR = "connection_error"
    HTTP_ERROR = "http_error"
    AUTHENTICATION_ERROR = "auth_error"
    RATE_LIMIT = "rate_limit"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    UNKNOWN = "unknown"


@dataclass
class ServiceMetrics:
    """Comprehensive service health metrics."""
    success_count: int = 0
    failure_count: int = 0
    timeout_count: int = 0
    response_times: deque = field(default_factory=lambda: deque(maxlen=100))
    failure_types: Dict[FailureType, int] = field(default_factory=lambda: defaultdict(int))
    last_success: Optional[float] = None
    last_failure: Optional[float] = None
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    
    def success_rate(self) -> float:
        """Calculate success rate."""
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.0
    
    def average_response_time(self) -> float:
        """Calculate average response time."""
        return statistics.mean(self.response_times) if self.response_times else 0.0
    
    def p95_response_time(self) -> float:
        """Calculate 95th percentile response time."""
        if len(self.response_times) < 5:
            return self.average_response_time()
        
        sorted_times = sorted(self.response_times)
        index = int(0.95 * len(sorted_times))
        return sorted_times[index]


@dataclass
class BulkheadConfig:
    """Configuration for bulkhead pattern."""
    max_concurrent_calls: int = 10
    max_wait_time: float = 5.0  # Maximum time to wait for a slot
    timeout_per_call: float = 30.0  # Timeout for individual calls


class AdaptiveCircuitBreaker(CircuitBreaker):
    """Enhanced circuit breaker with adaptive thresholds and health monitoring."""
    
    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        if config is None:
            config = CircuitBreakerConfig()
        super().__init__(config)
        self.name = name
        
        # Enhanced metrics
        self.metrics = ServiceMetrics()
        
        # Adaptive threshold parameters
        self.base_failure_threshold = self.config.failure_threshold
        self.adaptive_enabled = True
        self.historical_window = 3600  # 1 hour window
        
        # Health scoring
        self.health_weights = {
            'success_rate': 0.4,
            'response_time': 0.3,
            'failure_pattern': 0.2,
            'availability': 0.1
        }
        
        # Cascading protection
        self.dependencies: Set[str] = set()
        self.dependent_services: Set[str] = set()
        
    def _on_success(self, execution_time: float) -> None:
        """Enhanced success recording with response time tracking."""
        super()._on_success(execution_time)
        
        current_time = time.time()
        self.metrics.success_count += 1
        self.metrics.consecutive_successes += 1
        self.metrics.consecutive_failures = 0
        self.metrics.last_success = current_time
        self.metrics.response_times.append(execution_time)
        
        # Adaptive threshold adjustment
        if self.adaptive_enabled:
            self._adjust_thresholds()
    
    def _on_failure(self, exception: Exception, execution_time: float) -> None:
        """Enhanced failure recording with failure type tracking."""
        super()._on_failure(exception, execution_time)
        
        # Determine failure type from exception
        failure_type = self._classify_failure(exception)
        
        current_time = time.time()
        self.metrics.failure_count += 1
        self.metrics.consecutive_failures += 1
        self.metrics.consecutive_successes = 0
        self.metrics.last_failure = current_time
        self.metrics.failure_types[failure_type] += 1
        
        if failure_type == FailureType.TIMEOUT:
            self.metrics.timeout_count += 1
        
        # Log failure pattern for analysis
        logger.warning(
            f"Circuit breaker {self.name} recorded failure: {failure_type.value} "
            f"(consecutive: {self.metrics.consecutive_failures})"
        )
        
        # Adaptive threshold adjustment
        if self.adaptive_enabled:
            self._adjust_thresholds()
    
    def _classify_failure(self, exception: Exception) -> FailureType:
        """Classify failure type from exception."""
        exception_str = str(exception).lower()
        exception_type = type(exception).__name__.lower()
        
        if "timeout" in exception_str or "timeout" in exception_type:
            return FailureType.TIMEOUT
        elif "connection" in exception_str or "connection" in exception_type:
            return FailureType.CONNECTION_ERROR
        elif "auth" in exception_str or "unauthorized" in exception_str:
            return FailureType.AUTHENTICATION_ERROR
        elif "rate" in exception_str and "limit" in exception_str:
            return FailureType.RATE_LIMIT
        elif "http" in exception_str or "status" in exception_str:
            return FailureType.HTTP_ERROR
        else:
            return FailureType.UNKNOWN
    
    # Keep backward compatibility methods
    def _record_success(self, response_time: float = 0.0):
        """Backward compatibility method."""
        self._on_success(response_time)
    
    def _record_failure(self, failure_type: FailureType = FailureType.UNKNOWN):
        """Backward compatibility method."""
        # Create a dummy exception for the failure type
        exception = Exception(f"Failure type: {failure_type.value}")
        self._on_failure(exception, 0.0)
    
    def _adjust_thresholds(self):
        """Adjust failure threshold based on historical performance."""
        if self.metrics.success_count + self.metrics.failure_count < 50:
            return  # Need more data
        
        success_rate = self.metrics.success_rate()
        avg_response_time = self.metrics.average_response_time()
        
        # Adjust threshold based on service health
        if success_rate > 0.95 and avg_response_time < 1.0:
            # Very healthy service - can tolerate more failures
            new_threshold = min(self.base_failure_threshold * 2, 15)
        elif success_rate > 0.8 and avg_response_time < 5.0:
            # Healthy service - normal threshold
            new_threshold = self.base_failure_threshold
        elif success_rate > 0.5:
            # Degraded service - lower threshold
            new_threshold = max(self.base_failure_threshold // 2, 2)
        else:
            # Unhealthy service - very low threshold
            new_threshold = 2
        
        if new_threshold != self.config.failure_threshold:
            logger.info(
                f"Adjusting circuit breaker {self.name} threshold: "
                f"{self.config.failure_threshold} -> {new_threshold}"
            )
            self.config.failure_threshold = new_threshold
    
    def calculate_health_score(self) -> float:
        """Calculate comprehensive health score (0.0 - 1.0)."""
        if self.metrics.success_count + self.metrics.failure_count == 0:
            return 1.0  # No data, assume healthy
        
        # Success rate component
        success_rate = self.metrics.success_rate()
        success_score = success_rate
        
        # Response time component
        avg_response_time = self.metrics.average_response_time()
        if avg_response_time == 0:
            response_score = 1.0
        elif avg_response_time < 1.0:
            response_score = 1.0
        elif avg_response_time < 5.0:
            response_score = 0.8
        elif avg_response_time < 10.0:
            response_score = 0.5
        else:
            response_score = 0.1
        
        # Failure pattern component
        recent_failures = self.metrics.consecutive_failures
        if recent_failures == 0:
            failure_pattern_score = 1.0
        elif recent_failures < 3:
            failure_pattern_score = 0.7
        elif recent_failures < 5:
            failure_pattern_score = 0.3
        else:
            failure_pattern_score = 0.0
        
        # Availability component (based on circuit state)
        if self.state == CircuitState.CLOSED:
            availability_score = 1.0
        elif self.state == CircuitState.HALF_OPEN:
            availability_score = 0.5
        else:  # OPEN
            availability_score = 0.0
        
        # Weighted score
        health_score = (
            success_score * self.health_weights['success_rate'] +
            response_score * self.health_weights['response_time'] +
            failure_pattern_score * self.health_weights['failure_pattern'] +
            availability_score * self.health_weights['availability']
        )
        
        return min(max(health_score, 0.0), 1.0)
    
    def get_health_status(self) -> HealthStatus:
        """Get overall health status."""
        health_score = self.calculate_health_score()
        
        if health_score >= 0.8:
            return HealthStatus.HEALTHY
        elif health_score >= 0.6:
            return HealthStatus.DEGRADED
        elif health_score >= 0.3:
            return HealthStatus.UNHEALTHY
        else:
            return HealthStatus.CRITICAL
    
    def add_dependency(self, service_name: str):
        """Add a service dependency for cascading protection."""
        self.dependencies.add(service_name)
    
    def add_dependent_service(self, service_name: str):
        """Add a dependent service that relies on this one."""
        self.dependent_services.add(service_name)
    
    def can_proceed(self) -> bool:
        """Check if the circuit allows requests to proceed."""
        if self.state == CircuitState.CLOSED:
            return True
        elif self.state == CircuitState.HALF_OPEN:
            return True  # Allow limited requests in half-open state
        else:  # OPEN
            return self._should_attempt_reset()


class BulkheadProtection:
    """Implement bulkhead pattern for resource isolation."""
    
    def __init__(self, name: str, config: BulkheadConfig):
        self.name = name
        self.config = config
        self.active_calls = 0
        self.waiting_calls = 0
        self.lock = threading.RLock()
        self.semaphore = threading.Semaphore(config.max_concurrent_calls)
        
        # Metrics
        self.total_calls = 0
        self.successful_calls = 0
        self.rejected_calls = 0
        self.timeout_calls = 0
        
    @contextmanager
    def acquire_slot(self, timeout: Optional[float] = None):
        """Acquire a slot for execution with timeout."""
        timeout = timeout or self.config.max_wait_time
        
        with self.lock:
            self.total_calls += 1
            self.waiting_calls += 1
        
        acquired = self.semaphore.acquire(timeout=timeout)
        
        with self.lock:
            self.waiting_calls -= 1
        
        if not acquired:
            with self.lock:
                self.rejected_calls += 1
            raise RuntimeError(f"Bulkhead {self.name}: No slots available within {timeout}s")
        
        try:
            with self.lock:
                self.active_calls += 1
            
            yield
            
            with self.lock:
                self.successful_calls += 1
                
        except Exception:
            with self.lock:
                self.timeout_calls += 1
            raise
        finally:
            with self.lock:
                self.active_calls -= 1
            self.semaphore.release()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get bulkhead metrics."""
        with self.lock:
            success_rate = (self.successful_calls / self.total_calls) if self.total_calls > 0 else 0.0
            rejection_rate = (self.rejected_calls / self.total_calls) if self.total_calls > 0 else 0.0
            
            return {
                'name': self.name,
                'active_calls': self.active_calls,
                'waiting_calls': self.waiting_calls,
                'total_calls': self.total_calls,
                'successful_calls': self.successful_calls,
                'rejected_calls': self.rejected_calls,
                'timeout_calls': self.timeout_calls,
                'success_rate': success_rate,
                'rejection_rate': rejection_rate,
                'utilization': self.active_calls / self.config.max_concurrent_calls
            }


class CascadingCircuitBreakerManager:
    """Manage cascading circuit breakers with dependency awareness."""
    
    def __init__(self):
        self.circuit_breakers: Dict[str, AdaptiveCircuitBreaker] = {}
        self.bulkheads: Dict[str, BulkheadProtection] = {}
        self.dependency_graph: Dict[str, Set[str]] = defaultdict(set)
        self.health_check_callbacks: Dict[str, Callable[[], bool]] = {}
        
        # Background monitoring
        self.monitoring_enabled = True
        self.monitoring_interval = 30.0  # seconds
        self.monitoring_thread = None
        
        self._start_monitoring()
    
    def register_circuit_breaker(self, name: str, config: Optional[CircuitBreakerConfig] = None) -> AdaptiveCircuitBreaker:
        """Register a new adaptive circuit breaker."""
        cb = AdaptiveCircuitBreaker(name, config)
        self.circuit_breakers[name] = cb
        logger.info(f"Registered adaptive circuit breaker: {name}")
        return cb
    
    def register_bulkhead(self, name: str, config: BulkheadConfig) -> BulkheadProtection:
        """Register a new bulkhead for resource isolation."""
        bulkhead = BulkheadProtection(name, config)
        self.bulkheads[name] = bulkhead
        logger.info(f"Registered bulkhead: {name}")
        return bulkhead
    
    def add_dependency(self, service: str, depends_on: str):
        """Add service dependency for cascading protection."""
        self.dependency_graph[service].add(depends_on)
        
        if service in self.circuit_breakers and depends_on in self.circuit_breakers:
            self.circuit_breakers[service].add_dependency(depends_on)
            self.circuit_breakers[depends_on].add_dependent_service(service)
        
        logger.info(f"Added dependency: {service} depends on {depends_on}")
    
    def register_health_check(self, service: str, callback: Callable[[], bool]):
        """Register a health check callback for a service."""
        self.health_check_callbacks[service] = callback
    
    def get_service_health(self, service: str) -> Tuple[HealthStatus, Dict[str, Any]]:
        """Get comprehensive service health information."""
        if service not in self.circuit_breakers:
            return HealthStatus.UNHEALTHY, {'error': 'Service not found'}
        
        cb = self.circuit_breakers[service]
        health_status = cb.get_health_status()
        health_score = cb.calculate_health_score()
        
        # Run health check if available
        health_check_passed = True
        if service in self.health_check_callbacks:
            try:
                health_check_passed = self.health_check_callbacks[service]()
            except Exception as e:
                logger.error(f"Health check failed for {service}: {e}")
                health_check_passed = False
        
        # Get bulkhead metrics if available
        bulkhead_metrics = None
        if service in self.bulkheads:
            bulkhead_metrics = self.bulkheads[service].get_metrics()
        
        details = {
            'health_score': health_score,
            'success_rate': cb.metrics.success_rate(),
            'avg_response_time': cb.metrics.average_response_time(),
            'p95_response_time': cb.metrics.p95_response_time(),
            'consecutive_failures': cb.metrics.consecutive_failures,
            'circuit_state': cb.state.value,
            'health_check_passed': health_check_passed,
            'bulkhead_metrics': bulkhead_metrics,
            'failure_types': dict(cb.metrics.failure_types),
            'dependencies': list(cb.dependencies),
            'dependent_services': list(cb.dependent_services)
        }
        
        return health_status, details
    
    def check_cascading_failures(self) -> List[Tuple[str, str]]:
        """Check for potential cascading failures."""
        cascading_risks = []
        
        for service, cb in self.circuit_breakers.items():
            if cb.state == CircuitState.OPEN:
                # Service is down, check dependents
                for dependent in cb.dependent_services:
                    if dependent in self.circuit_breakers:
                        dependent_cb = self.circuit_breakers[dependent]
                        if dependent_cb.get_health_status() == HealthStatus.DEGRADED:
                            cascading_risks.append((service, dependent))
                            logger.warning(
                                f"Potential cascading failure: {service} (down) -> {dependent} (degraded)"
                            )
        
        return cascading_risks
    
    def get_system_health_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive system health dashboard."""
        dashboard = {
            'timestamp': datetime.now().isoformat(),
            'services': {},
            'bulkheads': {},
            'cascading_risks': [],
            'system_summary': {
                'total_services': len(self.circuit_breakers),
                'healthy_services': 0,
                'degraded_services': 0,
                'unhealthy_services': 0,
                'critical_services': 0
            }
        }
        
        # Service health details
        for service_name in self.circuit_breakers:
            health_status, details = self.get_service_health(service_name)
            dashboard['services'][service_name] = {
                'status': health_status.value,
                **details
            }
            
            # Update summary counts
            dashboard['system_summary'][f'{health_status.value}_services'] += 1
        
        # Bulkhead metrics
        for bulkhead_name, bulkhead in self.bulkheads.items():
            dashboard['bulkheads'][bulkhead_name] = bulkhead.get_metrics()
        
        # Cascading failure risks
        dashboard['cascading_risks'] = self.check_cascading_failures()
        
        # Overall system health score
        if dashboard['system_summary']['total_services'] > 0:
            total_health_score = sum(
                details['health_score'] 
                for details in dashboard['services'].values()
            )
            dashboard['system_summary']['overall_health_score'] = (
                total_health_score / dashboard['system_summary']['total_services']
            )
        else:
            dashboard['system_summary']['overall_health_score'] = 1.0
        
        return dashboard
    
    def _start_monitoring(self):
        """Start background monitoring thread."""
        def monitor():
            while self.monitoring_enabled:
                try:
                    # Run periodic health checks
                    for service, callback in self.health_check_callbacks.items():
                        try:
                            callback()
                        except Exception as e:
                            logger.error(f"Health check error for {service}: {e}")
                    
                    # Check for cascading failures
                    cascading_risks = self.check_cascading_failures()
                    if cascading_risks:
                        logger.warning(f"Detected {len(cascading_risks)} cascading failure risks")
                    
                    # Log system summary
                    dashboard = self.get_system_health_dashboard()
                    summary = dashboard['system_summary']
                    
                    if summary['critical_services'] > 0 or summary['unhealthy_services'] > 0:
                        logger.warning(
                            f"System health alert: {summary['critical_services']} critical, "
                            f"{summary['unhealthy_services']} unhealthy, "
                            f"overall score: {summary['overall_health_score']:.2f}"
                        )
                    
                except Exception as e:
                    logger.error(f"Monitoring error: {e}")
                
                time.sleep(self.monitoring_interval)
        
        self.monitoring_thread = threading.Thread(target=monitor, daemon=True)
        self.monitoring_thread.start()
        logger.info("Started circuit breaker monitoring")
    
    def shutdown(self):
        """Shutdown the monitoring system."""
        self.monitoring_enabled = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        logger.info("Shutdown circuit breaker monitoring")


# Global manager instance
_global_manager = CascadingCircuitBreakerManager()


def get_circuit_breaker(name: str, config: Optional[CircuitBreakerConfig] = None) -> AdaptiveCircuitBreaker:
    """Get or create an adaptive circuit breaker."""
    if name not in _global_manager.circuit_breakers:
        return _global_manager.register_circuit_breaker(name, config)
    return _global_manager.circuit_breakers[name]


def get_bulkhead(name: str, config: Optional[BulkheadConfig] = None) -> BulkheadProtection:
    """Get or create a bulkhead for resource isolation."""
    if name not in _global_manager.bulkheads:
        if config is None:
            config = BulkheadConfig()
        return _global_manager.register_bulkhead(name, config)
    return _global_manager.bulkheads[name]


def add_service_dependency(service: str, depends_on: str):
    """Add service dependency for cascading protection."""
    _global_manager.add_dependency(service, depends_on)


def register_health_check(service: str, callback: Callable[[], bool]):
    """Register a health check callback."""
    _global_manager.register_health_check(service, callback)


def get_system_health_dashboard() -> Dict[str, Any]:
    """Get comprehensive system health dashboard."""
    return _global_manager.get_system_health_dashboard()


@contextmanager
def protected_call(circuit_breaker_name: str, bulkhead_name: Optional[str] = None,
                   timeout: Optional[float] = None, failure_type: FailureType = FailureType.UNKNOWN):
    """Context manager for protected calls with circuit breaker and optional bulkhead."""
    cb = get_circuit_breaker(circuit_breaker_name)
    
    # Check circuit breaker
    if not cb.can_proceed():
        raise RuntimeError(f"Circuit breaker {circuit_breaker_name} is open")
    
    # Acquire bulkhead slot if specified
    bulkhead_context = None
    if bulkhead_name:
        bulkhead = get_bulkhead(bulkhead_name)
        bulkhead_context = bulkhead.acquire_slot(timeout)
    
    start_time = time.time()
    
    try:
        if bulkhead_context:
            with bulkhead_context:
                yield
        else:
            yield
        
        # Record success
        response_time = time.time() - start_time
        cb._record_success(response_time)
        
    except Exception as e:
        # Record failure
        cb._record_failure(failure_type)
        
        # Determine failure type from exception
        if "timeout" in str(e).lower():
            failure_type = FailureType.TIMEOUT
        elif "connection" in str(e).lower():
            failure_type = FailureType.CONNECTION_ERROR
        elif "auth" in str(e).lower():
            failure_type = FailureType.AUTHENTICATION_ERROR
        
        raise


# Example usage functions
def create_database_circuit_breaker() -> AdaptiveCircuitBreaker:
    """Create circuit breaker for database connections."""
    config = CircuitBreakerConfig(
        failure_threshold=3,
        success_threshold=2,
        timeout_seconds=30.0
    )
    return get_circuit_breaker("database", config)


def create_api_bulkhead() -> BulkheadProtection:
    """Create bulkhead for API calls."""
    config = BulkheadConfig(
        max_concurrent_calls=20,
        max_wait_time=5.0,
        timeout_per_call=30.0
    )
    return get_bulkhead("api_calls", config)