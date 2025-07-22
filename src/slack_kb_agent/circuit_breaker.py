"""
Circuit breaker implementation for external service protection.

This module provides a robust circuit breaker pattern to prevent cascading failures
when external services become unavailable. It supports three states:
- CLOSED: Normal operation, all requests pass through
- OPEN: Service is failing, requests are rejected immediately  
- HALF_OPEN: Testing if service has recovered

Key features:
- Thread-safe operation using locks
- Configurable failure/success thresholds
- Automatic recovery testing with timeout
- Integration with existing monitoring and logging
- Graceful degradation with informative error messages
"""

import time
import logging
import threading
from typing import Callable, Any, Optional, Dict, TYPE_CHECKING
from dataclasses import dataclass, field
from enum import Enum
from contextlib import contextmanager

if TYPE_CHECKING:
    from .constants import NetworkDefaults

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states following the standard pattern."""
    CLOSED = "closed"      # Normal operation - requests pass through
    OPEN = "open"          # Failing - requests are rejected immediately
    HALF_OPEN = "half_open"  # Testing recovery - limited requests allowed


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""
    
    # Failure threshold before opening the circuit
    failure_threshold: int = 5
    
    # Success threshold to close circuit from half-open state
    success_threshold: int = 2
    
    # Time to wait before attempting recovery (seconds)
    timeout_seconds: float = 60.0
    
    # Maximum number of requests to allow in half-open state
    half_open_max_requests: int = 3
    
    # Window size for failure counting (seconds) - None means no window
    failure_window_seconds: Optional[float] = None
    
    # Service name for logging and monitoring
    service_name: str = "unknown_service"


class CircuitBreakerError(Exception):
    """Base exception for circuit breaker errors."""
    pass


class CircuitOpenError(CircuitBreakerError):
    """Raised when circuit breaker is open and blocking requests."""
    
    def __init__(self, service_name: str, state: CircuitState, retry_after: Optional[float] = None):
        self.service_name = service_name
        self.state = state
        self.retry_after = retry_after
        
        if retry_after:
            message = f"Circuit breaker for {service_name} is {state.value}. Retry after {retry_after:.1f} seconds."
        else:
            message = f"Circuit breaker for {service_name} is {state.value}."
        
        super().__init__(message)


class CircuitBreaker:
    """
    Circuit breaker implementation for protecting external service calls.
    
    Provides automatic failure detection and recovery testing to prevent
    cascading failures in distributed systems.
    """
    
    def __init__(self, config: CircuitBreakerConfig):
        """Initialize circuit breaker with given configuration."""
        self.config = config
        self.state = CircuitState.CLOSED
        
        # Failure tracking
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None
        self.failure_timestamps: list = []
        
        # Half-open state tracking
        self.half_open_requests = 0
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Metrics for monitoring
        self.total_requests = 0
        self.total_failures = 0
        self.total_successes = 0
        self.circuit_opened_count = 0
        self.last_state_change_time = time.time()
        
        logger.info(f"Circuit breaker initialized for {self.config.service_name}")
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection.
        
        Args:
            func: Function to execute
            *args: Positional arguments for function
            **kwargs: Keyword arguments for function
            
        Returns:
            Result of function call
            
        Raises:
            CircuitOpenError: When circuit is open
            Any exception: From the protected function call
        """
        with self._lock:
            self.total_requests += 1
            
            # Check if circuit should be opened based on current state
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self._transition_to_half_open()
                else:
                    retry_after = self._calculate_retry_after()
                    raise CircuitOpenError(
                        self.config.service_name, 
                        self.state, 
                        retry_after
                    )
            
            # Limit requests in half-open state
            if self.state == CircuitState.HALF_OPEN:
                if self.half_open_requests >= self.config.half_open_max_requests:
                    retry_after = self._calculate_retry_after()
                    raise CircuitOpenError(
                        self.config.service_name, 
                        self.state,
                        retry_after
                    )
                self.half_open_requests += 1
        
        # Execute the protected function
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            self._on_success(execution_time)
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self._on_failure(e, execution_time)
            raise
    
    @contextmanager
    def protect(self):
        """
        Context manager for circuit breaker protection.
        
        Example:
            with circuit_breaker.protect():
                # Protected code here
                result = external_service_call()
        """
        self.call(lambda: None)  # Check circuit state
        try:
            yield
            self._on_success(0)  # Mark as success
        except Exception as e:
            self._on_failure(e, 0)
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt circuit reset."""
        if not self.last_failure_time:
            return True
        
        time_since_failure = time.time() - self.last_failure_time
        return time_since_failure >= self.config.timeout_seconds
    
    def _calculate_retry_after(self) -> Optional[float]:
        """Calculate seconds until next retry attempt."""
        if not self.last_failure_time:
            return None
        
        time_since_failure = time.time() - self.last_failure_time
        retry_after = self.config.timeout_seconds - time_since_failure
        return max(0, retry_after)
    
    def _transition_to_half_open(self) -> None:
        """Transition circuit from OPEN to HALF_OPEN state."""
        self.state = CircuitState.HALF_OPEN
        self.half_open_requests = 0
        self.success_count = 0
        self.last_state_change_time = time.time()
        
        logger.info(f"Circuit breaker for {self.config.service_name} transitioned to HALF_OPEN")
    
    def _on_success(self, execution_time: float) -> None:
        """Handle successful function execution."""
        with self._lock:
            self.total_successes += 1
            
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                
                if self.success_count >= self.config.success_threshold:
                    # Recovery successful - close circuit
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
                    self.success_count = 0
                    self.half_open_requests = 0
                    self.last_state_change_time = time.time()
                    
                    logger.info(f"Circuit breaker for {self.config.service_name} is now CLOSED (recovery successful)")
            
            elif self.state == CircuitState.CLOSED:
                # Reset failure count on success in closed state
                self.failure_count = 0
                self._clean_failure_window()
    
    def _on_failure(self, exception: Exception, execution_time: float) -> None:
        """Handle failed function execution."""
        with self._lock:
            self.total_failures += 1
            current_time = time.time()
            
            # Track failure
            self.failure_count += 1
            self.last_failure_time = current_time
            
            if self.config.failure_window_seconds:
                self.failure_timestamps.append(current_time)
                self._clean_failure_window()
                
                # Use windowed failure count if configured
                failure_count_in_window = len(self.failure_timestamps)
            else:
                failure_count_in_window = self.failure_count
            
            # Log failure with context
            logger.warning(
                f"Circuit breaker for {self.config.service_name} recorded failure "
                f"({failure_count_in_window}/{self.config.failure_threshold}): {exception}"
            )
            
            # Check if circuit should open
            if self.state == CircuitState.HALF_OPEN:
                # Any failure in half-open state reopens circuit
                self.state = CircuitState.OPEN
                self.circuit_opened_count += 1
                self.last_state_change_time = current_time
                
                logger.error(f"Circuit breaker for {self.config.service_name} is now OPEN (half-open test failed)")
                
            elif failure_count_in_window >= self.config.failure_threshold:
                # Threshold exceeded - open circuit
                self.state = CircuitState.OPEN
                self.circuit_opened_count += 1
                self.last_state_change_time = current_time
                
                logger.error(f"Circuit breaker for {self.config.service_name} is now OPEN (failure threshold exceeded)")
    
    def _clean_failure_window(self) -> None:
        """Remove old failures outside the window."""
        if not self.config.failure_window_seconds:
            return
        
        current_time = time.time()
        cutoff_time = current_time - self.config.failure_window_seconds
        
        self.failure_timestamps = [
            timestamp for timestamp in self.failure_timestamps 
            if timestamp > cutoff_time
        ]
    
    def get_state(self) -> CircuitState:
        """Get current circuit breaker state."""
        with self._lock:
            return self.state
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get circuit breaker metrics for monitoring."""
        with self._lock:
            return {
                "service_name": self.config.service_name,
                "state": self.state.value,
                "total_requests": self.total_requests,
                "total_successes": self.total_successes,
                "total_failures": self.total_failures,
                "failure_rate": self.total_failures / max(1, self.total_requests),
                "current_failure_count": self.failure_count,
                "failure_threshold": self.config.failure_threshold,
                "circuit_opened_count": self.circuit_opened_count,
                "last_failure_time": self.last_failure_time,
                "last_state_change_time": self.last_state_change_time,
                "time_since_last_failure": time.time() - (self.last_failure_time or 0),
                "half_open_requests": self.half_open_requests if self.state == CircuitState.HALF_OPEN else 0,
            }
    
    def reset(self) -> None:
        """Manually reset circuit breaker to closed state."""
        with self._lock:
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.success_count = 0
            self.half_open_requests = 0
            self.failure_timestamps.clear()
            self.last_state_change_time = time.time()
            
            logger.info(f"Circuit breaker for {self.config.service_name} manually reset to CLOSED")
    
    def force_open(self) -> None:
        """Manually force circuit breaker to open state (for maintenance)."""
        with self._lock:
            self.state = CircuitState.OPEN
            self.last_failure_time = time.time()
            self.circuit_opened_count += 1
            self.last_state_change_time = time.time()
            
            logger.warning(f"Circuit breaker for {self.config.service_name} manually forced to OPEN")


class CircuitBreakerRegistry:
    """Registry for managing multiple circuit breakers."""
    
    def __init__(self):
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._lock = threading.RLock()
    
    def register(self, name: str, config: CircuitBreakerConfig) -> CircuitBreaker:
        """Register a new circuit breaker."""
        with self._lock:
            if name in self._breakers:
                logger.warning(f"Circuit breaker {name} already registered, returning existing instance")
                return self._breakers[name]
            
            config.service_name = name
            breaker = CircuitBreaker(config)
            self._breakers[name] = breaker
            
            logger.info(f"Registered circuit breaker: {name}")
            return breaker
    
    def get(self, name: str) -> Optional[CircuitBreaker]:
        """Get circuit breaker by name."""
        with self._lock:
            return self._breakers.get(name)
    
    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics for all registered circuit breakers."""
        with self._lock:
            return {
                name: breaker.get_metrics() 
                for name, breaker in self._breakers.items()
            }
    
    def reset_all(self) -> None:
        """Reset all circuit breakers to closed state."""
        with self._lock:
            for breaker in self._breakers.values():
                breaker.reset()


# Global registry instance
circuit_breaker_registry = CircuitBreakerRegistry()