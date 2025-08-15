"""
Robust error handling and validation for Slack KB Agent.
Implements comprehensive exception handling with recovery mechanisms.
"""

import logging
import time
import traceback
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels for proper escalation."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SlackKBError(Exception):
    """Base exception class for Slack KB Agent."""

    def __init__(self, message: str, severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                 context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.severity = severity
        self.context = context or {}
        self.timestamp = time.time()


class ValidationError(SlackKBError):
    """Raised when input validation fails."""
    pass


class SearchError(SlackKBError):
    """Raised when search operations fail."""
    pass


class DocumentError(SlackKBError):
    """Raised when document operations fail."""
    pass


class DatabaseError(SlackKBError):
    """Raised when database operations fail."""
    pass


class NetworkError(SlackKBError):
    """Raised when network operations fail."""
    pass


class AuthenticationError(SlackKBError):
    """Raised when authentication fails."""
    pass


def retry_with_backoff(max_retries: int = 3, backoff_factor: float = 2.0,
                      exceptions: tuple = (Exception,)):
    """
    Decorator to retry operations with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        backoff_factor: Exponential backoff multiplier
        exceptions: Tuple of exceptions to catch and retry
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e

                    if attempt == max_retries:
                        logger.error(f"Function {func.__name__} failed after {max_retries} retries: {e}")
                        raise

                    wait_time = backoff_factor ** attempt
                    logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}, retrying in {wait_time}s: {e}")
                    time.sleep(wait_time)

            # Should never reach here, but for type safety
            raise last_exception
        return wrapper
    return decorator


def handle_exceptions(default_return=None, log_error: bool = True,
                     reraise: bool = False):
    """
    Decorator to handle exceptions gracefully with optional recovery.
    
    Args:
        default_return: Value to return on exception
        log_error: Whether to log the exception
        reraise: Whether to reraise the exception after handling
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_error:
                    logger.error(f"Exception in {func.__name__}: {e}")
                    logger.debug(f"Full traceback: {traceback.format_exc()}")

                if reraise:
                    raise

                return default_return
        return wrapper
    return decorator


class ErrorAggregator:
    """Aggregates and analyzes error patterns for system health monitoring."""

    def __init__(self, max_errors: int = 1000):
        self.errors: List[Dict[str, Any]] = []
        self.max_errors = max_errors
        self.error_counts: Dict[str, int] = {}

    def record_error(self, error: Exception, context: Optional[Dict[str, Any]] = None):
        """Record an error for analysis."""
        error_info = {
            'type': type(error).__name__,
            'message': str(error),
            'timestamp': time.time(),
            'context': context or {},
            'traceback': traceback.format_exc()
        }

        self.errors.append(error_info)

        # Maintain size limit
        if len(self.errors) > self.max_errors:
            self.errors.pop(0)

        # Update counts
        error_type = type(error).__name__
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1

    def get_error_patterns(self, time_window: int = 3600) -> Dict[str, Any]:
        """Analyze error patterns within a time window."""
        current_time = time.time()
        recent_errors = [
            e for e in self.errors
            if current_time - e['timestamp'] <= time_window
        ]

        return {
            'total_errors': len(recent_errors),
            'unique_errors': len(set(e['type'] for e in recent_errors)),
            'most_common': max(self.error_counts.items(), key=lambda x: x[1]) if self.error_counts else None,
            'error_rate': len(recent_errors) / (time_window / 60),  # errors per minute
        }


class SafeExecutor:
    """Safely execute operations with comprehensive error handling."""

    def __init__(self, error_aggregator: Optional[ErrorAggregator] = None):
        self.error_aggregator = error_aggregator or ErrorAggregator()

    def safe_execute(self, operation: Callable, *args,
                    default_return=None, context: Optional[Dict[str, Any]] = None,
                    **kwargs) -> Any:
        """
        Execute an operation safely with error handling.
        
        Args:
            operation: Function to execute
            *args: Positional arguments for the operation
            default_return: Default value to return on error
            context: Additional context for error reporting
            **kwargs: Keyword arguments for the operation
        """
        try:
            return operation(*args, **kwargs)
        except Exception as e:
            self.error_aggregator.record_error(e, context)
            logger.error(f"Safe execution failed for {operation.__name__}: {e}")
            return default_return

    @retry_with_backoff(max_retries=3)
    def safe_execute_with_retry(self, operation: Callable, *args, **kwargs) -> Any:
        """Execute operation with automatic retry on failure."""
        return operation(*args, **kwargs)


def validate_input(validator: Callable, error_message: str = "Validation failed"):
    """
    Decorator to validate function inputs.
    
    Args:
        validator: Function that takes the same arguments and returns True if valid
        error_message: Error message to raise on validation failure
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not validator(*args, **kwargs):
                raise ValidationError(error_message, context={
                    'function': func.__name__,
                    'args': str(args)[:100],  # Truncate for safety
                    'kwargs': str(kwargs)[:100]
                })
            return func(*args, **kwargs)
        return wrapper
    return decorator


def sanitize_input(input_value: Any, max_length: int = 10000,
                  allowed_types: tuple = (str, int, float, bool, list, dict)) -> Any:
    """
    Sanitize input to prevent injection attacks and ensure data integrity.
    
    Args:
        input_value: Value to sanitize
        max_length: Maximum string length
        allowed_types: Tuple of allowed data types
    """
    if input_value is None:
        return None

    # Type validation
    if not isinstance(input_value, allowed_types):
        raise ValidationError(f"Invalid input type: {type(input_value).__name__}")

    # String sanitization
    if isinstance(input_value, str):
        if len(input_value) > max_length:
            raise ValidationError(f"Input exceeds maximum length of {max_length}")

        # Remove potentially dangerous characters
        dangerous_chars = ['<', '>', '"', "'", '&', ';', '(', ')', '{', '}']
        for char in dangerous_chars:
            input_value = input_value.replace(char, '')

    # List/dict recursion
    elif isinstance(input_value, list):
        return [sanitize_input(item, max_length, allowed_types) for item in input_value]
    elif isinstance(input_value, dict):
        return {
            sanitize_input(k, max_length, allowed_types): sanitize_input(v, max_length, allowed_types)
            for k, v in input_value.items()
        }

    return input_value


# Global error aggregator instance
global_error_aggregator = ErrorAggregator()
safe_executor = SafeExecutor(global_error_aggregator)


def get_system_health() -> Dict[str, Any]:
    """Get overall system health based on error patterns."""
    error_patterns = global_error_aggregator.get_error_patterns()

    # Determine health status
    if error_patterns['total_errors'] == 0:
        health_status = "excellent"
    elif error_patterns['error_rate'] < 1:  # Less than 1 error per minute
        health_status = "good"
    elif error_patterns['error_rate'] < 5:
        health_status = "fair"
    else:
        health_status = "poor"

    return {
        'health_status': health_status,
        'error_patterns': error_patterns,
        'recommendations': _get_health_recommendations(error_patterns)
    }


def _get_health_recommendations(error_patterns: Dict[str, Any]) -> List[str]:
    """Generate health recommendations based on error patterns."""
    recommendations = []

    if error_patterns['error_rate'] > 5:
        recommendations.append("High error rate detected - investigate system stability")

    if error_patterns['unique_errors'] > 10:
        recommendations.append("Many different error types - review error handling coverage")

    if error_patterns['total_errors'] > 100:
        recommendations.append("High error volume - consider system optimization")

    return recommendations
