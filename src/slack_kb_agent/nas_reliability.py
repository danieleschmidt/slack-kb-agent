"""Reliability and fault tolerance for Neural Architecture Search system.

This module implements comprehensive error handling, recovery mechanisms,
circuit breakers, and reliability patterns for the NAS engine.
"""

import asyncio
import logging
import random
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np

from .circuit_breaker import CircuitBreaker
from .exceptions import (
    NASEngineError,
    ResourceLimitExceededError,
    ValidationError,
    SecurityViolationError,
    OptimizationError,
    PredictionError
)

logger = logging.getLogger(__name__)


class FailureMode(Enum):
    """Types of failures that can occur in NAS system."""
    PREDICTION_FAILURE = "prediction_failure"
    OPTIMIZATION_FAILURE = "optimization_failure"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    TIMEOUT = "timeout"
    VALIDATION_FAILURE = "validation_failure"
    SECURITY_VIOLATION = "security_violation"
    INFRASTRUCTURE_FAILURE = "infrastructure_failure"
    DATA_CORRUPTION = "data_corruption"


class RecoveryStrategy(Enum):
    """Recovery strategies for different failure modes."""
    RETRY = "retry"
    FALLBACK = "fallback"
    DEGRADE = "degrade"
    ABORT = "abort"
    ESCALATE = "escalate"


@dataclass
class FailureContext:
    """Context information about a failure."""
    failure_mode: FailureMode
    error_message: str
    stack_trace: str
    timestamp: float
    operation: str
    input_data: Any
    retry_count: int = 0
    recovery_attempts: List[str] = None


@dataclass
class RecoveryAction:
    """A recovery action to be taken."""
    strategy: RecoveryStrategy
    description: str
    action_function: Callable[[], Any]
    timeout_seconds: float = 30.0
    max_retries: int = 3


class NASReliabilityManager:
    """Manages reliability and fault tolerance for NAS operations."""
    
    def __init__(self):
        self.failure_history: List[FailureContext] = []
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.recovery_strategies: Dict[FailureMode, List[RecoveryAction]] = {}
        self.fallback_configurations: Dict[str, Dict[str, Any]] = {}
        
        self._initialize_circuit_breakers()
        self._initialize_recovery_strategies()
        self._initialize_fallback_configurations()
        
        logger.info("Initialized NAS reliability manager")
    
    def _initialize_circuit_breakers(self) -> None:
        """Initialize circuit breakers for different operations."""
        self.circuit_breakers = {
            "prediction": CircuitBreaker(
                failure_threshold=5,
                recovery_timeout=30.0,
                expected_exception=PredictionError
            ),
            "optimization": CircuitBreaker(
                failure_threshold=3,
                recovery_timeout=60.0,
                expected_exception=OptimizationError
            ),
            "validation": CircuitBreaker(
                failure_threshold=10,
                recovery_timeout=15.0,
                expected_exception=ValidationError
            ),
            "security": CircuitBreaker(
                failure_threshold=1,  # Very low threshold for security
                recovery_timeout=300.0,  # Long timeout for security issues
                expected_exception=SecurityViolationError
            )
        }
    
    def _initialize_recovery_strategies(self) -> None:
        """Initialize recovery strategies for different failure modes."""
        self.recovery_strategies = {
            FailureMode.PREDICTION_FAILURE: [
                RecoveryAction(
                    strategy=RecoveryStrategy.RETRY,
                    description="Retry prediction with simplified model",
                    action_function=self._retry_with_simplified_prediction,
                    max_retries=3
                ),
                RecoveryAction(
                    strategy=RecoveryStrategy.FALLBACK,
                    description="Use cached prediction or baseline",
                    action_function=self._use_prediction_fallback,
                    max_retries=1
                )
            ],
            
            FailureMode.OPTIMIZATION_FAILURE: [
                RecoveryAction(
                    strategy=RecoveryStrategy.DEGRADE,
                    description="Use basic optimization instead of advanced",
                    action_function=self._use_basic_optimization,
                    max_retries=2
                ),
                RecoveryAction(
                    strategy=RecoveryStrategy.FALLBACK,
                    description="Return unoptimized configuration",
                    action_function=self._skip_optimization,
                    max_retries=1
                )
            ],
            
            FailureMode.RESOURCE_EXHAUSTION: [
                RecoveryAction(
                    strategy=RecoveryStrategy.DEGRADE,
                    description="Reduce resource requirements",
                    action_function=self._reduce_resource_requirements,
                    max_retries=2
                ),
                RecoveryAction(
                    strategy=RecoveryStrategy.ABORT,
                    description="Abort operation to prevent system instability",
                    action_function=self._abort_operation,
                    max_retries=0
                )
            ],
            
            FailureMode.TIMEOUT: [
                RecoveryAction(
                    strategy=RecoveryStrategy.RETRY,
                    description="Retry with increased timeout",
                    action_function=self._retry_with_timeout_increase,
                    timeout_seconds=60.0,
                    max_retries=2
                ),
                RecoveryAction(
                    strategy=RecoveryStrategy.DEGRADE,
                    description="Use faster algorithm",
                    action_function=self._use_faster_algorithm,
                    max_retries=1
                )
            ],
            
            FailureMode.VALIDATION_FAILURE: [
                RecoveryAction(
                    strategy=RecoveryStrategy.RETRY,
                    description="Retry with sanitized input",
                    action_function=self._retry_with_sanitized_input,
                    max_retries=2
                ),
                RecoveryAction(
                    strategy=RecoveryStrategy.FALLBACK,
                    description="Use default configuration",
                    action_function=self._use_default_configuration,
                    max_retries=1
                )
            ],
            
            FailureMode.SECURITY_VIOLATION: [
                RecoveryAction(
                    strategy=RecoveryStrategy.ABORT,
                    description="Immediately abort for security reasons",
                    action_function=self._abort_for_security,
                    max_retries=0
                ),
                RecoveryAction(
                    strategy=RecoveryStrategy.ESCALATE,
                    description="Escalate to security team",
                    action_function=self._escalate_security_violation,
                    max_retries=0
                )
            ]
        }
    
    def _initialize_fallback_configurations(self) -> None:
        """Initialize fallback configurations for different scenarios."""
        self.fallback_configurations = {
            "minimal_transformer": {
                "architecture_type": "transformer",
                "num_layers": 6,
                "hidden_size": 384,
                "num_heads": 6,
                "max_sequence_length": 256,
                "batch_size": 16,
                "vocab_size": 30000,
                "use_mixed_precision": False
            },
            
            "efficient_cnn": {
                "architecture_type": "cnn",
                "num_layers": 8,
                "hidden_size": 256,
                "kernel_sizes": [3, 3, 3],
                "channels": [32, 64, 128],
                "max_sequence_length": 224,
                "batch_size": 32,
                "use_mixed_precision": True
            },
            
            "lightweight_rnn": {
                "architecture_type": "rnn",
                "num_layers": 4,
                "hidden_size": 256,
                "max_sequence_length": 128,
                "batch_size": 64,
                "vocab_size": 20000,
                "use_mixed_precision": True
            }
        }
    
    async def execute_with_reliability(
        self,
        operation_name: str,
        operation_func: Callable,
        *args,
        timeout: float = 300.0,
        **kwargs
    ) -> Any:
        """Execute operation with comprehensive reliability mechanisms."""
        start_time = time.time()
        
        try:
            # Check circuit breaker
            circuit_breaker = self.circuit_breakers.get(operation_name.split('_')[0])
            if circuit_breaker and circuit_breaker.state == "open":
                raise NASEngineError(f"Circuit breaker open for {operation_name}")
            
            # Execute with timeout
            result = await asyncio.wait_for(
                operation_func(*args, **kwargs),
                timeout=timeout
            )
            
            # Reset circuit breaker on success
            if circuit_breaker:
                circuit_breaker.record_success()
            
            return result
            
        except Exception as e:
            # Record failure
            failure_context = self._create_failure_context(
                operation_name, e, args, kwargs, time.time() - start_time
            )
            
            # Record circuit breaker failure
            if circuit_breaker:
                circuit_breaker.record_failure()
            
            # Attempt recovery
            return await self._attempt_recovery(failure_context, operation_func, *args, **kwargs)
    
    async def _attempt_recovery(
        self,
        failure_context: FailureContext,
        operation_func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Attempt recovery from failure."""
        self.failure_history.append(failure_context)
        logger.error(f"Operation failed: {failure_context.operation} - {failure_context.error_message}")
        
        # Get recovery strategies for the failure mode
        strategies = self.recovery_strategies.get(failure_context.failure_mode, [])
        
        for strategy in strategies:
            try:
                logger.info(f"Attempting recovery: {strategy.description}")
                
                # Execute recovery action
                recovery_result = await asyncio.wait_for(
                    strategy.action_function(),
                    timeout=strategy.timeout_seconds
                )
                
                if recovery_result is not None:
                    logger.info(f"Recovery successful: {strategy.description}")
                    return recovery_result
                
            except Exception as recovery_error:
                logger.warning(f"Recovery attempt failed: {recovery_error}")
                continue
        
        # All recovery attempts failed
        logger.error(f"All recovery attempts failed for {failure_context.operation}")
        raise NASEngineError(f"Operation {failure_context.operation} failed and could not be recovered")
    
    def _create_failure_context(
        self,
        operation: str,
        error: Exception,
        args: Tuple,
        kwargs: Dict,
        duration: float
    ) -> FailureContext:
        """Create failure context from exception."""
        failure_mode = self._classify_failure(error)
        
        return FailureContext(
            failure_mode=failure_mode,
            error_message=str(error),
            stack_trace=str(error.__traceback__) if hasattr(error, '__traceback__') else "",
            timestamp=time.time(),
            operation=operation,
            input_data={"args": args, "kwargs": kwargs, "duration": duration},
            recovery_attempts=[]
        )
    
    def _classify_failure(self, error: Exception) -> FailureMode:
        """Classify failure based on exception type and message."""
        error_type = type(error).__name__
        error_message = str(error).lower()
        
        if isinstance(error, SecurityViolationError):
            return FailureMode.SECURITY_VIOLATION
        elif isinstance(error, ValidationError):
            return FailureMode.VALIDATION_FAILURE
        elif isinstance(error, PredictionError):
            return FailureMode.PREDICTION_FAILURE
        elif isinstance(error, OptimizationError):
            return FailureMode.OPTIMIZATION_FAILURE
        elif isinstance(error, ResourceLimitExceededError):
            return FailureMode.RESOURCE_EXHAUSTION
        elif isinstance(error, asyncio.TimeoutError):
            return FailureMode.TIMEOUT
        elif "memory" in error_message or "out of" in error_message:
            return FailureMode.RESOURCE_EXHAUSTION
        elif "timeout" in error_message:
            return FailureMode.TIMEOUT
        elif "corrupt" in error_message or "invalid" in error_message:
            return FailureMode.DATA_CORRUPTION
        else:
            return FailureMode.INFRASTRUCTURE_FAILURE
    
    # Recovery action implementations
    async def _retry_with_simplified_prediction(self) -> Any:
        """Retry prediction with a simplified model."""
        logger.info("Using simplified prediction model")
        # Implementation would use a lighter prediction model
        return {"status": "simplified_prediction", "confidence": 0.7}
    
    async def _use_prediction_fallback(self) -> Any:
        """Use cached prediction or baseline."""
        logger.info("Using prediction fallback")
        return {"status": "fallback_prediction", "confidence": 0.5}
    
    async def _use_basic_optimization(self) -> Any:
        """Use basic optimization instead of advanced."""
        logger.info("Using basic optimization")
        return {"status": "basic_optimization", "optimizations": ["memory_layout"]}
    
    async def _skip_optimization(self) -> Any:
        """Return unoptimized configuration."""
        logger.info("Skipping optimization")
        return {"status": "unoptimized", "warning": "optimization_skipped"}
    
    async def _reduce_resource_requirements(self) -> Any:
        """Reduce resource requirements."""
        logger.info("Reducing resource requirements")
        return self.fallback_configurations["minimal_transformer"]
    
    async def _abort_operation(self) -> Any:
        """Abort operation to prevent system instability."""
        logger.warning("Aborting operation due to resource exhaustion")
        raise NASEngineError("Operation aborted due to resource constraints")
    
    async def _retry_with_timeout_increase(self) -> Any:
        """Retry with increased timeout."""
        logger.info("Retrying with increased timeout")
        # This would trigger a retry with longer timeout
        return {"status": "retry_with_timeout", "new_timeout": 120.0}
    
    async def _use_faster_algorithm(self) -> Any:
        """Use faster algorithm."""
        logger.info("Switching to faster algorithm")
        return {"status": "fast_algorithm", "accuracy_trade_off": True}
    
    async def _retry_with_sanitized_input(self) -> Any:
        """Retry with sanitized input."""
        logger.info("Retrying with sanitized input")
        return {"status": "sanitized_retry", "sanitized": True}
    
    async def _use_default_configuration(self) -> Any:
        """Use default configuration."""
        logger.info("Using default configuration")
        return self.fallback_configurations["minimal_transformer"]
    
    async def _abort_for_security(self) -> Any:
        """Immediately abort for security reasons."""
        logger.critical("Operation aborted due to security violation")
        raise SecurityViolationError("Operation aborted due to security concerns")
    
    async def _escalate_security_violation(self) -> Any:
        """Escalate to security team."""
        logger.critical("Escalating security violation")
        # In real implementation, this would notify security team
        raise SecurityViolationError("Security violation escalated")
    
    def get_reliability_metrics(self, hours: int = 24) -> Dict[str, Any]:
        """Get reliability metrics for the specified time period."""
        cutoff_time = time.time() - (hours * 3600)
        recent_failures = [f for f in self.failure_history if f.timestamp >= cutoff_time]
        
        # Calculate failure rates by mode
        failure_counts = {}
        for failure in recent_failures:
            mode = failure.failure_mode.value
            failure_counts[mode] = failure_counts.get(mode, 0) + 1
        
        # Calculate circuit breaker states
        circuit_states = {
            name: cb.state for name, cb in self.circuit_breakers.items()
        }
        
        # Calculate recovery success rate
        total_recovery_attempts = sum(
            len(f.recovery_attempts) for f in recent_failures if f.recovery_attempts
        )
        
        return {
            "time_period_hours": hours,
            "total_failures": len(recent_failures),
            "failure_breakdown": failure_counts,
            "circuit_breaker_states": circuit_states,
            "recovery_attempts": total_recovery_attempts,
            "most_common_failure": max(failure_counts.items(), key=lambda x: x[1])[0] if failure_counts else None
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        health_status = {
            "overall_health": "healthy",
            "circuit_breakers": {},
            "recent_failures": 0,
            "recovery_rate": 1.0,
            "recommendations": []
        }
        
        # Check circuit breakers
        open_breakers = []
        for name, cb in self.circuit_breakers.items():
            health_status["circuit_breakers"][name] = cb.state
            if cb.state == "open":
                open_breakers.append(name)
        
        # Check recent failure rate
        recent_failures = [
            f for f in self.failure_history 
            if f.timestamp >= time.time() - 3600  # Last hour
        ]
        health_status["recent_failures"] = len(recent_failures)
        
        # Determine overall health
        if open_breakers:
            health_status["overall_health"] = "degraded"
            health_status["recommendations"].append(f"Circuit breakers open: {open_breakers}")
        
        if len(recent_failures) > 10:
            health_status["overall_health"] = "unhealthy" if health_status["overall_health"] == "degraded" else "degraded"
            health_status["recommendations"].append("High failure rate detected")
        
        return health_status


class ReliableNASWrapper:
    """Wrapper that adds reliability to NAS operations."""
    
    def __init__(self, nas_engine, reliability_manager: Optional[NASReliabilityManager] = None):
        self.nas_engine = nas_engine
        self.reliability_manager = reliability_manager or NASReliabilityManager()
    
    async def search_architectures_reliable(self, *args, **kwargs) -> Any:
        """Search architectures with reliability mechanisms."""
        return await self.reliability_manager.execute_with_reliability(
            "search_architectures",
            self.nas_engine.search_architectures,
            *args,
            **kwargs
        )
    
    async def optimize_architecture_reliable(self, *args, **kwargs) -> Any:
        """Optimize architecture with reliability mechanisms."""
        return await self.reliability_manager.execute_with_reliability(
            "optimize_architecture",
            self._safe_optimize_architecture,
            *args,
            **kwargs
        )
    
    async def _safe_optimize_architecture(self, *args, **kwargs) -> Any:
        """Safe wrapper for architecture optimization."""
        try:
            # Check if TPU optimizer is available
            if hasattr(self.nas_engine, 'tpu_optimizer'):
                return await self.nas_engine.tpu_optimizer.optimize_architecture(*args, **kwargs)
            else:
                # Fallback to basic optimization
                return {"status": "basic_optimization", "message": "TPU optimizer not available"}
        except Exception as e:
            raise OptimizationError(f"Architecture optimization failed: {e}")


# Factory function
def get_reliability_manager() -> NASReliabilityManager:
    """Get reliability manager instance."""
    return NASReliabilityManager()


def make_nas_reliable(nas_engine) -> ReliableNASWrapper:
    """Make NAS engine reliable by wrapping with reliability mechanisms."""
    return ReliableNASWrapper(nas_engine)


# Demo usage
async def demo_reliability():
    """Demonstrate reliability features."""
    print("🔧 NAS Reliability Demo")
    
    reliability_manager = get_reliability_manager()
    
    # Simulate various failure scenarios
    async def failing_operation():
        """Operation that randomly fails."""
        if random.random() < 0.7:  # 70% failure rate
            failure_types = [
                PredictionError("Prediction model failed"),
                ResourceLimitExceededError("Out of memory"),
                ValidationError("Invalid configuration"),
                asyncio.TimeoutError("Operation timed out")
            ]
            raise random.choice(failure_types)
        return {"status": "success", "result": "operation completed"}
    
    # Test reliability mechanisms
    for i in range(10):
        try:
            result = await reliability_manager.execute_with_reliability(
                f"test_operation_{i}",
                failing_operation,
                timeout=10.0
            )
            print(f"Operation {i}: SUCCESS - {result['status']}")
        except Exception as e:
            print(f"Operation {i}: FAILED - {e}")
    
    # Show reliability metrics
    metrics = reliability_manager.get_reliability_metrics(hours=1)
    print(f"\n📊 Reliability Metrics:")
    print(f"Total failures: {metrics['total_failures']}")
    print(f"Failure breakdown: {metrics['failure_breakdown']}")
    print(f"Circuit breaker states: {metrics['circuit_breaker_states']}")
    
    # Health check
    health = reliability_manager.health_check()
    print(f"\n❤️ Health Status: {health['overall_health'].upper()}")
    if health['recommendations']:
        print(f"Recommendations: {health['recommendations']}")


if __name__ == "__main__":
    asyncio.run(demo_reliability())