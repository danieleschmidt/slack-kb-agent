#!/usr/bin/env python3
"""
Validation script for Redis circuit breaker implementation.

This script checks that the Redis circuit breaker is properly integrated
into the CacheManager class without requiring actual Redis or dependencies.
"""

import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def check_imports():
    """Check that all required imports work."""
    try:
        from slack_kb_agent.cache import CacheManager, CacheConfig
        from slack_kb_agent.circuit_breaker import CircuitBreaker, CircuitBreakerConfig, CircuitState
        from slack_kb_agent.constants import CircuitBreakerDefaults
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def check_circuit_breaker_constants():
    """Check that Redis circuit breaker constants exist."""
    try:
        from slack_kb_agent.constants import CircuitBreakerDefaults
        
        required_constants = [
            'REDIS_FAILURE_THRESHOLD',
            'REDIS_SUCCESS_THRESHOLD', 
            'REDIS_TIMEOUT_SECONDS',
            'REDIS_HALF_OPEN_MAX_REQUESTS'
        ]
        
        for const in required_constants:
            if not hasattr(CircuitBreakerDefaults, const):
                print(f"✗ Missing constant: {const}")
                return False
            value = getattr(CircuitBreakerDefaults, const)
            print(f"✓ {const} = {value}")
        
        return True
    except Exception as e:
        print(f"✗ Error checking constants: {e}")
        return False

def check_cache_manager_structure():
    """Check that CacheManager has circuit breaker integration."""
    try:
        from slack_kb_agent.cache import CacheManager, CacheConfig
        
        # Create CacheManager with disabled cache (no Redis needed)
        config = CacheConfig(enabled=False)
        cache_manager = CacheManager(config)
        
        # Check for required methods and attributes
        required_methods = ['_get_circuit_breaker', 'is_available']
        for method in required_methods:
            if not hasattr(cache_manager, method):
                print(f"✗ Missing method: {method}")
                return False
            print(f"✓ Method {method} exists")
        
        # Check for circuit_breaker property
        if not hasattr(cache_manager, 'circuit_breaker'):
            print("✗ Missing circuit_breaker property")
            return False
        print("✓ circuit_breaker property exists")
        
        return True
    except Exception as e:
        print(f"✗ Error checking CacheManager structure: {e}")
        return False

def check_circuit_breaker_config():
    """Check that circuit breaker config is properly created."""
    try:
        from slack_kb_agent.cache import CacheManager, CacheConfig
        from slack_kb_agent.constants import CircuitBreakerDefaults
        
        config = CacheConfig(enabled=False)
        cache_manager = CacheManager(config)
        
        # Get circuit breaker configuration
        circuit_breaker = cache_manager._get_circuit_breaker()
        
        # Validate configuration values
        expected_values = {
            'failure_threshold': CircuitBreakerDefaults.REDIS_FAILURE_THRESHOLD,
            'success_threshold': CircuitBreakerDefaults.REDIS_SUCCESS_THRESHOLD,
            'timeout_seconds': CircuitBreakerDefaults.REDIS_TIMEOUT_SECONDS,
            'half_open_max_requests': CircuitBreakerDefaults.REDIS_HALF_OPEN_MAX_REQUESTS,
            'service_name': 'redis'
        }
        
        for attr, expected_value in expected_values.items():
            actual_value = getattr(circuit_breaker.config, attr)
            if actual_value != expected_value:
                print(f"✗ Config mismatch - {attr}: expected {expected_value}, got {actual_value}")
                return False
            print(f"✓ Config {attr} = {actual_value}")
        
        return True
    except Exception as e:
        print(f"✗ Error checking circuit breaker config: {e}")
        return False

def main():
    """Run all validation checks."""
    print("Redis Circuit Breaker Implementation Validation")
    print("=" * 50)
    
    checks = [
        ("Import validation", check_imports),
        ("Circuit breaker constants", check_circuit_breaker_constants),
        ("CacheManager structure", check_cache_manager_structure),
        ("Circuit breaker configuration", check_circuit_breaker_config),
    ]
    
    all_passed = True
    for check_name, check_func in checks:
        print(f"\n{check_name}:")
        if not check_func():
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("✅ ALL CHECKS PASSED - Redis circuit breaker implementation is valid!")
        return 0
    else:
        print("❌ SOME CHECKS FAILED - Redis circuit breaker implementation needs fixes")
        return 1

if __name__ == "__main__":
    sys.exit(main())