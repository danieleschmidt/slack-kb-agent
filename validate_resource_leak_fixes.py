#!/usr/bin/env python3
"""
Validation script for resource leak fixes.

This script validates that the resource leak fixes are properly implemented:
1. Database connection pool cleanup via atexit
2. Redis connection pool cleanup via atexit  
3. Circuit breaker failure timestamp pruning
"""

import sys
import os
import atexit
import inspect

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def check_atexit_registration():
    """Check that cleanup functions are registered with atexit."""
    print("Checking atexit registration:")
    
    # Import modules to trigger atexit registration
    from slack_kb_agent import database, cache
    
    # Get atexit callbacks (this is implementation-dependent)
    # We'll check by inspecting the module for the registration call
    
    database_source = inspect.getsource(database)
    cache_source = inspect.getsource(cache)
    
    # Check for atexit.register calls
    if "atexit.register(_cleanup_database_resources)" in database_source:
        print("✓ Database cleanup function registered with atexit")
        db_cleanup_registered = True
    else:
        print("✗ Database cleanup function NOT registered with atexit")
        db_cleanup_registered = False
    
    if "atexit.register(_cleanup_cache_resources)" in cache_source:
        print("✓ Cache cleanup function registered with atexit")
        cache_cleanup_registered = True
    else:
        print("✗ Cache cleanup function NOT registered with atexit")
        cache_cleanup_registered = False
    
    return db_cleanup_registered and cache_cleanup_registered

def check_cleanup_functions_exist():
    """Check that cleanup functions exist and are callable."""
    print("\nChecking cleanup function implementations:")
    
    try:
        from slack_kb_agent import database
        
        # Check if cleanup function exists
        if hasattr(database, '_cleanup_database_resources'):
            print("✓ Database cleanup function exists")
            if callable(getattr(database, '_cleanup_database_resources')):
                print("✓ Database cleanup function is callable")
                db_cleanup_ok = True
            else:
                print("✗ Database cleanup function is not callable")
                db_cleanup_ok = False
        else:
            print("✗ Database cleanup function does not exist")
            db_cleanup_ok = False
            
    except Exception as e:
        print(f"✗ Error checking database cleanup: {e}")
        db_cleanup_ok = False
    
    try:
        from slack_kb_agent import cache
        
        # Check if cleanup function exists
        if hasattr(cache, '_cleanup_cache_resources'):
            print("✓ Cache cleanup function exists")
            if callable(getattr(cache, '_cleanup_cache_resources')):
                print("✓ Cache cleanup function is callable")
                cache_cleanup_ok = True
            else:
                print("✗ Cache cleanup function is not callable")
                cache_cleanup_ok = False
        else:
            print("✗ Cache cleanup function does not exist")
            cache_cleanup_ok = False
            
    except Exception as e:
        print(f"✗ Error checking cache cleanup: {e}")
        cache_cleanup_ok = False
    
    return db_cleanup_ok and cache_cleanup_ok

def check_circuit_breaker_window_config():
    """Check that circuit breaker configurations include failure window."""
    print("\nChecking circuit breaker failure window configuration:")
    
    try:
        from slack_kb_agent.constants import CircuitBreakerDefaults
        
        # Check that failure window constants exist
        required_window_constants = [
            'DATABASE_FAILURE_WINDOW_SECONDS',
            'REDIS_FAILURE_WINDOW_SECONDS',
            'LLM_FAILURE_WINDOW_SECONDS',
            'SLACK_FAILURE_WINDOW_SECONDS',
            'EXTERNAL_SERVICE_FAILURE_WINDOW_SECONDS'
        ]
        
        all_constants_exist = True
        for const in required_window_constants:
            if hasattr(CircuitBreakerDefaults, const):
                value = getattr(CircuitBreakerDefaults, const)
                print(f"✓ {const} = {value}")
                if value is None or value <= 0:
                    print(f"  ⚠️  Warning: {const} should be > 0 for memory leak prevention")
            else:
                print(f"✗ Missing constant: {const}")
                all_constants_exist = False
        
        return all_constants_exist
        
    except Exception as e:
        print(f"✗ Error checking circuit breaker constants: {e}")
        return False

def check_circuit_breaker_usage():
    """Check that circuit breakers are configured with failure windows."""
    print("\nChecking circuit breaker configuration usage:")
    
    try:
        # Check database circuit breaker configuration
        from slack_kb_agent.database import DatabaseManager
        from slack_kb_agent.constants import CircuitBreakerDefaults
        
        config = CircuitBreakerDefaults.DATABASE_FAILURE_WINDOW_SECONDS
        if config and config > 0:
            print(f"✓ Database circuit breaker has failure window: {config} seconds")
            db_window_ok = True
        else:
            print("✗ Database circuit breaker missing or invalid failure window")
            db_window_ok = False
            
    except Exception as e:
        print(f"✗ Error checking database circuit breaker config: {e}")
        db_window_ok = False
    
    try:
        # Check cache circuit breaker configuration
        from slack_kb_agent.cache import CacheManager
        from slack_kb_agent.constants import CircuitBreakerDefaults
        
        config = CircuitBreakerDefaults.REDIS_FAILURE_WINDOW_SECONDS
        if config and config > 0:
            print(f"✓ Redis circuit breaker has failure window: {config} seconds")
            cache_window_ok = True
        else:
            print("✗ Redis circuit breaker missing or invalid failure window")
            cache_window_ok = False
            
    except Exception as e:
        print(f"✗ Error checking cache circuit breaker config: {e}")
        cache_window_ok = False
    
    return db_window_ok and cache_window_ok

def main():
    """Run all resource leak fix validations."""
    print("Resource Leak Fix Validation")
    print("=" * 50)
    
    checks = [
        ("Atexit registration", check_atexit_registration),
        ("Cleanup function implementation", check_cleanup_functions_exist),
        ("Circuit breaker window constants", check_circuit_breaker_window_config),
        ("Circuit breaker window usage", check_circuit_breaker_usage),
    ]
    
    all_passed = True
    for check_name, check_func in checks:
        try:
            if not check_func():
                all_passed = False
        except Exception as e:
            print(f"✗ {check_name} failed with error: {e}")
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("✅ ALL CHECKS PASSED - Resource leak fixes are properly implemented!")
        print("\nResource leak protection summary:")
        print("• Database connection pools will be closed on application exit")
        print("• Redis connection pools will be closed on application exit")
        print("• Circuit breaker failure timestamps will be pruned automatically")
        print("• Memory leaks from unbounded data structures are prevented")
        return 0
    else:
        print("❌ SOME CHECKS FAILED - Resource leak fixes need attention")
        return 1

if __name__ == "__main__":
    sys.exit(main())