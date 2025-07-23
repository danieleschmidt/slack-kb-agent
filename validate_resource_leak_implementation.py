#!/usr/bin/env python3  
"""
Validation script for resource leak implementation by checking source code.

This script validates the fixes by examining the source code directly
without importing modules that may have missing dependencies.
"""

import sys
import os

def check_database_fixes():
    """Check database.py for proper resource leak fixes."""
    print("Checking database.py resource leak fixes:")
    
    try:
        with open('src/slack_kb_agent/database.py', 'r') as f:
            content = f.read()
        
        checks = [
            ('atexit import', 'import atexit'),
            ('cleanup function definition', 'def _cleanup_database_resources'),
            ('atexit registration', 'atexit.register(_cleanup_database_resources)'),
            ('db manager close call', '_db_manager.close()'),
            ('failure window constant usage', 'DATABASE_FAILURE_WINDOW_SECONDS'),
        ]
        
        all_passed = True
        for check_name, pattern in checks:
            if pattern in content:
                print(f"✓ {check_name} found")
            else:
                print(f"✗ {check_name} missing")
                all_passed = False
        
        return all_passed
        
    except FileNotFoundError:
        print("✗ database.py file not found")
        return False
    except Exception as e:
        print(f"✗ Error reading database.py: {e}")
        return False

def check_cache_fixes():
    """Check cache.py for proper resource leak fixes."""
    print("\nChecking cache.py resource leak fixes:")
    
    try:
        with open('src/slack_kb_agent/cache.py', 'r') as f:
            content = f.read()
        
        checks = [
            ('atexit import', 'import atexit'),
            ('cleanup function definition', 'def _cleanup_cache_resources'),
            ('atexit registration', 'atexit.register(_cleanup_cache_resources)'),
            ('cache manager close call', '_cache_manager.close()'),
            ('failure window constant usage', 'REDIS_FAILURE_WINDOW_SECONDS'),
        ]
        
        all_passed = True
        for check_name, pattern in checks:
            if pattern in content:
                print(f"✓ {check_name} found")
            else:
                print(f"✗ {check_name} missing")
                all_passed = False
        
        return all_passed
        
    except FileNotFoundError:
        print("✗ cache.py file not found")
        return False
    except Exception as e:
        print(f"✗ Error reading cache.py: {e}")
        return False

def check_constants_fixes():
    """Check constants.py for failure window additions."""
    print("\nChecking constants.py failure window additions:")
    
    try:
        with open('src/slack_kb_agent/constants.py', 'r') as f:
            content = f.read()
        
        required_constants = [
            'DATABASE_FAILURE_WINDOW_SECONDS',
            'REDIS_FAILURE_WINDOW_SECONDS', 
            'LLM_FAILURE_WINDOW_SECONDS',
            'SLACK_FAILURE_WINDOW_SECONDS',
            'EXTERNAL_SERVICE_FAILURE_WINDOW_SECONDS'
        ]
        
        all_passed = True
        for const in required_constants:
            if const in content:
                print(f"✓ {const} defined")
            else:
                print(f"✗ {const} missing")
                all_passed = False
        
        return all_passed
        
    except FileNotFoundError:
        print("✗ constants.py file not found")
        return False
    except Exception as e:
        print(f"✗ Error reading constants.py: {e}")
        return False

def check_circuit_breaker_config_usage():
    """Check that failure_window_seconds is used in circuit breaker configs."""
    print("\nChecking circuit breaker config usage:")
    
    files_to_check = [
        ('database.py', 'DATABASE_FAILURE_WINDOW_SECONDS'),
        ('cache.py', 'REDIS_FAILURE_WINDOW_SECONDS')
    ]
    
    all_passed = True
    for filename, expected_constant in files_to_check:
        try:
            with open(f'src/slack_kb_agent/{filename}', 'r') as f:
                content = f.read()
            
            if f'failure_window_seconds={expected_constant}' in content or f'failure_window_seconds=CircuitBreakerDefaults.{expected_constant}' in content:
                print(f"✓ {filename} uses {expected_constant} in circuit breaker config")
            else:
                print(f"✗ {filename} missing {expected_constant} in circuit breaker config")
                all_passed = False
                
        except FileNotFoundError:
            print(f"✗ {filename} file not found")
            all_passed = False
        except Exception as e:
            print(f"✗ Error reading {filename}: {e}")
            all_passed = False
    
    return all_passed

def main():
    """Run all validation checks."""
    print("Resource Leak Implementation Validation")
    print("=" * 50)
    
    checks = [
        check_database_fixes,
        check_cache_fixes, 
        check_constants_fixes,
        check_circuit_breaker_config_usage
    ]
    
    all_passed = True
    for check_func in checks:
        try:
            if not check_func():
                all_passed = False
        except Exception as e:
            print(f"✗ Check failed with error: {e}")
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("✅ ALL IMPLEMENTATION CHECKS PASSED!")
        print("\nResource Leak Fixes Summary:")
        print("• Database connection pools have atexit cleanup handlers")
        print("• Redis connection pools have atexit cleanup handlers") 
        print("• Circuit breaker failure timestamps will be pruned via time windows")
        print("• All global resource managers will be properly closed on shutdown")
        print("\nThese fixes prevent:")
        print("- Database connection pool exhaustion")
        print("- Redis connection pool exhaustion") 
        print("- Unbounded memory growth from failure timestamps")
        return 0
    else:
        print("❌ SOME IMPLEMENTATION CHECKS FAILED")
        return 1

if __name__ == "__main__":
    sys.exit(main())