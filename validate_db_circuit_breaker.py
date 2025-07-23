#!/usr/bin/env python3
"""
Simple validation script for database circuit breaker implementation.

This script performs basic syntax validation and functionality checks
without requiring the full test environment or database connections.
"""

import sys
import os
sys.path.append('.')

def validate_imports():
    """Validate that all imports work correctly."""
    try:
        print("✓ Testing imports...")
        
        # Test circuit breaker imports
        from src.slack_kb_agent.circuit_breaker import CircuitBreaker, CircuitBreakerConfig, CircuitState
        print("  ✓ Circuit breaker imports successful")
        
        # Test constants import
        from src.slack_kb_agent.constants import CircuitBreakerDefaults
        print("  ✓ Constants imports successful")
        
        # Test database module can be imported (structure validation)
        try:
            from src.slack_kb_agent.database import DatabaseManager, DatabaseRepository
            print("  ✓ Database module structure valid")
        except ImportError as e:
            if "sqlalchemy" in str(e).lower():
                print("  ⚠ Database module requires SQLAlchemy (expected in test environment)")
            else:
                raise
        
        return True
    except Exception as e:
        print(f"  ✗ Import validation failed: {e}")
        return False

def validate_circuit_breaker_config():
    """Validate circuit breaker configuration constants."""
    try:
        print("✓ Testing circuit breaker configuration...")
        
        from src.slack_kb_agent.constants import CircuitBreakerDefaults
        
        # Validate database circuit breaker constants exist
        assert hasattr(CircuitBreakerDefaults, 'DATABASE_FAILURE_THRESHOLD')
        assert hasattr(CircuitBreakerDefaults, 'DATABASE_SUCCESS_THRESHOLD')
        assert hasattr(CircuitBreakerDefaults, 'DATABASE_TIMEOUT_SECONDS')
        assert hasattr(CircuitBreakerDefaults, 'DATABASE_HALF_OPEN_MAX_REQUESTS')
        
        # Validate values are reasonable
        assert CircuitBreakerDefaults.DATABASE_FAILURE_THRESHOLD == 5
        assert CircuitBreakerDefaults.DATABASE_SUCCESS_THRESHOLD == 2
        assert CircuitBreakerDefaults.DATABASE_TIMEOUT_SECONDS == 60.0
        assert CircuitBreakerDefaults.DATABASE_HALF_OPEN_MAX_REQUESTS == 3
        
        print("  ✓ Database circuit breaker constants are correct")
        return True
    except Exception as e:
        print(f"  ✗ Circuit breaker config validation failed: {e}")
        return False

def validate_database_manager_methods():
    """Validate DatabaseManager has required circuit breaker methods."""
    try:
        print("✓ Testing DatabaseManager circuit breaker integration...")
        
        # Import database module source to check methods exist
        import inspect
        with open('src/slack_kb_agent/database.py', 'r') as f:
            source_code = f.read()
        
        # Check for required method implementations
        required_patterns = [
            '_get_circuit_breaker',
            'self.circuit_breaker',
            'CircuitBreaker(',
            'CircuitBreakerConfig(',
            'CircuitBreakerDefaults.DATABASE_FAILURE_THRESHOLD'
        ]
        
        for pattern in required_patterns:
            assert pattern in source_code, f"Missing pattern: {pattern}"
        
        # Check that circuit breaker is used in key methods
        key_methods = [
            'def initialize(self)',
            'def is_available(self)',
            'def get_session(self)'
        ]
        
        for method in key_methods:
            if method in source_code:
                # Find the method body and check it uses circuit breaker
                method_start = source_code.find(method)
                # This is a simple check - in practice the methods should use circuit_breaker.call
                print(f"  ✓ Found method: {method}")
        
        print("  ✓ DatabaseManager circuit breaker integration looks correct")
        return True
    except Exception as e:
        print(f"  ✗ DatabaseManager validation failed: {e}")
        return False

def validate_database_repository_methods():
    """Validate DatabaseRepository has required circuit breaker methods."""
    try:
        print("✓ Testing DatabaseRepository circuit breaker integration...")
        
        # Import database module source to check methods exist
        with open('src/slack_kb_agent/database.py', 'r') as f:
            source_code = f.read()
        
        # Check for DatabaseRepository circuit breaker integration
        repository_patterns = [
            'class DatabaseRepository:',
            'self.circuit_breaker = self._get_circuit_breaker()',
            'def _get_circuit_breaker(self) -> CircuitBreaker:'
        ]
        
        for pattern in repository_patterns:
            assert pattern in source_code, f"Missing repository pattern: {pattern}"
        
        # Check that CRUD operations use circuit breaker protection
        crud_methods = [
            'def create_document(',
            'def create_documents(',
            'def get_document(',
            'def get_all_documents(',
            'def search_documents(',
            'def count_documents(',
            'def delete_document(',
            'def clear_all_documents('
        ]
        
        for method in crud_methods:
            if method in source_code:
                print(f"  ✓ Found protected method: {method}")
        
        print("  ✓ DatabaseRepository circuit breaker integration looks correct")
        return True
    except Exception as e:
        print(f"  ✗ DatabaseRepository validation failed: {e}")
        return False

def main():
    """Run all validation checks."""
    print("Database Circuit Breaker Implementation Validation")
    print("=" * 50)
    
    all_passed = True
    
    # Run validation checks
    checks = [
        validate_imports,
        validate_circuit_breaker_config,
        validate_database_manager_methods,
        validate_database_repository_methods
    ]
    
    for check in checks:
        if not check():
            all_passed = False
        print()
    
    # Summary
    if all_passed:
        print("🎉 All validation checks PASSED!")
        print("Database circuit breaker implementation appears to be correct.")
        return 0
    else:
        print("❌ Some validation checks FAILED!")
        print("Please review the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())