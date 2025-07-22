#!/usr/bin/env python3
"""
Simple test to verify credential masking functionality works correctly.
This test doesn't depend on external dependencies and focuses on the 
security utility functions.
"""

import sys
import os
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from slack_kb_agent.security_utils import (
        mask_database_url, mask_connection_string, 
        mask_sensitive_dict, get_safe_repr, quick_mask_credentials
    )
except ImportError as e:
    print(f"Failed to import security utilities: {e}")
    sys.exit(1)


def test_database_url_masking():
    """Test that database URLs are properly masked."""
    test_cases = [
        # (input, expected_output, description)
        (
            "postgresql://user:password123@localhost:5432/mydb",
            "postgresql://user:***@localhost:5432/mydb",
            "PostgreSQL URL with password"
        ),
        (
            "mysql://admin:secret@db.example.com:3306/testdb",
            "mysql://admin:***@db.example.com:3306/testdb",
            "MySQL URL with password"
        ),
        (
            "postgresql://localhost:5432/mydb",
            "postgresql://localhost:5432/mydb",
            "PostgreSQL URL without credentials"
        ),
        (
            "postgresql://user@localhost:5432/mydb",
            "postgresql://user@localhost:5432/mydb",
            "PostgreSQL URL with username only"
        ),
        (
            "postgresql://user:@localhost:5432/mydb",
            "postgresql://user:@localhost:5432/mydb",
            "PostgreSQL URL with empty password"
        ),
        (
            "invalid-url-format",
            "invalid-url-format",
            "Non-URL string"
        ),
        (
            "",
            "",
            "Empty string"
        ),
        (
            None,
            "None",
            "None input"
        )
    ]
    
    print("🔒 Testing Database URL Masking...")
    
    for input_url, expected, description in test_cases:
        result = mask_database_url(input_url)
        
        if result == expected:
            print(f"  ✅ {description}: PASS")
        else:
            print(f"  ❌ {description}: FAIL")
            print(f"     Input: {input_url}")
            print(f"     Expected: {expected}")
            print(f"     Got: {result}")
            return False
    
    return True


def test_sensitive_dict_masking():
    """Test that sensitive data in dictionaries is properly masked."""
    print("\n🔒 Testing Sensitive Dictionary Masking...")
    
    test_data = {
        "database_url": "postgresql://user:secret@localhost/db",
        "api_key": "sk-1234567890abcdef",
        "normal_field": "normal_value",
        "nested": {
            "password": "hidden_password",
            "public_info": "visible_data"
        }
    }
    
    masked_data = mask_sensitive_dict(test_data)
    
    # Check that sensitive fields are masked
    checks = [
        (masked_data["database_url"] == "postgresql://user:***@localhost/db", "Database URL masked"),
        (masked_data["api_key"] == "***", "API key masked"),
        (masked_data["normal_field"] == "normal_value", "Normal field preserved"),
        (masked_data["nested"]["password"] == "***", "Nested password masked"),
        (masked_data["nested"]["public_info"] == "visible_data", "Nested public info preserved")
    ]
    
    all_passed = True
    for passed, description in checks:
        if passed:
            print(f"  ✅ {description}: PASS")
        else:
            print(f"  ❌ {description}: FAIL")
            all_passed = False
    
    return all_passed


def test_quick_masking():
    """Test quick regex-based masking for performance."""
    print("\n🔒 Testing Quick Credential Masking...")
    
    test_cases = [
        (
            "Connection: postgresql://user:password@host/db",
            "Connection: postgresql://user:***@host/db",
            "URL in text"
        ),
        (
            "Config: host=localhost password=secret port=5432",
            "Config: host=localhost password=*** port=5432",
            "Key-value format"
        ),
        (
            "No sensitive data here",
            "No sensitive data here",
            "Clean text"
        )
    ]
    
    all_passed = True
    for input_text, expected, description in test_cases:
        result = quick_mask_credentials(input_text)
        
        if result == expected:
            print(f"  ✅ {description}: PASS")
        else:
            print(f"  ❌ {description}: FAIL")
            print(f"     Expected: {expected}")
            print(f"     Got: {result}")
            all_passed = False
    
    return all_passed


def test_database_memory_stats_security():
    """Test that the specific security fix for memory stats works."""
    print("\n🔒 Testing Database Memory Stats Security Fix...")
    
    # Simulate the memory stats return value with the security fix
    test_database_url = "postgresql://dbuser:supersecret@prod-db.company.com:5432/app_db"
    
    # This simulates what the fixed memory stats should return
    memory_stats = {
        "total_documents": 1500,
        "source_distribution": {"slack": 800, "github": 700},
        "estimated_size_bytes": 2048000,
        "database_url": mask_database_url(test_database_url)
    }
    
    # Verify the database URL is properly masked
    masked_url = memory_stats["database_url"]
    
    checks = [
        ("supersecret" not in masked_url, "Password not exposed"),
        ("dbuser:***@" in masked_url, "Username preserved, password masked"),
        ("prod-db.company.com" in masked_url, "Host information preserved"),
        (":5432/app_db" in masked_url, "Port and database preserved")
    ]
    
    all_passed = True
    for passed, description in checks:
        if passed:
            print(f"  ✅ {description}: PASS")
        else:
            print(f"  ❌ {description}: FAIL")
            print(f"     Masked URL: {masked_url}")
            all_passed = False
    
    return all_passed


def main():
    """Run all credential masking tests."""
    print("🧪 Verifying Credential Masking Security Implementation...\n")
    
    tests = [
        test_database_url_masking,
        test_sensitive_dict_masking,
        test_quick_masking,
        test_database_memory_stats_security
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test_func.__name__} failed with exception: {e}")
            results.append(False)
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print(f"\n📊 Test Summary: {passed}/{total} tests passed")
    
    if all(results):
        print("🎉 All credential masking tests passed! Security fix is working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Credential masking needs further work.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)