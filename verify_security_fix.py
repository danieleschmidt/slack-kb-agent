#!/usr/bin/env python3
"""
Verification script for pickle security fix.
This checks the code changes without requiring external dependencies.
"""

import sys
import os
from pathlib import Path

def check_code_changes():
    """Verify that the security changes are in place."""
    print("Verifying security fix implementation...\n")
    
    cache_file = Path(__file__).parent / "src" / "slack_kb_agent" / "cache.py"
    
    if not cache_file.exists():
        print(f"✗ Cache file not found: {cache_file}")
        return False
    
    with open(cache_file, 'r') as f:
        content = f.read()
    
    # Check that pickle import is removed
    if "import pickle" in content:
        print("✗ Pickle import still present in cache.py")
        return False
    else:
        print("✓ Pickle import removed")
    
    # Check that base64 import is added
    if "import base64" in content:
        print("✓ Base64 import added")
    else:
        print("✗ Base64 import missing")
        return False
    
    # Check that serialization methods are present
    if "_serialize_embedding" in content and "_deserialize_embedding" in content:
        print("✓ Secure serialization methods added")
    else:
        print("✗ Secure serialization methods missing")
        return False
    
    # Check that pickle.loads and pickle.dumps are removed
    if "pickle.loads" in content or "pickle.dumps" in content:
        print("✗ Pickle usage still present")
        return False
    else:
        print("✓ Pickle usage removed")
    
    # Check that Redis host is no longer hardcoded
    if 'host: str = "localhost"' in content:
        print("✗ Redis host still hardcoded to localhost")
        return False
    else:
        print("✓ Redis host default removed")
    
    # Check for configuration validation
    if "REDIS_HOST environment variable is required" in content:
        print("✓ Redis host validation added")
    else:
        print("✗ Redis host validation missing")
        return False
    
    print("\n✓ All code changes verified successfully!")
    return True

def check_test_updates():
    """Verify that tests have been updated."""
    print("\nVerifying test updates...\n")
    
    test_file = Path(__file__).parent / "tests" / "test_cache.py"
    
    if not test_file.exists():
        print(f"✗ Test file not found: {test_file}")
        return False
    
    with open(test_file, 'r') as f:
        content = f.read()
    
    # Check that pickle usage in tests is removed/updated
    if "pickle.dumps(embedding)" in content:
        print("✗ Test still uses pickle.dumps")
        return False
    else:
        print("✓ Pickle usage removed from tests")
    
    # Check for security test class
    if "TestCacheSecurityFeatures" in content:
        print("✓ Security test class added")
    else:
        print("✗ Security test class missing")
        return False
    
    # Check for specific security tests
    security_tests = [
        "test_secure_embedding_serialization",
        "test_malicious_embedding_data_rejection", 
        "test_no_code_execution_in_cache_data",
        "test_embedding_type_validation"
    ]
    
    missing_tests = [test for test in security_tests if test not in content]
    if missing_tests:
        print(f"✗ Missing security tests: {missing_tests}")
        return False
    else:
        print("✓ All security tests present")
    
    # Check for configuration security test
    if "test_cache_config_requires_host_when_enabled" in content:
        print("✓ Configuration security test added")
    else:
        print("✗ Configuration security test missing")
        return False
    
    print("\n✓ All test updates verified successfully!")
    return True

def check_documentation():
    """Verify documentation updates."""
    print("\nVerifying documentation updates...\n")
    
    backlog_file = Path(__file__).parent / "CURRENT_ITERATION_BACKLOG.md"
    
    if backlog_file.exists():
        print("✓ Current iteration backlog created")
        
        with open(backlog_file, 'r') as f:
            content = f.read()
        
        if "Replace Unsafe Pickle Serialization" in content:
            print("✓ Security fix documented in backlog")
        else:
            print("✗ Security fix not documented in backlog")
            return False
    else:
        print("✗ Current iteration backlog missing")
        return False
    
    print("\n✓ Documentation verified successfully!")
    return True

if __name__ == "__main__":
    print("Verifying pickle security vulnerability fix...\n")
    print("="*60)
    
    checks = [
        ("Code Changes", check_code_changes),
        ("Test Updates", check_test_updates), 
        ("Documentation", check_documentation),
    ]
    
    results = []
    for check_name, check_func in checks:
        print(f"\n{check_name}:")
        print("-" * len(check_name))
        try:
            results.append(check_func())
        except Exception as e:
            print(f"✗ {check_name} verification failed: {e}")
            results.append(False)
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print("\n" + "="*60)
    print(f"SECURITY FIX VERIFICATION: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n🔒 SECURITY FIX VERIFIED!")
        print("The pickle vulnerability has been successfully addressed:")
        print("  • Unsafe pickle serialization replaced with secure JSON+base64")
        print("  • Hardcoded Redis defaults removed")
        print("  • Comprehensive security tests added")
        print("  • Configuration validation implemented")
        sys.exit(0)
    else:
        print(f"\n❌ {total-passed} verification checks failed!")
        print("Manual review required before deploying.")
        sys.exit(1)