#!/usr/bin/env python3
"""
Test script to verify empty exception handlers have been fixed.
"""

import sys
import os
from pathlib import Path

def test_cache_deletion_error_handling():
    """Test that cache deletion errors are now logged."""
    print("Testing cache deletion error handling...")
    
    cache_file = Path(__file__).parent / "src" / "slack_kb_agent" / "cache.py"
    
    with open(cache_file, 'r') as f:
        content = f.read()
    
    # Check that the empty exception handler is fixed
    if "except Exception:\n                pass" in content:
        print("✗ Found empty exception handler in cache.py")
        return False
    
    # Check that proper error logging is added
    if "Failed to delete corrupted cache entry" in content:
        print("✓ Cache deletion errors are now logged")
    else:
        print("✗ Cache deletion error logging missing")
        return False
    
    return True

def test_metrics_error_handling():
    """Test that metrics collection errors are now logged."""
    print("\nTesting metrics collection error handling...")
    
    files_to_check = [
        ("src/slack_kb_agent/knowledge_base.py", "Failed to update knowledge base metrics"),
        ("src/slack_kb_agent/query_processor.py", "Failed to update query processor metrics")
    ]
    
    for file_path, expected_message in files_to_check:
        full_path = Path(__file__).parent / file_path
        
        with open(full_path, 'r') as f:
            content = f.read()
        
        # Check that empty exception handlers are fixed
        if "except Exception:\n            # Don't let metrics collection crash the application\n            pass" in content:
            print(f"✗ Found empty exception handler in {file_path}")
            return False
        
        # Check that proper error logging is added
        if expected_message in content:
            print(f"✓ {file_path} now logs metrics errors")
        else:
            print(f"✗ {file_path} missing metrics error logging")
            return False
    
    return True

def test_pass_statement_removal():
    """Test that unnecessary pass statements have been removed."""
    print("\nTesting pass statement removal...")
    
    monitoring_file = Path(__file__).parent / "src" / "slack_kb_agent" / "monitoring.py"
    
    with open(monitoring_file, 'r') as f:
        content = f.read()
    
    # Count remaining pass statements (excluding class definitions)
    lines = content.split('\n')
    pass_count = 0
    
    for i, line in enumerate(lines):
        if line.strip() == "pass":
            # Check if this is after a class or function definition (acceptable)
            prev_lines = lines[max(0, i-3):i]
            is_class_or_func = any("class " in l or "def " in l for l in prev_lines)
            if not is_class_or_func:
                pass_count += 1
    
    if pass_count == 0:
        print("✓ No unnecessary pass statements found in monitoring.py")
    else:
        print(f"✗ Found {pass_count} unnecessary pass statements")
        return False
    
    return True

def test_error_logging_consistency():
    """Test that error logging is consistent across modules."""
    print("\nTesting error logging consistency...")
    
    modules = [
        "src/slack_kb_agent/cache.py",
        "src/slack_kb_agent/knowledge_base.py", 
        "src/slack_kb_agent/query_processor.py"
    ]
    
    consistent_patterns = 0
    
    for module in modules:
        module_file = Path(__file__).parent / module
        
        with open(module_file, 'r') as f:
            content = f.read()
        
        # Check for consistent error logging patterns
        if "type(e).__name__" in content:
            consistent_patterns += 1
        
        # Check that errors have context
        if "Failed to" in content:
            consistent_patterns += 1
    
    if consistent_patterns >= 4:  # At least good patterns in multiple modules
        print(f"✓ Error logging patterns are consistent ({consistent_patterns} patterns found)")
        return True
    else:
        print(f"✗ Error logging patterns inconsistent ({consistent_patterns} patterns found)")
        return False

def test_no_silent_failures():
    """Test that there are no remaining silent failures."""
    print("\nTesting for silent failures...")
    
    src_dir = Path(__file__).parent / "src" / "slack_kb_agent"
    silent_failures = 0
    
    for py_file in src_dir.glob("*.py"):
        with open(py_file, 'r') as f:
            content = f.read()
        
        # Look for problematic patterns
        if "except Exception:\n        pass" in content:
            silent_failures += 1
            print(f"✗ Found silent failure in {py_file.name}")
        
        if "except Exception:\n            pass" in content:
            silent_failures += 1
            print(f"✗ Found silent failure in {py_file.name}")
    
    if silent_failures == 0:
        print("✓ No silent failures found")
        return True
    else:
        print(f"✗ Found {silent_failures} silent failures")
        return False

if __name__ == "__main__":
    print("Testing empty exception handler fixes...\n")
    print("="*60)
    
    tests = [
        ("Cache Deletion Error Handling", test_cache_deletion_error_handling),
        ("Metrics Error Handling", test_metrics_error_handling),
        ("Pass Statement Removal", test_pass_statement_removal),
        ("Error Logging Consistency", test_error_logging_consistency),
        ("No Silent Failures", test_no_silent_failures),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            results.append(test_func())
        except Exception as e:
            print(f"✗ {test_name} test failed with exception: {e}")
            results.append(False)
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print("\n" + "="*60)
    print(f"EMPTY EXCEPTION HANDLER FIXES: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✅ ALL EMPTY EXCEPTION HANDLER TESTS PASSED!")
        print("Improvements implemented:")
        print("  • Cache deletion errors are now logged with context")
        print("  • Metrics collection failures use debug-level logging")
        print("  • Eliminated silent failure patterns")
        print("  • Consistent error logging with exception type names")
        print("  • Removed unnecessary pass statements")
        sys.exit(0)
    else:
        print(f"\n❌ {total-passed} empty exception handler tests failed!")
        print("Manual review required.")
        sys.exit(1)