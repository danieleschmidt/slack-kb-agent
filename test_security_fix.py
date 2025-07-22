#!/usr/bin/env python3
"""
Quick test script to verify the security fix for pickle vulnerability.
"""

import sys
import os
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import the fixed cache module
from slack_kb_agent.cache import CacheManager, CacheConfig

def test_secure_serialization():
    """Test that the cache no longer uses pickle."""
    print("Testing secure serialization...")
    
    # Create a cache manager without Redis (just test serialization methods)
    config = CacheConfig(enabled=False)  # Disable to avoid Redis requirement
    cache_manager = CacheManager(config)
    
    # Test numpy array serialization
    test_embedding = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
    
    try:
        # Test serialization
        serialized = cache_manager._serialize_embedding(test_embedding)
        print(f"✓ Serialization successful, size: {len(serialized)} bytes")
        
        # Verify it's JSON format
        import json
        metadata = json.loads(serialized.decode('utf-8'))
        print(f"✓ Serialized data is valid JSON with keys: {list(metadata.keys())}")
        
        # Test deserialization
        deserialized = cache_manager._deserialize_embedding(serialized)
        print(f"✓ Deserialization successful, shape: {deserialized.shape}")
        
        # Verify data integrity
        np.testing.assert_array_equal(test_embedding, deserialized)
        print("✓ Data integrity verified - arrays are identical")
        
        # Verify no pickle-like patterns in serialized data
        serialized_str = serialized.decode('utf-8')
        dangerous_patterns = ['pickle', 'eval', 'exec', '__reduce__', '__setstate__']
        found_patterns = [p for p in dangerous_patterns if p in serialized_str.lower()]
        
        if found_patterns:
            print(f"✗ Found dangerous patterns: {found_patterns}")
            return False
        else:
            print("✓ No dangerous code execution patterns found")
        
        print("✓ All security tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False

def test_malicious_data_rejection():
    """Test that malicious data is rejected."""
    print("\nTesting malicious data rejection...")
    
    config = CacheConfig(enabled=False)
    cache_manager = CacheManager(config)
    
    # Test various malicious inputs
    malicious_inputs = [
        b"invalid json",
        b'{"missing": "fields"}',
        b'{"dtype": "float32", "shape": [3], "data": "invalid_base64!"}',
    ]
    
    for i, malicious_data in enumerate(malicious_inputs):
        try:
            cache_manager._deserialize_embedding(malicious_data)
            print(f"✗ Malicious input {i+1} was not rejected!")
            return False
        except (ValueError, TypeError, json.JSONDecodeError):
            print(f"✓ Malicious input {i+1} correctly rejected")
        except Exception as e:
            print(f"✗ Unexpected error for input {i+1}: {e}")
            return False
    
    print("✓ All malicious data rejection tests passed!")
    return True

def test_config_security():
    """Test that Redis configuration is secure."""
    print("\nTesting Redis configuration security...")
    
    # Test that Redis host is required when enabled
    os.environ.pop('REDIS_HOST', None)  # Remove if set
    os.environ['CACHE_ENABLED'] = 'true'
    
    try:
        config = CacheConfig.from_env()
        print("✗ Configuration allowed enabled cache without Redis host!")
        return False
    except ValueError as e:
        if "REDIS_HOST environment variable is required" in str(e):
            print("✓ Configuration correctly requires Redis host when enabled")
        else:
            print(f"✗ Unexpected error: {e}")
            return False
    
    # Test that disabled cache doesn't require host
    os.environ['CACHE_ENABLED'] = 'false'
    try:
        config = CacheConfig.from_env()
        print("✓ Disabled cache works without Redis host")
    except Exception as e:
        print(f"✗ Disabled cache failed: {e}")
        return False
    
    print("✓ All configuration security tests passed!")
    return True

if __name__ == "__main__":
    print("Running security fix verification tests...\n")
    
    # Run all tests
    tests = [
        test_secure_serialization,
        test_malicious_data_rejection,
        test_config_security,
    ]
    
    results = []
    for test_func in tests:
        try:
            results.append(test_func())
        except Exception as e:
            print(f"✗ Test {test_func.__name__} failed with exception: {e}")
            results.append(False)
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print(f"\n{'='*50}")
    print(f"Security Fix Verification Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🔒 All security tests passed! The pickle vulnerability has been fixed.")
        sys.exit(0)
    else:
        print("❌ Some security tests failed! Manual review required.")
        sys.exit(1)