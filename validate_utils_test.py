#!/usr/bin/env python3
"""
Simple validation script for utils.py test coverage.
This script validates the utility functions without requiring pytest.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from slack_kb_agent.utils import add


def test_add_basic_functionality():
    """Test basic add function functionality."""
    print("Testing add function basic functionality...")
    
    # Test normal integer addition
    assert add(5, 3) == 8
    assert add(-5, -3) == -8
    assert add(10, -3) == 7
    assert add(-10, 3) == -7
    print("✓ Basic integer addition works")
    
    # Test with zero
    assert add(5, 0) == 5
    assert add(0, 7) == 7
    assert add(0, 0) == 0
    print("✓ Addition with zero works")
    
    # Test with None (should treat as 0)
    assert add(None, 5) == 5
    assert add(10, None) == 10
    assert add(None, None) == 0
    print("✓ Addition with None works")
    
    # Test edge cases
    assert add(0, None) == 0
    assert add(None, 0) == 0
    print("✓ Edge cases work")


def test_add_return_types():
    """Test return types."""
    print("Testing return types...")
    
    assert isinstance(add(1, 2), int)
    assert isinstance(add(None, None), int)
    assert isinstance(add(None, 5), int)
    assert isinstance(add(10, None), int)
    print("✓ Return types are correct")


def test_add_comprehensive_cases():
    """Test comprehensive test cases."""
    print("Testing comprehensive cases...")
    
    test_cases = [
        (1, 2, 3),
        (1, None, 1),
        (None, 2, 2),
        (None, None, 0),
        (0, 0, 0),
        (0, None, 0),
        (None, 0, 0),
        (-5, 5, 0),
        (-10, None, -10),
        (None, -15, -15),
        (100, 200, 300),
        (-100, -200, -300)
    ]
    
    for a, b, expected in test_cases:
        result = add(a, b)
        assert result == expected, f"add({a}, {b}) = {result}, expected {expected}"
    
    print("✓ All comprehensive test cases pass")


def test_add_practical_usage():
    """Test practical usage scenarios."""
    print("Testing practical usage scenarios...")
    
    # Simulate processing a list with potential None values
    values = [1, 2, None, 4, None, 6]
    total = 0
    for i in range(0, len(values), 2):
        a = values[i] if i < len(values) else None
        b = values[i + 1] if i + 1 < len(values) else None
        total += add(a, b)
    
    # Expected: add(1,2)=3 + add(None,4)=4 + add(None,6)=6 = 13
    assert total == 13
    print("✓ Practical list processing works")
    
    # Test chaining
    result = add(add(1, 2), add(3, None))
    assert result == 6  # (1+2) + (3+0) = 6
    print("✓ Function chaining works")


def test_add_documentation():
    """Test that function has proper documentation."""
    print("Testing documentation...")
    
    assert add.__doc__ is not None
    assert "sum" in add.__doc__.lower()
    assert "none" in add.__doc__.lower()
    print("✓ Function documentation exists and is appropriate")


if __name__ == "__main__":
    print("=== Validating utils.py test coverage ===")
    
    try:
        test_add_basic_functionality()
        test_add_return_types()
        test_add_comprehensive_cases()
        test_add_practical_usage()
        test_add_documentation()
        
        print("\n✅ All utils validation tests passed!")
        print("The utils.py module has comprehensive test coverage.")
        print("The add function handles all expected inputs correctly including None values.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)