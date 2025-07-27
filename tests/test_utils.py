#!/usr/bin/env python3
"""
Test suite for utility functions in the Slack KB Agent.

This module tests the utility functions used throughout the system.
"""

import pytest
from typing import Optional

# Import the utility functions to test
from slack_kb_agent.utils import add


class TestAddFunction:
    """Test the add utility function."""
    
    def test_add_two_positive_integers(self):
        """Test adding two positive integers."""
        result = add(5, 3)
        assert result == 8
    
    def test_add_two_negative_integers(self):
        """Test adding two negative integers."""
        result = add(-5, -3)
        assert result == -8
    
    def test_add_positive_and_negative(self):
        """Test adding positive and negative integers."""
        result = add(10, -3)
        assert result == 7
        
        result = add(-10, 3)
        assert result == -7
    
    def test_add_with_zero(self):
        """Test adding with zero."""
        result = add(5, 0)
        assert result == 5
        
        result = add(0, 7)
        assert result == 7
        
        result = add(0, 0)
        assert result == 0
    
    def test_add_with_none_values(self):
        """Test adding with None values (should treat as 0)."""
        result = add(None, 5)
        assert result == 5
        
        result = add(10, None)
        assert result == 10
        
        result = add(None, None)
        assert result == 0
    
    def test_add_large_numbers(self):
        """Test adding large numbers."""
        result = add(1000000, 2000000)
        assert result == 3000000
        
        result = add(-1000000, -2000000)
        assert result == -3000000
    
    def test_add_maximum_integer_values(self):
        """Test adding very large integer values."""
        import sys
        
        # Test with large values (but not overflow)
        large_val = sys.maxsize // 2
        result = add(large_val, large_val)
        assert result == large_val * 2
    
    def test_add_return_type(self):
        """Test that add function returns an integer."""
        result = add(1, 2)
        assert isinstance(result, int)
        
        result = add(None, None)
        assert isinstance(result, int)
        
        result = add(None, 5)
        assert isinstance(result, int)
    
    def test_add_with_zero_and_none(self):
        """Test adding zero and None (both should be treated as 0)."""
        result = add(0, None)
        assert result == 0
        
        result = add(None, 0)
        assert result == 0
    
    def test_add_edge_case_combinations(self):
        """Test various edge case combinations."""
        # Test cases that verify None is treated as 0
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
            (None, -15, -15)
        ]
        
        for a, b, expected in test_cases:
            result = add(a, b)
            assert result == expected, f"add({a}, {b}) = {result}, expected {expected}"
    
    def test_add_function_signature(self):
        """Test that the function accepts the correct parameter types."""
        # These should all work without type errors
        add(1, 2)
        add(None, 2)
        add(1, None)
        add(None, None)
        
        # The function should handle all valid inputs
        assert callable(add)


class TestUtilsModuleStructure:
    """Test the structure and organization of the utils module."""
    
    def test_module_imports(self):
        """Test that the utils module can be imported properly."""
        import slack_kb_agent.utils
        
        # Check that the module has the expected function
        assert hasattr(slack_kb_agent.utils, 'add')
        assert callable(slack_kb_agent.utils.add)
    
    def test_function_docstring(self):
        """Test that the add function has proper documentation."""
        assert add.__doc__ is not None
        assert "sum" in add.__doc__.lower()
        assert "none" in add.__doc__.lower()
    
    def test_type_annotations(self):
        """Test that the function has proper type annotations."""
        import inspect
        
        sig = inspect.signature(add)
        
        # Check parameter annotations
        assert 'a' in sig.parameters
        assert 'b' in sig.parameters
        
        # Check return annotation
        assert sig.return_annotation == int


class TestUtilsIntegration:
    """Test integration and usage patterns of utility functions."""
    
    def test_add_in_calculations(self):
        """Test using add function in typical calculation scenarios."""
        # Simulate a scenario where we might have optional values
        values = [1, 2, None, 4, None, 6]
        
        # Sum pairs using the add function
        total = 0
        for i in range(0, len(values), 2):
            a = values[i] if i < len(values) else None
            b = values[i + 1] if i + 1 < len(values) else None
            total += add(a, b)
        
        # Expected: add(1,2)=3 + add(None,4)=4 + add(None,6)=6 = 13
        assert total == 13
    
    def test_add_with_user_input_simulation(self):
        """Test add function with simulated user input scenarios."""
        # Simulate scenarios where user input might be missing/None
        
        # User provides both values
        user_a, user_b = 10, 20
        result = add(user_a, user_b)
        assert result == 30
        
        # User provides only one value
        user_a, user_b = 15, None
        result = add(user_a, user_b)
        assert result == 15
        
        # User provides no values
        user_a, user_b = None, None
        result = add(user_a, user_b)
        assert result == 0
    
    def test_add_chaining(self):
        """Test chaining multiple add operations."""
        # Test that we can chain multiple add operations
        result = add(add(1, 2), add(3, None))
        assert result == 6  # (1+2) + (3+0) = 6
        
        result = add(add(None, None), add(5, 10))
        assert result == 15  # (0+0) + (5+10) = 15


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])