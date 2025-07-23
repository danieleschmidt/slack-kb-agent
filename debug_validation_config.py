#!/usr/bin/env python3
"""Debug validation configuration."""

import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from slack_kb_agent.validation import get_validator

def debug_config():
    """Debug validation configuration."""
    validator = get_validator()
    
    print("Validation Configuration:")
    print("=" * 30)
    print(f"Enabled: {validator.config.enabled}")
    print(f"Block SQL injection: {validator.config.block_sql_injection}")
    print(f"Block command injection: {validator.config.block_command_injection}")
    print(f"Block XSS: {validator.config.block_xss}")
    print(f"Normalize whitespace: {validator.config.normalize_whitespace}")
    print(f"Strip HTML: {validator.config.strip_html}")
    print(f"Allow Unicode: {validator.config.allow_unicode}")
    
    # Test the dangerous query step by step
    query = "test; rm -rf /"
    print(f"\nDebugging query: '{query}'")
    
    # Step 1: Sanitization
    sanitized = validator._sanitize_text(query)
    print(f"After sanitization: '{sanitized}'")
    
    # Step 2: Check original for command injection
    original_contains_command = validator._contains_command_injection(query)
    print(f"Original contains command injection: {original_contains_command}")
    
    # Step 3: Check sanitized for command injection  
    sanitized_contains_command = validator._contains_command_injection(sanitized)
    print(f"Sanitized contains command injection: {sanitized_contains_command}")
    
    # Step 4: Full validation
    result = validator.validate_query(query)
    print(f"Full validation result: is_valid={result.is_valid}, error='{result.error_message}'")

if __name__ == "__main__":
    debug_config()