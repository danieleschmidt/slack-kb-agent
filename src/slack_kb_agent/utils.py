"""Utility functions for slack_kb_agent."""

from typing import Optional


def add(a: Optional[int], b: Optional[int]) -> int:
    """Return the sum of ``a`` and ``b`` treating ``None`` as ``0``."""
    return (a or 0) + (b or 0)
