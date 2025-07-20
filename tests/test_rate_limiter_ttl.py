"""Tests for TTL cleanup functionality in RateLimiter."""

import time
import pytest
from unittest.mock import patch
from slack_kb_agent.auth import RateLimiter


def test_rate_limiter_basic_functionality():
    """Test that basic rate limiting still works with TTL cleanup."""
    limiter = RateLimiter(max_requests=3, window_seconds=60)
    
    # Should allow first 3 requests
    assert limiter.is_allowed("user1") == True
    assert limiter.is_allowed("user1") == True
    assert limiter.is_allowed("user1") == True
    
    # Should deny 4th request
    assert limiter.is_allowed("user1") == False


def test_rate_limiter_ttl_cleanup():
    """Test that expired identifiers are cleaned up."""
    # Short cleanup interval for testing
    limiter = RateLimiter(max_requests=3, window_seconds=10, cleanup_interval=5)
    
    # Add requests for multiple identifiers
    limiter.is_allowed("user1")
    limiter.is_allowed("user2")
    limiter.is_allowed("user3")
    
    # Should have 3 identifiers
    assert len(limiter.requests) == 3
    
    # Mock time to simulate passage of time beyond window + cleanup interval
    with patch('time.time') as mock_time:
        # Set time 20 seconds in future (beyond window + cleanup interval)
        mock_time.return_value = time.time() + 20
        
        # This should trigger cleanup
        limiter.is_allowed("user4")
        
        # Old identifiers should be cleaned up, only user4 should remain
        assert len(limiter.requests) == 1
        assert "user4" in limiter.requests


def test_rate_limiter_partial_cleanup():
    """Test that only truly expired identifiers are cleaned up."""
    limiter = RateLimiter(max_requests=3, window_seconds=10, cleanup_interval=5)
    
    # Add old requests
    limiter.is_allowed("user1")
    limiter.is_allowed("user2")
    
    # Mock time to add recent request for user2
    with patch('time.time') as mock_time:
        # Set time 5 seconds in future
        mock_time.return_value = time.time() + 5
        limiter.is_allowed("user2")  # Recent request for user2
        
        # Set time 15 seconds in future (beyond cleanup interval)
        mock_time.return_value = time.time() + 15
        
        # This should trigger cleanup
        limiter.is_allowed("user3")
        
        # user1 should be cleaned (no recent requests), user2 and user3 should remain
        assert "user1" not in limiter.requests
        assert "user2" in limiter.requests
        assert "user3" in limiter.requests


def test_rate_limiter_cleanup_interval():
    """Test that cleanup only happens after interval passes."""
    limiter = RateLimiter(max_requests=3, window_seconds=5, cleanup_interval=10)
    
    limiter.is_allowed("user1")
    initial_time = limiter.last_cleanup
    
    # Mock time to be just before cleanup interval
    with patch('time.time') as mock_time:
        mock_time.return_value = initial_time + 9  # 1 second before interval
        
        limiter.is_allowed("user2")
        
        # Cleanup should not have happened yet
        assert limiter.last_cleanup == initial_time
        
        # Now set time beyond cleanup interval
        mock_time.return_value = initial_time + 11
        
        limiter.is_allowed("user3")
        
        # Cleanup should have happened
        assert limiter.last_cleanup > initial_time


def test_rate_limiter_force_cleanup():
    """Test the cleanup_now method."""
    limiter = RateLimiter(max_requests=3, window_seconds=5, cleanup_interval=3600)
    
    # Add some expired requests
    limiter.is_allowed("user1")
    limiter.is_allowed("user2")
    limiter.is_allowed("user3")
    
    # Mock time to make all requests expired
    with patch('time.time') as mock_time:
        mock_time.return_value = time.time() + 10  # Beyond window
        
        # Force cleanup
        cleaned_count = limiter.cleanup_now()
        
        # Should have cleaned all 3 identifiers
        assert cleaned_count == 3
        assert len(limiter.requests) == 0


def test_rate_limiter_stats():
    """Test the get_stats method."""
    limiter = RateLimiter(max_requests=5, window_seconds=60)
    
    # Add requests
    limiter.is_allowed("user1")
    limiter.is_allowed("user1")
    limiter.is_allowed("user2")
    
    stats = limiter.get_stats()
    
    assert stats["active_identifiers"] == 2
    assert stats["total_requests"] == 3
    assert stats["total_tracked_identifiers"] == 2


def test_rate_limiter_no_cleanup_when_not_needed():
    """Test that cleanup doesn't happen unnecessarily."""
    limiter = RateLimiter(max_requests=3, window_seconds=60, cleanup_interval=3600)
    
    limiter.is_allowed("user1")
    initial_cleanup_time = limiter.last_cleanup
    
    # Make another request immediately
    limiter.is_allowed("user1")
    
    # Cleanup time should not have changed
    assert limiter.last_cleanup == initial_cleanup_time


def test_rate_limiter_empty_requests_cleanup():
    """Test cleanup of identifiers with empty request lists."""
    limiter = RateLimiter(max_requests=3, window_seconds=5, cleanup_interval=1)
    
    # Manually add empty entry (simulating edge case)
    limiter.requests["empty_user"] = []
    limiter.is_allowed("user1")
    
    # Force time passage and cleanup
    with patch('time.time') as mock_time:
        mock_time.return_value = time.time() + 10
        
        limiter.is_allowed("user2")
        
        # Empty user should be cleaned up
        assert "empty_user" not in limiter.requests
        assert "user2" in limiter.requests


def test_rate_limiter_zero_cleanup_interval():
    """Test behavior with zero cleanup interval (cleanup every call)."""
    limiter = RateLimiter(max_requests=3, window_seconds=5, cleanup_interval=0)
    
    limiter.is_allowed("user1")
    
    # Mock time to make request expired
    with patch('time.time') as mock_time:
        mock_time.return_value = time.time() + 10
        
        # This should cleanup immediately due to zero interval
        limiter.is_allowed("user2")
        
        # user1 should be cleaned, only user2 should remain
        assert "user1" not in limiter.requests
        assert "user2" in limiter.requests
        assert len(limiter.requests) == 1


def test_rate_limiter_memory_efficiency():
    """Test that memory usage doesn't grow indefinitely."""
    limiter = RateLimiter(max_requests=2, window_seconds=1, cleanup_interval=1)
    
    # Simulate many different users over time
    with patch('time.time') as mock_time:
        current_time = time.time()
        
        for i in range(100):
            # Each iteration represents 2 seconds passing
            mock_time.return_value = current_time + (i * 2)
            limiter.is_allowed(f"user{i}")
        
        # Should have much fewer than 100 identifiers due to cleanup
        assert len(limiter.requests) < 10  # Should be very few due to aggressive cleanup


def test_rate_limiter_mixed_activity():
    """Test cleanup with mix of active and inactive identifiers."""
    limiter = RateLimiter(max_requests=5, window_seconds=10, cleanup_interval=5)
    
    # Add requests for multiple users
    for i in range(5):
        limiter.is_allowed(f"user{i}")
    
    assert len(limiter.requests) == 5
    
    # Mock time progression and keep some users active
    with patch('time.time') as mock_time:
        base_time = time.time()
        
        # 8 seconds later - keep user0 and user2 active
        mock_time.return_value = base_time + 8
        limiter.is_allowed("user0")
        limiter.is_allowed("user2")
        
        # 15 seconds later - should trigger cleanup
        mock_time.return_value = base_time + 15
        limiter.is_allowed("user_new")
        
        # Only user0, user2, and user_new should remain
        remaining_users = set(limiter.requests.keys())
        expected_users = {"user0", "user2", "user_new"}
        assert remaining_users == expected_users