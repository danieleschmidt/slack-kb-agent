#!/usr/bin/env python3
"""
Rate limiting module for Slack KB Agent.

Provides rate limiting functionality to prevent abuse of bot commands
and API endpoints.
"""

import logging
import os
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, NamedTuple, Optional

from .constants import EnvironmentConfig, RateLimitDefaults

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RateLimitResult(NamedTuple):
    """Result of rate limit check."""
    allowed: bool
    error_message: Optional[str] = None
    retry_after: Optional[int] = None  # seconds until retry allowed
    remaining_requests: Optional[int] = None


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    enabled: bool = True
    requests_per_minute: int = RateLimitDefaults.REQUESTS_PER_MINUTE
    requests_per_hour: int = RateLimitDefaults.REQUESTS_PER_HOUR
    requests_per_day: int = RateLimitDefaults.REQUESTS_PER_DAY
    burst_limit: int = RateLimitDefaults.BURST_LIMIT  # Allow short bursts above the per-minute limit
    cleanup_interval: int = RateLimitDefaults.CLEANUP_INTERVAL_SECONDS  # seconds between cleanup of old entries

    @classmethod
    def from_env(cls) -> 'RateLimitConfig':
        """Create RateLimitConfig from environment variables."""
        return cls(
            enabled=os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true",
            requests_per_minute=EnvironmentConfig.get_requests_per_minute(),
            requests_per_hour=EnvironmentConfig.get_requests_per_hour(),
            requests_per_day=int(os.getenv("RATE_LIMIT_REQUESTS_PER_DAY", str(RateLimitDefaults.REQUESTS_PER_DAY))),
            burst_limit=int(os.getenv("RATE_LIMIT_BURST_LIMIT", str(RateLimitDefaults.BURST_LIMIT))),
            cleanup_interval=int(os.getenv("RATE_LIMIT_CLEANUP_INTERVAL", str(RateLimitDefaults.CLEANUP_INTERVAL_SECONDS)))
        )


class TimeWindow:
    """Helper class to track requests in a time window."""

    def __init__(self, window_seconds: int):
        self.window_seconds = window_seconds
        self.requests: deque = deque()
        self.lock = threading.Lock()

    def add_request(self, timestamp: float = None) -> None:
        """Add a request timestamp."""
        if timestamp is None:
            timestamp = time.time()

        with self.lock:
            self.requests.append(timestamp)
            self._cleanup_old_requests()

    def get_request_count(self) -> int:
        """Get current request count in the window."""
        with self.lock:
            self._cleanup_old_requests()
            return len(self.requests)

    def _cleanup_old_requests(self) -> None:
        """Remove requests outside the time window."""
        cutoff = time.time() - self.window_seconds
        while self.requests and self.requests[0] < cutoff:
            self.requests.popleft()

    def time_until_next_allowed(self) -> int:
        """Get seconds until next request would be allowed."""
        if not self.requests:
            return 0

        oldest_request = self.requests[0]
        cutoff = time.time() - self.window_seconds

        if oldest_request <= cutoff:
            return 0

        return int(oldest_request - cutoff) + 1


class RateLimiter:
    """Rate limiter implementation with multiple time windows."""

    def __init__(self, config: RateLimitConfig):
        self.config = config

        # Track requests per user per time window
        self.minute_windows: Dict[str, TimeWindow] = defaultdict(lambda: TimeWindow(60))
        self.hour_windows: Dict[str, TimeWindow] = defaultdict(lambda: TimeWindow(3600))
        self.day_windows: Dict[str, TimeWindow] = defaultdict(lambda: TimeWindow(86400))

        # Burst tracking (shorter window for burst detection)
        self.burst_windows: Dict[str, TimeWindow] = defaultdict(lambda: TimeWindow(10))

        # Last cleanup time
        self.last_cleanup = time.time()
        self.cleanup_lock = threading.Lock()

    def check_rate_limit(self, user_id: str, request_type: str = "query") -> RateLimitResult:
        """Check if request is allowed under rate limits."""
        if not self.config.enabled:
            return RateLimitResult(allowed=True)

        # Cleanup old entries periodically
        self._periodic_cleanup()

        # Check burst limit (10-second window)
        burst_count = self.burst_windows[user_id].get_request_count()
        if burst_count >= self.config.burst_limit:
            retry_after = self.burst_windows[user_id].time_until_next_allowed()
            return RateLimitResult(
                allowed=False,
                error_message=f"Too many requests in quick succession. Please wait {retry_after} seconds.",
                retry_after=retry_after
            )

        # Check per-minute limit
        minute_count = self.minute_windows[user_id].get_request_count()
        if minute_count >= self.config.requests_per_minute:
            retry_after = self.minute_windows[user_id].time_until_next_allowed()
            return RateLimitResult(
                allowed=False,
                error_message=f"Rate limit exceeded: {self.config.requests_per_minute} requests per minute. Please wait {retry_after} seconds.",
                retry_after=retry_after,
                remaining_requests=0
            )

        # Check per-hour limit
        hour_count = self.hour_windows[user_id].get_request_count()
        if hour_count >= self.config.requests_per_hour:
            retry_after = self.hour_windows[user_id].time_until_next_allowed()
            return RateLimitResult(
                allowed=False,
                error_message=f"Hourly rate limit exceeded: {self.config.requests_per_hour} requests per hour. Please try again later.",
                retry_after=retry_after,
                remaining_requests=0
            )

        # Check per-day limit
        day_count = self.day_windows[user_id].get_request_count()
        if day_count >= self.config.requests_per_day:
            retry_after = self.day_windows[user_id].time_until_next_allowed()
            return RateLimitResult(
                allowed=False,
                error_message=f"Daily rate limit exceeded: {self.config.requests_per_day} requests per day. Please try again tomorrow.",
                retry_after=retry_after,
                remaining_requests=0
            )

        # Request is allowed - record it
        self.record_request(user_id, request_type)

        # Calculate remaining requests (use the most restrictive limit)
        remaining_minute = self.config.requests_per_minute - minute_count - 1
        remaining_hour = self.config.requests_per_hour - hour_count - 1
        remaining_day = self.config.requests_per_day - day_count - 1
        remaining = min(remaining_minute, remaining_hour, remaining_day)

        return RateLimitResult(
            allowed=True,
            remaining_requests=remaining
        )

    def record_request(self, user_id: str, request_type: str = "query") -> None:
        """Record a request for rate limiting purposes."""
        if not self.config.enabled:
            return

        timestamp = time.time()
        self.burst_windows[user_id].add_request(timestamp)
        self.minute_windows[user_id].add_request(timestamp)
        self.hour_windows[user_id].add_request(timestamp)
        self.day_windows[user_id].add_request(timestamp)

        logger.debug(f"Recorded {request_type} request for user {user_id}")

    def get_request_count(self, user_id: str, window: str = "minute") -> int:
        """Get current request count for a user in the specified window."""
        if window == "minute":
            return self.minute_windows[user_id].get_request_count()
        elif window == "hour":
            return self.hour_windows[user_id].get_request_count()
        elif window == "day":
            return self.day_windows[user_id].get_request_count()
        elif window == "burst":
            return self.burst_windows[user_id].get_request_count()
        else:
            raise ValueError(f"Unknown window: {window}")

    def reset_user_limits(self, user_id: str) -> None:
        """Reset rate limits for a specific user (admin function)."""
        if user_id in self.minute_windows:
            del self.minute_windows[user_id]
        if user_id in self.hour_windows:
            del self.hour_windows[user_id]
        if user_id in self.day_windows:
            del self.day_windows[user_id]
        if user_id in self.burst_windows:
            del self.burst_windows[user_id]

        logger.info(f"Reset rate limits for user {user_id}")

    def get_user_stats(self, user_id: str) -> Dict[str, int]:
        """Get rate limiting statistics for a user."""
        return {
            "requests_last_minute": self.get_request_count(user_id, "minute"),
            "requests_last_hour": self.get_request_count(user_id, "hour"),
            "requests_last_day": self.get_request_count(user_id, "day"),
            "burst_requests": self.get_request_count(user_id, "burst"),
            "limit_per_minute": self.config.requests_per_minute,
            "limit_per_hour": self.config.requests_per_hour,
            "limit_per_day": self.config.requests_per_day,
            "burst_limit": self.config.burst_limit
        }

    def _periodic_cleanup(self) -> None:
        """Periodically clean up old entries to prevent memory leaks."""
        now = time.time()

        with self.cleanup_lock:
            if now - self.last_cleanup < self.config.cleanup_interval:
                return

            self.last_cleanup = now

        # Cleanup is handled automatically by TimeWindow._cleanup_old_requests()
        # when get_request_count() is called, so we don't need to do anything here
        logger.debug("Performed periodic rate limiter cleanup")


class UserRateLimiter:
    """High-level rate limiter for user interactions."""

    def __init__(self, config: Optional[RateLimitConfig] = None):
        self.config = config or RateLimitConfig.from_env()
        self.rate_limiter = RateLimiter(self.config)

    def check_user_rate_limit(self, user_id: str, query: str) -> RateLimitResult:
        """Check rate limit for a user query."""
        # Different request types could have different limits in the future
        request_type = "query"

        # You could implement query complexity scoring here
        # For now, treat all queries the same

        return self.rate_limiter.check_rate_limit(user_id, request_type)

    def record_request(self, user_id: str, request_type: str = "query") -> None:
        """Record a user request."""
        self.rate_limiter.record_request(user_id, request_type)

    def is_user_blocked(self, user_id: str) -> bool:
        """Check if user is currently rate limited."""
        result = self.rate_limiter.check_rate_limit(user_id, "query")
        return not result.allowed

    def get_user_stats(self, user_id: str) -> Dict[str, int]:
        """Get user rate limiting statistics."""
        return self.rate_limiter.get_user_stats(user_id)


# Global rate limiter instance
_global_rate_limiter: Optional[RateLimiter] = None
_global_user_rate_limiter: Optional[UserRateLimiter] = None


def get_rate_limiter() -> RateLimiter:
    """Get global rate limiter instance."""
    global _global_rate_limiter

    if _global_rate_limiter is None:
        config = RateLimitConfig.from_env()
        _global_rate_limiter = RateLimiter(config)

    return _global_rate_limiter


def get_user_rate_limiter() -> UserRateLimiter:
    """Get global user rate limiter instance."""
    global _global_user_rate_limiter

    if _global_user_rate_limiter is None:
        config = RateLimitConfig.from_env()
        _global_user_rate_limiter = UserRateLimiter(config)

    return _global_user_rate_limiter


def check_rate_limit(user_id: str, request_type: str = "query") -> RateLimitResult:
    """Convenience function to check rate limit."""
    limiter = get_rate_limiter()
    return limiter.check_rate_limit(user_id, request_type)


def record_request(user_id: str, request_type: str = "query") -> None:
    """Convenience function to record request."""
    limiter = get_rate_limiter()
    limiter.record_request(user_id, request_type)
