#!/usr/bin/env python3
"""
Authentication and authorization module for Slack KB Agent.

Provides authentication middleware for monitoring endpoints and
access control for bot commands.
"""

import os
import hashlib
import hmac
import base64
import logging
import time
from typing import Dict, List, Optional, NamedTuple

from .password_hash import PasswordHasher, is_bcrypt_hash
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AuthResult(NamedTuple):
    """Result of authentication attempt."""
    is_authenticated: bool
    user: Optional[str] = None
    error_message: Optional[str] = None
    requires_audit: bool = False


@dataclass
class AuthConfig:
    """Configuration for authentication system."""
    enabled: bool = True
    method: str = "basic"  # "basic", "api_key", or "mixed"
    basic_auth_users: Dict[str, str] = field(default_factory=dict)
    api_keys: List[str] = field(default_factory=list)
    protected_endpoints: List[str] = field(default_factory=lambda: ["/metrics", "/status", "/health"])
    rate_limit_requests: int = 100
    rate_limit_window: int = 3600  # seconds
    audit_enabled: bool = True
    
    @classmethod
    def from_env(cls) -> 'AuthConfig':
        """Create AuthConfig from environment variables."""
        enabled = os.getenv("MONITORING_AUTH_ENABLED", "false").lower() == "true"
        method = os.getenv("MONITORING_AUTH_METHOD", "basic")
        
        # Parse basic auth users (format: "user1:pass1,user2:pass2")
        basic_auth_str = os.getenv("MONITORING_BASIC_AUTH_USERS", "")
        basic_auth_users = {}
        if basic_auth_str:
            for user_pass in basic_auth_str.split(","):
                if ":" in user_pass:
                    user, password = user_pass.strip().split(":", 1)
                    basic_auth_users[user] = password
        
        # Parse API keys (format: "key1,key2,key3")
        api_keys_str = os.getenv("MONITORING_API_KEYS", "")
        api_keys = [key.strip() for key in api_keys_str.split(",") if key.strip()]
        
        # Parse protected endpoints
        endpoints_str = os.getenv("MONITORING_PROTECTED_ENDPOINTS", "/metrics,/status,/health")
        protected_endpoints = [ep.strip() for ep in endpoints_str.split(",") if ep.strip()]
        
        return cls(
            enabled=enabled,
            method=method,
            basic_auth_users=basic_auth_users,
            api_keys=api_keys,
            protected_endpoints=protected_endpoints,
            rate_limit_requests=int(os.getenv("MONITORING_RATE_LIMIT_REQUESTS", "100")),
            rate_limit_window=int(os.getenv("MONITORING_RATE_LIMIT_WINDOW", "3600")),
            audit_enabled=os.getenv("MONITORING_AUDIT_ENABLED", "true").lower() == "true"
        )


class RateLimiter:
    """Simple in-memory rate limiter with TTL cleanup."""
    
    def __init__(self, max_requests: int, window_seconds: int, cleanup_interval: int = 3600):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.cleanup_interval = cleanup_interval  # Seconds between cleanup runs
        self.requests: Dict[str, List[float]] = {}
        self.last_cleanup = time.time()
    
    def is_allowed(self, identifier: str) -> bool:
        """Check if request is allowed for the given identifier."""
        now = time.time()
        
        # Perform periodic cleanup to prevent memory growth
        self._cleanup_if_needed(now)
        
        window_start = now - self.window_seconds
        
        # Clean old requests for this identifier
        if identifier in self.requests:
            self.requests[identifier] = [
                req_time for req_time in self.requests[identifier]
                if req_time > window_start
            ]
        else:
            self.requests[identifier] = []
        
        # Check if under limit
        if len(self.requests[identifier]) < self.max_requests:
            self.requests[identifier].append(now)
            # Update metrics periodically (not on every request to avoid overhead)
            if len(self.requests[identifier]) == 1:  # New identifier added
                self._update_memory_metrics()
            return True
        
        return False

    def _cleanup_if_needed(self, now: float) -> None:
        """Perform cleanup if enough time has passed since last cleanup."""
        if now - self.last_cleanup >= self.cleanup_interval:
            self._cleanup_expired_identifiers(now)
            self.last_cleanup = now

    def _cleanup_expired_identifiers(self, now: float) -> None:
        """Remove identifiers with no recent requests (TTL cleanup)."""
        window_start = now - self.window_seconds
        expired_identifiers = []
        
        for identifier, request_times in self.requests.items():
            # Filter out expired requests
            active_requests = [
                req_time for req_time in request_times 
                if req_time > window_start
            ]
            
            if active_requests:
                # Update with only active requests
                self.requests[identifier] = active_requests
            else:
                # Mark for removal if no active requests
                expired_identifiers.append(identifier)
        
        # Remove expired identifiers
        for identifier in expired_identifiers:
            del self.requests[identifier]
        
        if expired_identifiers:
            logger.info(f"Rate limiter cleaned up {len(expired_identifiers)} expired identifiers")
            
        # Update memory metrics after cleanup
        self._update_memory_metrics()

    def cleanup_now(self) -> int:
        """Force immediate cleanup and return number of cleaned identifiers."""
        now = time.time()
        before_count = len(self.requests)
        self._cleanup_expired_identifiers(now)
        self.last_cleanup = now
        cleaned_count = before_count - len(self.requests)
        return cleaned_count

    def get_stats(self) -> Dict[str, int]:
        """Get current rate limiter statistics."""
        now = time.time()
        active_identifiers = 0
        total_requests = 0
        
        for request_times in self.requests.values():
            if request_times:
                active_identifiers += 1
                total_requests += len(request_times)
        
        return {
            "active_identifiers": active_identifiers,
            "total_requests": total_requests,
            "total_tracked_identifiers": len(self.requests)
        }

    def _update_memory_metrics(self) -> None:
        """Update memory usage metrics for monitoring."""
        try:
            # Import here to avoid circular imports
            from .monitoring import get_global_metrics
            metrics = get_global_metrics()
            
            stats = self.get_stats()
            metrics.set_gauge("rate_limiter_active_identifiers", stats["active_identifiers"])
            metrics.set_gauge("rate_limiter_total_requests", stats["total_requests"])
            metrics.set_gauge("rate_limiter_tracked_identifiers", stats["total_tracked_identifiers"])
            
        except (ImportError, AttributeError) as e:
            # Metrics module may not be available or properly configured
            logger.debug(f"Metrics collection unavailable for rate limiter: {type(e).__name__}: {e}")
        except Exception as e:
            # Log unexpected errors but don't crash the application
            logger.warning(f"Unexpected error updating rate limiter metrics: {type(e).__name__}: {e}")


class AuditLogger:
    """Audit logger for security events."""
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.logger = logging.getLogger(f"{__name__}.audit")
    
    def log_auth_attempt(self, 
                        endpoint: str, 
                        user: Optional[str], 
                        success: bool, 
                        client_ip: str = "unknown",
                        user_agent: str = "unknown"):
        """Log authentication attempt."""
        if not self.enabled:
            return
        
        event = {
            "event_type": "auth_attempt",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "endpoint": endpoint,
            "user": user or "anonymous",
            "success": success,
            "client_ip": client_ip,
            "user_agent": user_agent
        }
        
        level = logging.INFO if success else logging.WARNING
        self.logger.log(level, f"Auth attempt: {event}")
    
    def log_rate_limit(self, identifier: str, endpoint: str):
        """Log rate limit violation."""
        if not self.enabled:
            return
        
        event = {
            "event_type": "rate_limit_exceeded",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "identifier": identifier,
            "endpoint": endpoint
        }
        
        self.logger.warning(f"Rate limit exceeded: {event}")


class BasicAuthenticator:
    """Basic authentication handler with secure password hashing."""
    
    def __init__(self, config: AuthConfig):
        self.config = config
        self.password_hasher = PasswordHasher()
        
        # Store users with hashed passwords
        self.users: Dict[str, str] = {}
        self.default_password_hash: Optional[str] = None
        
        # Hash and store user passwords
        if config.basic_auth_users:
            for username, password in config.basic_auth_users.items():
                if is_bcrypt_hash(password):
                    self.users[username] = password
                else:
                    self.users[username] = self.password_hasher.hash_password(password)
        
        # Handle default password (used for 'admin' user)
        default_password = getattr(config, 'basic_password', None)
        if default_password:
            if is_bcrypt_hash(default_password):
                self.default_password_hash = default_password
            else:
                self.default_password_hash = self.password_hasher.hash_password(default_password)
    
    def verify_basic_auth(self, username: str, password: str) -> bool:
        """Verify basic authentication credentials."""
        # Check named users
        if username in self.users:
            return self.password_hasher.verify_password(password, self.users[username])
        
        # Check default password for 'admin' user
        if username == "admin" and self.default_password_hash:
            return self.password_hasher.verify_password(password, self.default_password_hash)
        
        return False


class AuthMiddleware:
    """Authentication middleware for HTTP requests."""
    
    def __init__(self, config: AuthConfig):
        self.config = config
        self.rate_limiter = RateLimiter(config.rate_limit_requests, config.rate_limit_window)
        self.audit_logger = AuditLogger(config.audit_enabled)
        
        # Initialize password hasher
        self.password_hasher = PasswordHasher()
        
        # Hash any plaintext passwords for security
        self._hash_plaintext_passwords()
    
    def authenticate_request(self, 
                           path: str, 
                           headers: Dict[str, str],
                           client_ip: str = "unknown") -> AuthResult:
        """Authenticate incoming HTTP request."""
        
        # If auth is disabled, allow all requests
        if not self.config.enabled:
            return AuthResult(is_authenticated=True)
        
        # Check if endpoint needs protection
        if not self._is_protected_endpoint(path):
            return AuthResult(is_authenticated=True)
        
        # Check rate limiting first
        identifier = self._get_client_identifier(headers, client_ip)
        if not self.rate_limiter.is_allowed(identifier):
            self.audit_logger.log_rate_limit(identifier, path)
            return AuthResult(
                is_authenticated=False,
                error_message="Rate limit exceeded. Please try again later.",
                requires_audit=True
            )
        
        # Perform authentication
        if self.config.method in ["basic", "mixed"]:
            result = self._authenticate_basic(headers)
            if result.is_authenticated:
                self.audit_logger.log_auth_attempt(
                    path, result.user, True, client_ip, 
                    headers.get("User-Agent", "unknown")
                )
                return result
        
        if self.config.method in ["api_key", "mixed"]:
            result = self._authenticate_api_key(headers)
            if result.is_authenticated:
                self.audit_logger.log_auth_attempt(
                    path, result.user, True, client_ip,
                    headers.get("User-Agent", "unknown")
                )
                return result
        
        # Authentication failed
        self.audit_logger.log_auth_attempt(
            path, None, False, client_ip,
            headers.get("User-Agent", "unknown")
        )
        
        return AuthResult(
            is_authenticated=False,
            error_message="Authentication required. Use Basic auth or API key.",
            requires_audit=True
        )
    
    def _is_protected_endpoint(self, path: str) -> bool:
        """Check if endpoint requires authentication."""
        return any(path.startswith(endpoint) for endpoint in self.config.protected_endpoints)
    
    def _get_client_identifier(self, headers: Dict[str, str], client_ip: str) -> str:
        """Get unique identifier for rate limiting."""
        # Use X-Forwarded-For if available, otherwise use provided IP
        forwarded_for = headers.get("X-Forwarded-For", "")
        if forwarded_for:
            client_ip = forwarded_for.split(",")[0].strip()
        
        return f"{client_ip}_{headers.get('User-Agent', 'unknown')}"
    
    def _authenticate_basic(self, headers: Dict[str, str]) -> AuthResult:
        """Authenticate using basic auth."""
        auth_header = headers.get("Authorization", "")
        
        if not auth_header.startswith("Basic "):
            return AuthResult(
                is_authenticated=False,
                error_message="Basic authentication required"
            )
        
        try:
            # Decode base64 credentials
            auth_data = auth_header[6:]  # Remove "Basic "
            decoded = base64.b64decode(auth_data).decode('utf-8')
            username, password = decoded.split(":", 1)
            
            # Check credentials
            if (username in self.config.basic_auth_users and 
                self._verify_password(password, self.config.basic_auth_users[username])):
                return AuthResult(is_authenticated=True, user=username)
            
            return AuthResult(
                is_authenticated=False,
                error_message="Invalid username or password"
            )
            
        except (ValueError, UnicodeDecodeError):
            return AuthResult(
                is_authenticated=False,
                error_message="Invalid basic auth format"
            )
    
    def _authenticate_api_key(self, headers: Dict[str, str]) -> AuthResult:
        """Authenticate using API key."""
        # Try different case variations of the API key header
        api_key = (headers.get("X-API-Key", "") or 
                  headers.get("X-Api-Key", "") or
                  headers.get("x-api-key", ""))
        
        if not api_key:
            return AuthResult(
                is_authenticated=False,
                error_message="API key required in X-API-Key header"
            )
        
        if api_key in self.config.api_keys:
            return AuthResult(is_authenticated=True, user="api_key_user")
        
        return AuthResult(
            is_authenticated=False,
            error_message="Invalid API key"
        )
    
    def _verify_password(self, provided: str, stored: str) -> bool:
        """Verify password against hash using secure comparison."""
        return self.password_hasher.verify_password(provided, stored)
    
    def _hash_plaintext_passwords(self) -> None:
        """Hash any plaintext passwords in the configuration."""
        # Hash basic_auth_users passwords if they aren't already hashed
        if self.config.basic_auth_users:
            for username, password in list(self.config.basic_auth_users.items()):
                if not is_bcrypt_hash(password):
                    hashed = self.password_hasher.hash_password(password)
                    self.config.basic_auth_users[username] = hashed
                    logger.info(f"Migrated plaintext password to secure hash for user: {username}")


def create_default_auth_config() -> AuthConfig:
    """Create default authentication configuration."""
    # Use environment variables for all credentials - no hardcoded defaults
    default_api_key = os.getenv("MONITORING_DEFAULT_API_KEY")
    default_username = os.getenv("MONITORING_DEFAULT_USERNAME", "admin")
    default_password = os.getenv("MONITORING_DEFAULT_PASSWORD")
    
    # Only create basic auth if both username and password are provided
    basic_auth_users = {}
    if default_username and default_password:
        basic_auth_users[default_username] = default_password
    
    # Only include API key if provided
    api_keys = []
    if default_api_key:
        api_keys.append(default_api_key)
    
    # If no credentials are provided, disable auth by default (safer than hardcoded creds)
    enabled = bool(basic_auth_users or api_keys)
    
    if not enabled:
        logger.warning(
            "No authentication credentials provided via environment variables. "
            "Authentication is DISABLED. Set MONITORING_DEFAULT_API_KEY or "
            "MONITORING_DEFAULT_USERNAME/MONITORING_DEFAULT_PASSWORD to enable."
        )
    
    return AuthConfig(
        enabled=enabled,
        method="mixed" if (basic_auth_users and api_keys) else "basic" if basic_auth_users else "api_key",
        basic_auth_users=basic_auth_users,
        api_keys=api_keys,
        protected_endpoints=["/metrics", "/status", "/health", "/metrics.json"]
    )


# Global auth instance (initialized lazily)
_auth_middleware: Optional[AuthMiddleware] = None


def get_auth_middleware() -> AuthMiddleware:
    """Get global authentication middleware instance."""
    global _auth_middleware
    
    if _auth_middleware is None:
        config = AuthConfig.from_env()
        _auth_middleware = AuthMiddleware(config)
    
    return _auth_middleware


def require_auth(func):
    """Decorator to require authentication for a function."""
    def wrapper(*args, **kwargs):
        # This decorator can be used for future auth requirements
        return func(*args, **kwargs)
    return wrapper