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
    """Simple in-memory rate limiter."""
    
    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: Dict[str, List[float]] = {}
    
    def is_allowed(self, identifier: str) -> bool:
        """Check if request is allowed for the given identifier."""
        now = time.time()
        window_start = now - self.window_seconds
        
        # Clean old requests
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
            return True
        
        return False


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


class AuthMiddleware:
    """Authentication middleware for HTTP requests."""
    
    def __init__(self, config: AuthConfig):
        self.config = config
        self.rate_limiter = RateLimiter(config.rate_limit_requests, config.rate_limit_window)
        self.audit_logger = AuditLogger(config.audit_enabled)
    
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
        api_key = headers.get("X-API-Key", "")
        
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
        """Verify password with timing-safe comparison."""
        # For production, passwords should be hashed
        # This is a simple implementation for the MVP
        return hmac.compare_digest(provided, stored)


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