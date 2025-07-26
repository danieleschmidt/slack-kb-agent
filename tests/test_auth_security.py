#!/usr/bin/env python3
"""
Comprehensive security tests for the authentication system.
Tests authentication bypass attempts, timing attacks, and security configurations.
"""

import pytest
import base64
import time
import os
from unittest.mock import Mock, patch, MagicMock
from typing import Dict

# Import the modules to test
from slack_kb_agent.auth import (
    AuthConfig, AuthMiddleware, BasicAuthenticator, RateLimiter, 
    AuditLogger, AuthResult, create_default_auth_config
)

try:
    from slack_kb_agent.password_hash import PasswordHasher, is_bcrypt_hash
    BCRYPT_AVAILABLE = True
except ImportError:
    BCRYPT_AVAILABLE = False


class TestAuthenticationSecurity:
    """Test authentication security and bypass attempts."""
    
    @pytest.mark.skipif(not BCRYPT_AVAILABLE, reason="bcrypt not available")
    def test_empty_credentials_rejected(self):
        """Test that empty credentials are properly rejected."""
        config = AuthConfig(
            enabled=True,
            method="basic",
            basic_auth_users={"admin": "secure_password"}
        )
        
        middleware = AuthMiddleware(config)
        
        # Test empty authorization header
        result = middleware.authenticate_request("/metrics", {})
        assert not result.is_authenticated
        assert "Authentication required" in result.error_message
    
    @pytest.mark.skipif(not BCRYPT_AVAILABLE, reason="bcrypt not available")
    def test_malformed_base64_rejected(self):
        """Test that malformed base64 credentials are rejected."""
        config = AuthConfig(
            enabled=True,
            method="basic", 
            basic_auth_users={"admin": "secure_password"}
        )
        
        middleware = AuthMiddleware(config)
        
        # Test malformed base64
        malformed_headers = {
            "Authorization": "Basic invalid_base64!!!"
        }
        result = middleware.authenticate_request("/metrics", malformed_headers)
        assert not result.is_authenticated
        assert "Invalid basic auth format" in result.error_message
    
    @pytest.mark.skipif(not BCRYPT_AVAILABLE, reason="bcrypt not available")
    def test_missing_colon_separator_rejected(self):
        """Test that credentials without colon separator are rejected."""
        config = AuthConfig(
            enabled=True,
            method="basic",
            basic_auth_users={"admin": "secure_password"}
        )
        
        middleware = AuthMiddleware(config)
        
        # Test missing colon in credentials
        no_colon = base64.b64encode(b"adminpassword").decode()
        no_colon_headers = {
            "Authorization": f"Basic {no_colon}"
        }
        result = middleware.authenticate_request("/metrics", no_colon_headers)
        assert not result.is_authenticated
        assert "Invalid basic auth format" in result.error_message
    
    @pytest.mark.skipif(not BCRYPT_AVAILABLE, reason="bcrypt not available")
    def test_case_sensitive_basic_keyword(self):
        """Test that 'Basic' keyword is case sensitive."""
        config = AuthConfig(
            enabled=True,
            method="basic",
            basic_auth_users={"admin": "secure_password"}
        )
        
        middleware = AuthMiddleware(config)
        
        # Test lowercase 'basic'
        case_headers = {
            "Authorization": "basic " + base64.b64encode(b"admin:secure_password").decode()
        }
        result = middleware.authenticate_request("/metrics", case_headers)
        assert not result.is_authenticated
        assert "Basic authentication required" in result.error_message
    
    @pytest.mark.skipif(not BCRYPT_AVAILABLE, reason="bcrypt not available")
    def test_valid_credentials_accepted(self):
        """Test that valid credentials are accepted."""
        config = AuthConfig(
            enabled=True,
            method="basic",
            basic_auth_users={"admin": "secure_password"}
        )
        
        middleware = AuthMiddleware(config)
        
        # Test valid credentials
        valid_auth = base64.b64encode(b"admin:secure_password").decode()
        valid_headers = {
            "Authorization": f"Basic {valid_auth}"
        }
        result = middleware.authenticate_request("/metrics", valid_headers)
        assert result.is_authenticated
        assert result.user == "admin"


class TestPasswordHashingSecurity:
    """Test password hashing security implementation."""
    
    @pytest.mark.skipif(not BCRYPT_AVAILABLE, reason="bcrypt not available")
    def test_bcrypt_format_used(self):
        """Test that bcrypt format is used for password hashing."""
        hasher = PasswordHasher()
        test_hash = hasher.hash_password("test123")
        
        assert test_hash.startswith(('$2a$', '$2b$', '$2x$', '$2y$'))
        assert is_bcrypt_hash(test_hash)
    
    @pytest.mark.skipif(not BCRYPT_AVAILABLE, reason="bcrypt not available")
    def test_adequate_cost_factor(self):
        """Test that adequate bcrypt cost factor is used."""
        hasher = PasswordHasher()
        
        # Default cost should be >= 10 for security
        assert hasher.cost >= 10
    
    @pytest.mark.skipif(not BCRYPT_AVAILABLE, reason="bcrypt not available")
    def test_salt_uniqueness(self):
        """Test that identical passwords produce different hashes due to salting."""
        hasher = PasswordHasher()
        
        hash1 = hasher.hash_password("identical")
        hash2 = hasher.hash_password("identical")
        
        # Hashes should be different due to unique salts
        assert hash1 != hash2
        
        # But both should verify correctly
        assert hasher.verify_password("identical", hash1)
        assert hasher.verify_password("identical", hash2)
    
    @pytest.mark.skipif(not BCRYPT_AVAILABLE, reason="bcrypt not available")
    def test_long_password_handling(self):
        """Test that long passwords are handled properly."""
        hasher = PasswordHasher()
        
        # bcrypt truncates at 72 bytes
        long_password = "a" * 100
        hash_result = hasher.hash_password(long_password)
        
        # Should still produce valid hash
        assert is_bcrypt_hash(hash_result)
        assert hasher.verify_password(long_password, hash_result)
    
    @pytest.mark.skipif(not BCRYPT_AVAILABLE, reason="bcrypt not available") 
    def test_password_verification_timing(self):
        """Test for potential timing attack vulnerabilities."""
        hasher = PasswordHasher(cost=4)  # Lower cost for faster testing
        
        test_password = "correct_password"
        correct_hash = hasher.hash_password(test_password)
        
        # Measure timing for correct password (small sample for CI efficiency)
        times_correct = []
        for _ in range(10):
            start = time.time()
            hasher.verify_password(test_password, correct_hash)
            times_correct.append(time.time() - start)
        
        # Measure timing for wrong password
        times_wrong = []
        for _ in range(10):
            start = time.time()
            hasher.verify_password("wrong_password", correct_hash)
            times_wrong.append(time.time() - start)
        
        avg_correct = sum(times_correct) / len(times_correct)
        avg_wrong = sum(times_wrong) / len(times_wrong)
        
        # Timing should be relatively consistent (allowing for system variance)
        timing_ratio = max(avg_correct, avg_wrong) / min(avg_correct, avg_wrong)
        
        # bcrypt should provide consistent timing
        assert timing_ratio < 2.0, f"Timing difference too large: {timing_ratio:.2f}x"


class TestRateLimitingSecurity:
    """Test rate limiting security and bypass attempts."""
    
    def test_basic_rate_limiting(self):
        """Test that rate limiting works correctly."""
        rate_limiter = RateLimiter(max_requests=2, window_seconds=60)
        
        identifier = "test_client"
        
        # Should allow first two requests
        assert rate_limiter.is_allowed(identifier)
        assert rate_limiter.is_allowed(identifier)
        
        # Should block third request
        assert not rate_limiter.is_allowed(identifier)
    
    def test_identifier_separation(self):
        """Test that different identifiers are treated separately."""
        rate_limiter = RateLimiter(max_requests=1, window_seconds=60)
        
        # Different identifiers should be independent
        assert rate_limiter.is_allowed("client1")
        assert rate_limiter.is_allowed("client2")
        
        # But same identifier should be limited
        assert not rate_limiter.is_allowed("client1")
    
    def test_memory_cleanup(self):
        """Test that rate limiter cleans up expired entries."""
        rate_limiter = RateLimiter(max_requests=10, window_seconds=1, cleanup_interval=1)
        
        # Add some identifiers
        for i in range(10):
            rate_limiter.is_allowed(f"client_{i}")
        
        initial_count = len(rate_limiter.requests)
        assert initial_count > 0
        
        # Wait for window to expire
        time.sleep(1.1)
        
        # Force cleanup
        cleaned = rate_limiter.cleanup_now()
        
        # Should have cleaned up expired entries
        assert cleaned > 0
        assert len(rate_limiter.requests) < initial_count
    
    def test_rate_limiter_stats(self):
        """Test rate limiter statistics functionality."""
        rate_limiter = RateLimiter(max_requests=5, window_seconds=60)
        
        # Add some requests
        for i in range(3):
            rate_limiter.is_allowed(f"client_{i}")
        
        stats = rate_limiter.get_stats()
        
        assert stats["active_identifiers"] == 3
        assert stats["total_requests"] == 3
        assert stats["total_tracked_identifiers"] == 3


class TestConfigurationSecurity:
    """Test authentication configuration security."""
    
    def test_default_config_security(self):
        """Test that default configuration is secure."""
        # Clear environment to test defaults
        with patch.dict(os.environ, {}, clear=True):
            config = create_default_auth_config()
            
            # Should be disabled if no credentials provided
            assert not config.enabled
    
    def test_environment_parsing(self):
        """Test environment variable parsing for security issues."""
        # Test with potentially malicious environment variables
        malicious_env = {
            "MONITORING_BASIC_AUTH_USERS": "admin:pass,normal:user",
            "MONITORING_API_KEYS": "key1,key2",
            "MONITORING_PROTECTED_ENDPOINTS": "/metrics,/status"
        }
        
        with patch.dict(os.environ, malicious_env):
            config = AuthConfig.from_env()
            
            # Should parse correctly without security issues
            assert len(config.basic_auth_users) == 2
            assert len(config.api_keys) == 2
            assert len(config.protected_endpoints) == 2
    
    def test_suspicious_config_patterns(self):
        """Test detection of suspicious configuration patterns."""
        suspicious_env = {
            "MONITORING_BASIC_AUTH_USERS": "../../../etc/passwd:hack",
            "MONITORING_PROTECTED_ENDPOINTS": "../../../admin"
        }
        
        with patch.dict(os.environ, suspicious_env):
            config = AuthConfig.from_env()
            
            # Config should still parse but values should be validated elsewhere
            assert "../../../etc/passwd" in config.basic_auth_users
            assert "../../../admin" in config.protected_endpoints


class TestAuditLogging:
    """Test audit logging security."""
    
    def test_audit_log_format(self):
        """Test that audit logs don't leak sensitive information."""
        audit_logger = AuditLogger(enabled=True)
        
        # Mock the logger to capture messages
        with patch.object(audit_logger, 'logger') as mock_logger:
            audit_logger.log_auth_attempt(
                endpoint="/metrics",
                user="admin", 
                success=False,
                client_ip="192.168.1.100",
                user_agent="TestAgent/1.0"
            )
            
            # Verify log was called
            mock_logger.log.assert_called_once()
            
            # Get the logged message
            call_args = mock_logger.log.call_args
            message = call_args[0][1]
            
            # Should not contain sensitive information like passwords
            assert "password" not in message.lower()
            assert "secret" not in message.lower()
            assert "hash" not in message.lower()
    
    def test_rate_limit_logging(self):
        """Test rate limit violation logging."""
        audit_logger = AuditLogger(enabled=True)
        
        with patch.object(audit_logger, 'logger') as mock_logger:
            audit_logger.log_rate_limit("malicious_client", "/metrics")
            
            mock_logger.warning.assert_called_once()
            
            # Verify the log contains necessary information
            call_args = mock_logger.warning.call_args
            message = call_args[0][0]
            
            assert "rate_limit_exceeded" in message
            assert "malicious_client" in message
            assert "/metrics" in message


class TestSessionManagement:
    """Test session management security."""
    
    @pytest.mark.skipif(not BCRYPT_AVAILABLE, reason="bcrypt not available")
    def test_stateless_authentication(self):
        """Test that authentication is stateless."""
        config = AuthConfig(
            enabled=True,
            method="basic",
            basic_auth_users={"admin": "password"}
        )
        
        middleware = AuthMiddleware(config)
        
        # Authenticate successfully
        valid_headers = {
            "Authorization": "Basic " + base64.b64encode(b"admin:password").decode()
        }
        
        result1 = middleware.authenticate_request("/metrics", valid_headers)
        assert result1.is_authenticated
        
        # Try to access without credentials (should fail)
        result2 = middleware.authenticate_request("/metrics", {})
        assert not result2.is_authenticated
        
        # No session state should be maintained
        assert result1.is_authenticated and not result2.is_authenticated


# Test that can be run even without bcrypt
class TestDependencyHandling:
    """Test handling of missing dependencies."""
    
    def test_bcrypt_dependency_detection(self):
        """Test that bcrypt dependency is properly detected."""
        # This test always runs to verify dependency handling
        try:
            import bcrypt
            bcrypt_installed = True
        except ImportError:
            bcrypt_installed = False
        
        # Test matches the actual availability
        assert BCRYPT_AVAILABLE == bcrypt_installed
    
    def test_graceful_degradation_without_bcrypt(self):
        """Test graceful handling when bcrypt is not available."""
        if BCRYPT_AVAILABLE:
            pytest.skip("bcrypt is available, cannot test graceful degradation")
        
        # Should handle missing bcrypt gracefully
        with pytest.raises(ImportError, match="bcrypt library is required"):
            from slack_kb_agent.password_hash import PasswordHasher
            PasswordHasher()


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])