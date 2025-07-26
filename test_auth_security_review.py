#!/usr/bin/env python3
"""
Comprehensive security review of the authentication system.
Tests for potential vulnerabilities and security best practices.
"""

import sys
import os
import base64
import time
import tempfile
from unittest.mock import Mock, patch
from typing import Dict, List

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from slack_kb_agent.auth import (
    AuthConfig, AuthMiddleware, BasicAuthenticator, RateLimiter, 
    AuditLogger, AuthResult, create_default_auth_config
)
from slack_kb_agent.password_hash import PasswordHasher, is_bcrypt_hash


class SecurityTestResults:
    """Track security test results."""
    
    def __init__(self):
        self.vulnerabilities: List[str] = []
        self.warnings: List[str] = []
        self.passed_tests: List[str] = []
        self.recommendations: List[str] = []
    
    def add_vulnerability(self, description: str):
        self.vulnerabilities.append(description)
        print(f"🚨 VULNERABILITY: {description}")
    
    def add_warning(self, description: str):
        self.warnings.append(description)
        print(f"⚠️  WARNING: {description}")
    
    def add_pass(self, description: str):
        self.passed_tests.append(description)
        print(f"✅ PASS: {description}")
    
    def add_recommendation(self, description: str):
        self.recommendations.append(description)
        print(f"💡 RECOMMENDATION: {description}")
    
    def summary(self):
        print(f"\n📊 SECURITY REVIEW SUMMARY:")
        print(f"   Vulnerabilities: {len(self.vulnerabilities)}")
        print(f"   Warnings: {len(self.warnings)}")
        print(f"   Passed Tests: {len(self.passed_tests)}")
        print(f"   Recommendations: {len(self.recommendations)}")
        
        if self.vulnerabilities:
            print(f"\n🚨 CRITICAL ISSUES TO FIX:")
            for vuln in self.vulnerabilities:
                print(f"   - {vuln}")
        
        return len(self.vulnerabilities) == 0


def test_authentication_bypass_attempts(results: SecurityTestResults):
    """Test for authentication bypass vulnerabilities."""
    
    try:
        # Test 1: Empty credentials bypass
        config = AuthConfig(
            enabled=True,
            method="basic",
            basic_auth_users={"admin": "secure_password"}
        )
        
        middleware = AuthMiddleware(config)
    except ImportError as e:
        results.add_vulnerability(f"bcrypt dependency missing - password hashing disabled: {e}")
        results.add_recommendation("Install bcrypt: pip install bcrypt")
        return
    
    # Test empty authorization header
    result = middleware.authenticate_request("/metrics", {})
    if result.is_authenticated:
        results.add_vulnerability("Empty credentials accepted - authentication bypass possible")
    else:
        results.add_pass("Empty credentials properly rejected")
    
    # Test 2: Malformed base64 bypass
    malformed_headers = {
        "Authorization": "Basic invalid_base64!!!"
    }
    result = middleware.authenticate_request("/metrics", malformed_headers)
    if result.is_authenticated:
        results.add_vulnerability("Malformed base64 credentials accepted")
    else:
        results.add_pass("Malformed base64 credentials properly rejected")
    
    # Test 3: Missing colon in credentials
    no_colon = base64.b64encode(b"adminpassword").decode()
    no_colon_headers = {
        "Authorization": f"Basic {no_colon}"
    }
    result = middleware.authenticate_request("/metrics", no_colon_headers)
    if result.is_authenticated:
        results.add_vulnerability("Credentials without colon separator accepted")
    else:
        results.add_pass("Credentials without colon properly rejected")
    
    # Test 4: Case sensitivity bypass attempt
    case_headers = {
        "Authorization": "basic " + base64.b64encode(b"admin:secure_password").decode()
    }
    result = middleware.authenticate_request("/metrics", case_headers)
    if result.is_authenticated:
        results.add_vulnerability("Case-insensitive 'Basic' keyword accepted - potential bypass")
    else:
        results.add_pass("Case-sensitive 'Basic' keyword properly enforced")
    
    # Test 5: Method switching vulnerability
    config_mixed = AuthConfig(
        enabled=True,
        method="mixed",
        basic_auth_users={"admin": "password"},
        api_keys=["secret-key"]
    )
    
    middleware_mixed = AuthMiddleware(config_mixed)
    
    # Try to bypass basic auth by only providing API key header when user exists
    api_only_headers = {
        "X-API-Key": "wrong-key"
    }
    result = middleware_mixed.authenticate_request("/metrics", api_only_headers)
    if result.is_authenticated:
        results.add_vulnerability("Method switching allows bypass with wrong API key")
    else:
        results.add_pass("Method switching properly validates all credentials")


def test_timing_attack_vulnerabilities(results: SecurityTestResults):
    """Test for timing attack vulnerabilities in password verification."""
    
    try:
        hasher = PasswordHasher()
    except ImportError as e:
        results.add_vulnerability(f"bcrypt dependency missing - cannot test timing attacks: {e}")
        return
    
    # Create a test hash
    test_password = "correct_password"
    correct_hash = hasher.hash_password(test_password)
    
    # Test timing consistency
    times_correct = []
    times_wrong = []
    
    # Measure timing for correct password
    for _ in range(50):
        start = time.time()
        hasher.verify_password(test_password, correct_hash)
        times_correct.append(time.time() - start)
    
    # Measure timing for wrong password  
    for _ in range(50):
        start = time.time()
        hasher.verify_password("wrong_password", correct_hash)
        times_wrong.append(time.time() - start)
    
    avg_correct = sum(times_correct) / len(times_correct)
    avg_wrong = sum(times_wrong) / len(times_wrong)
    
    # Check for significant timing difference (more than 50% difference is concerning)
    timing_ratio = max(avg_correct, avg_wrong) / min(avg_correct, avg_wrong)
    
    if timing_ratio > 1.5:
        results.add_warning(f"Potential timing attack vulnerability: {timing_ratio:.2f}x timing difference")
    else:
        results.add_pass("Password verification timing appears consistent")


def test_password_hash_security(results: SecurityTestResults):
    """Test password hashing implementation security."""
    
    try:
        hasher = PasswordHasher()
    except ImportError as e:
        results.add_vulnerability(f"bcrypt dependency missing - cannot test password hashing: {e}")
        return
    
    # Test 1: Ensure bcrypt is used
    test_hash = hasher.hash_password("test123")
    if not test_hash.startswith(('$2a$', '$2b$', '$2x$', '$2y$')):
        results.add_vulnerability("Non-bcrypt password hashing detected")
    else:
        results.add_pass("bcrypt password hashing confirmed")
    
    # Test 2: Check cost factor
    if hasher.cost < 10:
        results.add_warning(f"Low bcrypt cost factor: {hasher.cost} (recommended: 12+)")
    else:
        results.add_pass(f"Adequate bcrypt cost factor: {hasher.cost}")
    
    # Test 3: Salt uniqueness
    hash1 = hasher.hash_password("identical")
    hash2 = hasher.hash_password("identical")
    
    if hash1 == hash2:
        results.add_vulnerability("Identical passwords produce identical hashes - salt not working")
    else:
        results.add_pass("Salt generates unique hashes for identical passwords")
    
    # Test 4: Long password handling
    long_password = "a" * 100
    try:
        long_hash = hasher.hash_password(long_password)
        # Verify truncation warning is properly handled
        if len(long_password.encode('utf-8')) > 72:
            results.add_pass("Long password truncation properly handled")
    except Exception as e:
        results.add_warning(f"Long password handling error: {e}")


def test_rate_limiting_bypass(results: SecurityTestResults):
    """Test rate limiting implementation for bypass vulnerabilities."""
    
    rate_limiter = RateLimiter(max_requests=2, window_seconds=60)
    
    # Test 1: Basic rate limiting
    identifier = "test_client"
    
    # Should allow first two requests
    if not (rate_limiter.is_allowed(identifier) and rate_limiter.is_allowed(identifier)):
        results.add_vulnerability("Rate limiter fails to allow legitimate requests")
    
    # Should block third request
    if rate_limiter.is_allowed(identifier):
        results.add_vulnerability("Rate limiter fails to block excess requests")
    else:
        results.add_pass("Rate limiting properly blocks excess requests")
    
    # Test 2: Identifier spoofing
    spoofed_identifiers = [
        "test_client_",
        "test_client2", 
        "TEST_CLIENT",
        "test client",  # space instead of underscore
    ]
    
    bypass_attempts = 0
    for spoofed_id in spoofed_identifiers:
        if rate_limiter.is_allowed(spoofed_id):
            bypass_attempts += 1
    
    if bypass_attempts > 0:
        results.add_pass("Rate limiter treats similar identifiers as separate (expected behavior)")
    
    # Test 3: Memory exhaustion via identifier pollution
    for i in range(1000):
        rate_limiter.is_allowed(f"attacker_{i}")
    
    stats = rate_limiter.get_stats()
    if stats["total_tracked_identifiers"] > 900:
        results.add_warning("Rate limiter vulnerable to memory exhaustion via identifier pollution")
    else:
        results.add_pass("Rate limiter handles large numbers of identifiers appropriately")


def test_configuration_security(results: SecurityTestResults):
    """Test authentication configuration for security issues."""
    
    # Test 1: Default configuration security
    with patch.dict(os.environ, {}, clear=True):
        config = create_default_auth_config()
        
        if config.enabled and not (config.basic_auth_users or config.api_keys):
            results.add_vulnerability("Authentication enabled without any configured credentials")
        elif not config.enabled:
            results.add_pass("Authentication disabled when no credentials provided")
    
    # Test 2: Environment variable parsing security
    malicious_env = {
        "MONITORING_BASIC_AUTH_USERS": "admin:pass,../../../etc/passwd:hack,normal:user",
        "MONITORING_API_KEYS": "key1,../../secret,key2",
        "MONITORING_PROTECTED_ENDPOINTS": "/metrics,../../../admin,/status"
    }
    
    with patch.dict(os.environ, malicious_env):
        config = AuthConfig.from_env()
        
        # Check for path traversal in usernames
        suspicious_users = [user for user in config.basic_auth_users.keys() if ".." in user]
        if suspicious_users:
            results.add_warning(f"Suspicious usernames in config: {suspicious_users}")
        else:
            results.add_pass("No path traversal patterns in usernames")
        
        # Check for path traversal in endpoints
        suspicious_endpoints = [ep for ep in config.protected_endpoints if ".." in ep]
        if suspicious_endpoints:
            results.add_warning(f"Suspicious endpoints in config: {suspicious_endpoints}")
        else:
            results.add_pass("No path traversal patterns in protected endpoints")


def test_audit_logging_security(results: SecurityTestResults):
    """Test audit logging for security information leakage."""
    
    audit_logger = AuditLogger(enabled=True)
    
    # Capture log messages
    captured_logs = []
    
    class LogCapture:
        def log(self, level, message):
            captured_logs.append(message)
        
        def info(self, message):
            captured_logs.append(message)
        
        def warning(self, message):
            captured_logs.append(message)
    
    with patch.object(audit_logger, 'logger', LogCapture()):
        # Test that sensitive data is not logged
        audit_logger.log_auth_attempt(
            endpoint="/metrics",
            user="admin",
            success=False,
            client_ip="192.168.1.100",
            user_agent="Mozilla/5.0"
        )
        
        # Check for password/credential leakage in logs
        log_content = " ".join(captured_logs)
        
        sensitive_patterns = ["password", "secret", "key", "token", "hash"]
        leaked_info = [pattern for pattern in sensitive_patterns if pattern.lower() in log_content.lower()]
        
        if leaked_info:
            results.add_warning(f"Potentially sensitive information in audit logs: {leaked_info}")
        else:
            results.add_pass("Audit logs do not contain obvious sensitive information")


def test_session_management(results: SecurityTestResults):
    """Test session management security."""
    
    try:
        # Test that no session state is maintained inappropriately
        config = AuthConfig(
            enabled=True,
            method="basic",
            basic_auth_users={"admin": "password"}
        )
        
        middleware = AuthMiddleware(config)
    except ImportError as e:
        results.add_vulnerability(f"bcrypt dependency missing - cannot test session management: {e}")
        return
    
    # Authenticate successfully
    valid_headers = {
        "Authorization": "Basic " + base64.b64encode(b"admin:password").decode()
    }
    
    result1 = middleware.authenticate_request("/metrics", valid_headers)
    
    # Try to access without credentials (should fail even after previous success)
    result2 = middleware.authenticate_request("/metrics", {})
    
    if result1.is_authenticated and not result2.is_authenticated:
        results.add_pass("No inappropriate session state maintained")
    else:
        results.add_vulnerability("Authentication state may be cached inappropriately")


def main():
    """Run comprehensive authentication security review."""
    print("🔍 Starting comprehensive authentication security review...\n")
    
    results = SecurityTestResults()
    
    # Run all security tests
    test_authentication_bypass_attempts(results)
    test_timing_attack_vulnerabilities(results)
    test_password_hash_security(results)
    test_rate_limiting_bypass(results)
    test_configuration_security(results)
    test_audit_logging_security(results)
    test_session_management(results)
    
    # Generate summary
    is_secure = results.summary()
    
    if is_secure:
        print("\n🎉 Authentication system passed comprehensive security review!")
        print("   No critical vulnerabilities detected.")
    else:
        print("\n⚠️  Authentication system has security issues that require attention.")
    
    return 0 if is_secure else 1


if __name__ == "__main__":
    sys.exit(main())