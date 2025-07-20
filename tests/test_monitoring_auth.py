#!/usr/bin/env python3
"""
Test suite for monitoring server authentication middleware.

Tests authentication and authorization for monitoring endpoints.
"""

import os
import json
import base64
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from slack_kb_agent.auth import AuthMiddleware, AuthConfig
from slack_kb_agent.monitoring import MonitoringConfig


class TestMonitoringAuth(unittest.TestCase):
    """Test authentication for monitoring endpoints."""
    
    def setUp(self):
        """Set up test environment."""
        # Test credentials (safe for testing - never use in production)
        self.test_username = "test_user"
        self.test_password = "test_password_123"  # Test-only credential
        self.test_api_key = "test_api_key_abc123"  # Test-only credential
        
        # Set up auth config
        self.auth_config = AuthConfig(
            enabled=True,
            method="basic",  # or "api_key"
            basic_auth_users={self.test_username: self.test_password},
            api_keys=[self.test_api_key],
            protected_endpoints=["/metrics", "/status", "/health"]
        )
    
    def test_auth_config_creation(self):
        """Test AuthConfig creation and validation."""
        self.assertTrue(self.auth_config.enabled)
        self.assertEqual(self.auth_config.method, "basic")
        self.assertIn(self.test_username, self.auth_config.basic_auth_users)
        self.assertIn("/metrics", self.auth_config.protected_endpoints)
    
    def test_auth_config_from_env(self):
        """Test AuthConfig creation from environment variables."""
        with patch.dict(os.environ, {
            'MONITORING_AUTH_ENABLED': 'true',
            'MONITORING_AUTH_METHOD': 'api_key',
            'MONITORING_API_KEYS': 'key1,key2,key3',
            'MONITORING_BASIC_AUTH_USERS': 'user1:pass1,user2:pass2'
        }):
            config = AuthConfig.from_env()
            self.assertTrue(config.enabled)
            self.assertEqual(config.method, "api_key")
            self.assertEqual(len(config.api_keys), 3)
            self.assertEqual(len(config.basic_auth_users), 2)
    
    def test_auth_middleware_basic_auth_valid(self):
        """Test basic auth with valid credentials."""
        middleware = AuthMiddleware(self.auth_config)
        
        # Create mock request with valid basic auth
        auth_string = f"{self.test_username}:{self.test_password}"
        auth_bytes = base64.b64encode(auth_string.encode()).decode()
        headers = {"Authorization": f"Basic {auth_bytes}"}
        
        # Test authentication
        result = middleware.authenticate_request("/metrics", headers)
        self.assertTrue(result.is_authenticated)
        self.assertEqual(result.user, self.test_username)
        self.assertIsNone(result.error_message)
    
    def test_auth_middleware_basic_auth_invalid(self):
        """Test basic auth with invalid credentials."""
        middleware = AuthMiddleware(self.auth_config)
        
        # Create mock request with invalid basic auth
        auth_string = "invalid_user:invalid_password"
        auth_bytes = base64.b64encode(auth_string.encode()).decode()
        headers = {"Authorization": f"Basic {auth_bytes}"}
        
        # Test authentication
        result = middleware.authenticate_request("/metrics", headers)
        self.assertFalse(result.is_authenticated)
        self.assertIsNotNone(result.error_message)
    
    def test_auth_middleware_api_key_valid(self):
        """Test API key authentication with valid key."""
        config = AuthConfig(
            enabled=True,
            method="api_key",
            api_keys=[self.test_api_key],
            protected_endpoints=["/metrics"]
        )
        middleware = AuthMiddleware(config)
        
        headers = {"X-API-Key": self.test_api_key}
        result = middleware.authenticate_request("/metrics", headers)
        
        self.assertTrue(result.is_authenticated)
        self.assertEqual(result.user, "api_key_user")
    
    def test_auth_middleware_api_key_invalid(self):
        """Test API key authentication with invalid key."""
        config = AuthConfig(
            enabled=True,
            method="api_key", 
            api_keys=[self.test_api_key],
            protected_endpoints=["/metrics"]
        )
        middleware = AuthMiddleware(config)
        
        headers = {"X-API-Key": "invalid_key"}
        result = middleware.authenticate_request("/metrics", headers)
        
        self.assertFalse(result.is_authenticated)
        self.assertIsNotNone(result.error_message)
    
    def test_auth_middleware_missing_auth(self):
        """Test request without authentication headers."""
        middleware = AuthMiddleware(self.auth_config)
        
        headers = {}
        result = middleware.authenticate_request("/metrics", headers)
        
        self.assertFalse(result.is_authenticated)
        self.assertIn("Authentication required", result.error_message)
    
    def test_auth_middleware_unprotected_endpoint(self):
        """Test that unprotected endpoints don't require auth."""
        middleware = AuthMiddleware(self.auth_config)
        
        headers = {}
        result = middleware.authenticate_request("/unprotected", headers)
        
        self.assertTrue(result.is_authenticated)  # Should pass for unprotected endpoints
    
    def test_auth_disabled(self):
        """Test that authentication is bypassed when disabled."""
        config = AuthConfig(enabled=False)
        middleware = AuthMiddleware(config)
        
        headers = {}
        result = middleware.authenticate_request("/metrics", headers)
        
        self.assertTrue(result.is_authenticated)  # Should pass when auth is disabled


class TestMonitoringServerWithAuth(unittest.TestCase):
    """Integration tests for monitoring server with authentication."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test server."""
        # This would set up a test monitoring server with auth
        # For now, we'll just test the auth middleware in isolation
        pass
    
    def test_integration_placeholder(self):
        """Placeholder for integration tests."""
        # TODO: Add integration tests with actual HTTP server
        # This would test:
        # 1. Unauthenticated requests return 401
        # 2. Valid auth returns metrics
        # 3. Rate limiting works
        # 4. Audit logging works
        pass


if __name__ == "__main__":
    unittest.main()