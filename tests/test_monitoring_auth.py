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
    
    def test_http_server_unauthenticated_request(self):
        """Test that unauthenticated requests to protected endpoints return 401."""
        import threading
        import time
        import requests
        from http.server import HTTPServer, BaseHTTPRequestHandler
        from slack_kb_agent.monitoring import get_monitoring_endpoints
        from slack_kb_agent.auth import AuthMiddleware, AuthConfig
        
        # Configure auth
        auth_config = AuthConfig(
            enabled=True,
            method="basic",
            basic_username="admin",
            basic_password="secret123",
            protected_endpoints=["/metrics", "/health"]
        )
        auth_middleware = AuthMiddleware(auth_config)
        
        # Get monitoring endpoints
        endpoints = get_monitoring_endpoints()
        
        class TestHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                # Apply authentication middleware
                headers = dict(self.headers)
                auth_result = auth_middleware.authenticate_request(
                    self.path, headers, self.client_address[0]
                )
                
                if not auth_result.is_authenticated:
                    self.send_response(401)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(b'{"error": "Unauthorized"}')
                    return
                
                # Handle endpoint
                if self.path in endpoints:
                    handler = endpoints[self.path]
                    try:
                        response, status_code, headers_dict = handler()
                        self.send_response(status_code)
                        for header, value in headers_dict.items():
                            self.send_header(header, value)
                        self.end_headers()
                        self.wfile.write(response.encode())
                    except Exception as e:
                        self.send_response(500)
                        self.send_header('Content-Type', 'application/json')
                        self.end_headers()
                        self.wfile.write(f'{{"error": "{str(e)}"}}'.encode())
                else:
                    self.send_response(404)
                    self.end_headers()
            
            def log_message(self, format, *args):
                # Suppress server logs during testing
                pass
        
        # Start test server
        server = HTTPServer(("localhost", 0), TestHandler)
        port = server.server_address[1]
        server_thread = threading.Thread(target=server.serve_forever)
        server_thread.daemon = True
        server_thread.start()
        
        try:
            # Wait for server to start
            time.sleep(0.1)
            
            # Test unauthenticated request
            response = requests.get(f"http://localhost:{port}/metrics", timeout=5)
            self.assertEqual(response.status_code, 401)
            self.assertIn("error", response.json())
            
        finally:
            server.shutdown()
            server_thread.join(timeout=1)

    def test_http_server_authenticated_request(self):
        """Test that authenticated requests to protected endpoints return data."""
        import threading
        import time
        import requests
        from http.server import HTTPServer, BaseHTTPRequestHandler
        from slack_kb_agent.monitoring import get_monitoring_endpoints
        from slack_kb_agent.auth import AuthMiddleware, AuthConfig
        import base64
        
        # Configure auth
        auth_config = AuthConfig(
            enabled=True,
            method="basic",
            basic_username="admin",
            basic_password="secret123",
            protected_endpoints=["/metrics", "/health"]
        )
        auth_middleware = AuthMiddleware(auth_config)
        
        # Get monitoring endpoints
        endpoints = get_monitoring_endpoints()
        
        class TestHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                # Apply authentication middleware
                headers = dict(self.headers)
                auth_result = auth_middleware.authenticate_request(
                    self.path, headers, self.client_address[0]
                )
                
                if not auth_result.is_authenticated:
                    self.send_response(401)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(b'{"error": "Unauthorized"}')
                    return
                
                # Handle endpoint
                if self.path in endpoints:
                    handler = endpoints[self.path]
                    try:
                        response, status_code, headers_dict = handler()
                        self.send_response(status_code)
                        for header, value in headers_dict.items():
                            self.send_header(header, value)
                        self.end_headers()
                        self.wfile.write(response.encode())
                    except Exception as e:
                        self.send_response(500)
                        self.send_header('Content-Type', 'application/json')
                        self.end_headers()
                        self.wfile.write(f'{{"error": "{str(e)}"}}'.encode())
                else:
                    self.send_response(404)
                    self.end_headers()
            
            def log_message(self, format, *args):
                # Suppress server logs during testing
                pass
        
        # Start test server
        server = HTTPServer(("localhost", 0), TestHandler)
        port = server.server_address[1]
        server_thread = threading.Thread(target=server.serve_forever)
        server_thread.daemon = True
        server_thread.start()
        
        try:
            # Wait for server to start
            time.sleep(0.1)
            
            # Test authenticated request
            credentials = base64.b64encode(b"admin:secret123").decode('ascii')
            headers = {"Authorization": f"Basic {credentials}"}
            
            response = requests.get(f"http://localhost:{port}/metrics", headers=headers, timeout=5)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.headers['Content-Type'], 'text/plain')
            
            # Test health endpoint
            response = requests.get(f"http://localhost:{port}/health", headers=headers, timeout=5)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.headers['Content-Type'], 'application/json')
            health_data = response.json()
            self.assertIn("status", health_data)
            self.assertIn("timestamp", health_data)
            
        finally:
            server.shutdown()
            server_thread.join(timeout=1)

    def test_http_server_rate_limiting(self):
        """Test that rate limiting works with HTTP server."""
        import threading
        import time
        import requests
        from http.server import HTTPServer, BaseHTTPRequestHandler
        from slack_kb_agent.monitoring import get_monitoring_endpoints
        from slack_kb_agent.auth import AuthMiddleware, AuthConfig
        import base64
        
        # Configure auth with low rate limits
        auth_config = AuthConfig(
            enabled=True,
            method="basic",
            basic_username="admin",
            basic_password="secret123",
            protected_endpoints=["/metrics"],
            rate_limit_requests=2,  # Very low limit
            rate_limit_window=60
        )
        auth_middleware = AuthMiddleware(auth_config)
        
        # Get monitoring endpoints
        endpoints = get_monitoring_endpoints()
        
        class TestHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                # Apply authentication middleware
                headers = dict(self.headers)
                auth_result = auth_middleware.authenticate_request(
                    self.path, headers, self.client_address[0]
                )
                
                if not auth_result.is_authenticated:
                    self.send_response(429 if "rate limit" in auth_result.error_message.lower() else 401)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    error_msg = auth_result.error_message or "Unauthorized"
                    self.wfile.write(f'{{"error": "{error_msg}"}}'.encode())
                    return
                
                # Handle endpoint
                if self.path in endpoints:
                    handler = endpoints[self.path]
                    try:
                        response, status_code, headers_dict = handler()
                        self.send_response(status_code)
                        for header, value in headers_dict.items():
                            self.send_header(header, value)
                        self.end_headers()
                        self.wfile.write(response.encode())
                    except Exception as e:
                        self.send_response(500)
                        self.send_header('Content-Type', 'application/json')
                        self.end_headers()
                        self.wfile.write(f'{{"error": "{str(e)}"}}'.encode())
                else:
                    self.send_response(404)
                    self.end_headers()
            
            def log_message(self, format, *args):
                # Suppress server logs during testing
                pass
        
        # Start test server
        server = HTTPServer(("localhost", 0), TestHandler)
        port = server.server_address[1]
        server_thread = threading.Thread(target=server.serve_forever)
        server_thread.daemon = True
        server_thread.start()
        
        try:
            # Wait for server to start
            time.sleep(0.1)
            
            credentials = base64.b64encode(b"admin:secret123").decode('ascii')
            headers = {"Authorization": f"Basic {credentials}"}
            
            # First request should succeed
            response = requests.get(f"http://localhost:{port}/metrics", headers=headers, timeout=5)
            self.assertEqual(response.status_code, 200)
            
            # Second request should succeed
            response = requests.get(f"http://localhost:{port}/metrics", headers=headers, timeout=5)
            self.assertEqual(response.status_code, 200)
            
            # Third request should be rate limited
            response = requests.get(f"http://localhost:{port}/metrics", headers=headers, timeout=5)
            self.assertEqual(response.status_code, 429)
            self.assertIn("rate limit", response.json()["error"].lower())
            
        finally:
            server.shutdown()
            server_thread.join(timeout=1)

    def test_http_server_api_key_auth(self):
        """Test API key authentication with HTTP server."""
        import threading
        import time
        import requests
        from http.server import HTTPServer, BaseHTTPRequestHandler
        from slack_kb_agent.monitoring import get_monitoring_endpoints
        from slack_kb_agent.auth import AuthMiddleware, AuthConfig
        
        # Configure API key auth
        auth_config = AuthConfig(
            enabled=True,
            method="api_key",
            api_keys=["test-key-123", "another-key-456"],
            protected_endpoints=["/metrics.json"]
        )
        auth_middleware = AuthMiddleware(auth_config)
        
        # Get monitoring endpoints
        endpoints = get_monitoring_endpoints()
        
        class TestHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                # Apply authentication middleware
                headers = dict(self.headers)
                auth_result = auth_middleware.authenticate_request(
                    self.path, headers, self.client_address[0]
                )
                
                if not auth_result.is_authenticated:
                    self.send_response(401)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(b'{"error": "Unauthorized"}')
                    return
                
                # Handle endpoint
                if self.path in endpoints:
                    handler = endpoints[self.path]
                    try:
                        response, status_code, headers_dict = handler()
                        self.send_response(status_code)
                        for header, value in headers_dict.items():
                            self.send_header(header, value)
                        self.end_headers()
                        self.wfile.write(response.encode())
                    except Exception as e:
                        self.send_response(500)
                        self.send_header('Content-Type', 'application/json')
                        self.end_headers()
                        self.wfile.write(f'{{"error": "{str(e)}"}}'.encode())
                else:
                    self.send_response(404)
                    self.end_headers()
            
            def log_message(self, format, *args):
                # Suppress server logs during testing
                pass
        
        # Start test server
        server = HTTPServer(("localhost", 0), TestHandler)
        port = server.server_address[1]
        server_thread = threading.Thread(target=server.serve_forever)
        server_thread.daemon = True
        server_thread.start()
        
        try:
            # Wait for server to start
            time.sleep(0.1)
            
            # Test with valid API key
            headers = {"X-API-Key": "test-key-123"}
            response = requests.get(f"http://localhost:{port}/metrics.json", headers=headers, timeout=5)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.headers['Content-Type'], 'application/json')
            
            # Test with invalid API key
            headers = {"X-API-Key": "invalid-key"}
            response = requests.get(f"http://localhost:{port}/metrics.json", headers=headers, timeout=5)
            self.assertEqual(response.status_code, 401)
            
        finally:
            server.shutdown()
            server_thread.join(timeout=1)

    def test_http_server_unprotected_endpoints(self):
        """Test that unprotected endpoints work without authentication."""
        import threading
        import time
        import requests
        from http.server import HTTPServer, BaseHTTPRequestHandler
        from slack_kb_agent.monitoring import get_monitoring_endpoints
        from slack_kb_agent.auth import AuthMiddleware, AuthConfig
        
        # Configure auth with only some endpoints protected
        auth_config = AuthConfig(
            enabled=True,
            method="basic",
            basic_username="admin",
            basic_password="secret123",
            protected_endpoints=["/metrics"]  # Only /metrics is protected
        )
        auth_middleware = AuthMiddleware(auth_config)
        
        # Get monitoring endpoints
        endpoints = get_monitoring_endpoints()
        
        class TestHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                # Apply authentication middleware
                headers = dict(self.headers)
                auth_result = auth_middleware.authenticate_request(
                    self.path, headers, self.client_address[0]
                )
                
                if not auth_result.is_authenticated:
                    self.send_response(401)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(b'{"error": "Unauthorized"}')
                    return
                
                # Handle endpoint
                if self.path in endpoints:
                    handler = endpoints[self.path]
                    try:
                        response, status_code, headers_dict = handler()
                        self.send_response(status_code)
                        for header, value in headers_dict.items():
                            self.send_header(header, value)
                        self.end_headers()
                        self.wfile.write(response.encode())
                    except Exception as e:
                        self.send_response(500)
                        self.send_header('Content-Type', 'application/json')
                        self.end_headers()
                        self.wfile.write(f'{{"error": "{str(e)}"}}'.encode())
                else:
                    self.send_response(404)
                    self.end_headers()
            
            def log_message(self, format, *args):
                # Suppress server logs during testing
                pass
        
        # Start test server
        server = HTTPServer(("localhost", 0), TestHandler)
        port = server.server_address[1]
        server_thread = threading.Thread(target=server.serve_forever)
        server_thread.daemon = True
        server_thread.start()
        
        try:
            # Wait for server to start
            time.sleep(0.1)
            
            # Test unprotected endpoint (health) - should work without auth
            response = requests.get(f"http://localhost:{port}/health", timeout=5)
            self.assertEqual(response.status_code, 200)
            
            # Test protected endpoint (metrics) - should require auth
            response = requests.get(f"http://localhost:{port}/metrics", timeout=5)
            self.assertEqual(response.status_code, 401)
            
        finally:
            server.shutdown()
            server_thread.join(timeout=1)


if __name__ == "__main__":
    unittest.main()