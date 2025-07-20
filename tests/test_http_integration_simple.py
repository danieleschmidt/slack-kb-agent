"""Simplified HTTP server integration tests that don't require external dependencies."""

import unittest
import threading
import time
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.request import urlopen, Request
from urllib.error import HTTPError
import base64

from slack_kb_agent.monitoring import get_monitoring_endpoints
from slack_kb_agent.auth import AuthMiddleware, AuthConfig


class HTTPIntegrationTests(unittest.TestCase):
    """Integration tests for HTTP server with authentication middleware."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.server = None
        self.server_thread = None
        self.port = None
    
    def tearDown(self):
        """Clean up test fixtures."""
        if self.server:
            self.server.shutdown()
        if self.server_thread:
            self.server_thread.join(timeout=1)
    
    def _start_test_server(self, auth_config):
        """Start a test HTTP server with authentication."""
        auth_middleware = AuthMiddleware(auth_config)
        endpoints = get_monitoring_endpoints()
        
        class TestHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                # Apply authentication middleware
                headers = dict(self.headers)
                
                
                auth_result = auth_middleware.authenticate_request(
                    self.path, headers, self.client_address[0]
                )
                
                if not auth_result.is_authenticated:
                    status_code = 429 if auth_result.error_message and "rate limit" in auth_result.error_message.lower() else 401
                    self.send_response(status_code)
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
        self.server = HTTPServer(("localhost", 0), TestHandler)
        self.port = self.server.server_address[1]
        self.server_thread = threading.Thread(target=self.server.serve_forever)
        self.server_thread.daemon = True
        self.server_thread.start()
        
        # Wait for server to start
        time.sleep(0.1)
    
    def _make_request(self, path, headers=None):
        """Make HTTP request using urllib."""
        url = f"http://localhost:{self.port}{path}"
        req = Request(url)
        
        if headers:
            for key, value in headers.items():
                req.add_header(key, value)
        
        try:
            with urlopen(req) as response:
                return {
                    'status_code': response.getcode(),
                    'headers': dict(response.headers),
                    'content': response.read().decode('utf-8')
                }
        except HTTPError as e:
            return {
                'status_code': e.code,
                'headers': dict(e.headers) if hasattr(e, 'headers') else {},
                'content': e.read().decode('utf-8') if hasattr(e, 'read') else ''
            }
    
    def test_unauthenticated_request_returns_401(self):
        """Test that unauthenticated requests to protected endpoints return 401."""
        auth_config = AuthConfig(
            enabled=True,
            method="basic",
            basic_auth_users={"admin": "secret123"},
            protected_endpoints=["/metrics", "/health"]
        )
        
        self._start_test_server(auth_config)
        
        # Test unauthenticated request
        response = self._make_request("/metrics")
        self.assertEqual(response['status_code'], 401)
        
        response_data = json.loads(response['content'])
        self.assertIn("error", response_data)
    
    def test_authenticated_request_returns_data(self):
        """Test that authenticated requests to protected endpoints return data."""
        auth_config = AuthConfig(
            enabled=True,
            method="basic",
            basic_auth_users={"admin": "secret123"},
            protected_endpoints=["/metrics", "/health"]
        )
        
        self._start_test_server(auth_config)
        
        # Test authenticated request
        credentials = base64.b64encode(b"admin:secret123").decode('ascii')
        headers = {"Authorization": f"Basic {credentials}"}
        
        response = self._make_request("/metrics", headers)
        self.assertEqual(response['status_code'], 200)
        self.assertEqual(response['headers']['Content-Type'], 'text/plain')
        
        # Test health endpoint
        response = self._make_request("/health", headers)
        self.assertEqual(response['status_code'], 200)
        self.assertEqual(response['headers']['Content-Type'], 'application/json')
        
        health_data = json.loads(response['content'])
        self.assertIn("status", health_data)
        self.assertIn("timestamp", health_data)
    
    def test_rate_limiting_enforcement(self):
        """Test that rate limiting works with HTTP server."""
        auth_config = AuthConfig(
            enabled=True,
            method="basic",
            basic_auth_users={"admin": "secret123"},
            protected_endpoints=["/metrics"],
            rate_limit_requests=2,  # Very low limit
            rate_limit_window=60
        )
        
        self._start_test_server(auth_config)
        
        credentials = base64.b64encode(b"admin:secret123").decode('ascii')
        headers = {"Authorization": f"Basic {credentials}"}
        
        # First request should succeed
        response = self._make_request("/metrics", headers)
        self.assertEqual(response['status_code'], 200)
        
        # Second request should succeed
        response = self._make_request("/metrics", headers)
        self.assertEqual(response['status_code'], 200)
        
        # Third request should be rate limited
        response = self._make_request("/metrics", headers)
        self.assertEqual(response['status_code'], 429)
        
        response_data = json.loads(response['content'])
        self.assertIn("rate limit", response_data["error"].lower())
    
    def test_api_key_authentication(self):
        """Test API key authentication with HTTP server."""
        auth_config = AuthConfig(
            enabled=True,
            method="api_key",
            api_keys=["test-key-123", "another-key-456"],
            protected_endpoints=["/metrics.json"]
        )
        
        self._start_test_server(auth_config)
        
        # Test with valid API key
        headers = {"X-API-Key": "test-key-123"}
        response = self._make_request("/metrics.json", headers)
        
        
        self.assertEqual(response['status_code'], 200)
        self.assertEqual(response['headers']['Content-Type'], 'application/json')
        
        # Test with invalid API key
        headers = {"X-API-Key": "invalid-key"}
        response = self._make_request("/metrics.json", headers)
        self.assertEqual(response['status_code'], 401)
    
    def test_unprotected_endpoints_work_without_auth(self):
        """Test that unprotected endpoints work without authentication."""
        auth_config = AuthConfig(
            enabled=True,
            method="basic",
            basic_auth_users={"admin": "secret123"},
            protected_endpoints=["/metrics"]  # Only /metrics is protected
        )
        
        self._start_test_server(auth_config)
        
        # Test unprotected endpoint (health) - should work without auth
        response = self._make_request("/health")
        self.assertEqual(response['status_code'], 200)
        
        # Test protected endpoint (metrics) - should require auth
        response = self._make_request("/metrics")
        self.assertEqual(response['status_code'], 401)
    
    def test_disabled_auth_allows_all_requests(self):
        """Test that disabled auth allows all requests."""
        auth_config = AuthConfig(
            enabled=False,  # Auth disabled
            protected_endpoints=["/metrics", "/health"]
        )
        
        self._start_test_server(auth_config)
        
        # All endpoints should work without auth when disabled
        response = self._make_request("/metrics")
        self.assertEqual(response['status_code'], 200)
        
        response = self._make_request("/health")
        self.assertEqual(response['status_code'], 200)
    
    def test_mixed_auth_method(self):
        """Test mixed authentication (both basic and API key)."""
        auth_config = AuthConfig(
            enabled=True,
            method="mixed",
            basic_auth_users={"admin": "secret123"},
            api_keys=["test-key-123"],
            protected_endpoints=["/metrics"]
        )
        
        self._start_test_server(auth_config)
        
        # Test with basic auth
        credentials = base64.b64encode(b"admin:secret123").decode('ascii')
        headers = {"Authorization": f"Basic {credentials}"}
        response = self._make_request("/metrics", headers)
        self.assertEqual(response['status_code'], 200)
        
        # Test with API key
        headers = {"X-API-Key": "test-key-123"}
        response = self._make_request("/metrics", headers)
        self.assertEqual(response['status_code'], 200)
        
        # Test without auth
        response = self._make_request("/metrics")
        self.assertEqual(response['status_code'], 401)


if __name__ == "__main__":
    unittest.main()