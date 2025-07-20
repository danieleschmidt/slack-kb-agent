#!/usr/bin/env python3
"""
Monitoring Server for Slack KB Agent

Provides HTTP endpoints for metrics and health checks.
Useful for production deployments with load balancers and monitoring systems.

Usage:
    python monitoring_server.py

Environment Variables:
    MONITORING_PORT     - Port for monitoring server (default: 9090)
    KB_DATA_PATH        - Path to knowledge base JSON file (default: kb.json)
    MONITORING_ENABLED  - Enable/disable monitoring (default: true)
"""

import sys
import os
import json
import time
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

# Add src to path for local development
sys.path.insert(0, str(Path(__file__).parent / "src"))

from slack_kb_agent import (
    KnowledgeBase,
    setup_monitoring,
    get_global_metrics,
    HealthChecker,
    MonitoringConfig
)
from slack_kb_agent.auth import get_auth_middleware, AuthConfig


class MonitoringHandler(BaseHTTPRequestHandler):
    """HTTP handler for monitoring endpoints."""
    
    def __init__(self, *args, **kwargs):
        self.metrics = get_global_metrics()
        self.health_checker = HealthChecker()
        self.auth_middleware = get_auth_middleware()
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        """Handle GET requests."""
        path = urlparse(self.path).path
        
        # Authenticate request
        headers = dict(self.headers)
        client_ip = self.client_address[0] if self.client_address else "unknown"
        
        auth_result = self.auth_middleware.authenticate_request(path, headers, client_ip)
        
        if not auth_result.is_authenticated:
            self.send_auth_error(auth_result.error_message)
            return
        
        try:
            if path == "/metrics":
                self.serve_metrics()
            elif path == "/health":
                self.serve_health()
            elif path == "/metrics.json":
                self.serve_metrics_json()
            elif path == "/status":
                self.serve_status()
            else:
                self.send_error(404, "Not Found")
        except Exception as e:
            self.log_error(f"Error handling request: {e}")
            self.send_error(500, "Internal Server Error")
    
    def serve_metrics(self):
        """Serve Prometheus metrics."""
        content = self.metrics.export_prometheus()
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content.encode('utf-8'))
    
    def serve_metrics_json(self):
        """Serve metrics in JSON format."""
        content = self.metrics.export_json()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content.encode('utf-8'))
    
    def serve_health(self):
        """Serve health check."""
        # Load knowledge base for health check
        kb_path = Path(os.getenv("KB_DATA_PATH", "kb.json"))
        kb = None
        
        if kb_path.exists():
            try:
                kb = KnowledgeBase.load(kb_path)
            except Exception as e:
                logger.warning(f"Failed to load knowledge base for health check: {e}")
                # Health check will handle the kb=None case
        
        health_status = self.health_checker.get_health_status(kb)
        content = json.dumps(health_status, indent=2)
        
        # Set status code based on health
        status_code = 200 if health_status["status"] == "healthy" else 503
        
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content.encode('utf-8'))
    
    def serve_status(self):
        """Serve general status information."""
        kb_path = Path(os.getenv("KB_DATA_PATH", "kb.json"))
        kb = None
        doc_count = 0
        
        if kb_path.exists():
            try:
                kb = KnowledgeBase.load(kb_path)
                doc_count = len(kb.documents)
            except Exception as e:
                logger.warning(f"Failed to load knowledge base for status: {e}")
                # Continue with doc_count = 0
        
        config = MonitoringConfig.from_env()
        
        status = {
            "service": "slack-kb-agent",
            "version": "1.3.0",
            "monitoring_enabled": config.enabled,
            "knowledge_base": {
                "documents": doc_count,
                "vector_search": kb.enable_vector_search if kb else False,
                "file_exists": kb_path.exists()
            },
            "endpoints": [
                "/health - Health check",
                "/metrics - Prometheus metrics",
                "/metrics.json - JSON metrics",
                "/status - This status page"
            ]
        }
        
        content = json.dumps(status, indent=2)
        
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content.encode('utf-8'))
    
    def send_auth_error(self, message: str):
        """Send authentication error response."""
        self.send_response(401)
        self.send_header("Content-Type", "application/json")
        self.send_header("WWW-Authenticate", 'Basic realm="Monitoring Server"')
        self.end_headers()
        
        error_response = {
            "error": "Authentication required",
            "message": message,
            "timestamp": json.dumps(time.time())
        }
        
        content = json.dumps(error_response, indent=2)
        self.wfile.write(content.encode('utf-8'))
    
    def log_message(self, format, *args):
        """Override to reduce log spam."""
        # Only log errors, not every request
        pass


def main():
    """Main entry point for monitoring server."""
    print("📊 Slack KB Agent Monitoring Server")
    print("=" * 40)
    
    # Set up monitoring
    monitoring = setup_monitoring()
    if monitoring["status"] != "enabled":
        print("❌ Monitoring is disabled")
        print("Set MONITORING_ENABLED=true to enable monitoring")
        return 1
    
    config = MonitoringConfig.from_env()
    port = int(os.getenv("MONITORING_PORT", config.metrics_port))
    
    # Show authentication configuration
    auth_config = AuthConfig.from_env()
    if auth_config.enabled:
        print(f"🔒 Authentication: {auth_config.method}")
        if auth_config.method in ["basic", "mixed"]:
            print(f"   Basic auth users: {len(auth_config.basic_auth_users)}")
        if auth_config.method in ["api_key", "mixed"]:
            print(f"   API keys configured: {len(auth_config.api_keys)}")
        print(f"   Protected endpoints: {', '.join(auth_config.protected_endpoints)}")
    else:
        print("⚠️  Authentication is DISABLED")
    
    # Check if knowledge base exists
    kb_path = Path(os.getenv("KB_DATA_PATH", "kb.json"))
    if kb_path.exists():
        try:
            kb = KnowledgeBase.load(kb_path)
            print(f"📚 Knowledge base loaded: {len(kb.documents)} documents")
        except Exception as e:
            print(f"⚠️  Failed to load knowledge base: {e}")
    else:
        print(f"⚠️  Knowledge base not found: {kb_path}")
    
    # Start HTTP server
    try:
        server = HTTPServer(("", port), MonitoringHandler)
        print(f"🚀 Monitoring server starting on port {port}")
        print(f"📊 Metrics: http://localhost:{port}/metrics")
        print(f"🏥 Health: http://localhost:{port}/health")
        print(f"📈 Status: http://localhost:{port}/status")
        print("\nPress Ctrl+C to stop")
        
        server.serve_forever()
        
    except KeyboardInterrupt:
        print("\n⏹️  Monitoring server stopped")
        return 0
    except Exception as e:
        print(f"❌ Failed to start monitoring server: {e}")
        return 1


if __name__ == "__main__":
    exit(main())