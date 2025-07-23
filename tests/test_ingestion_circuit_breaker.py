#!/usr/bin/env python3
"""
Tests for HTTP circuit breaker integration in ingestion system.

This test suite verifies that circuit breaker properly protects HTTP calls
in GitHubIngester and WebDocumentationCrawler classes against external
service failures.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import time
from typing import List

from src.slack_kb_agent.ingestion import (
    GitHubIngester, WebDocumentationCrawler, ContentProcessor
)
from src.slack_kb_agent.circuit_breaker import (
    CircuitBreaker, CircuitBreakerConfig, CircuitState, CircuitOpenError
)
from src.slack_kb_agent.constants import CircuitBreakerDefaults
from src.slack_kb_agent.models import Document

# Handle optional requests dependency like ingestion module does
try:
    import requests
except ImportError:
    requests = None


class TestGitHubIngesterCircuitBreaker(unittest.TestCase):
    """Test circuit breaker integration with GitHub API calls."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_repo = "test-org/test-repo"
        self.test_token = "ghp_test_token"
        
        # Circuit breaker config for external services
        self.circuit_config = CircuitBreakerConfig(
            failure_threshold=CircuitBreakerDefaults.EXTERNAL_SERVICE_FAILURE_THRESHOLD,
            success_threshold=CircuitBreakerDefaults.EXTERNAL_SERVICE_SUCCESS_THRESHOLD,
            timeout_seconds=CircuitBreakerDefaults.EXTERNAL_SERVICE_TIMEOUT_SECONDS,
            half_open_max_requests=CircuitBreakerDefaults.EXTERNAL_SERVICE_HALF_OPEN_MAX_REQUESTS,
            service_name="github_api"
        )

    def test_circuit_breaker_config_from_constants(self):
        """Test that circuit breaker uses proper configuration from constants."""
        config = CircuitBreakerConfig(
            failure_threshold=CircuitBreakerDefaults.EXTERNAL_SERVICE_FAILURE_THRESHOLD,
            success_threshold=CircuitBreakerDefaults.EXTERNAL_SERVICE_SUCCESS_THRESHOLD,
            timeout_seconds=CircuitBreakerDefaults.EXTERNAL_SERVICE_TIMEOUT_SECONDS,
            half_open_max_requests=CircuitBreakerDefaults.EXTERNAL_SERVICE_HALF_OPEN_MAX_REQUESTS,
            service_name="github_api"
        )
        
        self.assertEqual(config.failure_threshold, 5)
        self.assertEqual(config.success_threshold, 2)
        self.assertEqual(config.timeout_seconds, 60.0)
        self.assertEqual(config.half_open_max_requests, 2)
        self.assertEqual(config.service_name, "github_api")

    @patch('src.slack_kb_agent.ingestion.requests.get')
    def test_github_circuit_breaker_protects_successful_calls(self, mock_get):
        """Test that successful GitHub API calls pass through circuit breaker."""
        # Mock successful GitHub API responses
        mock_issues_response = Mock()
        mock_issues_response.status_code = 200
        mock_issues_response.json.return_value = [
            {
                "title": "Test Issue",
                "body": "Test issue body",
                "number": 1,
                "state": "open",
                "labels": [{"name": "bug"}]
            }
        ]
        mock_issues_response.raise_for_status.return_value = None
        mock_get.return_value = mock_issues_response
        
        with patch('src.slack_kb_agent.ingestion.GitHubIngester._get_circuit_breaker') as mock_cb:
            circuit_breaker = CircuitBreaker(self.circuit_config)
            mock_cb.return_value = circuit_breaker
            
            ingester = GitHubIngester(token=self.test_token)
            documents = ingester.ingest_issues(self.test_repo)
            
            self.assertGreater(len(documents), 0)
            self.assertEqual(circuit_breaker.state, CircuitState.CLOSED)
            self.assertEqual(circuit_breaker.failure_count, 0)

    @patch('src.slack_kb_agent.ingestion.requests.get')
    def test_github_circuit_breaker_handles_api_failures(self, mock_get):
        """Test that circuit breaker opens after repeated GitHub API failures."""
        # Mock API failure  
        if requests:
            mock_get.side_effect = requests.exceptions.ConnectionError("GitHub API unavailable")
        else:
            mock_get.side_effect = Exception("GitHub API unavailable")
        
        with patch('src.slack_kb_agent.ingestion.GitHubIngester._get_circuit_breaker') as mock_cb:
            circuit_breaker = CircuitBreaker(self.circuit_config)
            mock_cb.return_value = circuit_breaker
            
            ingester = GitHubIngester(token=self.test_token)
            
            # Generate failures to trigger circuit breaker
            for i in range(CircuitBreakerDefaults.EXTERNAL_SERVICE_FAILURE_THRESHOLD):
                expected_exceptions = (CircuitOpenError, Exception)
                if requests:
                    expected_exceptions = (requests.exceptions.ConnectionError, CircuitOpenError)
                with self.assertRaises(expected_exceptions):
                    ingester.ingest_issues(self.test_repo)
            
            # Circuit should now be open
            self.assertEqual(circuit_breaker.state, CircuitState.OPEN)
            
            # Next call should be rejected by circuit breaker
            with self.assertRaises(CircuitOpenError):
                ingester.ingest_issues(self.test_repo)

    @patch('src.slack_kb_agent.ingestion.requests.get')
    def test_github_circuit_breaker_protects_readme_calls(self, mock_get):
        """Test that circuit breaker protects GitHub README API calls."""
        # Mock successful README API responses
        mock_readme_response = Mock()
        mock_readme_response.status_code = 200
        mock_readme_response.json.return_value = {
            "download_url": "https://raw.githubusercontent.com/test-org/test-repo/main/README.md"
        }
        
        mock_content_response = Mock()
        mock_content_response.text = "# Test README\n\nThis is a test README file."
        mock_content_response.raise_for_status.return_value = None
        
        mock_get.side_effect = [mock_readme_response, mock_content_response]
        
        with patch('src.slack_kb_agent.ingestion.GitHubIngester._get_circuit_breaker') as mock_cb:
            circuit_breaker = CircuitBreaker(self.circuit_config)
            mock_cb.return_value = circuit_breaker
            
            ingester = GitHubIngester(token=self.test_token)
            documents = ingester.ingest_readme(self.test_repo)
            
            self.assertGreater(len(documents), 0)
            self.assertEqual(circuit_breaker.state, CircuitState.CLOSED)
            # Should have made 2 successful calls (README API + content download)
            self.assertEqual(circuit_breaker.success_count, 2)

    @patch('src.slack_kb_agent.ingestion.requests.get')
    def test_github_circuit_breaker_handles_rate_limiting(self, mock_get):
        """Test that circuit breaker handles GitHub API rate limiting."""
        # Mock rate limiting response
        mock_response = Mock()
        mock_response.status_code = 403
        if requests:
            mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("403 rate limit exceeded")
        else:
            mock_response.raise_for_status.side_effect = Exception("403 rate limit exceeded")
        mock_get.return_value = mock_response
        
        with patch('src.slack_kb_agent.ingestion.GitHubIngester._get_circuit_breaker') as mock_cb:
            circuit_breaker = CircuitBreaker(self.circuit_config)
            mock_cb.return_value = circuit_breaker
            
            ingester = GitHubIngester(token=self.test_token)
            
            # First few calls should fail but not trigger circuit breaker
            for i in range(3):
                expected_exception = Exception
                if requests:
                    expected_exception = requests.exceptions.HTTPError
                with self.assertRaises(expected_exception):
                    ingester.ingest_issues(self.test_repo)
            
            # Circuit breaker should record failures
            self.assertGreater(circuit_breaker.failure_count, 0)


class TestWebCrawlerCircuitBreaker(unittest.TestCase):
    """Test circuit breaker integration with web crawling requests."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_url = "https://example.com/docs"
        
        # Circuit breaker config for web crawler
        self.circuit_config = CircuitBreakerConfig(
            failure_threshold=CircuitBreakerDefaults.EXTERNAL_SERVICE_FAILURE_THRESHOLD,
            success_threshold=CircuitBreakerDefaults.EXTERNAL_SERVICE_SUCCESS_THRESHOLD,
            timeout_seconds=CircuitBreakerDefaults.EXTERNAL_SERVICE_TIMEOUT_SECONDS,
            half_open_max_requests=CircuitBreakerDefaults.EXTERNAL_SERVICE_HALF_OPEN_MAX_REQUESTS,
            service_name="web_crawler"
        )

    @patch('src.slack_kb_agent.ingestion.requests.get')
    @patch('src.slack_kb_agent.ingestion.BeautifulSoup')
    def test_web_crawler_circuit_breaker_protects_successful_calls(self, mock_soup, mock_get):
        """Test that successful web crawling calls pass through circuit breaker."""
        # Mock successful HTTP response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "text/html"}
        mock_response.text = "<html><body><h1>Test Page</h1><p>Test content</p></body></html>"
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        # Mock BeautifulSoup parsing
        mock_soup_instance = Mock()
        mock_soup_instance.get_text.return_value = "Test Page Test content"
        mock_soup_instance.find_all.return_value = []
        mock_soup.return_value = mock_soup_instance
        
        with patch('src.slack_kb_agent.ingestion.WebDocumentationCrawler._get_circuit_breaker') as mock_cb:
            circuit_breaker = CircuitBreaker(self.circuit_config)
            mock_cb.return_value = circuit_breaker
            
            crawler = WebDocumentationCrawler()
            documents = crawler.crawl_site(self.test_url, max_pages=1)
            
            self.assertGreater(len(documents), 0)
            self.assertEqual(circuit_breaker.state, CircuitState.CLOSED)
            self.assertEqual(circuit_breaker.failure_count, 0)

    @patch('src.slack_kb_agent.ingestion.requests.get')
    def test_web_crawler_circuit_breaker_handles_connection_failures(self, mock_get):
        """Test that circuit breaker opens after repeated connection failures."""
        # Mock connection failure
        if requests:
            mock_get.side_effect = requests.exceptions.ConnectionError("Connection refused")
        else:
            mock_get.side_effect = Exception("Connection refused")
        
        with patch('src.slack_kb_agent.ingestion.WebDocumentationCrawler._get_circuit_breaker') as mock_cb:
            circuit_breaker = CircuitBreaker(self.circuit_config)
            mock_cb.return_value = circuit_breaker
            
            crawler = WebDocumentationCrawler()
            
            # Generate failures to trigger circuit breaker
            for i in range(CircuitBreakerDefaults.EXTERNAL_SERVICE_FAILURE_THRESHOLD):
                documents = crawler.crawl_site(self.test_url, max_pages=1)
                # Should return empty list on failure
                self.assertEqual(len(documents), 0)
            
            # Circuit should now be open
            self.assertEqual(circuit_breaker.state, CircuitState.OPEN)

    @patch('src.slack_kb_agent.ingestion.requests.get')
    def test_web_crawler_circuit_breaker_handles_timeout_errors(self, mock_get):
        """Test that circuit breaker handles timeout errors."""
        # Mock timeout error
        if requests:
            mock_get.side_effect = requests.exceptions.Timeout("Request timeout")
        else:
            mock_get.side_effect = Exception("Request timeout")
        
        with patch('src.slack_kb_agent.ingestion.WebDocumentationCrawler._get_circuit_breaker') as mock_cb:
            circuit_breaker = CircuitBreaker(self.circuit_config)
            mock_cb.return_value = circuit_breaker
            
            crawler = WebDocumentationCrawler()
            
            # Multiple timeouts should be recorded as failures
            for i in range(3):
                documents = crawler.crawl_site(self.test_url, max_pages=1)
                self.assertEqual(len(documents), 0)
            
            # Circuit breaker should record failures
            self.assertGreater(circuit_breaker.failure_count, 0)

    @patch('src.slack_kb_agent.ingestion.requests.get')
    def test_web_crawler_circuit_breaker_recovery(self, mock_get):
        """Test circuit breaker recovery after service recovers."""
        with patch('src.slack_kb_agent.ingestion.WebDocumentationCrawler._get_circuit_breaker') as mock_cb:
            circuit_breaker = CircuitBreaker(self.circuit_config)
            mock_cb.return_value = circuit_breaker
            
            # Force circuit to open state
            circuit_breaker.state = CircuitState.OPEN
            circuit_breaker.failure_count = CircuitBreakerDefaults.EXTERNAL_SERVICE_FAILURE_THRESHOLD
            circuit_breaker.last_failure_time = time.time() - (CircuitBreakerDefaults.EXTERNAL_SERVICE_TIMEOUT_SECONDS + 1)
            
            # Mock successful response for recovery
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.headers = {"content-type": "text/html"}
            mock_response.text = "<html><body><h1>Recovered</h1></body></html>"
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response
            
            crawler = WebDocumentationCrawler()
            
            # First call should transition to half-open
            with patch('src.slack_kb_agent.ingestion.BeautifulSoup') as mock_soup:
                mock_soup_instance = Mock()
                mock_soup_instance.get_text.return_value = "Recovered"
                mock_soup_instance.find_all.return_value = []
                mock_soup.return_value = mock_soup_instance
                
                documents = crawler.crawl_site(self.test_url, max_pages=1)
                
                self.assertGreater(len(documents), 0)
                self.assertEqual(circuit_breaker.state, CircuitState.HALF_OPEN)


class TestCircuitBreakerIntegrationMetrics(unittest.TestCase):
    """Test circuit breaker metrics integration for ingestion services."""
    
    def test_github_circuit_breaker_metrics_tracking(self):
        """Test that GitHub circuit breaker metrics are properly tracked."""
        circuit_config = CircuitBreakerConfig(
            failure_threshold=3,
            success_threshold=2,
            timeout_seconds=30.0,
            half_open_max_requests=1,
            service_name="github_api"
        )
        
        with patch('src.slack_kb_agent.ingestion.GitHubIngester._get_circuit_breaker') as mock_cb:
            circuit_breaker = CircuitBreaker(circuit_config)
            mock_cb.return_value = circuit_breaker
            
            # Test metrics collection
            metrics = circuit_breaker.get_metrics()
            expected_keys = [
                'service_name', 'state', 'total_requests', 'total_successes', 'total_failures',
                'failure_rate', 'current_failure_count', 'failure_threshold', 'circuit_opened_count',
                'last_failure_time', 'last_state_change_time', 'time_since_last_failure', 'half_open_requests'
            ]
            
            for key in expected_keys:
                self.assertIn(key, metrics)
            
            self.assertEqual(metrics['service_name'], "github_api")
            self.assertEqual(metrics['state'], CircuitState.CLOSED.value)
            self.assertEqual(metrics['current_failure_count'], 0)

    def test_web_crawler_circuit_breaker_metrics_tracking(self):
        """Test that web crawler circuit breaker metrics are properly tracked."""
        circuit_config = CircuitBreakerConfig(
            failure_threshold=5,
            success_threshold=2,
            timeout_seconds=60.0,
            half_open_max_requests=2,
            service_name="web_crawler"
        )
        
        with patch('src.slack_kb_agent.ingestion.WebDocumentationCrawler._get_circuit_breaker') as mock_cb:
            circuit_breaker = CircuitBreaker(circuit_config)
            mock_cb.return_value = circuit_breaker
            
            # Test metrics collection
            metrics = circuit_breaker.get_metrics()
            
            self.assertEqual(metrics['service_name'], "web_crawler")
            self.assertEqual(metrics['failure_threshold'], 5)
            self.assertEqual(metrics['half_open_requests'], 0)


class TestCircuitBreakerPerformance(unittest.TestCase):
    """Test performance characteristics of circuit breaker integration."""
    
    def setUp(self):
        """Set up performance test fixtures."""
        self.circuit_config = CircuitBreakerConfig(
            failure_threshold=3,
            success_threshold=2,
            timeout_seconds=1.0,  # Short timeout for tests
            half_open_max_requests=1,
            service_name="test_service"
        )
    
    @patch('src.slack_kb_agent.ingestion.requests.get')
    def test_circuit_breaker_overhead_minimal_github(self, mock_get):
        """Test that circuit breaker adds minimal overhead to GitHub API calls."""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = []
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        with patch('src.slack_kb_agent.ingestion.GitHubIngester._get_circuit_breaker') as mock_cb:
            circuit_breaker = CircuitBreaker(self.circuit_config)
            mock_cb.return_value = circuit_breaker
            
            ingester = GitHubIngester(token="test_token")
            
            # Measure time with circuit breaker
            start_time = time.time()
            documents = ingester.ingest_issues("test/repo")
            elapsed_time = time.time() - start_time
            
            # Circuit breaker overhead should be minimal (< 10ms)
            self.assertLess(elapsed_time, 0.01)

    @patch('src.slack_kb_agent.ingestion.requests.get')
    def test_circuit_breaker_fast_failure_when_open(self, mock_get):
        """Test that circuit breaker provides fast failure when open."""
        with patch('src.slack_kb_agent.ingestion.WebDocumentationCrawler._get_circuit_breaker') as mock_cb:
            circuit_breaker = CircuitBreaker(self.circuit_config)
            mock_cb.return_value = circuit_breaker
            
            # Force circuit to open state
            circuit_breaker.state = CircuitState.OPEN
            circuit_breaker.failure_count = 3
            circuit_breaker.last_failure_time = time.time()
            
            crawler = WebDocumentationCrawler()
            
            # Measure fast failure time
            start_time = time.time()
            documents = crawler.crawl_site("https://example.com", max_pages=1)
            elapsed_time = time.time() - start_time
            
            self.assertEqual(len(documents), 0)
            # Should fail very quickly (< 1ms)
            self.assertLess(elapsed_time, 0.001)
            
            # Should not call the actual HTTP client
            mock_get.assert_not_called()


if __name__ == '__main__':
    unittest.main()