"""Tests for the advanced API server with REST and GraphQL endpoints."""

import pytest
import json
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch
from fastapi.testclient import TestClient
from httpx import AsyncClient

from src.slack_kb_agent.api_server import (
    APIServer, 
    QueryRequest, 
    DocumentRequest,
    create_app
)
from src.slack_kb_agent.knowledge_base import KnowledgeBase
from src.slack_kb_agent.query_processor import QueryProcessor
from src.slack_kb_agent.models import Document, DocumentType, SourceType


class TestAPIServer:
    """Test API server functionality."""
    
    @pytest.fixture
    def mock_knowledge_base(self):
        """Create mock knowledge base."""
        kb = Mock(spec=KnowledgeBase)
        kb.documents = {}
        kb.search.return_value = []
        kb.search_semantic.return_value = []
        kb.search_hybrid.return_value = []
        kb.add_document.return_value = "test_doc_id"
        kb.get_document.return_value = None
        kb.remove_document.return_value = True
        return kb
    
    @pytest.fixture
    def mock_query_processor(self):
        """Create mock query processor."""
        qp = Mock(spec=QueryProcessor)
        return qp
    
    @pytest.fixture
    def api_server(self, mock_knowledge_base, mock_query_processor):
        """Create API server for testing."""
        with patch('src.slack_kb_agent.api_server.DocumentRepository'), \
             patch('src.slack_kb_agent.api_server.AnalyticsRepository'), \
             patch('src.slack_kb_agent.api_server.get_auth_middleware', return_value=None), \
             patch('src.slack_kb_agent.api_server.get_rate_limiter', return_value=None), \
             patch('src.slack_kb_agent.api_server.get_global_metrics'):
            
            server = APIServer(
                knowledge_base=mock_knowledge_base,
                query_processor=mock_query_processor,
                enable_auth=False,
                enable_rate_limiting=False
            )
            return server
    
    @pytest.fixture
    def client(self, api_server):
        """Create test client."""
        return TestClient(api_server.app)
    
    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data
        assert "components" in data
        assert "uptime_seconds" in data
        
        # Check components structure
        components = data["components"]
        assert "knowledge_base" in components
        assert "database" in components
        assert "cache" in components


class TestSearchEndpoint:
    """Test search endpoint functionality."""
    
    @pytest.fixture
    def api_server_with_results(self, mock_knowledge_base, mock_query_processor):
        """Create API server with mock search results."""
        # Mock search results
        mock_doc = Mock()
        mock_doc.content = "Test document content about Python programming"
        mock_doc.source = "test_source"
        mock_doc.metadata = {
            "doc_id": "test_doc_1",
            "title": "Test Document",
            "doc_type": "text",
            "source_type": "manual",
            "priority": 1,
            "tags": ["python", "programming"],
            "created_at": datetime.utcnow().isoformat()
        }
        mock_doc.score = 0.85
        
        mock_knowledge_base.search_hybrid.return_value = [mock_doc]
        
        with patch('src.slack_kb_agent.api_server.DocumentRepository'), \
             patch('src.slack_kb_agent.api_server.AnalyticsRepository') as mock_analytics_repo, \
             patch('src.slack_kb_agent.api_server.get_auth_middleware', return_value=None), \
             patch('src.slack_kb_agent.api_server.get_rate_limiter', return_value=None), \
             patch('src.slack_kb_agent.api_server.get_global_metrics'):
            
            # Mock analytics repository
            mock_analytics_instance = Mock()
            mock_analytics_instance.record_event = AsyncMock()
            mock_analytics_instance.get_popular_queries = AsyncMock(return_value=[])
            mock_analytics_repo.return_value = mock_analytics_instance
            
            server = APIServer(
                knowledge_base=mock_knowledge_base,
                query_processor=mock_query_processor,
                enable_auth=False,
                enable_rate_limiting=False
            )
            return server
    
    @pytest.fixture
    def client_with_results(self, api_server_with_results):
        """Create test client with search results."""
        return TestClient(api_server_with_results.app)
    
    def test_search_basic(self, client_with_results):
        """Test basic search functionality."""
        query_data = {
            "query": "Python programming",
            "limit": 10,
            "include_sources": True
        }
        
        response = client_with_results.post("/search", json=query_data)
        
        assert response.status_code == 200
        data = response.json()
        
        # Check response structure
        assert data["query"] == "Python programming"
        assert "results" in data
        assert "total_results" in data
        assert "response_time_ms" in data
        assert "suggested_queries" in data
        assert "metadata" in data
        
        # Check results structure
        results = data["results"]
        assert len(results) > 0
        
        result = results[0]
        assert "id" in result
        assert "document" in result
        assert "score" in result
        assert "snippet" in result
    
    def test_search_with_user_context(self, client_with_results):
        """Test search with user context."""
        query_data = {
            "query": "deployment process",
            "limit": 5,
            "user_context": {
                "expertise_level": "expert",
                "interests": ["devops", "automation"]
            }
        }
        
        response = client_with_results.post("/search", json=query_data)
        
        assert response.status_code == 200
        data = response.json()
        
        # Should have routing metadata
        assert "routing" in data["metadata"]
    
    def test_search_validation_errors(self, client_with_results):
        """Test search input validation."""
        # Empty query
        response = client_with_results.post("/search", json={"query": ""})
        assert response.status_code == 422
        
        # Query too long
        long_query = "x" * 1001
        response = client_with_results.post("/search", json={"query": long_query})
        assert response.status_code == 422
        
        # Invalid limit
        response = client_with_results.post("/search", json={
            "query": "test",
            "limit": 0
        })
        assert response.status_code == 422
    
    def test_search_suggestions(self, client_with_results):
        """Test query suggestions in search response."""
        query_data = {"query": "Python programming"}
        
        response = client_with_results.post("/search", json=query_data)
        
        assert response.status_code == 200
        data = response.json()
        
        # Should have suggestions (even if empty)
        assert "suggested_queries" in data
        assert isinstance(data["suggested_queries"], list)


class TestDocumentEndpoints:
    """Test document management endpoints."""
    
    @pytest.fixture
    def api_server_for_docs(self, mock_knowledge_base, mock_query_processor):
        """Create API server for document testing."""
        with patch('src.slack_kb_agent.api_server.DocumentRepository') as mock_doc_repo, \
             patch('src.slack_kb_agent.api_server.AnalyticsRepository'), \
             patch('src.slack_kb_agent.api_server.get_auth_middleware', return_value=None), \
             patch('src.slack_kb_agent.api_server.get_rate_limiter', return_value=None), \
             patch('src.slack_kb_agent.api_server.get_global_metrics'):
            
            # Mock document repository
            mock_doc_instance = Mock()
            mock_doc_instance.create_from_document = AsyncMock()
            mock_doc_repo.return_value = mock_doc_instance
            
            server = APIServer(
                knowledge_base=mock_knowledge_base,
                query_processor=mock_query_processor,
                enable_auth=False,
                enable_rate_limiting=False
            )
            return server
    
    @pytest.fixture
    def client_for_docs(self, api_server_for_docs):
        """Create test client for document operations."""
        return TestClient(api_server_for_docs.app)
    
    def test_add_document(self, client_for_docs):
        """Test adding a new document."""
        doc_data = {
            "content": "This is a test document about API development",
            "source": "test_api_docs",
            "title": "API Development Guide",
            "doc_type": "markdown",
            "source_type": "manual_entry",
            "tags": ["api", "development", "guide"],
            "metadata": {"category": "documentation"},
            "priority": 3
        }
        
        response = client_for_docs.post("/documents", json=doc_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "document_id" in data
        assert data["status"] == "created"
    
    def test_get_document(self, client_for_docs, mock_knowledge_base):
        """Test retrieving a document."""
        # Mock document in knowledge base
        mock_doc = Mock()
        mock_doc.content = "Test document content"
        mock_doc.source = "test_source"
        mock_doc.title = "Test Document"
        mock_doc.metadata = {"author": "test_author"}
        mock_doc.created_at = datetime.utcnow()
        
        mock_knowledge_base.get_document.return_value = mock_doc
        
        response = client_for_docs.get("/documents/test_doc_id")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["id"] == "test_doc_id"
        assert data["content"] == "Test document content"
        assert data["source"] == "test_source"
        assert data["title"] == "Test Document"
    
    def test_get_document_not_found(self, client_for_docs, mock_knowledge_base):
        """Test retrieving non-existent document."""
        mock_knowledge_base.get_document.return_value = None
        
        response = client_for_docs.get("/documents/nonexistent")
        
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()
    
    def test_delete_document(self, client_for_docs, mock_knowledge_base):
        """Test deleting a document."""
        mock_knowledge_base.remove_document.return_value = True
        
        response = client_for_docs.delete("/documents/test_doc_id")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "deleted"
    
    def test_delete_document_not_found(self, client_for_docs, mock_knowledge_base):
        """Test deleting non-existent document."""
        mock_knowledge_base.remove_document.return_value = False
        
        response = client_for_docs.delete("/documents/nonexistent")
        
        assert response.status_code == 404
    
    def test_document_validation(self, client_for_docs):
        """Test document input validation."""
        # Missing required fields
        response = client_for_docs.post("/documents", json={})
        assert response.status_code == 422
        
        # Invalid priority
        doc_data = {
            "content": "Test content",
            "source": "test_source",
            "priority": 10  # Invalid - should be 1-5
        }
        response = client_for_docs.post("/documents", json=doc_data)
        assert response.status_code == 422


class TestAnalyticsEndpoint:
    """Test analytics endpoint functionality."""
    
    @pytest.fixture
    def api_server_with_analytics(self, mock_knowledge_base, mock_query_processor):
        """Create API server with mock analytics."""
        with patch('src.slack_kb_agent.api_server.DocumentRepository'), \
             patch('src.slack_kb_agent.api_server.AnalyticsRepository') as mock_analytics_repo, \
             patch('src.slack_kb_agent.api_server.get_auth_middleware', return_value=None), \
             patch('src.slack_kb_agent.api_server.get_rate_limiter', return_value=None), \
             patch('src.slack_kb_agent.api_server.get_global_metrics'):
            
            # Mock analytics repository
            mock_analytics_instance = Mock()
            mock_analytics_instance.get_usage_dashboard_data = AsyncMock(return_value={
                "current_week": {
                    "total_queries": 150,
                    "successful_queries": 135,
                    "success_rate": 0.9,
                    "avg_response_time": 850.5
                }
            })
            mock_analytics_instance.get_popular_queries = AsyncMock(return_value=[
                {"query": "how to deploy", "frequency": 25},
                {"query": "api authentication", "frequency": 18}
            ])
            mock_analytics_repo.return_value = mock_analytics_instance
            
            server = APIServer(
                knowledge_base=mock_knowledge_base,
                query_processor=mock_query_processor,
                enable_auth=False,
                enable_rate_limiting=False
            )
            
            # Mock knowledge gap analyzer
            server.gap_analyzer.identify_knowledge_gaps = Mock(return_value=[])
            
            return server
    
    @pytest.fixture
    def client_with_analytics(self, api_server_with_analytics):
        """Create test client with analytics."""
        return TestClient(api_server_with_analytics.app)
    
    def test_get_analytics_default_period(self, client_with_analytics):
        """Test analytics with default period."""
        response = client_with_analytics.get("/analytics")
        
        assert response.status_code == 200
        data = response.json()
        
        # Check response structure
        assert data["period"] == "7d"
        assert "total_queries" in data
        assert "successful_queries" in data
        assert "success_rate" in data
        assert "avg_response_time_ms" in data
        assert "popular_queries" in data
        assert "knowledge_gaps" in data
        assert "generated_at" in data
        
        # Check data values
        assert data["total_queries"] == 150
        assert data["success_rate"] == 0.9
        assert len(data["popular_queries"]) == 2
    
    def test_get_analytics_custom_period(self, client_with_analytics):
        """Test analytics with custom period."""
        response = client_with_analytics.get("/analytics?period=30d")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["period"] == "30d"
    
    def test_get_analytics_invalid_period(self, client_with_analytics):
        """Test analytics with invalid period."""
        # Should default to 7d for invalid periods
        response = client_with_analytics.get("/analytics?period=invalid")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["period"] == "invalid"  # API accepts it but uses default internally


class TestSuggestionsEndpoint:
    """Test query suggestions endpoint."""
    
    @pytest.fixture
    def api_server_with_suggestions(self, mock_knowledge_base, mock_query_processor):
        """Create API server with mock suggestions."""
        with patch('src.slack_kb_agent.api_server.DocumentRepository'), \
             patch('src.slack_kb_agent.api_server.AnalyticsRepository') as mock_analytics_repo, \
             patch('src.slack_kb_agent.api_server.get_auth_middleware', return_value=None), \
             patch('src.slack_kb_agent.api_server.get_rate_limiter', return_value=None), \
             patch('src.slack_kb_agent.api_server.get_global_metrics'):
            
            # Mock analytics repository for suggestions
            mock_analytics_instance = Mock()
            mock_analytics_instance.get_popular_queries = AsyncMock(return_value=[
                {"query": "how to deploy application"},
                {"query": "deployment pipeline setup"},
                {"query": "deploy with docker"},
                {"query": "python programming basics"},
                {"query": "programming best practices"}
            ])
            mock_analytics_repo.return_value = mock_analytics_instance
            
            server = APIServer(
                knowledge_base=mock_knowledge_base,
                query_processor=mock_query_processor,
                enable_auth=False,
                enable_rate_limiting=False
            )
            return server
    
    @pytest.fixture
    def client_with_suggestions(self, api_server_with_suggestions):
        """Create test client with suggestions."""
        return TestClient(api_server_with_suggestions.app)
    
    def test_get_suggestions(self, client_with_suggestions):
        """Test getting query suggestions."""
        response = client_with_suggestions.get("/suggestions?q=deploy")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "suggestions" in data
        suggestions = data["suggestions"]
        assert isinstance(suggestions, list)
        
        # Should return suggestions containing "deploy"
        if suggestions:  # Might be empty depending on mock data
            for suggestion in suggestions:
                assert "deploy" in suggestion.lower()
    
    def test_get_suggestions_with_limit(self, client_with_suggestions):
        """Test suggestions with custom limit."""
        response = client_with_suggestions.get("/suggestions?q=deploy&limit=2")
        
        assert response.status_code == 200
        data = response.json()
        
        suggestions = data["suggestions"]
        assert len(suggestions) <= 2
    
    def test_suggestions_validation(self, client_with_suggestions):
        """Test suggestions input validation."""
        # Missing query parameter
        response = client_with_suggestions.get("/suggestions")
        assert response.status_code == 422
        
        # Invalid limit
        response = client_with_suggestions.get("/suggestions?q=test&limit=0")
        assert response.status_code == 422


class TestMiddleware:
    """Test API middleware functionality."""
    
    def test_cors_headers(self, client):
        """Test CORS middleware headers."""
        response = client.options("/health")
        
        # Should have CORS headers
        assert "access-control-allow-origin" in response.headers
    
    def test_process_time_header(self, client):
        """Test process time header middleware."""
        response = client.get("/health")
        
        # Should have process time header
        assert "x-process-time" in response.headers
        
        # Should be a valid number
        process_time = float(response.headers["x-process-time"])
        assert process_time >= 0


class TestErrorHandling:
    """Test API error handling."""
    
    @pytest.fixture
    def api_server_with_errors(self, mock_knowledge_base, mock_query_processor):
        """Create API server that throws errors."""
        # Make knowledge base throw errors
        mock_knowledge_base.search_hybrid.side_effect = Exception("Search failed")
        
        with patch('src.slack_kb_agent.api_server.DocumentRepository'), \
             patch('src.slack_kb_agent.api_server.AnalyticsRepository') as mock_analytics_repo, \
             patch('src.slack_kb_agent.api_server.get_auth_middleware', return_value=None), \
             patch('src.slack_kb_agent.api_server.get_rate_limiter', return_value=None), \
             patch('src.slack_kb_agent.api_server.get_global_metrics'):
            
            # Mock analytics repository
            mock_analytics_instance = Mock()
            mock_analytics_instance.record_event = AsyncMock()
            mock_analytics_repo.return_value = mock_analytics_instance
            
            server = APIServer(
                knowledge_base=mock_knowledge_base,
                query_processor=mock_query_processor,
                enable_auth=False,
                enable_rate_limiting=False
            )
            return server
    
    @pytest.fixture
    def client_with_errors(self, api_server_with_errors):
        """Create test client with error scenarios."""
        return TestClient(api_server_with_errors.app)
    
    def test_search_error_handling(self, client_with_errors):
        """Test search error handling."""
        query_data = {"query": "test query"}
        
        response = client_with_errors.post("/search", json=query_data)
        
        # Should return 500 for internal errors
        assert response.status_code == 500
        
        # Should have error detail
        data = response.json()
        assert "detail" in data


class TestAppFactory:
    """Test application factory function."""
    
    def test_create_app(self):
        """Test basic app creation."""
        app = create_app()
        
        assert app.title == "Slack KB Agent API"
        assert app.version == "1.7.2"
        
        # Test basic endpoint
        client = TestClient(app)
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Slack KB Agent API"
        assert data["version"] == "1.7.2"


# Integration tests would go here for testing with real components
class TestIntegration:
    """Integration tests for API server."""
    
    @pytest.mark.integration
    def test_full_search_flow(self):
        """Test complete search flow with real components."""
        # This would test with real KnowledgeBase and QueryProcessor
        # Currently skipped as it requires full setup
        pytest.skip("Integration test requires full component setup")
    
    @pytest.mark.integration  
    def test_graphql_endpoint(self):
        """Test GraphQL endpoint functionality."""
        # This would test the GraphQL endpoint
        # Currently skipped as it requires strawberry setup
        pytest.skip("GraphQL test requires strawberry configuration")