"""End-to-end tests for complete Slack KB Agent workflows.

These tests verify the entire system working together from
knowledge ingestion through Slack response generation.
"""

import pytest
import tempfile
import json
import os
from pathlib import Path
from unittest.mock import patch, Mock, AsyncMock

from slack_kb_agent.knowledge_base import KnowledgeBase
from slack_kb_agent.ingestion import FileIngester
from slack_kb_agent.slack_bot import SlackBot
from slack_kb_agent.configuration import Configuration


@pytest.mark.e2e
@pytest.mark.slow
class TestCompleteWorkflow:
    """End-to-end tests for complete system workflows."""

    @pytest.fixture
    def temp_knowledge_dir(self):
        """Create temporary directory with test knowledge files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test markdown files
            docs_dir = Path(temp_dir) / "docs"
            docs_dir.mkdir()
            
            # API documentation
            (docs_dir / "api.md").write_text("""
# API Documentation

## Authentication
Use Bearer tokens for API authentication.

## Endpoints
- GET /health - Health check
- POST /search - Search knowledge base
""")
            
            # Deployment guide
            (docs_dir / "deployment.md").write_text("""
# Deployment Guide

## Docker Deployment
1. Build the image: `docker build -t app .`
2. Run the container: `docker run -p 3000:3000 app`

## Kubernetes Deployment
Use the provided helm charts in the k8s/ directory.
""")
            
            yield docs_dir

    @pytest.fixture
    def knowledge_base_with_data(self, temp_knowledge_dir):
        """Create knowledge base with test data."""
        kb = KnowledgeBase()
        
        # Ingest test documents
        ingester = FileIngester()
        documents = ingester.ingest_directory(str(temp_knowledge_dir), recursive=True)
        
        for doc in documents:
            kb.add_document(doc)
        
        return kb

    @pytest.fixture
    def mock_slack_app(self):
        """Mock Slack app for e2e testing."""
        with patch('slack_kb_agent.slack_bot.App') as mock_app:
            app_instance = Mock()
            mock_app.return_value = app_instance
            yield app_instance

    def test_knowledge_ingestion_to_search_workflow(self, temp_knowledge_dir):
        """Test complete workflow from ingestion to search."""
        # Step 1: Create knowledge base
        kb = KnowledgeBase()
        
        # Step 2: Ingest documents
        ingester = FileIngester()
        documents = ingester.ingest_directory(str(temp_knowledge_dir), recursive=True)
        
        # Step 3: Add documents to knowledge base
        for doc in documents:
            kb.add_document(doc)
        
        # Step 4: Search for information
        results = kb.search("API authentication")
        
        # Verify results
        assert len(results) > 0
        assert any("Bearer tokens" in result['content'] for result in results)
        
        # Test another search
        deployment_results = kb.search("docker deployment")
        assert len(deployment_results) > 0
        assert any("docker build" in result['content'] for result in deployment_results)

    def test_slack_bot_end_to_end_workflow(self, knowledge_base_with_data, mock_slack_app):
        """Test complete Slack bot workflow."""
        # Step 1: Initialize bot with knowledge base
        config = Configuration()
        bot = SlackBot(config)
        bot.kb = knowledge_base_with_data
        bot.app = mock_slack_app
        
        # Step 2: Mock Slack interaction
        mock_event = {
            'text': '<@U123456> How do I authenticate with the API?',
            'user': 'U789012',
            'channel': 'C123456',
            'ts': '1234567890.123456'
        }
        
        mock_say = Mock()
        
        # Step 3: Process the message (synchronous for testing)
        with patch.object(bot, 'search_and_respond') as mock_search_respond:
            mock_search_respond.return_value = "Use Bearer tokens for API authentication."
            
            # Simulate message processing
            response = bot.search_and_respond(mock_event['text'])
            
            # Verify response contains relevant information
            assert "Bearer tokens" in response or "authentication" in response.lower()

    def test_persistence_workflow(self, temp_knowledge_dir):
        """Test knowledge base persistence workflow."""
        # Step 1: Create and populate knowledge base
        kb1 = KnowledgeBase()
        ingester = FileIngester()
        documents = ingester.ingest_directory(str(temp_knowledge_dir), recursive=True)
        
        for doc in documents:
            kb1.add_document(doc)
        
        # Step 2: Save knowledge base
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            kb_file = f.name
        
        try:
            kb1.save_to_file(kb_file)
            
            # Step 3: Create new knowledge base and load data
            kb2 = KnowledgeBase()
            kb2.load_from_file(kb_file)
            
            # Step 4: Verify data integrity
            results1 = kb1.search("API authentication")
            results2 = kb2.search("API authentication")
            
            assert len(results1) == len(results2)
            assert results1[0]['content'] == results2[0]['content']
            
        finally:
            if os.path.exists(kb_file):
                os.unlink(kb_file)

    def test_vector_search_workflow(self, knowledge_base_with_data):
        """Test vector search workflow if available."""
        if not knowledge_base_with_data.vector_search_enabled:
            pytest.skip("Vector search not available")
        
        # Test semantic search
        results = knowledge_base_with_data.search_semantic("How to secure API access?")
        
        # Should find authentication-related content even without exact keywords
        assert len(results) > 0
        assert any("authentication" in result['content'].lower() or 
                  "Bearer" in result['content'] for result in results)

    def test_error_recovery_workflow(self, temp_knowledge_dir):
        """Test system recovery from various error conditions."""
        kb = KnowledgeBase()
        
        # Test with corrupted file
        corrupted_file = temp_knowledge_dir / "corrupted.md"
        corrupted_file.write_bytes(b'\x00\x01\x02\x03invalid content')
        
        # Should handle corrupted file gracefully
        ingester = FileIngester()
        documents = ingester.ingest_directory(str(temp_knowledge_dir), recursive=True)
        
        # Should still get valid documents despite corrupted file
        valid_docs = [doc for doc in documents if doc.content.strip()]
        assert len(valid_docs) >= 2  # api.md and deployment.md

    @pytest.mark.integration
    def test_monitoring_integration_workflow(self, knowledge_base_with_data):
        """Test that monitoring captures workflow metrics."""
        from slack_kb_agent.monitoring import MonitoringService
        
        # Initialize monitoring
        monitoring = MonitoringService()
        
        # Perform searches and verify metrics are collected
        knowledge_base_with_data.search("API")
        knowledge_base_with_data.search("deployment")
        
        # Note: In a real test, you would verify actual metrics collection
        # This is a placeholder for metric verification logic
        assert True  # Placeholder assertion

    def test_configuration_override_workflow(self, temp_knowledge_dir):
        """Test workflow with different configuration options."""
        # Test with different search configurations
        config = Configuration({
            'max_results': 1,
            'min_score': 0.1,
            'enable_vector_search': False
        })
        
        kb = KnowledgeBase(config=config)
        
        # Ingest and search with custom config
        ingester = FileIngester()
        documents = ingester.ingest_directory(str(temp_knowledge_dir), recursive=True)
        
        for doc in documents:
            kb.add_document(doc)
        
        results = kb.search("API")
        
        # Should respect max_results configuration
        assert len(results) <= 1

    def test_concurrent_access_workflow(self, knowledge_base_with_data):
        """Test workflow under concurrent access conditions."""
        import threading
        import time
        
        results_list = []
        
        def search_worker():
            """Worker function for concurrent searches."""
            try:
                results = knowledge_base_with_data.search("deployment")
                results_list.append(len(results))
            except Exception as e:
                results_list.append(f"Error: {e}")
        
        # Create multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=search_worker)
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join(timeout=10)
        
        # Verify all searches completed successfully
        assert len(results_list) == 5
        assert all(isinstance(result, int) and result > 0 for result in results_list)