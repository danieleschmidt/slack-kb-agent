"""Tests for knowledge source ingestion functionality."""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

from slack_kb_agent.knowledge_base import KnowledgeBase
from slack_kb_agent.models import Document


class TestIngestionSystem:
    """Test knowledge source ingestion capabilities."""

    def test_file_based_ingestion(self):
        """Test ingesting markdown and text files from directories."""
        from slack_kb_agent.ingestion import FileIngester
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test files
            (temp_path / "doc1.md").write_text("# Documentation\nThis is a guide.")
            (temp_path / "doc2.txt").write_text("Plain text content here.")
            (temp_path / "subdir").mkdir()
            (temp_path / "subdir" / "doc3.md").write_text("## Sub-section\nNested content.")
            
            ingester = FileIngester()
            documents = ingester.ingest_directory(temp_path, recursive=True)
            
            assert len(documents) == 3
            assert all(isinstance(doc, Document) for doc in documents)
            assert any("Documentation" in doc.content for doc in documents)
            assert any("Plain text content" in doc.content for doc in documents)
            assert any("Sub-section" in doc.content for doc in documents)

    def test_github_api_ingestion(self):
        """Test ingesting from GitHub repositories."""
        from slack_kb_agent.ingestion import GitHubIngester
        
        # Mock GitHub API responses
        mock_issues = [
            {
                "title": "Bug in authentication",
                "body": "The login system fails when...",
                "number": 123,
                "state": "closed",
                "labels": [{"name": "bug"}]
            },
            {
                "title": "Feature request: dark mode",
                "body": "Add dark mode support for better UX",
                "number": 124,
                "state": "open",
                "labels": [{"name": "enhancement"}]
            }
        ]
        
        with patch('requests.get') as mock_get:
            mock_response = MagicMock()
            mock_response.json.return_value = mock_issues
            mock_response.status_code = 200
            mock_get.return_value = mock_response
            
            ingester = GitHubIngester(token="fake-token")
            documents = ingester.ingest_repository("owner/repo", include_issues=True)
            
            assert len(documents) >= 2
            assert any("authentication" in doc.content.lower() for doc in documents)
            assert any("dark mode" in doc.content.lower() for doc in documents)

    def test_web_documentation_crawler(self):
        """Test crawling web documentation sites."""
        from slack_kb_agent.ingestion import WebDocumentationCrawler
        
        mock_html = """
        <html>
            <head><title>API Documentation</title></head>
            <body>
                <h1>Getting Started</h1>
                <p>This API allows you to...</p>
                <h2>Authentication</h2>
                <p>Use API keys for authentication.</p>
            </body>
        </html>
        """
        
        with patch('requests.get') as mock_get:
            mock_response = MagicMock()
            mock_response.text = mock_html
            mock_response.status_code = 200
            mock_response.headers = {"content-type": "text/html"}
            mock_get.return_value = mock_response
            
            crawler = WebDocumentationCrawler()
            documents = crawler.crawl_url("https://docs.example.com/api")
            
            assert len(documents) >= 1
            assert any("Getting Started" in doc.content for doc in documents)
            assert any("Authentication" in doc.content for doc in documents)

    def test_slack_history_ingestion(self):
        """Test ingesting Slack conversation history."""
        from slack_kb_agent.ingestion import SlackHistoryIngester
        
        mock_messages = [
            {
                "text": "How do we deploy to production?",
                "user": "U123456",
                "ts": "1234567890.123456",
                "channel": "C789"
            },
            {
                "text": "Use the deployment script in scripts/deploy.sh",
                "user": "U654321", 
                "ts": "1234567891.123456",
                "channel": "C789"
            }
        ]
        
        with patch('slack_sdk.WebClient') as mock_client:
            mock_client_instance = MagicMock()
            mock_client_instance.conversations_history.return_value = {
                "messages": mock_messages
            }
            mock_client.return_value = mock_client_instance
            
            ingester = SlackHistoryIngester(token="xoxb-fake-token")
            documents = ingester.ingest_channel("general", days=30)
            
            assert len(documents) >= 2
            assert any("deploy to production" in doc.content.lower() for doc in documents)

    def test_ingestion_command_line_interface(self):
        """Test the command line ingestion interface."""
        from slack_kb_agent.ingestion import main as ingestion_main
        
        # Test file ingestion
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            (temp_path / "test.md").write_text("# Test Document\nContent here.")
            
            kb = KnowledgeBase()
            args = ["--source", "files", "--path", str(temp_path), "--output", "kb.json"]
            
            with patch('sys.argv', ['ingest.py'] + args):
                with patch.object(KnowledgeBase, 'save') as mock_save:
                    ingestion_main()
                    mock_save.assert_called_once()

    def test_incremental_updates(self):
        """Test incremental ingestion that avoids duplicates."""
        from slack_kb_agent.ingestion import IngestionTracker
        
        kb = KnowledgeBase()
        tracker = IngestionTracker()
        
        # First ingestion
        doc1 = Document(content="Original content", source="file1.md")
        tracker.add_document(doc1, checksum="abc123")
        kb.add_document(doc1)
        
        initial_count = len(kb.documents)
        
        # Second ingestion with same content (should skip)
        result = tracker.should_ingest_document("file1.md", checksum="abc123")
        assert result is False
        
        # Third ingestion with changed content (should ingest)
        result = tracker.should_ingest_document("file1.md", checksum="def456")
        assert result is True

    def test_content_filtering_and_processing(self):
        """Test content filtering and processing during ingestion."""
        from slack_kb_agent.ingestion import ContentProcessor
        
        processor = ContentProcessor()
        
        # Test markdown processing
        markdown_content = "# Title\n\nSome **bold** text and `code`."
        processed = processor.process_markdown(markdown_content)
        assert "Title" in processed
        assert "bold" in processed
        assert "code" in processed
        
        # Test content filtering (remove sensitive patterns)
        sensitive_content = "API key: sk-1234567890abcdef\nPassword: secret123"
        filtered = processor.filter_sensitive_content(sensitive_content)
        assert "sk-1234567890abcdef" not in filtered
        assert "secret123" not in filtered
        assert "[REDACTED" in filtered

    def test_batch_ingestion_performance(self):
        """Test batch processing for large ingestion jobs."""
        from slack_kb_agent.ingestion import BatchIngester
        
        # Create mock documents
        documents = [
            Document(content=f"Document {i} content", source=f"doc{i}.md")
            for i in range(100)
        ]
        
        kb = KnowledgeBase()
        ingester = BatchIngester(kb, batch_size=20)
        
        ingester.ingest_batch(documents)
        
        assert len(kb.documents) == 100
        # Verify vector index was built efficiently
        if kb.enable_vector_search:
            assert kb.vector_engine.index is not None

    def test_error_handling_and_recovery(self):
        """Test error handling during ingestion."""
        from slack_kb_agent.ingestion import FileIngester
        
        ingester = FileIngester()
        
        # Test with non-existent directory
        documents = ingester.ingest_directory("/non/existent/path")
        assert documents == []
        
        # Test with permission denied
        with patch('pathlib.Path.read_text', side_effect=PermissionError):
            documents = ingester.ingest_directory("/tmp")
            # Should not crash, may return empty or partial results
            assert isinstance(documents, list)

    def test_ingestion_progress_tracking(self):
        """Test progress tracking for long ingestion jobs."""
        from slack_kb_agent.ingestion import IngestionProgress
        
        progress = IngestionProgress(total_items=100)
        
        assert progress.percentage == 0
        assert not progress.is_complete
        
        progress.update(processed=50)
        assert progress.percentage == 50
        
        progress.update(processed=100)
        assert progress.percentage == 100
        assert progress.is_complete