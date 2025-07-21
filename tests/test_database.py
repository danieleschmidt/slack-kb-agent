"""Tests for database functionality."""

import pytest
import os
import tempfile
from unittest.mock import patch, MagicMock
from pathlib import Path

from slack_kb_agent.models import Document
from slack_kb_agent.database import (
    DatabaseManager, 
    DatabaseRepository, 
    DocumentModel,
    get_database_manager,
    get_database_repository,
    is_database_available
)


@pytest.fixture
def mock_db_url():
    """Provide a test database URL."""
    return "sqlite:///:memory:"


@pytest.fixture
def db_manager(mock_db_url):
    """Create a test database manager."""
    manager = DatabaseManager(database_url=mock_db_url)
    manager.initialize()
    return manager


@pytest.fixture
def db_repository(db_manager):
    """Create a test database repository."""
    return DatabaseRepository(db_manager)


@pytest.fixture
def sample_documents():
    """Create sample documents for testing."""
    return [
        Document(
            content="This is a test document about Python programming",
            source="test_source_1",
            metadata={"category": "programming", "tags": ["python", "test"]}
        ),
        Document(
            content="Another document about machine learning",
            source="test_source_2",
            metadata={"category": "ml", "tags": ["ai", "learning"]}
        ),
        Document(
            content="Documentation for the API endpoints",
            source="test_source_1",
            metadata={"category": "api", "tags": ["docs", "endpoints"]}
        )
    ]


class TestDatabaseManager:
    """Test DatabaseManager functionality."""
    
    def test_init_with_default_url(self):
        """Test DatabaseManager initialization with default URL."""
        with patch.dict(os.environ, {'DATABASE_URL': 'sqlite:///:memory:'}):
            manager = DatabaseManager()
            assert 'sqlite:///:memory:' in manager.database_url
    
    def test_init_with_custom_url(self, mock_db_url):
        """Test DatabaseManager initialization with custom URL."""
        manager = DatabaseManager(database_url=mock_db_url)
        assert manager.database_url == mock_db_url
    
    def test_initialize_creates_schema(self, db_manager):
        """Test that initialize creates the database schema."""
        # Should not raise an exception
        db_manager.initialize()
        assert db_manager._initialized
    
    def test_is_available_returns_true_for_valid_db(self, db_manager):
        """Test is_available returns True for valid database."""
        assert db_manager.is_available()
    
    def test_get_session_context_manager(self, db_manager):
        """Test database session context manager."""
        with db_manager.get_session() as session:
            result = session.execute('SELECT 1').fetchone()
            assert result[0] == 1
    
    def test_get_engine_returns_engine(self, db_manager):
        """Test get_engine returns SQLAlchemy engine."""
        engine = db_manager.get_engine()
        assert engine is not None
    
    def test_close_disposes_engine(self, db_manager):
        """Test close disposes the engine."""
        engine = db_manager.engine
        db_manager.close()
        # Should not raise an exception
        assert True


class TestDocumentModel:
    """Test DocumentModel functionality."""
    
    def test_to_document_conversion(self):
        """Test conversion from DocumentModel to Document."""
        doc_model = DocumentModel(
            content="Test content",
            source="test_source",
            metadata={"key": "value"}
        )
        
        document = doc_model.to_document()
        
        assert isinstance(document, Document)
        assert document.content == "Test content"
        assert document.source == "test_source"
        assert document.metadata == {"key": "value"}
    
    def test_from_document_conversion(self):
        """Test conversion from Document to DocumentModel."""
        document = Document(
            content="Test content",
            source="test_source",
            metadata={"key": "value"}
        )
        
        doc_model = DocumentModel.from_document(document)
        
        assert doc_model.content == "Test content"
        assert doc_model.source == "test_source"
        assert doc_model.metadata == {"key": "value"}
    
    def test_round_trip_conversion(self):
        """Test round-trip conversion between Document and DocumentModel."""
        original = Document(
            content="Test content",
            source="test_source",
            metadata={"tags": ["test", "conversion"], "score": 0.95}
        )
        
        doc_model = DocumentModel.from_document(original)
        converted = doc_model.to_document()
        
        assert converted.content == original.content
        assert converted.source == original.source
        assert converted.metadata == original.metadata


class TestDatabaseRepository:
    """Test DatabaseRepository functionality."""
    
    def test_create_document(self, db_repository, sample_documents):
        """Test creating a single document."""
        document = sample_documents[0]
        doc_id = db_repository.create_document(document)
        
        assert isinstance(doc_id, int)
        assert doc_id > 0
    
    def test_create_documents(self, db_repository, sample_documents):
        """Test creating multiple documents."""
        doc_ids = db_repository.create_documents(sample_documents)
        
        assert len(doc_ids) == len(sample_documents)
        assert all(isinstance(doc_id, int) and doc_id > 0 for doc_id in doc_ids)
    
    def test_get_document(self, db_repository, sample_documents):
        """Test retrieving a document by ID."""
        document = sample_documents[0]
        doc_id = db_repository.create_document(document)
        
        retrieved = db_repository.get_document(doc_id)
        
        assert retrieved is not None
        assert retrieved.content == document.content
        assert retrieved.source == document.source
        assert retrieved.metadata == document.metadata
    
    def test_get_document_not_found(self, db_repository):
        """Test retrieving a non-existent document."""
        retrieved = db_repository.get_document(99999)
        assert retrieved is None
    
    def test_get_all_documents(self, db_repository, sample_documents):
        """Test retrieving all documents."""
        db_repository.create_documents(sample_documents)
        
        all_docs = db_repository.get_all_documents()
        
        assert len(all_docs) == len(sample_documents)
        assert all(isinstance(doc, Document) for doc in all_docs)
    
    def test_get_all_documents_with_limit(self, db_repository, sample_documents):
        """Test retrieving documents with limit."""
        db_repository.create_documents(sample_documents)
        
        limited_docs = db_repository.get_all_documents(limit=2)
        
        assert len(limited_docs) == 2
    
    def test_get_all_documents_with_offset(self, db_repository, sample_documents):
        """Test retrieving documents with offset."""
        db_repository.create_documents(sample_documents)
        
        offset_docs = db_repository.get_all_documents(offset=1)
        
        assert len(offset_docs) == len(sample_documents) - 1
    
    def test_get_documents_by_source(self, db_repository, sample_documents):
        """Test retrieving documents by source."""
        db_repository.create_documents(sample_documents)
        
        source_docs = db_repository.get_documents_by_source("test_source_1")
        
        # Should have 2 documents from test_source_1
        assert len(source_docs) == 2
        assert all(doc.source == "test_source_1" for doc in source_docs)
    
    def test_search_documents(self, db_repository, sample_documents):
        """Test searching documents by content."""
        db_repository.create_documents(sample_documents)
        
        results = db_repository.search_documents("Python")
        
        assert len(results) == 1
        assert "Python" in results[0].content
    
    def test_search_documents_case_insensitive(self, db_repository, sample_documents):
        """Test case-insensitive document search."""
        db_repository.create_documents(sample_documents)
        
        results = db_repository.search_documents("python")
        
        assert len(results) == 1
        assert "Python" in results[0].content
    
    def test_count_documents(self, db_repository, sample_documents):
        """Test counting total documents."""
        db_repository.create_documents(sample_documents)
        
        count = db_repository.count_documents()
        
        assert count == len(sample_documents)
    
    def test_count_documents_by_source(self, db_repository, sample_documents):
        """Test counting documents by source."""
        db_repository.create_documents(sample_documents)
        
        count = db_repository.count_documents_by_source("test_source_1")
        
        assert count == 2  # Two documents from test_source_1
    
    def test_delete_document(self, db_repository, sample_documents):
        """Test deleting a document."""
        document = sample_documents[0]
        doc_id = db_repository.create_document(document)
        
        deleted = db_repository.delete_document(doc_id)
        
        assert deleted is True
        
        # Verify document is deleted
        retrieved = db_repository.get_document(doc_id)
        assert retrieved is None
    
    def test_delete_document_not_found(self, db_repository):
        """Test deleting a non-existent document."""
        deleted = db_repository.delete_document(99999)
        assert deleted is False
    
    def test_delete_documents_by_source(self, db_repository, sample_documents):
        """Test deleting documents by source."""
        db_repository.create_documents(sample_documents)
        
        deleted_count = db_repository.delete_documents_by_source("test_source_1")
        
        assert deleted_count == 2  # Two documents from test_source_1
        
        # Verify documents are deleted
        remaining = db_repository.get_documents_by_source("test_source_1")
        assert len(remaining) == 0
    
    def test_clear_all_documents(self, db_repository, sample_documents):
        """Test clearing all documents."""
        db_repository.create_documents(sample_documents)
        
        cleared_count = db_repository.clear_all_documents()
        
        assert cleared_count == len(sample_documents)
        
        # Verify all documents are cleared
        all_docs = db_repository.get_all_documents()
        assert len(all_docs) == 0
    
    def test_get_memory_stats(self, db_repository, sample_documents):
        """Test getting database statistics."""
        db_repository.create_documents(sample_documents)
        
        stats = db_repository.get_memory_stats()
        
        assert isinstance(stats, dict)
        assert "total_documents" in stats
        assert stats["total_documents"] == len(sample_documents)
        assert "source_distribution" in stats
        assert "estimated_size_bytes" in stats


class TestGlobalFunctions:
    """Test global database functions."""
    
    @patch('slack_kb_agent.database._db_manager', None)
    @patch('slack_kb_agent.database.DatabaseManager')
    def test_get_database_manager_creates_instance(self, mock_db_manager_class):
        """Test get_database_manager creates new instance."""
        mock_manager = MagicMock()
        mock_db_manager_class.return_value = mock_manager
        
        result = get_database_manager()
        
        mock_db_manager_class.assert_called_once()
        mock_manager.initialize.assert_called_once()
        assert result == mock_manager
    
    @patch('slack_kb_agent.database._db_repository', None)
    @patch('slack_kb_agent.database.get_database_manager')
    @patch('slack_kb_agent.database.DatabaseRepository')
    def test_get_database_repository_creates_instance(self, mock_repo_class, mock_get_manager):
        """Test get_database_repository creates new instance."""
        mock_manager = MagicMock()
        mock_get_manager.return_value = mock_manager
        mock_repo = MagicMock()
        mock_repo_class.return_value = mock_repo
        
        result = get_database_repository()
        
        mock_repo_class.assert_called_once_with(mock_manager)
        assert result == mock_repo
    
    @patch('slack_kb_agent.database.get_database_manager')
    def test_is_database_available_true(self, mock_get_manager):
        """Test is_database_available returns True when database is available."""
        mock_manager = MagicMock()
        mock_manager.is_available.return_value = True
        mock_get_manager.return_value = mock_manager
        
        result = is_database_available()
        
        assert result is True
    
    @patch('slack_kb_agent.database.get_database_manager')
    def test_is_database_available_false_on_exception(self, mock_get_manager):
        """Test is_database_available returns False when exception occurs."""
        mock_get_manager.side_effect = Exception("Database error")
        
        result = is_database_available()
        
        assert result is False


class TestDatabaseIntegration:
    """Integration tests for database functionality."""
    
    def test_full_workflow(self, db_repository, sample_documents):
        """Test complete database workflow."""
        # Create documents
        doc_ids = db_repository.create_documents(sample_documents)
        assert len(doc_ids) == len(sample_documents)
        
        # Retrieve documents
        all_docs = db_repository.get_all_documents()
        assert len(all_docs) == len(sample_documents)
        
        # Search documents
        search_results = db_repository.search_documents("programming")
        assert len(search_results) == 1
        
        # Get statistics
        stats = db_repository.get_memory_stats()
        assert stats["total_documents"] == len(sample_documents)
        
        # Delete by source
        deleted = db_repository.delete_documents_by_source("test_source_2")
        assert deleted == 1
        
        # Verify deletion
        remaining = db_repository.get_all_documents()
        assert len(remaining) == len(sample_documents) - 1
        
        # Clear all
        cleared = db_repository.clear_all_documents()
        assert cleared == len(sample_documents) - 1
        
        # Verify empty
        final_docs = db_repository.get_all_documents()
        assert len(final_docs) == 0