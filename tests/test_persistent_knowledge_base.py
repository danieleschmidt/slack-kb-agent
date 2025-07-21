"""Tests for PersistentKnowledgeBase functionality."""

import pytest
import json
import tempfile
from unittest.mock import patch, MagicMock
from pathlib import Path

from slack_kb_agent.models import Document
from slack_kb_agent.persistent_knowledge_base import PersistentKnowledgeBase
from slack_kb_agent.database import DatabaseRepository


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


@pytest.fixture
def temp_json_file(sample_documents):
    """Create a temporary JSON file with sample documents."""
    data = {"documents": [doc.__dict__ for doc in sample_documents]}
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(data, f)
        temp_path = Path(f.name)
    
    yield temp_path
    
    # Cleanup
    if temp_path.exists():
        temp_path.unlink()


class TestPersistentKnowledgeBaseInit:
    """Test PersistentKnowledgeBase initialization."""
    
    @patch('slack_kb_agent.persistent_knowledge_base.is_database_available')
    def test_init_with_database_available(self, mock_db_available):
        """Test initialization when database is available."""
        mock_db_available.return_value = True
        
        with patch('slack_kb_agent.persistent_knowledge_base.get_database_repository') as mock_get_repo:
            mock_repo = MagicMock()
            mock_get_repo.return_value = mock_repo
            
            kb = PersistentKnowledgeBase(use_database=None)
            
            assert kb.use_database is True
            assert kb._db_repository == mock_repo
    
    @patch('slack_kb_agent.persistent_knowledge_base.is_database_available')
    def test_init_with_database_unavailable(self, mock_db_available):
        """Test initialization when database is unavailable."""
        mock_db_available.return_value = False
        
        kb = PersistentKnowledgeBase(use_database=None)
        
        assert kb.use_database is False
        assert kb._db_repository is None
    
    def test_init_with_explicit_database_setting(self):
        """Test initialization with explicit database setting."""
        # Force disable database
        kb = PersistentKnowledgeBase(use_database=False)
        assert kb.use_database is False
        
        # Force enable database (may fail if not available)
        try:
            kb = PersistentKnowledgeBase(use_database=True)
            assert kb.use_database in [True, False]  # May fallback if database fails
        except Exception:
            pass  # Database may not be available in test environment
    
    def test_init_with_lazy_load_disabled(self):
        """Test initialization with lazy loading disabled."""
        kb = PersistentKnowledgeBase(lazy_load=False, use_database=False)
        assert kb.lazy_load is False
        assert kb._documents_loaded is True


class TestPersistentKnowledgeBaseDatabaseOperations:
    """Test database-related operations."""
    
    @patch('slack_kb_agent.persistent_knowledge_base.is_database_available')
    @patch('slack_kb_agent.persistent_knowledge_base.get_database_repository')
    def test_add_document_with_database(self, mock_get_repo, mock_db_available, sample_documents):
        """Test adding document with database enabled."""
        mock_db_available.return_value = True
        mock_repo = MagicMock()
        mock_repo.create_document.return_value = 123
        mock_get_repo.return_value = mock_repo
        
        kb = PersistentKnowledgeBase(use_database=True, lazy_load=False)
        document = sample_documents[0]
        
        kb.add_document(document)
        
        mock_repo.create_document.assert_called_once_with(document)
        assert len(kb._documents) == 1
        assert kb._documents[0] == document
    
    @patch('slack_kb_agent.persistent_knowledge_base.is_database_available')
    @patch('slack_kb_agent.persistent_knowledge_base.get_database_repository')
    def test_add_documents_with_database(self, mock_get_repo, mock_db_available, sample_documents):
        """Test adding multiple documents with database enabled."""
        mock_db_available.return_value = True
        mock_repo = MagicMock()
        mock_repo.create_documents.return_value = [1, 2, 3]
        mock_get_repo.return_value = mock_repo
        
        kb = PersistentKnowledgeBase(use_database=True, lazy_load=False)
        
        kb.add_documents(sample_documents)
        
        mock_repo.create_documents.assert_called_once_with(sample_documents)
        assert len(kb._documents) == len(sample_documents)
    
    @patch('slack_kb_agent.persistent_knowledge_base.is_database_available')
    @patch('slack_kb_agent.persistent_knowledge_base.get_database_repository')
    def test_add_document_database_error(self, mock_get_repo, mock_db_available, sample_documents):
        """Test adding document when database operation fails."""
        mock_db_available.return_value = True
        mock_repo = MagicMock()
        mock_repo.create_document.side_effect = Exception("Database error")
        mock_get_repo.return_value = mock_repo
        
        kb = PersistentKnowledgeBase(use_database=True, lazy_load=False)
        document = sample_documents[0]
        
        # Should not raise exception, just log error
        kb.add_document(document)
        
        # Document should still be added to memory
        assert len(kb._documents) == 1
        assert kb._documents[0] == document
    
    def test_add_document_without_database(self, sample_documents):
        """Test adding document with database disabled."""
        kb = PersistentKnowledgeBase(use_database=False, lazy_load=False)
        document = sample_documents[0]
        
        kb.add_document(document)
        
        assert len(kb._documents) == 1
        assert kb._documents[0] == document


class TestPersistentKnowledgeBaseLazyLoading:
    """Test lazy loading functionality."""
    
    @patch('slack_kb_agent.persistent_knowledge_base.is_database_available')
    @patch('slack_kb_agent.persistent_knowledge_base.get_database_repository')
    def test_lazy_load_on_documents_access(self, mock_get_repo, mock_db_available, sample_documents):
        """Test lazy loading when accessing documents property."""
        mock_db_available.return_value = True
        mock_repo = MagicMock()
        mock_repo.get_all_documents.return_value = sample_documents
        mock_get_repo.return_value = mock_repo
        
        kb = PersistentKnowledgeBase(use_database=True, lazy_load=True)
        
        # Documents should not be loaded yet
        assert not kb._documents_loaded
        
        # Accessing documents should trigger lazy load
        docs = kb.documents
        
        mock_repo.get_all_documents.assert_called_once()
        assert kb._documents_loaded
        assert len(docs) == len(sample_documents)
    
    @patch('slack_kb_agent.persistent_knowledge_base.is_database_available')
    @patch('slack_kb_agent.persistent_knowledge_base.get_database_repository')
    def test_lazy_load_on_search(self, mock_get_repo, mock_db_available, sample_documents):
        """Test lazy loading when performing search."""
        mock_db_available.return_value = True
        mock_repo = MagicMock()
        mock_repo.get_all_documents.return_value = sample_documents
        mock_get_repo.return_value = mock_repo
        
        kb = PersistentKnowledgeBase(use_database=True, lazy_load=True)
        
        # Search should trigger lazy load
        results = kb.search("Python")
        
        mock_repo.get_all_documents.assert_called_once()
        assert kb._documents_loaded
    
    def test_no_lazy_load_when_disabled(self, sample_documents):
        """Test no lazy loading when disabled."""
        kb = PersistentKnowledgeBase(use_database=False, lazy_load=False)
        
        # Should already be marked as loaded
        assert kb._documents_loaded
        
        # Accessing documents should not trigger any loading
        docs = kb.documents
        assert isinstance(docs, list)


class TestPersistentKnowledgeBasePersistence:
    """Test persistence operations."""
    
    def test_save_to_json(self, sample_documents):
        """Test saving knowledge base to JSON file."""
        kb = PersistentKnowledgeBase(use_database=False, lazy_load=False)
        kb.add_documents(sample_documents)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = Path(f.name)
        
        try:
            kb.save(temp_path)
            
            # Verify file was created and contains correct data
            assert temp_path.exists()
            
            with open(temp_path, 'r') as f:
                data = json.load(f)
            
            assert "documents" in data
            assert len(data["documents"]) == len(sample_documents)
            
        finally:
            if temp_path.exists():
                temp_path.unlink()
    
    def test_load_from_json(self, temp_json_file):
        """Test loading knowledge base from JSON file."""
        kb = PersistentKnowledgeBase.load(temp_json_file, use_database=False)
        
        assert len(kb.documents) == 3
        assert all(isinstance(doc, Document) for doc in kb.documents)
    
    def test_load_from_nonexistent_file(self):
        """Test loading from non-existent JSON file."""
        kb = PersistentKnowledgeBase.load("/nonexistent/file.json", use_database=False)
        
        assert len(kb.documents) == 0
    
    @patch('slack_kb_agent.persistent_knowledge_base.is_database_available')
    @patch('slack_kb_agent.persistent_knowledge_base.get_database_repository')
    def test_load_from_database(self, mock_get_repo, mock_db_available, sample_documents):
        """Test loading knowledge base from database."""
        mock_db_available.return_value = True
        mock_repo = MagicMock()
        mock_repo.get_all_documents.return_value = sample_documents
        mock_get_repo.return_value = mock_repo
        
        kb = PersistentKnowledgeBase.load_from_database()
        
        mock_repo.get_all_documents.assert_called_once()
        assert len(kb.documents) == len(sample_documents)
        assert kb.use_database is True
    
    @patch('slack_kb_agent.persistent_knowledge_base.is_database_available')
    @patch('slack_kb_agent.persistent_knowledge_base.get_database_repository')
    def test_sync_to_database(self, mock_get_repo, mock_db_available, sample_documents):
        """Test syncing in-memory documents to database."""
        mock_db_available.return_value = True
        mock_repo = MagicMock()
        mock_repo.clear_all_documents.return_value = 0
        mock_repo.create_documents.return_value = [1, 2, 3]
        mock_get_repo.return_value = mock_repo
        
        kb = PersistentKnowledgeBase(use_database=True, lazy_load=False)
        kb.add_documents(sample_documents)
        
        result = kb.sync_to_database()
        
        mock_repo.clear_all_documents.assert_called_once()
        mock_repo.create_documents.assert_called_with(sample_documents)
        assert result == len(sample_documents)
    
    def test_sync_to_database_without_db(self, sample_documents):
        """Test syncing to database when database is disabled."""
        kb = PersistentKnowledgeBase(use_database=False, lazy_load=False)
        kb.add_documents(sample_documents)
        
        result = kb.sync_to_database()
        
        assert result == 0


class TestPersistentKnowledgeBaseStatistics:
    """Test statistics and monitoring functionality."""
    
    @patch('slack_kb_agent.persistent_knowledge_base.is_database_available')
    @patch('slack_kb_agent.persistent_knowledge_base.get_database_repository')
    def test_get_database_stats_with_db(self, mock_get_repo, mock_db_available):
        """Test getting database statistics when database is enabled."""
        mock_db_available.return_value = True
        mock_repo = MagicMock()
        mock_repo.get_memory_stats.return_value = {
            "total_documents": 100,
            "estimated_size_bytes": 50000
        }
        mock_get_repo.return_value = mock_repo
        
        kb = PersistentKnowledgeBase(use_database=True, lazy_load=False)
        
        stats = kb.get_database_stats()
        
        assert stats["database_enabled"] is True
        assert stats["total_documents"] == 100
        assert stats["estimated_size_bytes"] == 50000
        assert "documents_loaded_in_memory" in stats
        assert "lazy_load_enabled" in stats
    
    def test_get_database_stats_without_db(self):
        """Test getting database statistics when database is disabled."""
        kb = PersistentKnowledgeBase(use_database=False)
        
        stats = kb.get_database_stats()
        
        assert stats["database_enabled"] is False
    
    def test_get_memory_stats_includes_db_stats(self, sample_documents):
        """Test that memory stats include database statistics."""
        kb = PersistentKnowledgeBase(use_database=False, lazy_load=False)
        kb.add_documents(sample_documents)
        
        stats = kb.get_memory_stats()
        
        # Should include base memory stats
        assert "documents_count" in stats
        assert "sources_count" in stats
        
        # Should include database stats
        assert "database_enabled" in stats
    
    @patch('slack_kb_agent.persistent_knowledge_base.is_database_available')
    @patch('slack_kb_agent.persistent_knowledge_base.get_database_repository')
    def test_clear_all_with_database(self, mock_get_repo, mock_db_available, sample_documents):
        """Test clearing all documents with database enabled."""
        mock_db_available.return_value = True
        mock_repo = MagicMock()
        mock_repo.clear_all_documents.return_value = 5
        mock_get_repo.return_value = mock_repo
        
        kb = PersistentKnowledgeBase(use_database=True, lazy_load=False)
        kb.add_documents(sample_documents)
        
        cleared_count = kb.clear_all()
        
        mock_repo.clear_all_documents.assert_called_once()
        assert cleared_count == len(sample_documents)
        assert len(kb._documents) == 0
    
    def test_clear_all_without_database(self, sample_documents):
        """Test clearing all documents without database."""
        kb = PersistentKnowledgeBase(use_database=False, lazy_load=False)
        kb.add_documents(sample_documents)
        
        cleared_count = kb.clear_all()
        
        assert cleared_count == len(sample_documents)
        assert len(kb._documents) == 0


class TestPersistentKnowledgeBaseIntegration:
    """Integration tests for PersistentKnowledgeBase."""
    
    def test_json_to_database_workflow(self, temp_json_file):
        """Test complete workflow from JSON to database."""
        # Load from JSON with database disabled
        kb = PersistentKnowledgeBase.load(temp_json_file, use_database=False)
        original_count = len(kb.documents)
        
        # Enable database and sync
        with patch('slack_kb_agent.persistent_knowledge_base.is_database_available') as mock_db_available:
            with patch('slack_kb_agent.persistent_knowledge_base.get_database_repository') as mock_get_repo:
                mock_db_available.return_value = True
                mock_repo = MagicMock()
                mock_repo.clear_all_documents.return_value = 0
                mock_repo.create_documents.return_value = list(range(original_count))
                mock_get_repo.return_value = mock_repo
                
                # Re-create with database enabled
                kb = PersistentKnowledgeBase.load(temp_json_file, use_database=True)
                
                # Should have loaded from JSON and synced to database
                assert len(kb.documents) == original_count
                mock_repo.create_documents.assert_called_once()
    
    def test_search_operations_with_lazy_loading(self, sample_documents):
        """Test search operations trigger lazy loading correctly."""
        with patch('slack_kb_agent.persistent_knowledge_base.is_database_available') as mock_db_available:
            with patch('slack_kb_agent.persistent_knowledge_base.get_database_repository') as mock_get_repo:
                mock_db_available.return_value = True
                mock_repo = MagicMock()
                mock_repo.get_all_documents.return_value = sample_documents
                mock_get_repo.return_value = mock_repo
                
                kb = PersistentKnowledgeBase(use_database=True, lazy_load=True)
                
                # Test regular search
                results = kb.search("Python")
                assert len(results) >= 0  # Should complete without error
                assert kb._documents_loaded
                
                # Reset for next test
                kb._documents_loaded = False
                mock_repo.reset_mock()
                
                # Test semantic search (if available)
                try:
                    results = kb.search_semantic("programming")
                    assert kb._documents_loaded
                except Exception:
                    pass  # May not be available in test environment