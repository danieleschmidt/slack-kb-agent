"""Tests for backup and restore functionality."""

import pytest
import json
import gzip
import tempfile
from unittest.mock import patch, MagicMock
from pathlib import Path
from datetime import datetime, timezone

from slack_kb_agent.models import Document
from slack_kb_agent.backup import (
    BackupManager,
    create_backup,
    restore_backup,
    validate_backup,
    get_backup_info
)


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
def mock_repository(sample_documents):
    """Create a mock database repository."""
    repo = MagicMock()
    repo.get_all_documents.return_value = sample_documents
    repo.create_documents.return_value = [1, 2, 3]
    repo.clear_all_documents.return_value = 0
    return repo


@pytest.fixture
def valid_backup_data(sample_documents):
    """Create valid backup data structure."""
    return {
        "metadata": {
            "created_at": "2023-07-20T12:00:00+00:00",
            "version": "1.0",
            "source": "slack_kb_agent",
            "document_count": len(sample_documents),
            "compressed": False
        },
        "documents": [
            {
                "content": doc.content,
                "source": doc.source,
                "metadata": doc.metadata
            }
            for doc in sample_documents
        ]
    }


@pytest.fixture
def backup_file(valid_backup_data):
    """Create a temporary backup file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(valid_backup_data, f)
        temp_path = Path(f.name)
    
    yield temp_path
    
    # Cleanup
    if temp_path.exists():
        temp_path.unlink()


@pytest.fixture
def compressed_backup_file(valid_backup_data):
    """Create a temporary compressed backup file."""
    with tempfile.NamedTemporaryFile(suffix='.json.gz', delete=False) as f:
        temp_path = Path(f.name)
    
    with gzip.open(temp_path, 'wt', encoding='utf-8') as f:
        json.dump(valid_backup_data, f)
    
    yield temp_path
    
    # Cleanup
    if temp_path.exists():
        temp_path.unlink()


class TestBackupManager:
    """Test BackupManager functionality."""
    
    @patch('slack_kb_agent.backup.is_database_available')
    def test_init_with_custom_repository(self, mock_db_available, mock_repository):
        """Test BackupManager initialization with custom repository."""
        mock_db_available.return_value = True
        
        manager = BackupManager(repository=mock_repository)
        
        assert manager.repository == mock_repository
    
    @patch('slack_kb_agent.backup.is_database_available')
    @patch('slack_kb_agent.backup.get_database_repository')
    def test_init_with_default_repository(self, mock_get_repo, mock_db_available, mock_repository):
        """Test BackupManager initialization with default repository."""
        mock_db_available.return_value = True
        mock_get_repo.return_value = mock_repository
        
        manager = BackupManager()
        
        assert manager.repository == mock_repository
    
    @patch('slack_kb_agent.backup.is_database_available')
    def test_create_backup_database_unavailable(self, mock_db_available):
        """Test create_backup when database is unavailable."""
        mock_db_available.return_value = False
        
        manager = BackupManager()
        
        with pytest.raises(RuntimeError, match="Database is not available"):
            manager.create_backup("/tmp/test_backup.json")
    
    @patch('slack_kb_agent.backup.is_database_available')
    def test_create_backup_uncompressed(self, mock_db_available, mock_repository, sample_documents):
        """Test creating uncompressed backup."""
        mock_db_available.return_value = True
        
        manager = BackupManager(repository=mock_repository)
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            temp_path = Path(f.name)
        
        try:
            result = manager.create_backup(temp_path, compress=False)
            
            assert result["success"] is True
            assert result["document_count"] == len(sample_documents)
            assert result["compressed"] is False
            assert Path(result["backup_path"]).exists()
            
            # Verify file content
            with open(result["backup_path"], 'r') as f:
                data = json.load(f)
            
            assert "metadata" in data
            assert "documents" in data
            assert len(data["documents"]) == len(sample_documents)
            
        finally:
            if temp_path.exists():
                temp_path.unlink()
    
    @patch('slack_kb_agent.backup.is_database_available')
    def test_create_backup_compressed(self, mock_db_available, mock_repository, sample_documents):
        """Test creating compressed backup."""
        mock_db_available.return_value = True
        
        manager = BackupManager(repository=mock_repository)
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            temp_path = Path(f.name)
        
        try:
            result = manager.create_backup(temp_path, compress=True)
            
            assert result["success"] is True
            assert result["document_count"] == len(sample_documents)
            assert result["compressed"] is True
            assert result["backup_path"].endswith('.gz')
            assert Path(result["backup_path"]).exists()
            
            # Verify compressed file content
            with gzip.open(result["backup_path"], 'rt', encoding='utf-8') as f:
                data = json.load(f)
            
            assert "metadata" in data
            assert "documents" in data
            assert len(data["documents"]) == len(sample_documents)
            
        finally:
            if temp_path.exists():
                temp_path.unlink()
            gz_path = Path(str(temp_path) + '.gz')
            if gz_path.exists():
                gz_path.unlink()
    
    def test_restore_backup_file_not_found(self, mock_repository):
        """Test restore_backup with non-existent file."""
        manager = BackupManager(repository=mock_repository)
        
        with pytest.raises(FileNotFoundError):
            manager.restore_backup("/nonexistent/backup.json")
    
    def test_restore_backup_invalid_json(self, mock_repository):
        """Test restore_backup with invalid JSON file."""
        manager = BackupManager(repository=mock_repository)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("invalid json content")
            temp_path = Path(f.name)
        
        try:
            with pytest.raises(ValueError, match="Invalid backup file"):
                manager.restore_backup(temp_path)
        finally:
            if temp_path.exists():
                temp_path.unlink()
    
    @patch('slack_kb_agent.backup.is_database_available')
    def test_restore_backup_validate_only(self, mock_db_available, backup_file):
        """Test restore_backup with validate_only=True."""
        mock_db_available.return_value = True
        
        manager = BackupManager()
        
        result = manager.restore_backup(backup_file, validate_only=True)
        
        assert result["validated_only"] is True
        assert result["validation"]["valid"] is True
        assert result["document_count"] == 3
        assert "backup_metadata" in result
    
    @patch('slack_kb_agent.backup.is_database_available')
    def test_restore_backup_success(self, mock_db_available, mock_repository, backup_file):
        """Test successful backup restore."""
        mock_db_available.return_value = True
        
        manager = BackupManager(repository=mock_repository)
        
        result = manager.restore_backup(backup_file, clear_existing=False)
        
        assert result["success"] is True
        assert result["documents_restored"] == 3
        assert result["documents_cleared"] == 0
        
        mock_repository.create_documents.assert_called_once()
        mock_repository.clear_all_documents.assert_not_called()
    
    @patch('slack_kb_agent.backup.is_database_available')
    def test_restore_backup_with_clear(self, mock_db_available, mock_repository, backup_file):
        """Test backup restore with clear_existing=True."""
        mock_db_available.return_value = True
        mock_repository.clear_all_documents.return_value = 5
        
        manager = BackupManager(repository=mock_repository)
        
        result = manager.restore_backup(backup_file, clear_existing=True)
        
        assert result["success"] is True
        assert result["documents_restored"] == 3
        assert result["documents_cleared"] == 5
        
        mock_repository.clear_all_documents.assert_called_once()
        mock_repository.create_documents.assert_called_once()
    
    @patch('slack_kb_agent.backup.is_database_available')
    def test_restore_compressed_backup(self, mock_db_available, mock_repository, compressed_backup_file):
        """Test restoring compressed backup."""
        mock_db_available.return_value = True
        
        manager = BackupManager(repository=mock_repository)
        
        result = manager.restore_backup(compressed_backup_file)
        
        assert result["success"] is True
        assert result["documents_restored"] == 3
    
    def test_validate_backup_data_valid(self, valid_backup_data):
        """Test validation of valid backup data."""
        manager = BackupManager()
        
        result = manager._validate_backup_data(valid_backup_data)
        
        assert result["valid"] is True
        assert len(result["errors"]) == 0
        assert result["document_count"] == 3
    
    def test_validate_backup_data_invalid_structure(self):
        """Test validation of invalid backup data structure."""
        manager = BackupManager()
        
        # Test non-dict input
        result = manager._validate_backup_data("not a dict")
        assert result["valid"] is False
        assert "must be a dictionary" in result["errors"][0]
        
        # Test missing documents
        result = manager._validate_backup_data({"metadata": {}})
        assert result["valid"] is False
        
        # Test invalid documents type
        result = manager._validate_backup_data({"documents": "not a list"})
        assert result["valid"] is False
        assert "must be a list" in result["errors"][0]
    
    def test_validate_backup_data_document_errors(self):
        """Test validation of backup data with document errors."""
        manager = BackupManager()
        
        invalid_data = {
            "metadata": {"document_count": 2},
            "documents": [
                {"content": "valid document", "source": "test"},
                "not a dict",  # Invalid document
                {"source": "test"},  # Missing content
            ]
        }
        
        result = manager._validate_backup_data(invalid_data)
        
        assert result["valid"] is False
        assert len(result["errors"]) == 2  # One for invalid dict, one for missing content
    
    def test_validate_backup_data_warnings(self):
        """Test validation generates appropriate warnings."""
        manager = BackupManager()
        
        data_with_warnings = {
            "documents": [
                {"content": "test", "metadata": "not a dict"},  # Invalid metadata
                {"content": "test2"}  # Missing source
            ]
        }
        
        result = manager._validate_backup_data(data_with_warnings)
        
        assert result["valid"] is True  # No errors, just warnings
        assert len(result["warnings"]) >= 2
    
    def test_list_backup_info_success(self, backup_file):
        """Test getting backup info for valid file."""
        manager = BackupManager()
        
        info = manager.list_backup_info(backup_file)
        
        assert info["readable"] is True
        assert info["compressed"] is False
        assert info["document_count"] == 3
        assert "file_size_bytes" in info
        assert "backup_metadata" in info
        assert info["validation"]["valid"] is True
    
    def test_list_backup_info_compressed(self, compressed_backup_file):
        """Test getting backup info for compressed file."""
        manager = BackupManager()
        
        info = manager.list_backup_info(compressed_backup_file)
        
        assert info["readable"] is True
        assert info["compressed"] is True
        assert info["document_count"] == 3
    
    def test_list_backup_info_invalid_file(self):
        """Test getting backup info for invalid file."""
        manager = BackupManager()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("invalid json")
            temp_path = Path(f.name)
        
        try:
            info = manager.list_backup_info(temp_path)
            
            assert info["readable"] is False
            assert "error" in info
            assert "file_size_bytes" in info
        finally:
            if temp_path.exists():
                temp_path.unlink()
    
    def test_list_backup_info_nonexistent_file(self):
        """Test getting backup info for non-existent file."""
        manager = BackupManager()
        
        with pytest.raises(FileNotFoundError):
            manager.list_backup_info("/nonexistent/file.json")


class TestBackupConvenienceFunctions:
    """Test convenience functions for backup operations."""
    
    @patch('slack_kb_agent.backup.BackupManager')
    def test_create_backup_function(self, mock_manager_class):
        """Test create_backup convenience function."""
        mock_manager = MagicMock()
        mock_manager.create_backup.return_value = {"success": True}
        mock_manager_class.return_value = mock_manager
        
        result = create_backup("/tmp/backup.json", compress=True)
        
        mock_manager_class.assert_called_once()
        mock_manager.create_backup.assert_called_once_with("/tmp/backup.json", True)
        assert result["success"] is True
    
    @patch('slack_kb_agent.backup.BackupManager')
    def test_restore_backup_function(self, mock_manager_class):
        """Test restore_backup convenience function."""
        mock_manager = MagicMock()
        mock_manager.restore_backup.return_value = {"success": True}
        mock_manager_class.return_value = mock_manager
        
        result = restore_backup("/tmp/backup.json", clear_existing=True, validate_only=False)
        
        mock_manager_class.assert_called_once()
        mock_manager.restore_backup.assert_called_once_with("/tmp/backup.json", True, False)
        assert result["success"] is True
    
    @patch('slack_kb_agent.backup.BackupManager')
    def test_validate_backup_function(self, mock_manager_class):
        """Test validate_backup convenience function."""
        mock_manager = MagicMock()
        mock_manager.restore_backup.return_value = {"validation": {"valid": True}}
        mock_manager_class.return_value = mock_manager
        
        result = validate_backup("/tmp/backup.json")
        
        mock_manager_class.assert_called_once()
        mock_manager.restore_backup.assert_called_once_with("/tmp/backup.json", validate_only=True)
    
    @patch('slack_kb_agent.backup.BackupManager')
    def test_get_backup_info_function(self, mock_manager_class):
        """Test get_backup_info convenience function."""
        mock_manager = MagicMock()
        mock_manager.list_backup_info.return_value = {"readable": True}
        mock_manager_class.return_value = mock_manager
        
        result = get_backup_info("/tmp/backup.json")
        
        mock_manager_class.assert_called_once()
        mock_manager.list_backup_info.assert_called_once_with("/tmp/backup.json")
        assert result["readable"] is True


class TestBackupIntegration:
    """Integration tests for backup functionality."""
    
    @patch('slack_kb_agent.backup.is_database_available')
    def test_full_backup_restore_cycle(self, mock_db_available, sample_documents):
        """Test complete backup and restore cycle."""
        mock_db_available.return_value = True
        
        # Create mock repository for backup
        backup_repo = MagicMock()
        backup_repo.get_all_documents.return_value = sample_documents
        
        # Create mock repository for restore
        restore_repo = MagicMock()
        restore_repo.create_documents.return_value = [1, 2, 3]
        restore_repo.clear_all_documents.return_value = 0
        
        backup_manager = BackupManager(repository=backup_repo)
        restore_manager = BackupManager(repository=restore_repo)
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            temp_path = Path(f.name)
        
        try:
            # Create backup
            backup_result = backup_manager.create_backup(temp_path, compress=False)
            assert backup_result["success"] is True
            assert backup_result["document_count"] == len(sample_documents)
            
            # Restore backup
            restore_result = restore_manager.restore_backup(temp_path)
            assert restore_result["success"] is True
            assert restore_result["documents_restored"] == len(sample_documents)
            
            # Verify repository was called correctly
            restore_repo.create_documents.assert_called_once()
            created_docs = restore_repo.create_documents.call_args[0][0]
            assert len(created_docs) == len(sample_documents)
            
        finally:
            if temp_path.exists():
                temp_path.unlink()