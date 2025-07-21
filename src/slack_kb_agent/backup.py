"""Database backup and restore functionality for PostgreSQL persistence."""

from __future__ import annotations

import json
import gzip
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import asdict

from .models import Document
from .database import get_database_repository, is_database_available, DatabaseRepository

logger = logging.getLogger(__name__)


class BackupManager:
    """Manages database backup and restore operations."""
    
    def __init__(self, repository: Optional[DatabaseRepository] = None):
        """Initialize backup manager.
        
        Args:
            repository: Database repository instance. If None, uses global instance.
        """
        self.repository = repository or get_database_repository()
    
    def create_backup(self, backup_path: str | Path, compress: bool = True) -> Dict[str, Any]:
        """Create a backup of all documents.
        
        Args:
            backup_path: Path where backup file will be saved
            compress: Whether to compress the backup file with gzip
            
        Returns:
            Dict with backup metadata (timestamp, document count, file size, etc.)
            
        Raises:
            RuntimeError: If database is not available
            OSError: If backup file cannot be written
        """
        if not is_database_available():
            raise RuntimeError("Database is not available for backup")
        
        backup_path = Path(backup_path)
        timestamp = datetime.now(timezone.utc)
        
        # Get all documents from database
        logger.info("Starting database backup...")
        documents = self.repository.get_all_documents()
        
        # Create backup data structure
        backup_data = {
            "metadata": {
                "created_at": timestamp.isoformat(),
                "version": "1.0",
                "source": "slack_kb_agent",
                "document_count": len(documents),
                "compressed": compress
            },
            "documents": [asdict(doc) for doc in documents]
        }
        
        # Serialize to JSON
        json_data = json.dumps(backup_data, indent=2, ensure_ascii=False)
        
        # Write to file (compressed or uncompressed)
        if compress:
            backup_path = backup_path.with_suffix(backup_path.suffix + '.gz')
            with gzip.open(backup_path, 'wt', encoding='utf-8') as f:
                f.write(json_data)
        else:
            with open(backup_path, 'w', encoding='utf-8') as f:
                f.write(json_data)
        
        # Get file size
        file_size = backup_path.stat().st_size
        
        backup_metadata = {
            "backup_path": str(backup_path),
            "created_at": timestamp.isoformat(),
            "document_count": len(documents),
            "file_size_bytes": file_size,
            "compressed": compress,
            "success": True
        }
        
        logger.info(f"Backup completed: {len(documents)} documents, {file_size} bytes, {backup_path}")
        return backup_metadata
    
    def restore_backup(
        self, 
        backup_path: str | Path, 
        clear_existing: bool = False,
        validate_only: bool = False
    ) -> Dict[str, Any]:
        """Restore documents from a backup file.
        
        Args:
            backup_path: Path to backup file
            clear_existing: Whether to clear existing documents before restore
            validate_only: If True, only validate the backup without restoring
            
        Returns:
            Dict with restore metadata (documents restored, validation results, etc.)
            
        Raises:
            RuntimeError: If database is not available
            FileNotFoundError: If backup file doesn't exist
            ValueError: If backup file is invalid or corrupted
        """
        if not validate_only and not is_database_available():
            raise RuntimeError("Database is not available for restore")
        
        backup_path = Path(backup_path)
        if not backup_path.exists():
            raise FileNotFoundError(f"Backup file not found: {backup_path}")
        
        # Read backup file (handle both compressed and uncompressed)
        try:
            if backup_path.suffix.endswith('.gz'):
                with gzip.open(backup_path, 'rt', encoding='utf-8') as f:
                    backup_data = json.load(f)
            else:
                with open(backup_path, 'r', encoding='utf-8') as f:
                    backup_data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            raise ValueError(f"Invalid backup file: {e}")
        
        # Validate backup structure
        validation_result = self._validate_backup_data(backup_data)
        if not validation_result["valid"]:
            raise ValueError(f"Invalid backup data: {validation_result['errors']}")
        
        if validate_only:
            return {
                "validation": validation_result,
                "document_count": len(backup_data.get("documents", [])),
                "backup_metadata": backup_data.get("metadata", {}),
                "validated_only": True
            }
        
        # Convert backup documents to Document objects
        documents = []
        for doc_data in backup_data["documents"]:
            try:
                # Ensure required fields exist
                content = doc_data.get("content", "")
                source = doc_data.get("source", "unknown")
                metadata = doc_data.get("metadata", {})
                
                documents.append(Document(
                    content=content,
                    source=source,
                    metadata=metadata
                ))
            except Exception as e:
                logger.warning(f"Skipping invalid document in backup: {e}")
        
        # Clear existing documents if requested
        if clear_existing:
            logger.info("Clearing existing documents before restore...")
            cleared_count = self.repository.clear_all_documents()
            logger.info(f"Cleared {cleared_count} existing documents")
        
        # Restore documents
        logger.info(f"Restoring {len(documents)} documents from backup...")
        restored_ids = self.repository.create_documents(documents)
        
        restore_metadata = {
            "backup_path": str(backup_path),
            "restored_at": datetime.now(timezone.utc).isoformat(),
            "documents_restored": len(restored_ids),
            "documents_cleared": cleared_count if clear_existing else 0,
            "validation": validation_result,
            "backup_metadata": backup_data.get("metadata", {}),
            "success": True
        }
        
        logger.info(f"Restore completed: {len(restored_ids)} documents restored")
        return restore_metadata
    
    def _validate_backup_data(self, backup_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate backup data structure.
        
        Args:
            backup_data: Backup data to validate
            
        Returns:
            Dict with validation results
        """
        errors = []
        warnings = []
        
        # Check top-level structure
        if not isinstance(backup_data, dict):
            errors.append("Backup data must be a dictionary")
            return {"valid": False, "errors": errors, "warnings": warnings}
        
        # Check metadata
        metadata = backup_data.get("metadata", {})
        if not isinstance(metadata, dict):
            warnings.append("Missing or invalid metadata section")
        else:
            if "created_at" not in metadata:
                warnings.append("Missing backup creation timestamp")
            if "document_count" not in metadata:
                warnings.append("Missing document count in metadata")
        
        # Check documents
        documents = backup_data.get("documents", [])
        if not isinstance(documents, list):
            errors.append("Documents section must be a list")
        else:
            # Validate document structure
            for i, doc in enumerate(documents):
                if not isinstance(doc, dict):
                    errors.append(f"Document {i} is not a dictionary")
                    continue
                
                if "content" not in doc:
                    errors.append(f"Document {i} missing 'content' field")
                if "source" not in doc:
                    warnings.append(f"Document {i} missing 'source' field")
                if "metadata" in doc and not isinstance(doc["metadata"], dict):
                    warnings.append(f"Document {i} has invalid metadata field")
        
        # Check document count consistency
        expected_count = metadata.get("document_count")
        actual_count = len(documents)
        if expected_count is not None and expected_count != actual_count:
            warnings.append(f"Document count mismatch: expected {expected_count}, found {actual_count}")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "document_count": len(documents),
            "metadata": metadata
        }
    
    def list_backup_info(self, backup_path: str | Path) -> Dict[str, Any]:
        """Get information about a backup file without restoring it.
        
        Args:
            backup_path: Path to backup file
            
        Returns:
            Dict with backup information
        """
        backup_path = Path(backup_path)
        
        if not backup_path.exists():
            raise FileNotFoundError(f"Backup file not found: {backup_path}")
        
        # Get file stats
        file_stats = backup_path.stat()
        
        try:
            # Read and validate backup
            if backup_path.suffix.endswith('.gz'):
                with gzip.open(backup_path, 'rt', encoding='utf-8') as f:
                    backup_data = json.load(f)
            else:
                with open(backup_path, 'r', encoding='utf-8') as f:
                    backup_data = json.load(f)
            
            validation_result = self._validate_backup_data(backup_data)
            
            return {
                "file_path": str(backup_path),
                "file_size_bytes": file_stats.st_size,
                "file_modified": datetime.fromtimestamp(file_stats.st_mtime, timezone.utc).isoformat(),
                "compressed": backup_path.suffix.endswith('.gz'),
                "backup_metadata": backup_data.get("metadata", {}),
                "document_count": len(backup_data.get("documents", [])),
                "validation": validation_result,
                "readable": True
            }
        
        except Exception as e:
            return {
                "file_path": str(backup_path),
                "file_size_bytes": file_stats.st_size,
                "file_modified": datetime.fromtimestamp(file_stats.st_mtime, timezone.utc).isoformat(),
                "compressed": backup_path.suffix.endswith('.gz'),
                "readable": False,
                "error": str(e)
            }


# Convenience functions for global backup manager
def create_backup(backup_path: str | Path, compress: bool = True) -> Dict[str, Any]:
    """Create a backup using the global backup manager."""
    manager = BackupManager()
    return manager.create_backup(backup_path, compress)


def restore_backup(
    backup_path: str | Path, 
    clear_existing: bool = False,
    validate_only: bool = False
) -> Dict[str, Any]:
    """Restore a backup using the global backup manager."""
    manager = BackupManager()
    return manager.restore_backup(backup_path, clear_existing, validate_only)


def validate_backup(backup_path: str | Path) -> Dict[str, Any]:
    """Validate a backup file without restoring it."""
    manager = BackupManager()
    return manager.restore_backup(backup_path, validate_only=True)


def get_backup_info(backup_path: str | Path) -> Dict[str, Any]:
    """Get information about a backup file."""
    manager = BackupManager()
    return manager.list_backup_info(backup_path)