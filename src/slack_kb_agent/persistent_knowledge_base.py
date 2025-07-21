"""Enhanced knowledge base with PostgreSQL persistence support."""

from __future__ import annotations

import os
import json
import logging
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path
from dataclasses import asdict

from .models import Document
from .sources import BaseSource
from .vector_search import VectorSearchEngine, is_vector_search_available
from .cache import get_cache_manager
from .database import get_database_repository, is_database_available, DatabaseRepository
from .knowledge_base import KnowledgeBase

logger = logging.getLogger(__name__)

# Import metrics functionality
try:
    from .monitoring import get_global_metrics
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False
    get_global_metrics = None


class PersistentKnowledgeBase(KnowledgeBase):
    """Knowledge base with PostgreSQL persistence support.
    
    This enhanced version of KnowledgeBase provides:
    - PostgreSQL database persistence 
    - Automatic fallback to JSON file persistence
    - Lazy loading from database
    - Hybrid storage (database + in-memory for performance)
    """
    
    def __init__(
        self,
        enable_vector_search: bool = True,
        vector_model: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = 0.5,
        max_documents: Optional[int] = None,
        use_database: Optional[bool] = None,
        lazy_load: bool = True
    ) -> None:
        """Initialize persistent knowledge base.
        
        Args:
            enable_vector_search: Enable vector-based semantic search
            vector_model: Sentence transformer model name
            similarity_threshold: Minimum similarity score for vector search
            max_documents: Maximum number of documents to keep in memory
            use_database: Whether to use database persistence. If None, auto-detect
            lazy_load: Whether to lazy load documents from database
        """
        # Call parent constructor
        super().__init__(
            enable_vector_search=enable_vector_search,
            vector_model=vector_model,
            similarity_threshold=similarity_threshold,
            max_documents=max_documents
        )
        
        # Database configuration
        self.use_database = use_database if use_database is not None else is_database_available()
        self.lazy_load = lazy_load
        self._db_repository: Optional[DatabaseRepository] = None
        self._documents_loaded = False
        
        if self.use_database:
            try:
                self._db_repository = get_database_repository()
                logger.info("Database persistence enabled")
            except Exception as e:
                logger.warning(f"Database persistence disabled due to error: {e}")
                self.use_database = False
        
        if not self.use_database:
            logger.info("Using in-memory storage with JSON file persistence")
    
    @property
    def documents(self) -> List[Document]:
        """Get documents with lazy loading from database if needed."""
        if self.use_database and not self._documents_loaded and self.lazy_load:
            self._load_from_database()
        return self._documents
    
    @documents.setter
    def documents(self, value: List[Document]) -> None:
        """Set documents."""
        self._documents = value
        self._documents_loaded = True
    
    def _load_from_database(self) -> None:
        """Load documents from database into memory."""
        if not self.use_database or not self._db_repository:
            return
        
        try:
            logger.debug("Loading documents from database...")
            db_documents = self._db_repository.get_all_documents(limit=self.max_documents)
            self._documents = db_documents
            self._documents_loaded = True
            
            # Rebuild vector index if needed
            if self.enable_vector_search and self.vector_engine:
                self._rebuild_vector_index()
            
            logger.info(f"Loaded {len(db_documents)} documents from database")
        except Exception as e:
            logger.error(f"Failed to load documents from database: {e}")
            # Fall back to empty list
            self._documents = []
            self._documents_loaded = True
    
    def add_document(self, document: Document) -> None:
        """Add a single document to the knowledge base."""
        # Ensure documents are loaded
        _ = self.documents
        
        # Add to database first if available
        if self.use_database and self._db_repository:
            try:
                doc_id = self._db_repository.create_document(document)
                logger.debug(f"Saved document to database with ID: {doc_id}")
            except Exception as e:
                logger.error(f"Failed to save document to database: {e}")
        
        # Add to in-memory storage
        self._documents.append(document)
        self._enforce_document_limit()
        self._update_memory_metrics()
        
        # Update vector index
        if self.enable_vector_search and self.vector_engine:
            self.vector_engine.add_document(document)
        
        # Invalidate search cache
        cache_manager = get_cache_manager()
        invalidated = cache_manager.invalidate_search_cache()
        if invalidated > 0:
            logger.debug(f"Invalidated {invalidated} search cache entries after adding document")
    
    def add_documents(self, documents: List[Document]) -> None:
        """Add multiple documents to the knowledge base."""
        if not documents:
            return
        
        # Ensure documents are loaded
        _ = self.documents
        
        # Add to database first if available
        if self.use_database and self._db_repository:
            try:
                doc_ids = self._db_repository.create_documents(documents)
                logger.debug(f"Saved {len(documents)} documents to database")
            except Exception as e:
                logger.error(f"Failed to save documents to database: {e}")
        
        # Add to in-memory storage
        self._documents.extend(documents)
        self._enforce_document_limit()
        self._update_memory_metrics()
        
        # Rebuild vector index
        self._rebuild_vector_index()
        
        # Invalidate search cache
        cache_manager = get_cache_manager()
        invalidated = cache_manager.invalidate_search_cache()
        if invalidated > 0:
            logger.debug(f"Invalidated {invalidated} search cache entries after adding {len(documents)} documents")
    
    def index(self) -> None:
        """Load all documents from registered sources."""
        new_documents = []
        for source in self.sources:
            new_documents.extend(source.load())
        
        if new_documents:
            self.add_documents(new_documents)
    
    def search(self, query: str) -> List[Document]:
        """Search documents by content (ensures documents are loaded)."""
        # Ensure documents are loaded
        _ = self.documents
        return super().search(query)
    
    def search_semantic(self, query: str, threshold: Optional[float] = None, top_k: int = 10) -> List[Document]:
        """Semantic search with vector similarity (ensures documents are loaded)."""
        # Ensure documents are loaded
        _ = self.documents
        return super().search_semantic(query, threshold, top_k)
    
    def search_hybrid(self, query: str, vector_weight: float = 0.7, keyword_weight: float = 0.3, top_k: int = 10) -> List[Document]:
        """Hybrid search combining semantic and keyword approaches (ensures documents are loaded)."""
        # Ensure documents are loaded
        _ = self.documents
        return super().search_hybrid(query, vector_weight, keyword_weight, top_k)
    
    def save(self, path: str | Path) -> None:
        """Persist documents to JSON file (backup method)."""
        # Ensure documents are loaded
        _ = self.documents
        super().save(path)
        logger.info(f"Saved {len(self.documents)} documents to JSON file: {path}")
    
    @classmethod
    def load(cls, path: str | Path, max_documents: Optional[int] = None, use_database: Optional[bool] = None) -> "PersistentKnowledgeBase":
        """Load documents from JSON file and optionally sync to database."""
        # Create knowledge base instance
        kb = cls(max_documents=max_documents, use_database=use_database, lazy_load=False)
        
        # Try to load from JSON file
        try:
            text = Path(path).read_text(encoding="utf-8")
            data = json.loads(text)
            if isinstance(data, dict):
                for item in data.get("documents", []):
                    if isinstance(item, dict):
                        try:
                            document = Document(**item)
                            kb._documents.append(document)
                        except Exception as e:
                            logger.warning(f"Skipping invalid document: {e}")
                
                logger.info(f"Loaded {len(kb._documents)} documents from JSON file: {path}")
                
                # If database is available, sync documents to database
                if kb.use_database and kb._db_repository:
                    try:
                        # Clear existing documents in database
                        cleared = kb._db_repository.clear_all_documents()
                        logger.info(f"Cleared {cleared} existing documents from database")
                        
                        # Add documents to database
                        if kb._documents:
                            doc_ids = kb._db_repository.create_documents(kb._documents)
                            logger.info(f"Synced {len(doc_ids)} documents to database")
                    except Exception as e:
                        logger.error(f"Failed to sync documents to database: {e}")
        
        except (OSError, json.JSONDecodeError) as e:
            logger.warning(f"Failed to load from JSON file {path}: {e}")
        
        kb._documents_loaded = True
        kb._enforce_document_limit()
        kb._rebuild_vector_index()
        return kb
    
    @classmethod
    def load_from_database(cls, max_documents: Optional[int] = None) -> "PersistentKnowledgeBase":
        """Load knowledge base entirely from database."""
        kb = cls(max_documents=max_documents, use_database=True, lazy_load=False)
        kb._load_from_database()
        return kb
    
    def sync_to_database(self) -> int:
        """Sync all in-memory documents to database.
        
        Returns:
            Number of documents synced to database
        """
        if not self.use_database or not self._db_repository:
            logger.warning("Database not available for sync")
            return 0
        
        # Ensure documents are loaded
        _ = self.documents
        
        try:
            # Clear existing documents in database
            cleared = self._db_repository.clear_all_documents()
            logger.debug(f"Cleared {cleared} existing documents from database")
            
            # Add all current documents to database
            if self._documents:
                doc_ids = self._db_repository.create_documents(self._documents)
                logger.info(f"Synced {len(doc_ids)} documents to database")
                return len(doc_ids)
            
            return 0
        except Exception as e:
            logger.error(f"Failed to sync documents to database: {e}")
            return 0
    
    def clear_all(self) -> int:
        """Clear all documents from both memory and database.
        
        Returns:
            Number of documents cleared
        """
        # Count documents before clearing
        total_cleared = len(self.documents)
        
        # Clear database
        if self.use_database and self._db_repository:
            try:
                db_cleared = self._db_repository.clear_all_documents()
                logger.debug(f"Cleared {db_cleared} documents from database")
            except Exception as e:
                logger.error(f"Failed to clear database: {e}")
        
        # Clear in-memory storage
        self._documents = []
        self._documents_loaded = True
        
        # Clear vector index
        if self.enable_vector_search and self.vector_engine:
            self.vector_engine = VectorSearchEngine(
                model_name=self.vector_engine.model_name,
                similarity_threshold=self.vector_engine.similarity_threshold
            )
        
        # Invalidate caches
        cache_manager = get_cache_manager()
        cache_manager.invalidate_search_cache()
        
        self._update_memory_metrics()
        
        logger.info(f"Cleared all documents: {total_cleared} total")
        return total_cleared
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database-specific statistics."""
        if not self.use_database or not self._db_repository:
            return {"database_enabled": False}
        
        try:
            stats = self._db_repository.get_memory_stats()
            stats["database_enabled"] = True
            stats["documents_loaded_in_memory"] = len(self._documents) if self._documents_loaded else 0
            stats["lazy_load_enabled"] = self.lazy_load
            return stats
        except Exception as e:
            logger.error(f"Failed to get database stats: {e}")
            return {
                "database_enabled": True,
                "error": str(e)
            }
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory and database statistics."""
        # Get base memory stats
        stats = super().get_memory_stats()
        
        # Add database stats
        db_stats = self.get_database_stats()
        stats.update(db_stats)
        
        return stats