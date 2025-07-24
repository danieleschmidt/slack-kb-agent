"""Database models and connection management for PostgreSQL persistence."""

from __future__ import annotations

import atexit
import os
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
from contextlib import contextmanager
from dataclasses import asdict

from sqlalchemy import (
    create_engine, 
    Column, 
    Integer, 
    String, 
    Text, 
    DateTime, 
    JSON,
    Index,
    Engine,
    func
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
from sqlalchemy.exc import SQLAlchemyError, OperationalError, DatabaseError, IntegrityError

from .models import Document
from .security_utils import mask_database_url, get_safe_repr
from .circuit_breaker import CircuitBreaker, CircuitBreakerConfig
from .constants import CircuitBreakerDefaults

logger = logging.getLogger(__name__)

Base = declarative_base()


class DocumentModel(Base):
    """SQLAlchemy model for Document storage."""
    
    __tablename__ = 'documents'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    content = Column(Text, nullable=False)
    source = Column(String(255), nullable=False)
    metadata = Column(JSON, default=dict)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    
    # Add indexes for common queries
    __table_args__ = (
        Index('idx_documents_source', 'source'),
        Index('idx_documents_created_at', 'created_at'),
        Index('idx_documents_content_search', 'content'),  # For full-text search support
    )
    
    def to_document(self) -> Document:
        """Convert SQLAlchemy model to Document dataclass."""
        return Document(
            content=self.content,
            source=self.source,
            metadata=self.metadata or {}
        )
    
    @classmethod
    def from_document(cls, document: Document) -> DocumentModel:
        """Create SQLAlchemy model from Document dataclass."""
        return cls(
            content=document.content,
            source=document.source,
            metadata=document.metadata
        )


class DatabaseManager:
    """Manages PostgreSQL database connections and operations."""
    
    def __init__(
        self,
        database_url: Optional[str] = None,
        pool_size: int = 10,
        max_overflow: int = 20,
        pool_pre_ping: bool = True,
        echo: bool = False
    ):
        """Initialize database manager.
        
        Args:
            database_url: PostgreSQL connection URL. If None, reads from DATABASE_URL env var
            pool_size: Number of connections to keep open in the pool
            max_overflow: Additional connections that can be created beyond pool_size
            pool_pre_ping: Validate connections before use
            echo: Enable SQL query logging
        """
        self.database_url = database_url or os.getenv('DATABASE_URL')
        if not self.database_url:
            raise ValueError(
                "DATABASE_URL environment variable is required. "
                "Example: postgresql://user:password@localhost:5432/dbname"
            )
        
        # Initialize circuit breaker for database operations
        self.circuit_breaker = self._get_circuit_breaker()
        
        # Create engine with connection pooling using circuit breaker protection
        self.engine = self.circuit_breaker.call(
            create_engine,
            self.database_url,
            poolclass=QueuePool,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_pre_ping=pool_pre_ping,
            echo=echo,
            # Additional pool settings for production
            pool_recycle=3600,  # Recycle connections after 1 hour
            pool_timeout=30,    # Timeout when getting connection from pool
        )
        
        # Create session factory
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )
        
        self._initialized = False
    
    def _get_circuit_breaker(self) -> CircuitBreaker:
        """Get or create circuit breaker for database operations."""
        circuit_config = CircuitBreakerConfig(
            failure_threshold=CircuitBreakerDefaults.DATABASE_FAILURE_THRESHOLD,
            success_threshold=CircuitBreakerDefaults.DATABASE_SUCCESS_THRESHOLD,
            timeout_seconds=CircuitBreakerDefaults.DATABASE_TIMEOUT_SECONDS,
            half_open_max_requests=CircuitBreakerDefaults.DATABASE_HALF_OPEN_MAX_REQUESTS,
            failure_window_seconds=CircuitBreakerDefaults.DATABASE_FAILURE_WINDOW_SECONDS,
            service_name="database"
        )
        return CircuitBreaker(circuit_config)
        
    def initialize(self) -> None:
        """Initialize database schema."""
        try:
            # Create all tables with circuit breaker protection
            self.circuit_breaker.call(Base.metadata.create_all, bind=self.engine)
            self._initialized = True
            logger.info("Database schema initialized successfully")
        except SQLAlchemyError as e:
            logger.error(f"Failed to initialize database schema: {e}")
            raise
    
    def is_available(self) -> bool:
        """Check if database is available and accessible."""
        try:
            with self.get_session() as session:
                self.circuit_breaker.call(session.execute, 'SELECT 1')
                return True
        except (OperationalError, DatabaseError) as e:
            logger.warning(f"Database connection failed: {type(e).__name__}: {e}")
            return False
        except SQLAlchemyError as e:
            logger.error(f"Unexpected database error during availability check: {type(e).__name__}: {e}")
            return False
    
    @contextmanager
    def get_session(self):
        """Get a database session with automatic cleanup and circuit breaker protection."""
        session = self.circuit_breaker.call(self.SessionLocal)
        try:
            yield session
            self.circuit_breaker.call(session.commit)
        except (SQLAlchemyError, Exception) as e:
            self.circuit_breaker.call(session.rollback)
            if isinstance(e, SQLAlchemyError):
                logger.error(f"Database operation failed: {type(e).__name__}: {e}")
            raise
        finally:
            session.close()
    
    def get_engine(self) -> Engine:
        """Get the SQLAlchemy engine."""
        return self.engine
    
    def close(self) -> None:
        """Close database connections."""
        if hasattr(self, 'engine'):
            self.engine.dispose()
            logger.info("Database connections closed")
    
    def __str__(self) -> str:
        """Return a safe string representation without exposing credentials."""
        return get_safe_repr(self)
    
    def __repr__(self) -> str:
        """Return a safe representation without exposing credentials."""
        masked_url = mask_database_url(self.database_url)
        return f"DatabaseManager(database_url='{masked_url}')"


class DatabaseRepository:
    """Repository pattern for document operations."""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.circuit_breaker = self._get_circuit_breaker()
    
    def _get_circuit_breaker(self) -> CircuitBreaker:
        """Get or create circuit breaker for repository operations."""
        circuit_config = CircuitBreakerConfig(
            failure_threshold=CircuitBreakerDefaults.DATABASE_FAILURE_THRESHOLD,
            success_threshold=CircuitBreakerDefaults.DATABASE_SUCCESS_THRESHOLD,
            timeout_seconds=CircuitBreakerDefaults.DATABASE_TIMEOUT_SECONDS,
            half_open_max_requests=CircuitBreakerDefaults.DATABASE_HALF_OPEN_MAX_REQUESTS,
            failure_window_seconds=CircuitBreakerDefaults.DATABASE_FAILURE_WINDOW_SECONDS,
            service_name="database"
        )
        return CircuitBreaker(circuit_config)
        
    def create_document(self, document: Document) -> int:
        """Create a new document and return its ID."""
        def _create_document():
            with self.db_manager.get_session() as session:
                doc_model = DocumentModel.from_document(document)
                session.add(doc_model)
                session.flush()  # Flush to get the ID without committing
                return doc_model.id
        
        return self.circuit_breaker.call(_create_document)
    
    def create_documents(self, documents: List[Document]) -> List[int]:
        """Create multiple documents and return their IDs."""
        def _create_documents():
            with self.db_manager.get_session() as session:
                doc_models = [DocumentModel.from_document(doc) for doc in documents]
                session.add_all(doc_models)
                session.flush()
                return [doc.id for doc in doc_models]
        
        return self.circuit_breaker.call(_create_documents)
    
    def get_document(self, doc_id: int) -> Optional[Document]:
        """Get a document by ID."""
        def _get_document():
            with self.db_manager.get_session() as session:
                doc_model = session.query(DocumentModel).filter(DocumentModel.id == doc_id).first()
                return doc_model.to_document() if doc_model else None
        
        return self.circuit_breaker.call(_get_document)
    
    def get_all_documents(self, limit: Optional[int] = None, offset: int = 0) -> List[Document]:
        """Get all documents with optional pagination."""
        def _get_all_documents():
            with self.db_manager.get_session() as session:
                query = session.query(DocumentModel).order_by(DocumentModel.created_at.desc())
                
                if offset > 0:
                    query = query.offset(offset)
                if limit is not None:
                    query = query.limit(limit)
                    
                doc_models = query.all()
                return [doc.to_document() for doc in doc_models]
        
        return self.circuit_breaker.call(_get_all_documents)
    
    def get_documents_by_source(self, source: str, limit: Optional[int] = None) -> List[Document]:
        """Get documents from a specific source."""
        def _get_documents_by_source():
            with self.db_manager.get_session() as session:
                query = session.query(DocumentModel).filter(DocumentModel.source == source)
                
                if limit is not None:
                    query = query.limit(limit)
                    
                doc_models = query.all()
                return [doc.to_document() for doc in doc_models]
        
        return self.circuit_breaker.call(_get_documents_by_source)
    
    def search_documents(self, query: str, limit: int = 100) -> List[Document]:
        """Search documents by content (basic text search with SQL injection protection)."""
        # Input validation and sanitization
        if not isinstance(query, str):
            raise ValueError("Search query must be a string")
        
        # Limit query length to prevent DoS attacks
        if len(query) > 1000:
            raise ValueError("Search query too long (max 1000 characters)")
        
        # Strip dangerous characters but preserve search intent
        sanitized_query = query.strip()
        if not sanitized_query:
            return []
        
        def _search_documents():
            with self.db_manager.get_session() as session:
                try:
                    # Use SQLAlchemy's concat function for safe parameter binding
                    # This prevents SQL injection by using proper parameter substitution
                    search_pattern = func.concat('%', sanitized_query, '%')
                    
                    doc_models = (
                        session.query(DocumentModel)
                        .filter(func.lower(DocumentModel.content).like(func.lower(search_pattern)))
                        .order_by(DocumentModel.created_at.desc())
                        .limit(limit)
                        .all()
                    )
                    return [doc.to_document() for doc in doc_models]
                except SQLAlchemyError as e:
                    logger.error(f"Database search failed for query '{sanitized_query}': {type(e).__name__}: {e}")
                    raise
        
        return self.circuit_breaker.call(_search_documents)
    
    def count_documents(self) -> int:
        """Count total number of documents."""
        def _count_documents():
            with self.db_manager.get_session() as session:
                return session.query(DocumentModel).count()
        
        return self.circuit_breaker.call(_count_documents)
    
    def count_documents_by_source(self, source: str) -> int:
        """Count documents from a specific source."""
        def _count_documents_by_source():
            with self.db_manager.get_session() as session:
                return session.query(DocumentModel).filter(DocumentModel.source == source).count()
        
        return self.circuit_breaker.call(_count_documents_by_source)
    
    def delete_document(self, doc_id: int) -> bool:
        """Delete a document by ID. Returns True if deleted, False if not found."""
        def _delete_document():
            with self.db_manager.get_session() as session:
                doc_model = session.query(DocumentModel).filter(DocumentModel.id == doc_id).first()
                if doc_model:
                    session.delete(doc_model)
                    return True
                return False
        
        return self.circuit_breaker.call(_delete_document)
    
    def delete_documents_by_source(self, source: str) -> int:
        """Delete all documents from a specific source. Returns count of deleted documents."""
        def _delete_documents_by_source():
            with self.db_manager.get_session() as session:
                count = session.query(DocumentModel).filter(DocumentModel.source == source).count()
                session.query(DocumentModel).filter(DocumentModel.source == source).delete()
                return count
        
        return self.circuit_breaker.call(_delete_documents_by_source)
    
    def clear_all_documents(self) -> int:
        """Delete all documents. Returns count of deleted documents."""
        def _clear_all_documents():
            with self.db_manager.get_session() as session:
                count = session.query(DocumentModel).count()
                session.query(DocumentModel).delete()
                return count
        
        return self.circuit_breaker.call(_clear_all_documents)
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        def _get_memory_stats():
            with self.db_manager.get_session() as session:
                total_docs = session.query(DocumentModel).count()
                
                # Get source distribution
                source_stats = (
                    session.query(DocumentModel.source, session.query(DocumentModel).filter(DocumentModel.source == DocumentModel.source).count())
                    .group_by(DocumentModel.source)
                    .all()
                )
                
                # Estimate database size (rough approximation)
                size_query = session.execute(
                    "SELECT pg_total_relation_size('documents') as size"
                ).fetchone()
                estimated_size = size_query[0] if size_query else 0
                
                return {
                    "total_documents": total_docs,
                    "source_distribution": dict(source_stats),
                    "estimated_size_bytes": estimated_size,
                    "database_url": mask_database_url(self.db_manager.database_url)
                }
        
        return self.circuit_breaker.call(_get_memory_stats)


# Global database manager instance
_db_manager: Optional[DatabaseManager] = None
_db_repository: Optional[DatabaseRepository] = None


def _cleanup_database_resources() -> None:
    """Clean up global database resources on application shutdown."""
    global _db_manager, _db_repository
    
    if _db_manager is not None:
        try:
            _db_manager.close()
            logger.info("Database connection pool closed during cleanup")
        except Exception as e:
            logger.warning(f"Error closing database manager during cleanup: {e}")
        finally:
            _db_manager = None
    
    _db_repository = None


# Register cleanup function to run on application exit
atexit.register(_cleanup_database_resources)


def get_database_manager() -> DatabaseManager:
    """Get or create the global database manager."""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
        try:
            _db_manager.initialize()
        except (OperationalError, DatabaseError) as e:
            logger.warning(f"Database initialization failed: {type(e).__name__}: {e}")
        except SQLAlchemyError as e:
            logger.error(f"Unexpected database error during initialization: {type(e).__name__}: {e}")
    return _db_manager


def get_database_repository() -> DatabaseRepository:
    """Get or create the global database repository."""
    global _db_repository
    if _db_repository is None:
        _db_repository = DatabaseRepository(get_database_manager())
    return _db_repository


def is_database_available() -> bool:
    """Check if database functionality is available."""
    try:
        return get_database_manager().is_available()
    except (OperationalError, DatabaseError, SQLAlchemyError):
        # Database errors are expected when DB is not available
        return False
    except Exception as e:
        logger.error(f"Unexpected error checking database availability: {type(e).__name__}: {e}")
        return False