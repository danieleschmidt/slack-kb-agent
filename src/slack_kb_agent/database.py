"""Database models and connection management for PostgreSQL persistence."""

from __future__ import annotations

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
    Engine
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
from sqlalchemy.exc import SQLAlchemyError, OperationalError, DatabaseError, IntegrityError

from .models import Document

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
        
        # Create engine with connection pooling
        self.engine = create_engine(
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
        
    def initialize(self) -> None:
        """Initialize database schema."""
        try:
            # Create all tables
            Base.metadata.create_all(bind=self.engine)
            self._initialized = True
            logger.info("Database schema initialized successfully")
        except SQLAlchemyError as e:
            logger.error(f"Failed to initialize database schema: {e}")
            raise
    
    def is_available(self) -> bool:
        """Check if database is available and accessible."""
        try:
            with self.get_session() as session:
                session.execute('SELECT 1')
                return True
        except (OperationalError, DatabaseError) as e:
            logger.warning(f"Database connection failed: {type(e).__name__}: {e}")
            return False
        except SQLAlchemyError as e:
            logger.error(f"Unexpected database error during availability check: {type(e).__name__}: {e}")
            return False
    
    @contextmanager
    def get_session(self):
        """Get a database session with automatic cleanup."""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except (SQLAlchemyError, Exception) as e:
            session.rollback()
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


class DatabaseRepository:
    """Repository pattern for document operations."""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        
    def create_document(self, document: Document) -> int:
        """Create a new document and return its ID."""
        with self.db_manager.get_session() as session:
            doc_model = DocumentModel.from_document(document)
            session.add(doc_model)
            session.flush()  # Flush to get the ID without committing
            return doc_model.id
    
    def create_documents(self, documents: List[Document]) -> List[int]:
        """Create multiple documents and return their IDs."""
        with self.db_manager.get_session() as session:
            doc_models = [DocumentModel.from_document(doc) for doc in documents]
            session.add_all(doc_models)
            session.flush()
            return [doc.id for doc in doc_models]
    
    def get_document(self, doc_id: int) -> Optional[Document]:
        """Get a document by ID."""
        with self.db_manager.get_session() as session:
            doc_model = session.query(DocumentModel).filter(DocumentModel.id == doc_id).first()
            return doc_model.to_document() if doc_model else None
    
    def get_all_documents(self, limit: Optional[int] = None, offset: int = 0) -> List[Document]:
        """Get all documents with optional pagination."""
        with self.db_manager.get_session() as session:
            query = session.query(DocumentModel).order_by(DocumentModel.created_at.desc())
            
            if offset > 0:
                query = query.offset(offset)
            if limit is not None:
                query = query.limit(limit)
                
            doc_models = query.all()
            return [doc.to_document() for doc in doc_models]
    
    def get_documents_by_source(self, source: str, limit: Optional[int] = None) -> List[Document]:
        """Get documents from a specific source."""
        with self.db_manager.get_session() as session:
            query = session.query(DocumentModel).filter(DocumentModel.source == source)
            
            if limit is not None:
                query = query.limit(limit)
                
            doc_models = query.all()
            return [doc.to_document() for doc in doc_models]
    
    def search_documents(self, query: str, limit: int = 100) -> List[Document]:
        """Search documents by content (basic text search)."""
        with self.db_manager.get_session() as session:
            # Use PostgreSQL ILIKE for case-insensitive search
            doc_models = (
                session.query(DocumentModel)
                .filter(DocumentModel.content.ilike(f'%{query}%'))
                .order_by(DocumentModel.created_at.desc())
                .limit(limit)
                .all()
            )
            return [doc.to_document() for doc in doc_models]
    
    def count_documents(self) -> int:
        """Count total number of documents."""
        with self.db_manager.get_session() as session:
            return session.query(DocumentModel).count()
    
    def count_documents_by_source(self, source: str) -> int:
        """Count documents from a specific source."""
        with self.db_manager.get_session() as session:
            return session.query(DocumentModel).filter(DocumentModel.source == source).count()
    
    def delete_document(self, doc_id: int) -> bool:
        """Delete a document by ID. Returns True if deleted, False if not found."""
        with self.db_manager.get_session() as session:
            doc_model = session.query(DocumentModel).filter(DocumentModel.id == doc_id).first()
            if doc_model:
                session.delete(doc_model)
                return True
            return False
    
    def delete_documents_by_source(self, source: str) -> int:
        """Delete all documents from a specific source. Returns count of deleted documents."""
        with self.db_manager.get_session() as session:
            count = session.query(DocumentModel).filter(DocumentModel.source == source).count()
            session.query(DocumentModel).filter(DocumentModel.source == source).delete()
            return count
    
    def clear_all_documents(self) -> int:
        """Delete all documents. Returns count of deleted documents."""
        with self.db_manager.get_session() as session:
            count = session.query(DocumentModel).count()
            session.query(DocumentModel).delete()
            return count
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
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
                "database_url": self.db_manager.database_url.split('@')[-1] if '@' in self.db_manager.database_url else "local"
            }


# Global database manager instance
_db_manager: Optional[DatabaseManager] = None
_db_repository: Optional[DatabaseRepository] = None


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