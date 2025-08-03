"""SQLAlchemy database models for persistent storage."""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from sqlalchemy import (
    Column, String, Text, DateTime, Boolean, Integer, Float, JSON,
    Index, ForeignKey, UniqueConstraint, CheckConstraint
)
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from sqlalchemy.orm import relationship, Mapped, mapped_column

from .connection import Base
from ..models import DocumentType, SourceType


class DocumentModel(Base):
    """Database model for documents with full-text search support."""
    
    __tablename__ = 'documents'
    
    # Primary fields
    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4()))
    doc_id: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    
    # Source information
    source: Mapped[str] = mapped_column(String(512), nullable=False, index=True)
    source_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    doc_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    
    # Metadata
    title: Mapped[Optional[str]] = mapped_column(String(512))
    author: Mapped[Optional[str]] = mapped_column(String(255))
    url: Mapped[Optional[str]] = mapped_column(String(2048))
    language: Mapped[Optional[str]] = mapped_column(String(10))
    
    # Document properties
    word_count: Mapped[int] = mapped_column(Integer, default=0)
    char_count: Mapped[int] = mapped_column(Integer, default=0)
    priority: Mapped[int] = mapped_column(Integer, default=1)
    is_sensitive: Mapped[bool] = mapped_column(Boolean, default=False)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[Optional[datetime]] = mapped_column(DateTime, onupdate=datetime.utcnow)
    
    # JSON fields
    metadata: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    tags: Mapped[List[str]] = mapped_column(ARRAY(String), default=list)
    
    # Relationships
    search_results = relationship("SearchResultModel", back_populates="document", cascade="all, delete-orphan")
    analytics_events = relationship("AnalyticsEventModel", back_populates="document")
    
    # Constraints and indexes
    __table_args__ = (
        Index('idx_documents_source_type', 'source_type'),
        Index('idx_documents_doc_type', 'doc_type'),
        Index('idx_documents_created_at', 'created_at'),
        Index('idx_documents_tags', 'tags', postgresql_using='gin'),
        Index('idx_documents_metadata', 'metadata', postgresql_using='gin'),
        Index('idx_documents_content_search', 'content', postgresql_using='gin'),
        CheckConstraint('priority >= 1 AND priority <= 5', name='check_priority_range'),
        CheckConstraint('word_count >= 0', name='check_word_count_positive'),
        CheckConstraint('char_count >= 0', name='check_char_count_positive'),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary."""
        return {
            'id': self.id,
            'doc_id': self.doc_id,
            'content': self.content,
            'content_hash': self.content_hash,
            'source': self.source,
            'source_type': self.source_type,
            'doc_type': self.doc_type,
            'title': self.title,
            'author': self.author,
            'url': self.url,
            'language': self.language,
            'word_count': self.word_count,
            'char_count': self.char_count,
            'priority': self.priority,
            'is_sensitive': self.is_sensitive,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'metadata': self.metadata,
            'tags': self.tags
        }
    
    @classmethod
    def from_document(cls, document: 'Document') -> 'DocumentModel':
        """Create DocumentModel from Document dataclass."""
        return cls(
            doc_id=document.doc_id,
            content=document.content,
            content_hash=document.content_hash,
            source=document.source,
            source_type=document.source_type.value,
            doc_type=document.doc_type.value,
            title=document.title,
            author=document.author,
            url=document.url,
            language=document.language,
            word_count=document.word_count,
            char_count=document.char_count,
            priority=document.priority,
            is_sensitive=document.is_sensitive,
            created_at=document.created_at,
            updated_at=document.updated_at,
            metadata=document.metadata,
            tags=document.tags
        )


class SearchResultModel(Base):
    """Database model for storing search results and analytics."""
    
    __tablename__ = 'search_results'
    
    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4()))
    
    # Search information
    query: Mapped[str] = mapped_column(Text, nullable=False)
    query_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    
    # Document reference
    document_id: Mapped[str] = mapped_column(UUID(as_uuid=False), ForeignKey('documents.id'), nullable=False)
    
    # Result metadata
    score: Mapped[float] = mapped_column(Float, nullable=False)
    relevance_type: Mapped[str] = mapped_column(String(20), nullable=False)
    matched_snippets: Mapped[List[str]] = mapped_column(ARRAY(Text), default=list)
    explanation: Mapped[Optional[str]] = mapped_column(Text)
    
    # Context
    user_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    channel_id: Mapped[Optional[str]] = mapped_column(String(255), index=True)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    document = relationship("DocumentModel", back_populates="search_results")
    
    # Indexes
    __table_args__ = (
        Index('idx_search_results_query_hash', 'query_hash'),
        Index('idx_search_results_user_id', 'user_id'),
        Index('idx_search_results_created_at', 'created_at'),
        Index('idx_search_results_score', 'score'),
    )


class AnalyticsEventModel(Base):
    """Database model for analytics events and usage tracking."""
    
    __tablename__ = 'analytics_events'
    
    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4()))
    
    # Event information
    event_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    user_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    
    # Query information
    query: Mapped[Optional[str]] = mapped_column(Text)
    response_time_ms: Mapped[Optional[float]] = mapped_column(Float)
    result_count: Mapped[Optional[int]] = mapped_column(Integer)
    
    # Status
    success: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    error_message: Mapped[Optional[str]] = mapped_column(Text)
    
    # Optional document reference
    document_id: Mapped[Optional[str]] = mapped_column(UUID(as_uuid=False), ForeignKey('documents.id'))
    
    # Additional metadata
    metadata: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    
    # Timestamp
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    document = relationship("DocumentModel", back_populates="analytics_events")
    
    # Indexes
    __table_args__ = (
        Index('idx_analytics_events_type', 'event_type'),
        Index('idx_analytics_events_user_id', 'user_id'),
        Index('idx_analytics_events_timestamp', 'timestamp'),
        Index('idx_analytics_events_success', 'success'),
    )


class UserProfileModel(Base):
    """Database model for user profiles and preferences."""
    
    __tablename__ = 'user_profiles'
    
    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4()))
    user_id: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    
    # Profile information
    display_name: Mapped[Optional[str]] = mapped_column(String(255))
    email: Mapped[Optional[str]] = mapped_column(String(255))
    timezone: Mapped[Optional[str]] = mapped_column(String(50))
    
    # Preferences
    preferred_sources: Mapped[List[str]] = mapped_column(ARRAY(String), default=list)
    preferred_doc_types: Mapped[List[str]] = mapped_column(ARRAY(String), default=list)
    expertise_level: Mapped[str] = mapped_column(String(20), default='intermediate')
    
    # Usage statistics
    total_queries: Mapped[int] = mapped_column(Integer, default=0)
    successful_queries: Mapped[int] = mapped_column(Integer, default=0)
    last_active: Mapped[Optional[datetime]] = mapped_column(DateTime)
    
    # Settings
    notifications_enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    advanced_features: Mapped[bool] = mapped_column(Boolean, default=False)
    
    # JSON preferences
    custom_preferences: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[Optional[datetime]] = mapped_column(DateTime, onupdate=datetime.utcnow)
    
    # Constraints
    __table_args__ = (
        CheckConstraint("expertise_level IN ('beginner', 'intermediate', 'expert')", name='check_expertise_level'),
        CheckConstraint('total_queries >= 0', name='check_total_queries_positive'),
        CheckConstraint('successful_queries >= 0', name='check_successful_queries_positive'),
        CheckConstraint('successful_queries <= total_queries', name='check_success_ratio'),
    )


class KnowledgeBaseStatsModel(Base):
    """Database model for knowledge base statistics snapshots."""
    
    __tablename__ = 'kb_stats'
    
    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4()))
    
    # Snapshot timestamp
    snapshot_date: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    
    # Document statistics
    total_documents: Mapped[int] = mapped_column(Integer, default=0)
    total_words: Mapped[int] = mapped_column(Integer, default=0)
    total_characters: Mapped[int] = mapped_column(Integer, default=0)
    
    # Breakdowns
    documents_by_type: Mapped[Dict[str, int]] = mapped_column(JSON, default=dict)
    documents_by_source: Mapped[Dict[str, int]] = mapped_column(JSON, default=dict)
    
    # Performance metrics
    avg_query_time_ms: Mapped[Optional[float]] = mapped_column(Float)
    cache_hit_rate: Mapped[Optional[float]] = mapped_column(Float)
    index_size_mb: Mapped[float] = mapped_column(Float, default=0.0)
    
    # Date ranges
    oldest_document: Mapped[Optional[datetime]] = mapped_column(DateTime)
    newest_document: Mapped[Optional[datetime]] = mapped_column(DateTime)
    
    # Constraints
    __table_args__ = (
        CheckConstraint('total_documents >= 0', name='check_total_documents_positive'),
        CheckConstraint('total_words >= 0', name='check_total_words_positive'),
        CheckConstraint('index_size_mb >= 0', name='check_index_size_positive'),
        CheckConstraint('cache_hit_rate >= 0 AND cache_hit_rate <= 1', name='check_cache_hit_rate_range'),
    )


class CacheEntryModel(Base):
    """Database model for persistent cache entries."""
    
    __tablename__ = 'cache_entries'
    
    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4()))
    
    # Cache key and data
    cache_key: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    cache_value: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False)
    
    # Cache metadata
    cache_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    hit_count: Mapped[int] = mapped_column(Integer, default=0)
    
    # Expiration
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    last_accessed: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Constraints and indexes
    __table_args__ = (
        Index('idx_cache_entries_expires_at', 'expires_at'),
        Index('idx_cache_entries_type', 'cache_type'),
        Index('idx_cache_entries_last_accessed', 'last_accessed'),
    )


class BackupMetadataModel(Base):
    """Database model for backup metadata and history."""
    
    __tablename__ = 'backup_metadata'
    
    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4()))
    
    # Backup information
    backup_name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    backup_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    backup_type: Mapped[str] = mapped_column(String(50), nullable=False)  # 'full', 'incremental', 'export'
    
    # Backup statistics
    total_documents: Mapped[int] = mapped_column(Integer, default=0)
    total_events: Mapped[int] = mapped_column(Integer, default=0)
    backup_size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    
    # Status
    status: Mapped[str] = mapped_column(String(20), default='in_progress')  # 'in_progress', 'completed', 'failed'
    error_message: Mapped[Optional[str]] = mapped_column(Text)
    
    # Timestamps
    started_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    
    # Metadata
    metadata: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    
    # Constraints
    __table_args__ = (
        CheckConstraint("backup_type IN ('full', 'incremental', 'export')", name='check_backup_type'),
        CheckConstraint("status IN ('in_progress', 'completed', 'failed')", name='check_backup_status'),
        Index('idx_backup_metadata_started_at', 'started_at'),
        Index('idx_backup_metadata_status', 'status'),
    )


# Migration helper functions
async def create_all_tables(engine):
    """Create all database tables."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def drop_all_tables(engine):
    """Drop all database tables."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


# Database utility functions
def get_model_by_table_name(table_name: str):
    """Get SQLAlchemy model class by table name."""
    model_mapping = {
        'documents': DocumentModel,
        'search_results': SearchResultModel,
        'analytics_events': AnalyticsEventModel,
        'user_profiles': UserProfileModel,
        'kb_stats': KnowledgeBaseStatsModel,
        'cache_entries': CacheEntryModel,
        'backup_metadata': BackupMetadataModel
    }
    return model_mapping.get(table_name)