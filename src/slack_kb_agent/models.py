from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import hashlib
import json
from enum import Enum


class DocumentType(Enum):
    """Document type enumeration for categorization."""
    
    TEXT = "text"
    CODE = "code"
    MARKDOWN = "markdown"
    ISSUE = "issue"
    PULL_REQUEST = "pull_request"
    SLACK_MESSAGE = "slack_message"
    WEB_CONTENT = "web_content"
    API_DOCUMENTATION = "api_documentation"


class SourceType(Enum):
    """Source type enumeration for tracking data origins."""
    
    GITHUB = "github"
    SLACK = "slack"
    FILE_SYSTEM = "file_system"
    WEB_CRAWL = "web_crawl"
    MANUAL_ENTRY = "manual_entry"
    API_IMPORT = "api_import"


@dataclass
class Document:
    """A piece of content stored in the knowledge base with comprehensive metadata."""

    content: str
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    doc_type: DocumentType = DocumentType.TEXT
    source_type: SourceType = SourceType.MANUAL_ENTRY
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None
    author: Optional[str] = None
    title: Optional[str] = None
    url: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    language: Optional[str] = None
    priority: int = 1  # 1=low, 5=high
    is_sensitive: bool = False
    
    def __post_init__(self):
        """Initialize derived fields after creation."""
        if not self.title and self.content:
            # Extract title from first line or generate from content
            lines = self.content.strip().split('\n')
            if lines and lines[0].strip():
                self.title = lines[0].strip()[:100]  # First 100 chars as title
            else:
                self.title = f"Document from {self.source}"
    
    @property
    def doc_id(self) -> str:
        """Generate a unique document ID based on content and source."""
        content_hash = hashlib.sha256(
            f"{self.content}{self.source}{self.created_at.isoformat()}".encode()
        ).hexdigest()
        return f"{self.source_type.value}_{content_hash[:16]}"
    
    @property
    def content_hash(self) -> str:
        """Generate hash of content for deduplication."""
        return hashlib.sha256(self.content.encode()).hexdigest()
    
    @property
    def word_count(self) -> int:
        """Calculate word count of the document content."""
        return len(self.content.split())
    
    @property
    def char_count(self) -> int:
        """Calculate character count of the document content."""
        return len(self.content)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert document to dictionary for serialization."""
        return {
            'doc_id': self.doc_id,
            'content': self.content,
            'source': self.source,
            'metadata': self.metadata,
            'doc_type': self.doc_type.value,
            'source_type': self.source_type.value,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'author': self.author,
            'title': self.title,
            'url': self.url,
            'tags': self.tags,
            'language': self.language,
            'priority': self.priority,
            'is_sensitive': self.is_sensitive,
            'word_count': self.word_count,
            'char_count': self.char_count,
            'content_hash': self.content_hash
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Document':
        """Create document from dictionary."""
        # Handle datetime parsing
        created_at = datetime.fromisoformat(data['created_at']) if data.get('created_at') else datetime.utcnow()
        updated_at = datetime.fromisoformat(data['updated_at']) if data.get('updated_at') else None
        
        return cls(
            content=data['content'],
            source=data['source'],
            metadata=data.get('metadata', {}),
            doc_type=DocumentType(data.get('doc_type', 'text')),
            source_type=SourceType(data.get('source_type', 'manual_entry')),
            created_at=created_at,
            updated_at=updated_at,
            author=data.get('author'),
            title=data.get('title'),
            url=data.get('url'),
            tags=data.get('tags', []),
            language=data.get('language'),
            priority=data.get('priority', 1),
            is_sensitive=data.get('is_sensitive', False)
        )
    
    def update_content(self, new_content: str, author: Optional[str] = None):
        """Update document content and metadata."""
        self.content = new_content
        self.updated_at = datetime.utcnow()
        if author:
            self.author = author
    
    def add_tags(self, tags: List[str]):
        """Add tags to the document."""
        self.tags.extend([tag for tag in tags if tag not in self.tags])
    
    def remove_tags(self, tags: List[str]):
        """Remove tags from the document."""
        self.tags = [tag for tag in self.tags if tag not in tags]
    
    def mark_sensitive(self, is_sensitive: bool = True):
        """Mark document as sensitive or not."""
        self.is_sensitive = is_sensitive
        if is_sensitive and 'sensitive' not in self.tags:
            self.tags.append('sensitive')
    
    def extract_metadata_from_content(self):
        """Extract useful metadata from content automatically."""
        # Extract code language from content
        if self.doc_type == DocumentType.CODE:
            first_line = self.content.split('\n')[0] if self.content else ""
            if first_line.startswith('```'):
                self.language = first_line[3:].strip()
        
        # Extract GitHub issue/PR metadata
        if self.source_type == SourceType.GITHUB:
            if '#' in self.source:
                self.metadata['issue_number'] = self.source.split('#')[-1]
        
        # Extract URL metadata
        if self.url:
            self.metadata['domain'] = self.url.split('/')[2] if '/' in self.url else self.url


@dataclass 
class SearchResult:
    """Represents a search result with scoring and relevance information."""
    
    document: Document
    score: float
    relevance_type: str = "keyword"  # "keyword", "semantic", "hybrid"
    matched_snippets: List[str] = field(default_factory=list)
    highlights: List[tuple] = field(default_factory=list)  # (start, end) positions
    explanation: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert search result to dictionary."""
        return {
            'document': self.document.to_dict(),
            'score': self.score,
            'relevance_type': self.relevance_type,
            'matched_snippets': self.matched_snippets,
            'highlights': self.highlights,
            'explanation': self.explanation
        }


@dataclass
class QueryContext:
    """Context information for query processing."""
    
    user_id: str
    channel_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    conversation_history: List[str] = field(default_factory=list)
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    query_intent: Optional[str] = None
    entities: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert query context to dictionary."""
        return {
            'user_id': self.user_id,
            'channel_id': self.channel_id,
            'timestamp': self.timestamp.isoformat(),
            'conversation_history': self.conversation_history,
            'user_preferences': self.user_preferences,
            'query_intent': self.query_intent,
            'entities': self.entities
        }


@dataclass
class AnalyticsEvent:
    """Analytics event for tracking usage and performance."""
    
    event_type: str
    user_id: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    query: Optional[str] = None
    response_time_ms: Optional[float] = None
    result_count: Optional[int] = None
    success: bool = True
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert analytics event to dictionary."""
        return {
            'event_type': self.event_type,
            'user_id': self.user_id,
            'timestamp': self.timestamp.isoformat(),
            'query': self.query,
            'response_time_ms': self.response_time_ms,
            'result_count': self.result_count,
            'success': self.success,
            'error_message': self.error_message,
            'metadata': self.metadata
        }


@dataclass
class KnowledgeBaseStats:
    """Statistics about the knowledge base state."""
    
    total_documents: int = 0
    documents_by_type: Dict[str, int] = field(default_factory=dict)
    documents_by_source: Dict[str, int] = field(default_factory=dict)
    total_words: int = 0
    total_characters: int = 0
    last_updated: Optional[datetime] = None
    index_size_mb: float = 0.0
    most_recent_document: Optional[datetime] = None
    oldest_document: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary."""
        return {
            'total_documents': self.total_documents,
            'documents_by_type': self.documents_by_type,
            'documents_by_source': self.documents_by_source,
            'total_words': self.total_words,
            'total_characters': self.total_characters,
            'last_updated': self.last_updated.isoformat() if self.last_updated else None,
            'index_size_mb': self.index_size_mb,
            'most_recent_document': self.most_recent_document.isoformat() if self.most_recent_document else None,
            'oldest_document': self.oldest_document.isoformat() if self.oldest_document else None
        }
