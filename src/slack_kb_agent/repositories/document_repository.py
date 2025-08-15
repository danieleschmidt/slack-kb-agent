"""Document repository with advanced search and analytics capabilities."""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from sqlalchemy import func, or_, select, text

from ..database.connection import get_db_session
from ..database.models import DocumentModel
from ..exceptions import RepositoryError
from ..models import Document, DocumentType, SourceType
from ..utils import calculate_content_hash
from .base_repository import BaseRepository

logger = logging.getLogger(__name__)


class DocumentRepository(BaseRepository[DocumentModel]):
    """Repository for document CRUD operations with advanced search capabilities."""

    def __init__(self):
        super().__init__(DocumentModel)

    async def create_from_document(self, document: Document) -> DocumentModel:
        """
        Create a database document from a domain Document object.
        
        Args:
            document: Domain Document object
            
        Returns:
            Created DocumentModel instance
        """
        try:
            # Calculate content hash for deduplication
            content_hash = calculate_content_hash(document.content)

            # Check if document with same hash already exists
            existing = await self.find_by_content_hash(content_hash)
            if existing:
                logger.debug(f"Document with hash {content_hash} already exists")
                return existing

            # Create new document
            doc_data = {
                'doc_id': str(uuid4()),
                'content': document.content,
                'content_hash': content_hash,
                'source': document.source,
                'source_type': document.source_type.value,
                'doc_type': document.doc_type.value,
                'title': document.title,
                'author': document.author,
                'url': document.url,
                'language': document.language,
                'word_count': len(document.content.split()),
                'char_count': len(document.content),
                'priority': document.priority,
                'is_sensitive': document.is_sensitive,
                'tags': document.tags,
                'metadata': document.metadata,
                'created_at': document.created_at,
                'updated_at': document.updated_at
            }

            return await self.create(**doc_data)

        except Exception as e:
            logger.error(f"Failed to create document: {e}")
            raise RepositoryError(f"Document creation failed: {e}")

    async def find_by_content_hash(self, content_hash: str) -> Optional[DocumentModel]:
        """Find document by content hash."""
        try:
            async with get_db_session() as session:
                stmt = select(DocumentModel).where(DocumentModel.content_hash == content_hash)
                result = await session.execute(stmt)
                return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"Failed to find document by hash {content_hash}: {e}")
            return None

    async def search_by_keywords(
        self,
        keywords: List[str],
        limit: int = 10,
        offset: int = 0,
        source_types: Optional[List[SourceType]] = None,
        doc_types: Optional[List[DocumentType]] = None
    ) -> List[DocumentModel]:
        """
        Search documents by keywords using full-text search.
        
        Args:
            keywords: List of keywords to search for
            limit: Maximum number of results
            offset: Number of results to skip
            source_types: Filter by source types
            doc_types: Filter by document types
            
        Returns:
            List of matching documents ordered by relevance
        """
        try:
            async with get_db_session() as session:
                # Build full-text search query
                search_terms = ' & '.join(keywords)

                stmt = select(DocumentModel).where(
                    func.to_tsvector('english', DocumentModel.content).op('@@')(
                        func.plainto_tsquery('english', search_terms)
                    )
                )

                # Add filters
                if source_types:
                    source_values = [st.value for st in source_types]
                    stmt = stmt.where(DocumentModel.source_type.in_(source_values))

                if doc_types:
                    doc_type_values = [dt.value for dt in doc_types]
                    stmt = stmt.where(DocumentModel.doc_type.in_(doc_type_values))

                # Order by relevance and limit
                stmt = stmt.order_by(
                    func.ts_rank(
                        func.to_tsvector('english', DocumentModel.content),
                        func.plainto_tsquery('english', search_terms)
                    ).desc()
                ).limit(limit).offset(offset)

                result = await session.execute(stmt)
                return result.scalars().all()

        except Exception as e:
            logger.error(f"Keyword search failed: {e}")
            raise RepositoryError(f"Search operation failed: {e}")

    async def search_by_similarity(
        self,
        query_vector: List[float],
        similarity_threshold: float = 0.5,
        limit: int = 10
    ) -> List[Tuple[DocumentModel, float]]:
        """
        Search documents by vector similarity.
        
        Args:
            query_vector: Query embedding vector
            similarity_threshold: Minimum similarity score
            limit: Maximum number of results
            
        Returns:
            List of (document, similarity_score) tuples
        """
        try:
            async with get_db_session() as session:
                # Convert vector to PostgreSQL array format
                vector_str = '[' + ','.join(map(str, query_vector)) + ']'

                # Use cosine similarity with vector extension
                stmt = text("""
                    SELECT d.*, 
                           (d.embedding <=> :query_vector::vector) AS similarity
                    FROM documents d
                    WHERE d.embedding IS NOT NULL
                      AND (d.embedding <=> :query_vector::vector) < :threshold
                    ORDER BY similarity
                    LIMIT :limit
                """).bindparams(
                    query_vector=vector_str,
                    threshold=1 - similarity_threshold,  # Convert similarity to distance
                    limit=limit
                )

                result = await session.execute(stmt)
                rows = result.fetchall()

                # Convert to DocumentModel instances with similarity scores
                documents_with_scores = []
                for row in rows:
                    doc = DocumentModel(**{c.name: getattr(row, c.name) for c in DocumentModel.__table__.columns})
                    similarity = 1 - row.similarity  # Convert distance back to similarity
                    documents_with_scores.append((doc, similarity))

                return documents_with_scores

        except Exception as e:
            logger.error(f"Vector similarity search failed: {e}")
            raise RepositoryError(f"Vector search operation failed: {e}")

    async def get_documents_by_source(
        self,
        source: str,
        include_children: bool = False
    ) -> List[DocumentModel]:
        """Get all documents from a specific source."""
        try:
            async with get_db_session() as session:
                if include_children:
                    # Include documents from child sources (e.g., subdirectories)
                    stmt = select(DocumentModel).where(
                        DocumentModel.source.like(f"{source}%")
                    )
                else:
                    stmt = select(DocumentModel).where(DocumentModel.source == source)

                stmt = stmt.order_by(DocumentModel.created_at.desc())
                result = await session.execute(stmt)
                return result.scalars().all()

        except Exception as e:
            logger.error(f"Failed to get documents by source {source}: {e}")
            raise RepositoryError(f"Source query failed: {e}")

    async def get_recent_documents(
        self,
        days: int = 7,
        limit: int = 50
    ) -> List[DocumentModel]:
        """Get recently created or updated documents."""
        try:
            async with get_db_session() as session:
                cutoff_date = datetime.utcnow() - timedelta(days=days)

                stmt = select(DocumentModel).where(
                    or_(
                        DocumentModel.created_at >= cutoff_date,
                        DocumentModel.updated_at >= cutoff_date
                    )
                ).order_by(
                    func.coalesce(DocumentModel.updated_at, DocumentModel.created_at).desc()
                ).limit(limit)

                result = await session.execute(stmt)
                return result.scalars().all()

        except Exception as e:
            logger.error(f"Failed to get recent documents: {e}")
            raise RepositoryError(f"Recent documents query failed: {e}")

    async def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive document statistics."""
        try:
            async with get_db_session() as session:
                # Total counts
                total_docs = await session.scalar(select(func.count(DocumentModel.id)))

                # By source type
                source_type_stats = await session.execute(
                    select(
                        DocumentModel.source_type,
                        func.count(DocumentModel.id)
                    ).group_by(DocumentModel.source_type)
                )

                # By document type
                doc_type_stats = await session.execute(
                    select(
                        DocumentModel.doc_type,
                        func.count(DocumentModel.id)
                    ).group_by(DocumentModel.doc_type)
                )

                # Content statistics
                content_stats = await session.execute(
                    select(
                        func.avg(DocumentModel.word_count).label('avg_words'),
                        func.sum(DocumentModel.word_count).label('total_words'),
                        func.avg(DocumentModel.char_count).label('avg_chars'),
                        func.sum(DocumentModel.char_count).label('total_chars')
                    )
                )

                # Recent activity
                recent_count = await session.scalar(
                    select(func.count(DocumentModel.id)).where(
                        DocumentModel.created_at >= datetime.utcnow() - timedelta(days=7)
                    )
                )

                content_row = content_stats.first()

                return {
                    'total_documents': total_docs,
                    'by_source_type': dict(source_type_stats.fetchall()),
                    'by_doc_type': dict(doc_type_stats.fetchall()),
                    'content_stats': {
                        'avg_words': float(content_row.avg_words or 0),
                        'total_words': int(content_row.total_words or 0),
                        'avg_chars': float(content_row.avg_chars or 0),
                        'total_chars': int(content_row.total_chars or 0)
                    },
                    'recent_documents_7d': recent_count,
                    'generated_at': datetime.utcnow().isoformat()
                }

        except Exception as e:
            logger.error(f"Failed to get document statistics: {e}")
            raise RepositoryError(f"Statistics query failed: {e}")

    async def cleanup_old_documents(
        self,
        retention_days: int = 365,
        exclude_high_priority: bool = True
    ) -> int:
        """
        Clean up old documents based on retention policy.
        
        Args:
            retention_days: Documents older than this will be removed
            exclude_high_priority: If True, keep high priority docs (priority >= 4)
            
        Returns:
            Number of documents deleted
        """
        try:
            async with get_db_session() as session:
                cutoff_date = datetime.utcnow() - timedelta(days=retention_days)

                stmt = delete(DocumentModel).where(
                    DocumentModel.created_at < cutoff_date
                )

                if exclude_high_priority:
                    stmt = stmt.where(DocumentModel.priority < 4)

                result = await session.execute(stmt)
                await session.commit()

                deleted_count = result.rowcount
                logger.info(f"Cleaned up {deleted_count} old documents")
                return deleted_count

        except Exception as e:
            logger.error(f"Failed to cleanup old documents: {e}")
            raise RepositoryError(f"Cleanup operation failed: {e}")

    async def bulk_update_embeddings(
        self,
        document_embeddings: List[Tuple[str, List[float]]]
    ) -> int:
        """
        Bulk update document embeddings.
        
        Args:
            document_embeddings: List of (doc_id, embedding) tuples
            
        Returns:
            Number of documents updated
        """
        try:
            async with get_db_session() as session:
                updated_count = 0

                for doc_id, embedding in document_embeddings:
                    vector_str = '[' + ','.join(map(str, embedding)) + ']'

                    stmt = text("""
                        UPDATE documents 
                        SET embedding = :embedding::vector
                        WHERE doc_id = :doc_id
                    """).bindparams(
                        embedding=vector_str,
                        doc_id=doc_id
                    )

                    result = await session.execute(stmt)
                    if result.rowcount > 0:
                        updated_count += 1

                await session.commit()
                logger.info(f"Updated embeddings for {updated_count} documents")
                return updated_count

        except Exception as e:
            logger.error(f"Failed to bulk update embeddings: {e}")
            raise RepositoryError(f"Bulk embedding update failed: {e}")

    def to_domain_document(self, doc_model: DocumentModel) -> Document:
        """Convert DocumentModel to domain Document object."""
        return Document(
            content=doc_model.content,
            source=doc_model.source,
            metadata=doc_model.metadata or {},
            doc_type=DocumentType(doc_model.doc_type),
            source_type=SourceType(doc_model.source_type),
            created_at=doc_model.created_at,
            updated_at=doc_model.updated_at,
            author=doc_model.author,
            title=doc_model.title,
            url=doc_model.url,
            tags=doc_model.tags or [],
            language=doc_model.language,
            priority=doc_model.priority,
            is_sensitive=doc_model.is_sensitive
        )
