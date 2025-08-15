"""Core knowledge management service with advanced search and analytics capabilities."""

import logging
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from ..cache import CacheManager
from ..exceptions import KnowledgeBaseError, SearchError, ValidationError
from ..models import (
    AnalyticsEvent,
    Document,
    DocumentType,
    KnowledgeBaseStats,
    QueryContext,
    SearchResult,
)
from ..utils import extract_keywords, sanitize_input
from ..vector_search import VectorSearchEngine

logger = logging.getLogger(__name__)


class KnowledgeService:
    """
    Core service for knowledge base operations including document management,
    intelligent search, and analytics tracking.
    """

    def __init__(
        self,
        vector_engine: Optional[VectorSearchEngine] = None,
        cache_manager: Optional[CacheManager] = None,
        max_documents: int = 10000,
        enable_analytics: bool = True
    ):
        self.documents: Dict[str, Document] = {}
        self.document_index: Dict[str, set] = defaultdict(set)  # keyword -> doc_ids
        self.vector_engine = vector_engine
        self.cache_manager = cache_manager
        self.max_documents = max_documents
        self.enable_analytics = enable_analytics

        # Analytics storage
        self.analytics_events: List[AnalyticsEvent] = []
        self.query_history: List[Tuple[str, datetime, float]] = []
        self.popular_queries: Counter = Counter()

        # Performance metrics
        self.search_stats = {
            'total_searches': 0,
            'avg_response_time': 0.0,
            'cache_hit_rate': 0.0
        }

        logger.info(f"KnowledgeService initialized with max_documents={max_documents}")

    async def add_document(self, document: Document, update_vectors: bool = True) -> str:
        """
        Add a document to the knowledge base with automatic indexing.
        
        Args:
            document: Document to add
            update_vectors: Whether to generate vector embeddings
            
        Returns:
            Document ID
            
        Raises:
            KnowledgeBaseError: If document cannot be added
        """
        try:
            # Validate document
            if not document.content.strip():
                raise ValidationError("Document content cannot be empty")

            # Check for duplicates
            content_hash = document.content_hash
            existing_doc = self._find_duplicate(content_hash)
            if existing_doc:
                logger.info(f"Duplicate document found: {existing_doc.doc_id}")
                return existing_doc.doc_id

            # Apply document limits with FIFO eviction
            if len(self.documents) >= self.max_documents:
                await self._evict_oldest_document()

            # Extract metadata automatically
            document.extract_metadata_from_content()

            # Store document
            doc_id = document.doc_id
            self.documents[doc_id] = document

            # Update search index
            await self._index_document(document)

            # Generate vector embeddings if enabled
            if update_vectors and self.vector_engine:
                try:
                    await self.vector_engine.add_document(document)
                except Exception as e:
                    logger.warning(f"Vector indexing failed for {doc_id}: {e}")

            # Invalidate relevant caches
            if self.cache_manager:
                await self.cache_manager.invalidate_pattern("search:*")
                await self.cache_manager.invalidate("kb:stats")

            # Track analytics
            if self.enable_analytics:
                event = AnalyticsEvent(
                    event_type="document_added",
                    user_id="system",
                    metadata={
                        'doc_id': doc_id,
                        'doc_type': document.doc_type.value,
                        'source_type': document.source_type.value,
                        'word_count': document.word_count
                    }
                )
                self.analytics_events.append(event)

            logger.info(f"Document added successfully: {doc_id}")
            return doc_id

        except Exception as e:
            logger.error(f"Failed to add document: {e}")
            raise KnowledgeBaseError(f"Failed to add document: {str(e)}")

    async def search(
        self,
        query: str,
        context: Optional[QueryContext] = None,
        search_type: str = "hybrid",
        limit: int = 10,
        min_score: float = 0.1
    ) -> List[SearchResult]:
        """
        Perform intelligent search across the knowledge base.
        
        Args:
            query: Search query
            context: Optional query context for personalization
            search_type: "keyword", "semantic", or "hybrid"
            limit: Maximum number of results
            min_score: Minimum relevance score
            
        Returns:
            List of search results ranked by relevance
        """
        start_time = time.time()

        try:
            # Validate and sanitize query
            query = sanitize_input(query.strip())
            if not query:
                raise ValidationError("Search query cannot be empty")

            # Check cache first
            cache_key = f"search:{hash(query)}:{search_type}:{limit}"
            if self.cache_manager:
                cached_results = await self.cache_manager.get(cache_key)
                if cached_results:
                    self._update_search_stats(time.time() - start_time, cache_hit=True)
                    return cached_results

            # Perform search based on type
            if search_type == "keyword":
                results = await self._keyword_search(query, limit, min_score)
            elif search_type == "semantic" and self.vector_engine:
                results = await self._semantic_search(query, limit, min_score)
            elif search_type == "hybrid":
                results = await self._hybrid_search(query, limit, min_score)
            else:
                # Fallback to keyword search
                results = await self._keyword_search(query, limit, min_score)

            # Apply context-based filtering and ranking
            if context:
                results = self._apply_context_filtering(results, context)

            # Cache results
            if self.cache_manager:
                await self.cache_manager.set(cache_key, results, ttl=1800)  # 30 minutes

            # Track analytics
            response_time = time.time() - start_time
            self._track_search_analytics(query, len(results), response_time, context)
            self._update_search_stats(response_time, cache_hit=False)

            logger.info(f"Search completed: query='{query}', results={len(results)}, time={response_time:.3f}s")
            return results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            # Track error analytics
            if self.enable_analytics and context:
                event = AnalyticsEvent(
                    event_type="search_error",
                    user_id=context.user_id,
                    query=query,
                    success=False,
                    error_message=str(e)
                )
                self.analytics_events.append(event)

            raise SearchError(f"Search failed: {str(e)}")

    async def _keyword_search(self, query: str, limit: int, min_score: float) -> List[SearchResult]:
        """Perform TF-IDF based keyword search."""
        keywords = extract_keywords(query)
        doc_scores: Dict[str, float] = defaultdict(float)
        matched_snippets: Dict[str, List[str]] = defaultdict(list)

        # Calculate TF-IDF scores
        for keyword in keywords:
            if keyword in self.document_index:
                doc_ids = self.document_index[keyword]
                idf_score = len(self.documents) / len(doc_ids) if doc_ids else 1

                for doc_id in doc_ids:
                    if doc_id in self.documents:
                        doc = self.documents[doc_id]
                        tf_score = doc.content.lower().count(keyword.lower())
                        doc_scores[doc_id] += tf_score * idf_score

                        # Extract relevant snippets
                        snippet = self._extract_snippet(doc.content, keyword)
                        if snippet:
                            matched_snippets[doc_id].append(snippet)

        # Create search results
        results = []
        for doc_id, score in sorted(doc_scores.items(), key=lambda x: x[1], reverse=True):
            if score >= min_score and len(results) < limit:
                doc = self.documents[doc_id]
                result = SearchResult(
                    document=doc,
                    score=score,
                    relevance_type="keyword",
                    matched_snippets=matched_snippets[doc_id],
                    explanation=f"Keyword match score: {score:.2f}"
                )
                results.append(result)

        return results

    async def _semantic_search(self, query: str, limit: int, min_score: float) -> List[SearchResult]:
        """Perform vector-based semantic search."""
        if not self.vector_engine:
            logger.warning("Vector engine not available, falling back to keyword search")
            return await self._keyword_search(query, limit, min_score)

        try:
            # Get semantic search results
            vector_results = await self.vector_engine.search(query, limit=limit * 2)  # Get more for filtering

            results = []
            for doc_id, score in vector_results:
                if doc_id in self.documents and score >= min_score and len(results) < limit:
                    doc = self.documents[doc_id]
                    result = SearchResult(
                        document=doc,
                        score=score,
                        relevance_type="semantic",
                        explanation=f"Semantic similarity score: {score:.2f}"
                    )
                    results.append(result)

            return results

        except Exception as e:
            logger.warning(f"Semantic search failed, falling back to keyword: {e}")
            return await self._keyword_search(query, limit, min_score)

    async def _hybrid_search(self, query: str, limit: int, min_score: float) -> List[SearchResult]:
        """Combine keyword and semantic search results."""
        # Get results from both methods
        keyword_results = await self._keyword_search(query, limit * 2, min_score * 0.5)
        semantic_results = await self._semantic_search(query, limit * 2, min_score * 0.5)

        # Combine and re-rank results
        combined_scores: Dict[str, Tuple[float, SearchResult]] = {}

        # Weight keyword results (30%)
        for result in keyword_results:
            doc_id = result.document.doc_id
            weighted_score = result.score * 0.3
            combined_scores[doc_id] = (weighted_score, result)

        # Weight semantic results (70%) and combine
        for result in semantic_results:
            doc_id = result.document.doc_id
            weighted_score = result.score * 0.7

            if doc_id in combined_scores:
                # Combine scores
                existing_score, existing_result = combined_scores[doc_id]
                final_score = existing_score + weighted_score

                # Merge result information
                merged_result = SearchResult(
                    document=result.document,
                    score=final_score,
                    relevance_type="hybrid",
                    matched_snippets=existing_result.matched_snippets,
                    explanation=f"Hybrid score: {final_score:.2f} (keyword: {existing_score:.2f}, semantic: {weighted_score:.2f})"
                )
                combined_scores[doc_id] = (final_score, merged_result)
            else:
                combined_scores[doc_id] = (weighted_score, result)

        # Sort by combined score and return top results
        results = [
            result for score, result in sorted(combined_scores.values(), key=lambda x: x[0], reverse=True)
            if score >= min_score
        ][:limit]

        return results

    def _apply_context_filtering(self, results: List[SearchResult], context: QueryContext) -> List[SearchResult]:
        """Apply user context to filter and re-rank results."""
        # Apply user preferences
        if context.user_preferences:
            preferred_sources = context.user_preferences.get('preferred_sources', [])
            preferred_types = context.user_preferences.get('preferred_doc_types', [])

            for result in results:
                doc = result.document

                # Boost preferred sources
                if preferred_sources and doc.source_type.value in preferred_sources:
                    result.score *= 1.2

                # Boost preferred document types
                if preferred_types and doc.doc_type.value in preferred_types:
                    result.score *= 1.1

        # Re-sort by adjusted scores
        results.sort(key=lambda r: r.score, reverse=True)
        return results

    async def _index_document(self, document: Document):
        """Index document for keyword search."""
        keywords = extract_keywords(document.content)
        doc_id = document.doc_id

        for keyword in keywords:
            self.document_index[keyword].add(doc_id)

        # Index metadata fields
        if document.title:
            title_keywords = extract_keywords(document.title)
            for keyword in title_keywords:
                self.document_index[keyword].add(doc_id)

        # Index tags
        for tag in document.tags:
            tag_keywords = extract_keywords(tag)
            for keyword in tag_keywords:
                self.document_index[keyword].add(doc_id)

    def _extract_snippet(self, content: str, keyword: str, snippet_length: int = 200) -> str:
        """Extract relevant snippet around keyword."""
        content_lower = content.lower()
        keyword_lower = keyword.lower()

        start_pos = content_lower.find(keyword_lower)
        if start_pos == -1:
            return ""

        # Find snippet boundaries
        snippet_start = max(0, start_pos - snippet_length // 2)
        snippet_end = min(len(content), start_pos + len(keyword) + snippet_length // 2)

        snippet = content[snippet_start:snippet_end]

        # Add ellipsis if truncated
        if snippet_start > 0:
            snippet = "..." + snippet
        if snippet_end < len(content):
            snippet = snippet + "..."

        return snippet.strip()

    def _find_duplicate(self, content_hash: str) -> Optional[Document]:
        """Find duplicate document by content hash."""
        for doc in self.documents.values():
            if doc.content_hash == content_hash:
                return doc
        return None

    async def _evict_oldest_document(self):
        """Remove oldest document to make space."""
        if not self.documents:
            return

        oldest_doc = min(self.documents.values(), key=lambda d: d.created_at)
        await self.remove_document(oldest_doc.doc_id)
        logger.info(f"Evicted oldest document: {oldest_doc.doc_id}")

    async def remove_document(self, doc_id: str) -> bool:
        """Remove document from knowledge base."""
        if doc_id not in self.documents:
            return False

        doc = self.documents[doc_id]

        # Remove from document store
        del self.documents[doc_id]

        # Remove from keyword index
        keywords = extract_keywords(doc.content)
        for keyword in keywords:
            if keyword in self.document_index:
                self.document_index[keyword].discard(doc_id)
                if not self.document_index[keyword]:
                    del self.document_index[keyword]

        # Remove from vector index
        if self.vector_engine:
            try:
                await self.vector_engine.remove_document(doc_id)
            except Exception as e:
                logger.warning(f"Failed to remove document from vector index: {e}")

        # Invalidate caches
        if self.cache_manager:
            await self.cache_manager.invalidate_pattern("search:*")
            await self.cache_manager.invalidate("kb:stats")

        logger.info(f"Document removed: {doc_id}")
        return True

    def _track_search_analytics(self, query: str, result_count: int, response_time: float, context: Optional[QueryContext]):
        """Track search analytics for performance monitoring."""
        if not self.enable_analytics:
            return

        # Update query history
        self.query_history.append((query, datetime.utcnow(), response_time))
        self.popular_queries[query] += 1

        # Limit query history size
        if len(self.query_history) > 10000:
            self.query_history = self.query_history[-5000:]

        # Create analytics event
        if context:
            event = AnalyticsEvent(
                event_type="search_performed",
                user_id=context.user_id,
                query=query,
                response_time_ms=response_time * 1000,
                result_count=result_count,
                metadata={
                    'channel_id': context.channel_id,
                    'query_intent': context.query_intent
                }
            )
            self.analytics_events.append(event)

    def _update_search_stats(self, response_time: float, cache_hit: bool):
        """Update search performance statistics."""
        self.search_stats['total_searches'] += 1

        # Update average response time
        total = self.search_stats['total_searches']
        current_avg = self.search_stats['avg_response_time']
        self.search_stats['avg_response_time'] = (current_avg * (total - 1) + response_time) / total

        # Update cache hit rate (simplified calculation)
        if cache_hit:
            cache_hits = self.search_stats.get('cache_hits', 0) + 1
            self.search_stats['cache_hits'] = cache_hits
            self.search_stats['cache_hit_rate'] = cache_hits / total

    async def get_stats(self) -> KnowledgeBaseStats:
        """Get comprehensive knowledge base statistics."""
        cache_key = "kb:stats"
        if self.cache_manager:
            cached_stats = await self.cache_manager.get(cache_key)
            if cached_stats:
                return cached_stats

        # Calculate statistics
        stats = KnowledgeBaseStats()

        if self.documents:
            stats.total_documents = len(self.documents)
            stats.total_words = sum(doc.word_count for doc in self.documents.values())
            stats.total_characters = sum(doc.char_count for doc in self.documents.values())

            # Group by type and source
            for doc in self.documents.values():
                stats.documents_by_type[doc.doc_type.value] = stats.documents_by_type.get(doc.doc_type.value, 0) + 1
                stats.documents_by_source[doc.source_type.value] = stats.documents_by_source.get(doc.source_type.value, 0) + 1

            # Find date ranges
            all_dates = [doc.created_at for doc in self.documents.values()]
            stats.oldest_document = min(all_dates)
            stats.most_recent_document = max(all_dates)
            stats.last_updated = max(all_dates)

        # Estimate index size (rough calculation)
        stats.index_size_mb = len(str(self.document_index)) / (1024 * 1024)

        # Cache stats
        if self.cache_manager:
            await self.cache_manager.set(cache_key, stats, ttl=300)  # 5 minutes

        return stats

    async def get_popular_queries(self, limit: int = 10) -> List[Tuple[str, int]]:
        """Get most popular search queries."""
        return self.popular_queries.most_common(limit)

    async def get_analytics_summary(self, days: int = 30) -> Dict[str, Any]:
        """Get analytics summary for the specified period."""
        cutoff_date = datetime.utcnow() - timedelta(days=days)

        # Filter recent events
        recent_events = [
            event for event in self.analytics_events
            if event.timestamp >= cutoff_date
        ]

        # Calculate metrics
        total_searches = len([e for e in recent_events if e.event_type == "search_performed"])
        total_documents_added = len([e for e in recent_events if e.event_type == "document_added"])

        # Calculate average response time
        search_events = [e for e in recent_events if e.event_type == "search_performed" and e.response_time_ms]
        avg_response_time = sum(e.response_time_ms for e in search_events) / len(search_events) if search_events else 0

        # Get unique users
        unique_users = len(set(e.user_id for e in recent_events if e.user_id != "system"))

        return {
            'period_days': days,
            'total_searches': total_searches,
            'documents_added': total_documents_added,
            'unique_users': unique_users,
            'avg_response_time_ms': avg_response_time,
            'search_stats': self.search_stats,
            'popular_queries': await self.get_popular_queries(10)
        }

    async def clear_cache(self):
        """Clear all cached data."""
        if self.cache_manager:
            await self.cache_manager.clear()
            logger.info("Knowledge base cache cleared")

    async def export_data(self) -> Dict[str, Any]:
        """Export all knowledge base data for backup."""
        return {
            'documents': [doc.to_dict() for doc in self.documents.values()],
            'analytics_events': [event.to_dict() for event in self.analytics_events],
            'search_stats': self.search_stats,
            'export_timestamp': datetime.utcnow().isoformat(),
            'version': '1.0'
        }

    async def import_data(self, data: Dict[str, Any]):
        """Import knowledge base data from backup."""
        try:
            # Import documents
            for doc_data in data.get('documents', []):
                doc = Document.from_dict(doc_data)
                await self.add_document(doc, update_vectors=False)

            # Import analytics events
            for event_data in data.get('analytics_events', []):
                event = AnalyticsEvent(
                    event_type=event_data['event_type'],
                    user_id=event_data['user_id'],
                    timestamp=datetime.fromisoformat(event_data['timestamp']),
                    query=event_data.get('query'),
                    response_time_ms=event_data.get('response_time_ms'),
                    result_count=event_data.get('result_count'),
                    success=event_data.get('success', True),
                    error_message=event_data.get('error_message'),
                    metadata=event_data.get('metadata', {})
                )
                self.analytics_events.append(event)

            # Import search stats
            if 'search_stats' in data:
                self.search_stats.update(data['search_stats'])

            logger.info(f"Data import completed: {len(data.get('documents', []))} documents, {len(data.get('analytics_events', []))} events")

        except Exception as e:
            logger.error(f"Data import failed: {e}")
            raise KnowledgeBaseError(f"Import failed: {str(e)}")


class DocumentProcessor:
    """Utility class for document processing and content enhancement."""

    @staticmethod
    def enhance_document(document: Document) -> Document:
        """Enhance document with additional metadata and processing."""
        # Extract entities and keywords
        content_keywords = extract_keywords(document.content)
        document.add_tags(content_keywords[:10])  # Add top 10 keywords as tags

        # Set document type based on content
        if document.content.startswith('```') or any(lang in document.content for lang in ['def ', 'function ', 'class ', 'import ']):
            document.doc_type = DocumentType.CODE
        elif document.content.startswith('#') or '##' in document.content:
            document.doc_type = DocumentType.MARKDOWN

        # Extract and set language for code documents
        if document.doc_type == DocumentType.CODE:
            first_line = document.content.split('\n')[0]
            if first_line.startswith('```'):
                document.language = first_line[3:].strip()

        return document

    @staticmethod
    def sanitize_document(document: Document) -> Document:
        """Sanitize document content for security and quality."""
        # Remove sensitive information patterns
        sensitive_patterns = [
            r'password\s*=\s*["\'][^"\']+["\']',
            r'api_key\s*=\s*["\'][^"\']+["\']',
            r'secret\s*=\s*["\'][^"\']+["\']',
            r'token\s*=\s*["\'][^"\']+["\']'
        ]

        import re
        content = document.content
        for pattern in sensitive_patterns:
            content = re.sub(pattern, '[REDACTED]', content, flags=re.IGNORECASE)

        # Mark as sensitive if redaction occurred
        if '[REDACTED]' in content:
            document.mark_sensitive(True)

        document.content = content
        return document
