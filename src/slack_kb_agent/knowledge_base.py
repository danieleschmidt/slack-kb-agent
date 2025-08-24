"""In-memory knowledge base supporting multiple data sources."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from .cache import get_cache_manager
from .models import Document
from .search_index import SearchEngine
from .sources import BaseSource
from .vector_search import VectorSearchEngine, is_vector_search_available

logger = logging.getLogger(__name__)

# Import metrics functionality
try:
    from .monitoring import get_global_metrics
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False
    get_global_metrics = None


class KnowledgeBase:
    """Aggregate documents from various sources and provide simple search."""

    def __init__(
        self,
        enable_vector_search: bool = True,
        vector_model: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = 0.5,
        max_documents: Optional[int] = None,
        enable_indexed_search: bool = True
    ) -> None:
        self.sources: List[BaseSource] = []
        self.documents: List[Document] = []
        self.max_documents = max_documents
        self.enable_vector_search = enable_vector_search and is_vector_search_available()
        self.enable_indexed_search = enable_indexed_search

        # Initialize search engine
        self.search_engine = SearchEngine(enable_indexing=enable_indexed_search)

        if self.enable_vector_search:
            try:
                self.vector_engine = VectorSearchEngine(
                    model_name=vector_model,
                    similarity_threshold=similarity_threshold
                )
                logger.info("Vector search enabled")
            except ImportError as e:
                logger.warning(f"Vector search disabled due to missing dependencies: {e}")
                self.enable_vector_search = False
                self.vector_engine = None
        else:
            self.vector_engine = None
            if not is_vector_search_available():
                logger.info("Vector search disabled - dependencies not available")
            else:
                logger.info("Vector search disabled by configuration")

    def add_source(self, source: BaseSource) -> None:
        """Register a new data source."""
        self.sources.append(source)

    def index(self) -> None:
        """Load all documents from registered sources."""
        for source in self.sources:
            self.documents.extend(source.load())
        self._enforce_document_limit()
        self._update_memory_metrics()
        self._rebuild_vector_index()

    def add_document(self, document: Document) -> None:
        """Add a single document to the knowledge base."""
        self.documents.append(document)
        self._enforce_document_limit()
        self._update_memory_metrics()

        # Add to search engine
        self.search_engine.add_document(document)

        if self.enable_vector_search and self.vector_engine:
            self.vector_engine.add_document(document)

        # Invalidate search cache when documents are added
        cache_manager = get_cache_manager()
        invalidated = cache_manager.invalidate_search_cache()
        if invalidated > 0:
            logger.debug(f"Invalidated {invalidated} search cache entries after adding document")

    def add_documents(self, documents: List[Document]) -> None:
        """Add multiple documents to the knowledge base."""
        self.documents.extend(documents)
        self._enforce_document_limit()
        self._update_memory_metrics()

        # Add to search engine
        self.search_engine.add_documents(documents)

        # Performance optimization: Pass new documents for incremental indexing
        self._rebuild_vector_index(new_documents=documents)

        # Invalidate search cache when documents are added
        cache_manager = get_cache_manager()
        invalidated = cache_manager.invalidate_search_cache()
        if invalidated > 0:
            logger.debug(f"Invalidated {invalidated} search cache entries after adding {len(documents)} documents")

    def _rebuild_vector_index(self, new_documents: Optional[List[Document]] = None) -> None:
        """Rebuild or incrementally update the vector search index.
        
        Args:
            new_documents: If provided, attempt incremental update instead of full rebuild
        """
        if self.enable_vector_search and self.vector_engine and self.documents:
            # Performance optimization: Try incremental updates first
            if new_documents and hasattr(self.vector_engine, 'add_documents_incremental'):
                try:
                    self.vector_engine.add_documents_incremental(new_documents)
                    return
                except (AttributeError, NotImplementedError):
                    # Fall back to full rebuild if incremental not supported
                    pass
            
            # Full rebuild as fallback
            self.vector_engine.build_index(self.documents)

    def search(self, query: str) -> List[Document]:
        """Return documents containing the query string (keyword search)."""
        # Use indexed search engine for better performance
        return self.search_engine.search(query)

    def search_semantic(
        self,
        query: str,
        top_k: int = 10,
        threshold: Optional[float] = None
    ) -> List[Document]:
        """Search using vector similarity (semantic search).
        
        Args:
            query: Search query text
            top_k: Maximum number of results to return
            threshold: Minimum similarity score (defaults to instance threshold)
            
        Returns:
            List of documents sorted by semantic similarity
        """
        if not self.enable_vector_search or not self.vector_engine:
            logger.warning("Vector search not available, falling back to keyword search")
            return self.search(query)

        if not query.strip():
            return []

        # Ensure vector index is built
        if self.vector_engine.index is None and self.documents:
            self._rebuild_vector_index()

        results = self.vector_engine.search(query, top_k, threshold)
        return [doc for doc, score in results]

    def search_hybrid(
        self,
        query: str,
        top_k: int = 10,
        vector_weight: float = 0.7,
        keyword_weight: float = 0.3
    ) -> List[Document]:
        """Combine vector and keyword search results.
        
        Args:
            query: Search query text
            top_k: Maximum number of results to return
            vector_weight: Weight for vector search results (0-1)
            keyword_weight: Weight for keyword search results (0-1)
            
        Returns:
            Combined and deduplicated results
        """
        if not query.strip():
            return []

        if not self.enable_vector_search:
            return self.search(query)

        # Get vector search results with scores
        vector_results = []
        if self.vector_engine and self.vector_engine.index is not None:
            vector_raw = self.vector_engine.search(query, top_k * 2)  # Get more for deduplication
            vector_results = [(doc, score * vector_weight) for doc, score in vector_raw]

        # Get keyword search results (assign score based on position)
        keyword_docs = self.search(query)
        keyword_results = [
            (doc, keyword_weight * (1.0 - i / len(keyword_docs)))
            for i, doc in enumerate(keyword_docs[:top_k * 2])
        ]

        # Combine and deduplicate
        doc_scores = {}
        for doc, score in vector_results + keyword_results:
            doc_id = id(doc)  # Use object id as unique identifier
            if doc_id in doc_scores:
                doc_scores[doc_id] = (doc, max(doc_scores[doc_id][1], score))
            else:
                doc_scores[doc_id] = (doc, score)

        # Sort by combined score and return top_k
        sorted_results = sorted(doc_scores.values(), key=lambda x: x[1], reverse=True)
        return [doc for doc, score in sorted_results[:top_k]]

    def search_with_consciousness(self, query: str, use_consciousness: bool = True, top_k: int = 10) -> List[Document]:
        """Search with consciousness-enhanced ranking.
        
        This method integrates the transcendent AGI consciousness system to provide
        enhanced knowledge retrieval with consciousness-guided insights.
        
        Args:
            query: Search query
            use_consciousness: Whether to use consciousness enhancement
            top_k: Maximum results to return
            
        Returns:
            List of documents enhanced by consciousness insights
        """
        # Get base search results using hybrid search
        base_results = self.search_hybrid(query, top_k=top_k * 2)  # Get more candidates
        
        if not use_consciousness or not base_results:
            return base_results[:top_k]
        
        try:
            # Import consciousness system (lazy import to avoid circular dependencies)
            from .transcendent_agi_consciousness import get_transcendent_consciousness
            
            consciousness = get_transcendent_consciousness()
            
            # Use consciousness to enhance knowledge processing
            enhanced_context = {
                'query': query,
                'documents': [{'content': doc.content, 'source': doc.source} for doc in base_results],
                'search_type': 'knowledge_retrieval'
            }
            
            # Get consciousness insights (async call wrapped)
            import asyncio
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
            consciousness_result = loop.run_until_complete(
                consciousness.process_knowledge_with_consciousness(query, enhanced_context)
            )
            
            # Apply consciousness-driven re-ranking
            consciousness_weights = consciousness_result.get('relevance_weights', [1.0] * len(base_results))
            consciousness_insights = consciousness_result.get('insights', [])
            
            # Re-rank results based on consciousness analysis
            enhanced_results = []
            for i, doc in enumerate(base_results):
                weight = consciousness_weights[i] if i < len(consciousness_weights) else 1.0
                enhanced_results.append((doc, weight))
            
            # Sort by consciousness-enhanced scores
            enhanced_results.sort(key=lambda x: x[1], reverse=True)
            
            logger.info(f"Consciousness enhancement applied. Generated {len(consciousness_insights)} insights")
            return [doc for doc, _ in enhanced_results[:top_k]]
            
        except Exception as e:
            logger.warning(f"Consciousness enhancement failed, falling back to hybrid search: {e}")
            return base_results[:top_k]

    def _generate_embedding(self, text: str):
        """Generate embedding for text (for testing purposes)."""
        if not self.enable_vector_search or not self.vector_engine:
            raise AttributeError("Vector search not enabled")
        return self.vector_engine.generate_embedding(text)

    def _enforce_document_limit(self) -> None:
        """Enforce maximum document limit by removing oldest documents if needed."""
        if self.max_documents is None or len(self.documents) <= self.max_documents:
            return

        # Calculate how many documents to remove
        excess_count = len(self.documents) - self.max_documents

        # Remove oldest documents (FIFO eviction)
        removed_docs = self.documents[:excess_count]
        self.documents = self.documents[excess_count:]

        # Log the eviction for monitoring
        logger.info(f"Evicted {excess_count} documents to enforce limit of {self.max_documents}")

        # Update memory metrics
        self._update_memory_metrics()

        # Rebuild search engine index after document eviction
        self.search_engine.clear()
        self.search_engine.add_documents(self.documents)

        # If vector search is enabled, we need to rebuild the index since we can't
        # selectively remove documents from FAISS index
        if self.enable_vector_search and self.vector_engine and removed_docs:
            logger.debug("Rebuilding vector index after document eviction")
            self._rebuild_vector_index()

    def _update_memory_metrics(self) -> None:
        """Update memory usage metrics for monitoring."""
        if not METRICS_AVAILABLE or not get_global_metrics:
            return

        try:
            metrics = get_global_metrics()

            # Document count metrics
            metrics.set_gauge("kb_documents_count", len(self.documents))
            if self.max_documents:
                metrics.set_gauge("kb_documents_limit", self.max_documents)
                usage_percent = (len(self.documents) / self.max_documents) * 100
                metrics.set_gauge("kb_documents_usage_percent", usage_percent)

            # Estimate memory usage (rough approximation)
            estimated_bytes = sum(len(doc.content.encode('utf-8')) + len(doc.source.encode('utf-8'))
                                for doc in self.documents)
            metrics.set_gauge("kb_estimated_memory_bytes", estimated_bytes)

            # Source count
            metrics.set_gauge("kb_sources_count", len(self.sources))

        except Exception as e:
            # Don't let metrics collection crash the application
            logger.debug(f"Failed to update knowledge base metrics: {type(e).__name__}: {e}")

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics for this knowledge base."""
        stats = {
            "documents_count": len(self.documents),
            "sources_count": len(self.sources),
            "max_documents": self.max_documents,
        }

        if self.max_documents:
            stats["documents_usage_percent"] = (len(self.documents) / self.max_documents) * 100

        # Estimate memory usage
        estimated_bytes = sum(len(doc.content.encode('utf-8')) + len(doc.source.encode('utf-8'))
                            for doc in self.documents)
        stats["estimated_memory_bytes"] = estimated_bytes
        stats["estimated_memory_mb"] = estimated_bytes / (1024 * 1024)

        # Add search engine statistics
        search_stats = self.search_engine.get_stats()
        stats.update({f"search_{k}": v for k, v in search_stats.items()})

        return stats

    # Persistence helpers -------------------------------------------------

    def to_dict(self) -> dict[str, list[dict]]:
        """Return a serializable representation of all documents."""
        return {"documents": [asdict(d) for d in self.documents]}

    @classmethod
    def from_dict(cls, data: dict[str, list[dict]], max_documents: Optional[int] = None) -> KnowledgeBase:
        """Create a knowledge base from a dictionary."""
        kb = cls(max_documents=max_documents)
        if not isinstance(data, dict):
            return kb
        for item in data.get("documents", []):
            if not isinstance(item, dict):
                continue
            kb.add_document(Document(**item))
        return kb

    def save(self, path: str | Path) -> None:
        """Persist documents to ``path`` as JSON."""
        Path(path).write_text(json.dumps(self.to_dict()), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path, max_documents: Optional[int] = None) -> KnowledgeBase:
        """Load documents from ``path`` and return a new knowledge base."""
        try:
            text = Path(path).read_text(encoding="utf-8")
        except OSError:
            return cls(max_documents=max_documents)
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return cls(max_documents=max_documents)
        if not isinstance(data, dict):
            return cls(max_documents=max_documents)
        return cls.from_dict(data, max_documents=max_documents)
