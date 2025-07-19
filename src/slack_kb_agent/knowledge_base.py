"""In-memory knowledge base supporting multiple data sources."""

from __future__ import annotations

from typing import List, Optional, Tuple
import json
import logging
from dataclasses import asdict
from pathlib import Path

from .models import Document
from .sources import BaseSource
from .vector_search import VectorSearchEngine, is_vector_search_available

logger = logging.getLogger(__name__)


class KnowledgeBase:
    """Aggregate documents from various sources and provide simple search."""

    def __init__(
        self, 
        enable_vector_search: bool = True,
        vector_model: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = 0.5
    ) -> None:
        self.sources: List[BaseSource] = []
        self.documents: List[Document] = []
        self.enable_vector_search = enable_vector_search and is_vector_search_available()
        
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
        self._rebuild_vector_index()

    def add_document(self, document: Document) -> None:
        """Add a single document to the knowledge base."""
        self.documents.append(document)
        if self.enable_vector_search and self.vector_engine:
            self.vector_engine.add_document(document)

    def add_documents(self, documents: List[Document]) -> None:
        """Add multiple documents to the knowledge base."""
        self.documents.extend(documents)
        self._rebuild_vector_index()
    
    def _rebuild_vector_index(self) -> None:
        """Rebuild the vector search index with current documents."""
        if self.enable_vector_search and self.vector_engine and self.documents:
            self.vector_engine.build_index(self.documents)

    def search(self, query: str) -> List[Document]:
        """Return documents containing the query string (keyword search)."""
        q = query.lower()
        return [doc for doc in self.documents if q in doc.content.lower()]
    
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
    
    def _generate_embedding(self, text: str):
        """Generate embedding for text (for testing purposes)."""
        if not self.enable_vector_search or not self.vector_engine:
            raise AttributeError("Vector search not enabled")
        return self.vector_engine.generate_embedding(text)

    # Persistence helpers -------------------------------------------------

    def to_dict(self) -> dict[str, list[dict]]:
        """Return a serializable representation of all documents."""
        return {"documents": [asdict(d) for d in self.documents]}

    @classmethod
    def from_dict(cls, data: dict[str, list[dict]]) -> "KnowledgeBase":
        """Create a knowledge base from a dictionary."""
        kb = cls()
        for item in data.get("documents", []):
            if not isinstance(item, dict):
                continue
            kb.add_document(Document(**item))
        return kb

    def save(self, path: str | Path) -> None:
        """Persist documents to ``path`` as JSON."""
        Path(path).write_text(json.dumps(self.to_dict()), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "KnowledgeBase":
        """Load documents from ``path`` and return a new knowledge base."""
        try:
            text = Path(path).read_text(encoding="utf-8")
        except OSError:
            return cls()
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return cls()
        if not isinstance(data, dict):
            return cls()
        return cls.from_dict(data)
