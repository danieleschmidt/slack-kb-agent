"""Efficient search indexing for knowledge base documents."""

from __future__ import annotations

import re
import logging
from typing import Dict, List, Set, Optional
from collections import defaultdict
from dataclasses import dataclass

from .models import Document
from .configuration import get_search_config

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Search result with relevance scoring."""
    document: Document
    score: float
    matched_terms: Set[str]


class InvertedIndex:
    """Inverted index for efficient text search."""
    
    def __init__(self, min_word_length: Optional[int] = None, max_index_size: Optional[int] = None):
        """Initialize inverted index.
        
        Args:
            min_word_length: Minimum word length to index (defaults to config value)
            max_index_size: Maximum number of terms to index (defaults to config value)
        """
        config = get_search_config()
        self.min_word_length = min_word_length if min_word_length is not None else config.min_word_length
        self.max_index_size = max_index_size if max_index_size is not None else config.max_index_size
        
        # term -> set of document indices
        self.index: Dict[str, Set[int]] = defaultdict(set)
        
        # document index -> document
        self.documents: List[Document] = []
        
        # document index -> document content (for scoring)
        self.document_content: List[str] = []
        
        # Statistics
        self.total_terms = 0
        self._index_full = False
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into searchable terms.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of normalized terms
        """
        # Convert to lowercase and extract words
        text = text.lower()
        
        # Extract words (letters, numbers, and some special chars)
        words = re.findall(r'\b\w+\b', text)
        
        # Filter by minimum length
        words = [word for word in words if len(word) >= self.min_word_length]
        
        return words
    
    def add_document(self, document: Document) -> int:
        """Add a document to the index.
        
        Args:
            document: Document to add
            
        Returns:
            Document index (ID) in the index
        """
        doc_index = len(self.documents)
        self.documents.append(document)
        
        # Tokenize content
        content = f"{document.content} {document.source}"
        self.document_content.append(content.lower())
        
        # Add to inverted index
        terms = self._tokenize(content)
        
        for term in terms:
            # Check if we're approaching memory limits
            if len(self.index) >= self.max_index_size and term not in self.index:
                if not self._index_full:
                    logger.warning(f"Search index reached maximum size ({self.max_index_size} terms)")
                    self._index_full = True
                continue
            
            self.index[term].add(doc_index)
            self.total_terms += 1
        
        return doc_index
    
    def remove_document(self, doc_index: int) -> bool:
        """Remove a document from the index.
        
        Args:
            doc_index: Document index to remove
            
        Returns:
            True if document was found and removed
        """
        if doc_index >= len(self.documents):
            return False
        
        # Get document content for term extraction
        content = self.document_content[doc_index]
        terms = self._tokenize(content)
        
        # Remove from inverted index
        for term in terms:
            if term in self.index:
                self.index[term].discard(doc_index)
                if not self.index[term]:  # Remove empty sets
                    del self.index[term]
        
        # Note: We don't remove from documents list to maintain indices
        # Instead, we mark as None (tombstone)
        self.documents[doc_index] = None
        self.document_content[doc_index] = ""
        
        return True
    
    def search(self, query: str, max_results: Optional[int] = None) -> List[SearchResult]:
        """Search for documents matching the query.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return (defaults to config value)
            
        Returns:
            List of search results sorted by relevance
        """
        if max_results is None:
            max_results = get_search_config().max_results_default
            
        if not query.strip():
            # Return all valid documents for empty query
            results = []
            for i, doc in enumerate(self.documents):
                if doc is not None:
                    results.append(SearchResult(
                        document=doc,
                        score=1.0,
                        matched_terms=set()
                    ))
            return results[:max_results]
        
        query_terms = self._tokenize(query)
        if not query_terms:
            return []
        
        # Find documents that contain any query term
        candidate_docs: Set[int] = set()
        term_counts: Dict[str, int] = {}
        
        for term in query_terms:
            if term in self.index:
                candidate_docs.update(self.index[term])
                term_counts[term] = len(self.index[term])
        
        if not candidate_docs:
            return []
        
        # Score and rank results
        results = []
        for doc_index in candidate_docs:
            if doc_index >= len(self.documents) or self.documents[doc_index] is None:
                continue
            
            document = self.documents[doc_index]
            content = self.document_content[doc_index]
            
            # Calculate relevance score
            score, matched_terms = self._calculate_score(content, query_terms, term_counts)
            
            results.append(SearchResult(
                document=document,
                score=score,
                matched_terms=matched_terms
            ))
        
        # Sort by score (descending) and return top results
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:max_results]
    
    def _calculate_score(self, content: str, query_terms: List[str], term_counts: Dict[str, int]) -> tuple[float, Set[str]]:
        """Calculate relevance score for a document.
        
        Args:
            content: Document content (normalized)
            query_terms: Query terms to match
            term_counts: Number of documents containing each term
            
        Returns:
            Tuple of (score, matched_terms)
        """
        score = 0.0
        matched_terms = set()
        total_docs = len(self.documents)
        
        for term in query_terms:
            if term in content:
                matched_terms.add(term)
                
                # Term frequency in document
                tf = content.count(term)
                
                # Inverse document frequency
                doc_freq = term_counts.get(term, 1)
                idf = 1.0 + (total_docs / doc_freq) if doc_freq > 0 else 1.0
                
                # TF-IDF score
                score += tf * idf
                
                # Bonus for exact phrase matches
                if len(query_terms) > 1:
                    query_phrase = " ".join(query_terms)
                    if query_phrase in content:
                        score += 10.0  # Significant bonus for phrase match
        
        # Normalize score by query length
        if query_terms:
            score = score / len(query_terms)
        
        return score, matched_terms
    
    def get_stats(self) -> Dict[str, int]:
        """Get index statistics.
        
        Returns:
            Dictionary with index statistics
        """
        valid_docs = sum(1 for doc in self.documents if doc is not None)
        
        return {
            "total_documents": len(self.documents),
            "valid_documents": valid_docs,
            "total_terms": len(self.index),
            "total_term_occurrences": self.total_terms,
            "index_full": self._index_full,
            "max_index_size": self.max_index_size
        }
    
    def clear(self) -> None:
        """Clear the entire index."""
        self.index.clear()
        self.documents.clear()
        self.document_content.clear()
        self.total_terms = 0
        self._index_full = False


class SearchEngine:
    """High-level search engine with caching and optimization."""
    
    def __init__(self, enable_indexing: Optional[bool] = None, cache_size: Optional[int] = None):
        """Initialize search engine.
        
        Args:
            enable_indexing: Whether to use inverted index for search (defaults to config)
            cache_size: Size of search result cache (defaults to config)
        """
        config = get_search_config()
        self.enable_indexing = enable_indexing if enable_indexing is not None else config.enable_indexing
        self.cache_size = cache_size if cache_size is not None else config.cache_size
        
        # Search components
        self.index = InvertedIndex() if self.enable_indexing else None
        self.documents: List[Document] = []
        
        # Simple LRU cache for search results
        self._search_cache: Dict[str, List[Document]] = {}
        self._cache_order: List[str] = []
        
        # Statistics
        self.search_count = 0
        self.cache_hits = 0
    
    def add_document(self, document: Document) -> None:
        """Add a document to the search engine."""
        self.documents.append(document)
        
        if self.index:
            self.index.add_document(document)
        
        # Clear cache when documents are added
        self._clear_cache()
    
    def add_documents(self, documents: List[Document]) -> None:
        """Add multiple documents to the search engine."""
        for document in documents:
            self.documents.append(document)
            if self.index:
                self.index.add_document(document)
        
        # Clear cache when documents are added
        self._clear_cache()
    
    def search(self, query: str, max_results: Optional[int] = None) -> List[Document]:
        """Search for documents matching the query.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return (defaults to config value)
            
        Returns:
            List of matching documents
        """
        if max_results is None:
            max_results = get_search_config().max_results_default
        self.search_count += 1
        
        # Check cache first
        cache_key = f"{query.lower()}:{max_results}"
        if cache_key in self._search_cache:
            self.cache_hits += 1
            return self._search_cache[cache_key]
        
        # Perform search
        if self.index and self.enable_indexing:
            # Use indexed search
            search_results = self.index.search(query, max_results)
            documents = [result.document for result in search_results]
        else:
            # Fall back to linear search
            documents = self._linear_search(query, max_results)
        
        # Cache result
        self._cache_result(cache_key, documents)
        
        return documents
    
    def _linear_search(self, query: str, max_results: int) -> List[Document]:
        """Fallback linear search implementation."""
        q = query.lower()
        results = []
        
        for doc in self.documents:
            if doc is not None and q in doc.content.lower():
                results.append(doc)
                if len(results) >= max_results:
                    break
        
        return results
    
    def _cache_result(self, cache_key: str, documents: List[Document]) -> None:
        """Cache search result with LRU eviction."""
        # Remove if already exists
        if cache_key in self._search_cache:
            self._cache_order.remove(cache_key)
        
        # Add to cache
        self._search_cache[cache_key] = documents
        self._cache_order.append(cache_key)
        
        # Evict oldest if cache is full
        while len(self._search_cache) > self.cache_size:
            oldest_key = self._cache_order.pop(0)
            del self._search_cache[oldest_key]
    
    def _clear_cache(self) -> None:
        """Clear the search cache."""
        self._search_cache.clear()
        self._cache_order.clear()
    
    def clear(self) -> None:
        """Clear all documents and index."""
        self.documents.clear()
        if self.index:
            self.index.clear()
        self._clear_cache()
    
    def get_stats(self) -> Dict[str, any]:
        """Get search engine statistics."""
        stats = {
            "total_documents": len(self.documents),
            "search_count": self.search_count,
            "cache_hits": self.cache_hits,
            "cache_hit_rate": self.cache_hits / max(self.search_count, 1),
            "cache_size": len(self._search_cache),
            "indexing_enabled": self.enable_indexing
        }
        
        if self.index:
            stats.update(self.index.get_stats())
        
        return stats