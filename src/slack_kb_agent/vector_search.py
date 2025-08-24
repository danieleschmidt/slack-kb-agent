"""Vector-based semantic search using sentence transformers and FAISS."""

from __future__ import annotations

import logging
import warnings
from typing import List, Optional, Tuple

try:
    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer
    VECTOR_DEPS_AVAILABLE = True
except ImportError:
    VECTOR_DEPS_AVAILABLE = False
    np = None
    faiss = None
    SentenceTransformer = None

from .cache import get_cache_manager
from .configuration import get_vector_search_config
from .models import Document

logger = logging.getLogger(__name__)


class VectorSearchEngine:
    """Handles vector embeddings and semantic search using FAISS."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = 0.5
    ):
        """Initialize vector search engine.
        
        Args:
            model_name: Sentence transformer model name
            similarity_threshold: Minimum similarity score for results
        """
        if not VECTOR_DEPS_AVAILABLE:
            raise ImportError(
                "Vector search dependencies not available. "
                "Install with: pip install sentence-transformers faiss-cpu"
            )

        self.model_name = model_name
        self.similarity_threshold = similarity_threshold
        self.model: Optional[SentenceTransformer] = None
        self.index: Optional[faiss.IndexFlatIP] = None
        self.documents: List[Document] = []
        self._embeddings: Optional[np.ndarray] = None

    def _ensure_model_loaded(self) -> None:
        """Lazy load the sentence transformer model."""
        if self.model is None:
            logger.info(f"Loading sentence transformer model: {self.model_name}")
            # Suppress transformers warnings about token usage
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.model = SentenceTransformer(self.model_name)

    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate vector embedding for text with caching.
        
        Args:
            text: Input text to embed
            
        Returns:
            Normalized embedding vector
        """
        if not text.strip():
            # Return zero vector for empty text
            self._ensure_model_loaded()
            embedding_dim = self.model.get_sentence_embedding_dimension()
            return np.zeros(embedding_dim, dtype=np.float32)

        # Try to get from cache first
        cache_manager = get_cache_manager()
        cached_embedding = cache_manager.get_embedding(text, self.model_name)
        if cached_embedding is not None:
            return cached_embedding

        # Generate new embedding
        self._ensure_model_loaded()
        embedding = self.model.encode([text], normalize_embeddings=True)[0]
        embedding = embedding.astype(np.float32)

        # Cache the result
        cache_manager.set_embedding(text, self.model_name, embedding)

        return embedding

    def build_index(self, documents: List[Document]) -> None:
        """Build FAISS index from documents.
        
        Args:
            documents: List of documents to index
        """
        if not documents:
            logger.warning("No documents provided for indexing")
            return

        logger.info(f"Building vector index for {len(documents)} documents")
        self.documents = documents

        # Generate embeddings for all documents
        texts = [doc.content for doc in documents]
        self._ensure_model_loaded()

        # Generate embeddings in batches for efficiency
        config = get_vector_search_config()
        batch_size = config.batch_size
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.model.encode(batch, normalize_embeddings=True)
            embeddings.extend(batch_embeddings)

        self._embeddings = np.array(embeddings, dtype=np.float32)

        # Create FAISS index for inner product (cosine similarity with normalized vectors)
        dimension = self._embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(self._embeddings)

        logger.info(f"Vector index built with {self.index.ntotal} documents")

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None
    ) -> List[Tuple[Document, float]]:
        """Search for similar documents using vector similarity with caching.
        
        Args:
            query: Search query text
            top_k: Maximum number of results to return (defaults to config value)
            threshold: Minimum similarity score (defaults to instance threshold)
            
        Returns:
            List of (document, similarity_score) tuples sorted by relevance
        """
        config = get_vector_search_config()
        if top_k is None:
            top_k = config.top_k_default
        if not query.strip():
            return []

        if self.index is None or not self.documents:
            logger.warning("Vector index not built. Call build_index() first.")
            return []

        threshold = threshold if threshold is not None else self.similarity_threshold

        # Generate cache key for search parameters
        cache_manager = get_cache_manager()
        search_params = {
            "model_name": self.model_name,
            "top_k": top_k,
            "threshold": threshold,
            "num_documents": len(self.documents)
        }
        cache_key = cache_manager.generate_search_hash(query, search_params)

        # Try to get results from cache
        cached_results = cache_manager.get_search_results(cache_key)
        if cached_results is not None:
            # Convert cached results back to Document objects
            results = []
            for result_data in cached_results:
                doc_data = result_data["document"]
                score = result_data["score"]
                # Reconstruct Document object
                document = Document(
                    content=doc_data["content"],
                    source=doc_data["source"],
                    metadata=doc_data.get("metadata", {})
                )
                results.append((document, score))
            return results

        # Generate query embedding
        query_embedding = self.generate_embedding(query)
        query_vector = query_embedding.reshape(1, -1)

        # Search with FAISS
        scores, indices = self.index.search(query_vector, min(top_k, len(self.documents)))

        # Filter by threshold and create results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if score >= threshold:
                results.append((self.documents[idx], float(score)))

        # Cache the results
        if results:
            cached_data = []
            for document, score in results:
                doc_dict = {
                    "content": document.content,
                    "source": document.source,
                    "metadata": document.metadata
                }
                cached_data.append({
                    "document": doc_dict,
                    "score": score
                })
            cache_manager.set_search_results(cache_key, cached_data)

        return results

    def add_document(self, document: Document) -> None:
        """Add a single document to the index.
        
        Args:
            document: Document to add
        """
        # For simplicity, rebuild the entire index
        # In production, you'd want incremental updates
        self.documents.append(document)
        if len(self.documents) > 1:
            self.build_index(self.documents)

    def get_embedding_dimension(self) -> int:
        """Get the embedding dimension of the current model."""
        self._ensure_model_loaded()
        return self.model.get_sentence_embedding_dimension()
    
    def add_documents_incremental(self, new_documents: List[Document]) -> None:
        """Add documents incrementally without full rebuild.
        
        Args:
            new_documents: New documents to add to the index
        """
        if not new_documents:
            return
            
        logger.info(f"Incrementally adding {len(new_documents)} documents to vector index")
        
        # Add documents to our list
        self.documents.extend(new_documents)
        
        # Generate embeddings for new documents only
        texts = [doc.content for doc in new_documents]
        self._ensure_model_loaded()
        
        config = get_vector_search_config()
        batch_size = config.batch_size
        new_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.model.encode(batch, normalize_embeddings=True)
            new_embeddings.extend(batch_embeddings)
        
        new_embeddings_array = np.array(new_embeddings, dtype=np.float32)
        
        # If we don't have an existing index, create one
        if self.index is None or self._embeddings is None:
            self.build_index(self.documents)
            return
        
        # Add new embeddings to existing index
        self.index.add(new_embeddings_array)
        
        # Update stored embeddings
        self._embeddings = np.vstack([self._embeddings, new_embeddings_array])
        
        logger.info(f"Incremental index update complete. Total documents: {self.index.ntotal}")


def is_vector_search_available() -> bool:
    """Check if vector search dependencies are available."""
    return VECTOR_DEPS_AVAILABLE
