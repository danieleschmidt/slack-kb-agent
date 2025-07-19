"""Vector-based semantic search using sentence transformers and FAISS."""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple, Union
import warnings

try:
    import numpy as np
    import faiss
    from sentence_transformers import SentenceTransformer
    VECTOR_DEPS_AVAILABLE = True
except ImportError:
    VECTOR_DEPS_AVAILABLE = False
    np = None
    faiss = None
    SentenceTransformer = None

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
        """Generate vector embedding for text.
        
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
        
        self._ensure_model_loaded()
        embedding = self.model.encode([text], normalize_embeddings=True)[0]
        return embedding.astype(np.float32)
    
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
        batch_size = 32
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
        top_k: int = 10,
        threshold: Optional[float] = None
    ) -> List[Tuple[Document, float]]:
        """Search for similar documents using vector similarity.
        
        Args:
            query: Search query text
            top_k: Maximum number of results to return
            threshold: Minimum similarity score (defaults to instance threshold)
            
        Returns:
            List of (document, similarity_score) tuples sorted by relevance
        """
        if not query.strip():
            return []
            
        if self.index is None or not self.documents:
            logger.warning("Vector index not built. Call build_index() first.")
            return []
        
        threshold = threshold if threshold is not None else self.similarity_threshold
        
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


def is_vector_search_available() -> bool:
    """Check if vector search dependencies are available."""
    return VECTOR_DEPS_AVAILABLE