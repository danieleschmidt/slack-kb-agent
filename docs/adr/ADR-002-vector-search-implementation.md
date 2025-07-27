# ADR-002: Vector Search Implementation

## Status
Accepted

## Context
The system requires semantic search capabilities to understand user queries beyond exact keyword matching. This involves:
- Converting documents and queries to vector embeddings
- Efficient similarity search across large document collections
- Integration with existing keyword search
- Graceful fallback when vector services unavailable

## Decision
We will use **FAISS** (Facebook AI Similarity Search) with **sentence-transformers** for vector search implementation.

## Rationale

### FAISS Advantages:
- **Performance**: Highly optimized similarity search algorithms
- **Scalability**: Handles millions of vectors efficiently
- **No External Dependencies**: Runs locally without API calls
- **Flexible Indexing**: Multiple index types for different use cases
- **Cost Effective**: No per-query costs unlike cloud services
- **Open Source**: Full control over implementation and data

### sentence-transformers Benefits:
- **Quality**: State-of-the-art semantic understanding
- **Variety**: Multiple pre-trained models available
- **Efficiency**: Optimized for inference performance
- **Local Execution**: No external API dependencies
- **Caching**: Embeddings can be cached for performance

### Considered Alternatives:

#### Pinecone
- **Pros**: Managed service, excellent performance
- **Cons**: Vendor lock-in, ongoing costs, external dependency

#### Weaviate
- **Pros**: Rich feature set, GraphQL interface
- **Cons**: Additional infrastructure complexity, resource overhead

#### OpenAI Embeddings
- **Pros**: High quality, easy integration
- **Cons**: API costs, rate limits, external dependency

#### Elasticsearch Vector Search
- **Pros**: Integrated with existing search infrastructure
- **Cons**: Higher resource requirements, complexity

## Implementation Details

### Model Selection:
- **Primary**: `all-MiniLM-L6-v2` (384 dimensions, good balance of speed/quality)
- **Alternative**: `all-mpnet-base-v2` (768 dimensions, higher quality)

### Index Configuration:
- **Index Type**: IndexFlatIP for exact search with small datasets
- **Upgrade Path**: IndexIVFFlat for larger datasets (>100k documents)
- **Distance Metric**: Inner Product (normalized vectors)

### Integration Strategy:
- **Hybrid Search**: Combine vector similarity with keyword relevance
- **Configurable Weights**: Adjust semantic vs keyword importance
- **Fallback Mechanism**: Graceful degradation to keyword-only search

## Architecture

```python
class VectorSearch:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.encoder = SentenceTransformer(model_name)
        self.index = faiss.IndexFlatIP(self.encoder.get_sentence_embedding_dimension())
    
    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        # Generate query embedding
        # Search FAISS index
        # Return ranked results
```

## Performance Considerations

### Memory Usage:
- 384-dim vectors: ~1.5KB per document
- 100k documents: ~150MB vector storage
- Model loading: ~90MB for MiniLM-L6-v2

### Search Performance:
- Exact search: O(n) for n documents
- Index search: O(log n) with IVF index
- Target: <100ms for queries on 100k documents

## Consequences

### Positive:
- No external API dependencies or costs
- Fast local similarity search
- Full control over vector generation and indexing
- Excellent performance for medium-scale deployments
- Easy to cache and optimize

### Negative:
- Requires local model storage (~90MB)
- CPU/memory requirements for embedding generation
- Need to manage model updates manually
- Scaling limits compared to managed services

## Migration Path
1. **Phase 1**: Implement with FAISS + local models
2. **Phase 2**: Add embedding caching for performance
3. **Phase 3**: Consider pgvector integration for unified storage
4. **Phase 4**: Evaluate managed services for very large deployments

## Monitoring
- Embedding generation time
- Search query performance
- Index size and memory usage
- Model accuracy metrics
- Fallback usage rates