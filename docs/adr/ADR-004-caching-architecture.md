# ADR-004: Caching Architecture Strategy

**Status**: Accepted  
**Date**: 2024-08-03  
**Authors**: Development Team  
**Reviewers**: Architecture Team, Performance Team  

## Context

The Slack Knowledge Base Agent requires an efficient caching strategy to optimize performance, reduce external API costs, and provide fast response times. We need to design a multi-layer caching architecture that handles different types of data with appropriate retention policies and invalidation strategies.

## Decision

We will implement a **hybrid multi-layer caching architecture** combining Redis distributed cache with intelligent in-memory caching:

### Core Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Application Layer                       │
├─────────────────────────────────────────────────────────────┤
│  L1 Cache (In-Memory)    │    L2 Cache (Redis)             │
│  - Query Results         │    - Search Results             │
│  - Vector Embeddings     │    - LLM Responses              │
│  - User Sessions         │    - Knowledge Base Metadata   │
│  - TTL: 5-30 minutes     │    - TTL: 1-24 hours           │
├─────────────────────────────────────────────────────────────┤
│                    Database Layer                           │
│  PostgreSQL (Persistent Storage) + FAISS (Vector Index)    │
└─────────────────────────────────────────────────────────────┘
```

### Implementation Strategy

1. **L1 (In-Memory) Cache**: Fast access for frequently used data
2. **L2 (Redis) Cache**: Distributed cache for shared data across instances
3. **Smart Eviction**: LRU with size and TTL-based eviction
4. **Cache Warming**: Proactive loading of popular content
5. **Intelligent Invalidation**: Event-driven cache updates

## Rationale

### Performance Benefits
- **Sub-millisecond**: L1 cache access times
- **Single-digit milliseconds**: L2 cache access times
- **Cost Reduction**: 80% reduction in LLM API calls through response caching
- **Scalability**: Distributed caching supports horizontal scaling

### Cache Hit Rate Targets
| Cache Type | Target Hit Rate | Benefit |
|------------|----------------|---------|
| Query Results | 85% | Fast response times |
| Vector Embeddings | 95% | Reduced computation |
| LLM Responses | 70% | Cost optimization |
| Search Results | 80% | Improved UX |

## Architecture Components

### L1 Cache (In-Memory)
```python
class InMemoryCache:
    def __init__(self, max_size: int = 1000, default_ttl: int = 300):
        self.cache = {}
        self.access_times = {}
        self.max_size = max_size
        self.default_ttl = default_ttl
        
    def get(self, key: str) -> Optional[Any]:
        if self._is_expired(key):
            self._evict(key)
            return None
        
        item = self.cache.get(key)
        if item:
            self.access_times[key] = time.time()
        return item
        
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        if len(self.cache) >= self.max_size:
            self._evict_lru()
        
        self.cache[key] = {
            'value': value,
            'created': time.time(),
            'ttl': ttl or self.default_ttl
        }
        self.access_times[key] = time.time()
```

### L2 Cache (Redis)
```python
class RedisCache:
    def __init__(self, redis_client, default_ttl: int = 3600):
        self.redis = redis_client
        self.default_ttl = default_ttl
        
    async def get(self, key: str) -> Optional[Any]:
        try:
            data = await self.redis.get(key)
            return pickle.loads(data) if data else None
        except Exception as e:
            logger.warning(f"Redis get failed: {e}")
            return None
            
    async def set(self, key: str, value: Any, ttl: Optional[int] = None):
        try:
            serialized = pickle.dumps(value)
            await self.redis.setex(key, ttl or self.default_ttl, serialized)
        except Exception as e:
            logger.warning(f"Redis set failed: {e}")
```

### Unified Cache Manager
```python
class CacheManager:
    def __init__(self, l1_cache: InMemoryCache, l2_cache: RedisCache):
        self.l1 = l1_cache
        self.l2 = l2_cache
        
    async def get(self, key: str) -> Optional[Any]:
        # Check L1 first
        result = self.l1.get(key)
        if result is not None:
            self.metrics.record_hit('l1')
            return result
            
        # Check L2
        result = await self.l2.get(key)
        if result is not None:
            self.metrics.record_hit('l2')
            # Promote to L1
            self.l1.set(key, result)
            return result
            
        self.metrics.record_miss()
        return None
        
    async def set(self, key: str, value: Any, ttl: Optional[int] = None):
        # Set in both layers
        self.l1.set(key, value, ttl)
        await self.l2.set(key, value, ttl)
```

## Cache Strategy by Data Type

### 1. Search Results Cache
```python
def cache_search_results(query: str, results: List[Document]) -> str:
    # Normalize query for consistent caching
    normalized_query = normalize_query(query)
    cache_key = f"search:{hash(normalized_query)}"
    
    # Cache with semantic similarity consideration
    cache_manager.set(
        cache_key, 
        results, 
        ttl=SEARCH_CACHE_TTL  # 30 minutes
    )
    return cache_key
```

### 2. LLM Response Cache
```python
def cache_llm_response(prompt_hash: str, context_hash: str, response: str):
    cache_key = f"llm:{prompt_hash}:{context_hash}"
    
    # Longer TTL for stable responses
    cache_manager.set(
        cache_key,
        response,
        ttl=LLM_CACHE_TTL  # 24 hours
    )
```

### 3. Vector Embeddings Cache
```python
def cache_embeddings(text: str, embeddings: np.ndarray):
    text_hash = hashlib.sha256(text.encode()).hexdigest()
    cache_key = f"embeddings:{text_hash}"
    
    # Long TTL since embeddings are stable
    cache_manager.set(
        cache_key,
        embeddings,
        ttl=EMBEDDINGS_CACHE_TTL  # 7 days
    )
```

### 4. Knowledge Base Metadata Cache
```python
def cache_kb_metadata(metadata: Dict):
    cache_key = "kb:metadata"
    
    # Shorter TTL for frequently changing data
    cache_manager.set(
        cache_key,
        metadata,
        ttl=METADATA_CACHE_TTL  # 5 minutes
    )
```

## Cache Invalidation Strategy

### Event-Driven Invalidation
```python
class CacheInvalidator:
    def __init__(self, cache_manager: CacheManager):
        self.cache_manager = cache_manager
        
    async def on_knowledge_update(self, event: KnowledgeUpdateEvent):
        # Invalidate related search results
        pattern = f"search:*{event.topic}*"
        await self._invalidate_pattern(pattern)
        
        # Invalidate metadata
        await self.cache_manager.invalidate("kb:metadata")
        
    async def on_document_ingestion(self, event: DocumentEvent):
        # Invalidate vector caches for updated documents
        await self._invalidate_document_cache(event.document_id)
        
    async def _invalidate_pattern(self, pattern: str):
        keys = await self.l2.redis.keys(pattern)
        if keys:
            await self.l2.redis.delete(*keys)
```

### Time-Based Invalidation
```python
# Configure TTL based on data volatility
CACHE_TTL_CONFIG = {
    'search_results': 1800,      # 30 minutes
    'llm_responses': 86400,      # 24 hours
    'embeddings': 604800,        # 7 days
    'user_sessions': 3600,       # 1 hour
    'kb_metadata': 300,          # 5 minutes
    'analytics': 900,            # 15 minutes
}
```

## Performance Optimizations

### Cache Warming
```python
class CacheWarmer:
    async def warm_popular_queries(self):
        # Pre-load frequently asked questions
        popular_queries = await self.analytics.get_popular_queries(limit=50)
        
        for query in popular_queries:
            if not await self.cache_manager.exists(f"search:{hash(query)}"):
                results = await self.search_service.search(query)
                await self.cache_manager.set(f"search:{hash(query)}", results)
                
    async def warm_embeddings(self):
        # Pre-compute embeddings for new documents
        new_docs = await self.kb.get_uncached_documents()
        
        for doc in new_docs:
            embeddings = await self.vector_service.get_embeddings(doc.content)
            await self.cache_manager.set(f"embeddings:{doc.hash}", embeddings)
```

### Cache Compression
```python
def compress_cache_value(value: Any) -> bytes:
    # Use gzip compression for large values
    serialized = pickle.dumps(value)
    if len(serialized) > COMPRESSION_THRESHOLD:
        return gzip.compress(serialized)
    return serialized

def decompress_cache_value(data: bytes) -> Any:
    try:
        # Try decompressing first
        decompressed = gzip.decompress(data)
        return pickle.loads(decompressed)
    except:
        # Fall back to direct deserialization
        return pickle.loads(data)
```

## Configuration

### Environment Variables
```env
# Redis Configuration
REDIS_URL=redis://localhost:6379/0
REDIS_MAX_CONNECTIONS=20
REDIS_SOCKET_TIMEOUT=5

# Cache Configuration
L1_CACHE_SIZE=1000
L1_CACHE_TTL=300
L2_CACHE_TTL=3600

# Cache TTL Overrides
SEARCH_CACHE_TTL=1800
LLM_CACHE_TTL=86400
EMBEDDINGS_CACHE_TTL=604800

# Performance Settings
CACHE_COMPRESSION_THRESHOLD=1024
CACHE_WARMING_ENABLED=true
CACHE_METRICS_ENABLED=true
```

### Cache Configuration
```yaml
# config/cache.yml
cache:
  enabled: true
  
  l1:
    max_size: 1000
    default_ttl: 300
    eviction_policy: 'lru'
    
  l2:
    host: 'localhost'
    port: 6379
    db: 0
    max_connections: 20
    default_ttl: 3600
    
  strategies:
    search_results:
      ttl: 1800
      compression: true
      
    llm_responses:
      ttl: 86400
      compression: true
      
    embeddings:
      ttl: 604800
      compression: false
      
  warming:
    enabled: true
    popular_queries_limit: 50
    warm_on_startup: true
```

## Monitoring & Metrics

### Cache Performance Metrics
```python
class CacheMetrics:
    def __init__(self):
        self.hit_counter = Counter()
        self.miss_counter = Counter()
        self.latency_histogram = Histogram()
        
    def record_hit(self, cache_level: str):
        self.hit_counter.labels(level=cache_level).inc()
        
    def record_miss(self):
        self.miss_counter.inc()
        
    def record_latency(self, operation: str, duration: float):
        self.latency_histogram.labels(operation=operation).observe(duration)
        
    def get_hit_rate(self, cache_level: str) -> float:
        hits = self.hit_counter.labels(level=cache_level)._value.sum()
        total = hits + self.miss_counter._value.sum()
        return hits / total if total > 0 else 0
```

### Health Checks
```python
async def cache_health_check() -> Dict[str, bool]:
    health = {}
    
    # Check L1 cache
    try:
        test_key = f"health_check:{time.time()}"
        cache_manager.l1.set(test_key, "test")
        health['l1_cache'] = cache_manager.l1.get(test_key) == "test"
    except:
        health['l1_cache'] = False
        
    # Check L2 cache (Redis)
    try:
        await cache_manager.l2.redis.ping()
        health['l2_cache'] = True
    except:
        health['l2_cache'] = False
        
    return health
```

## Security Considerations

### Cache Security
```python
class SecureCache:
    def __init__(self, encryption_key: bytes):
        self.cipher = Fernet(encryption_key)
        
    def encrypt_value(self, value: Any) -> bytes:
        serialized = pickle.dumps(value)
        return self.cipher.encrypt(serialized)
        
    def decrypt_value(self, encrypted_data: bytes) -> Any:
        decrypted = self.cipher.decrypt(encrypted_data)
        return pickle.loads(decrypted)
```

### Sensitive Data Handling
- Never cache sensitive user data (passwords, tokens)
- Implement automatic expiration for user sessions
- Use secure serialization methods
- Monitor cache access patterns for anomalies

## Testing Strategy

### Unit Tests
```python
class TestCacheManager:
    async def test_l1_cache_hit(self):
        cache_manager.set("test_key", "test_value")
        result = cache_manager.get("test_key")
        assert result == "test_value"
        assert cache_metrics.get_hit_rate('l1') > 0
        
    async def test_cache_eviction(self):
        # Fill cache beyond capacity
        for i in range(cache_manager.l1.max_size + 10):
            cache_manager.set(f"key_{i}", f"value_{i}")
            
        # Verify LRU eviction
        assert cache_manager.get("key_0") is None
        assert cache_manager.get(f"key_{cache_manager.l1.max_size + 5}") is not None
```

### Load Testing
```python
async def load_test_cache():
    # Simulate concurrent access
    tasks = []
    for i in range(1000):
        task = asyncio.create_task(cache_manager.get(f"key_{i % 100}"))
        tasks.append(task)
        
    results = await asyncio.gather(*tasks)
    
    # Verify performance under load
    assert cache_metrics.get_average_latency() < 0.001  # 1ms
```

## Migration Strategy

### Phase 1: Basic Implementation
- [ ] Implement L1 in-memory cache
- [ ] Add basic TTL and LRU eviction
- [ ] Integrate with existing search functionality

### Phase 2: Redis Integration
- [ ] Add Redis L2 cache layer
- [ ] Implement cache warming
- [ ] Add monitoring and metrics

### Phase 3: Optimization
- [ ] Add compression for large values
- [ ] Implement intelligent eviction strategies
- [ ] Add security features

### Phase 4: Advanced Features
- [ ] Event-driven invalidation
- [ ] Predictive cache warming
- [ ] Advanced analytics

## Alternative Solutions Considered

### Single-Layer Redis Cache
**Pros**: Simplicity, distributed by default
**Cons**: Network latency for all requests, single point of failure
**Decision**: Rejected in favor of hybrid approach for better performance

### Database Query Cache Only
**Pros**: Simplicity, database-native features
**Cons**: Limited control, doesn't help with API costs, slower than memory cache
**Decision**: Complementary, not sufficient alone

### Pure In-Memory Cache
**Pros**: Fastest possible access
**Cons**: No sharing between instances, lost on restart, memory limitations
**Decision**: Used as L1 but needs L2 for production

## Future Enhancements

### Intelligent Prefetching
- Machine learning-based prediction of cache needs
- User behavior analysis for proactive loading
- Time-based prefetching (morning prep for daily questions)

### Advanced Eviction Policies
- Adaptive replacement cache (ARC)
- Frequency-based eviction
- Cost-aware eviction (considering regeneration cost)

### Distributed Cache Coherence
- Event-driven invalidation across instances
- Cache versioning for consistency
- Conflict resolution strategies

## Conclusion

The hybrid multi-layer caching architecture provides optimal performance while maintaining flexibility and reliability. This approach balances speed, cost optimization, and scalability requirements for the Slack Knowledge Base Agent.

The implementation prioritizes cache hit rates and response times while providing robust monitoring and management capabilities.

## References

- [Redis Best Practices](https://redis.io/docs/manual/optimization/)
- [Cache Patterns and Strategies](internal-link)
- [Performance Benchmarking Results](internal-link)
- [Security Guidelines for Caching](internal-link)