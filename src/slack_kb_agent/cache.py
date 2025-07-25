"""Redis-based caching layer for performance optimization."""

from __future__ import annotations

import atexit
import json
import hashlib
import logging
import base64
from typing import Any, Optional, List, Union, Dict
from dataclasses import dataclass, asdict
from datetime import timedelta

try:
    import redis
    import numpy as np
    REDIS_AVAILABLE = True
except ImportError:
    redis = None
    np = None
    REDIS_AVAILABLE = False

from .models import Document
from .circuit_breaker import CircuitBreaker, CircuitBreakerConfig, CircuitState
from .constants import CircuitBreakerDefaults

logger = logging.getLogger(__name__)


@dataclass
class CacheConfig:
    """Configuration for Redis caching."""
    enabled: bool = True
    host: Optional[str] = None
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    socket_timeout: float = 5.0
    socket_connect_timeout: float = 5.0
    # Cache TTL settings (in seconds)
    embedding_ttl: int = 3600 * 24 * 7  # 7 days for embeddings
    query_expansion_ttl: int = 3600 * 24  # 1 day for query expansions
    search_results_ttl: int = 3600  # 1 hour for search results
    max_connections: int = 20
    
    @classmethod
    def from_env(cls) -> 'CacheConfig':
        """Create cache config from environment variables.
        
        Raises:
            ValueError: If cache is enabled but required Redis configuration is missing.
        """
        import os
        
        enabled = os.getenv("CACHE_ENABLED", "true").lower() == "true"
        host = os.getenv("REDIS_HOST")
        
        # If cache is enabled, require explicit Redis host configuration
        if enabled and not host:
            raise ValueError(
                "REDIS_HOST environment variable is required when caching is enabled. "
                "Set CACHE_ENABLED=false to disable caching or provide REDIS_HOST."
            )
        
        return cls(
            enabled=enabled,
            host=host,
            port=int(os.getenv("REDIS_PORT", "6379")),
            db=int(os.getenv("REDIS_DB", "0")),
            password=os.getenv("REDIS_PASSWORD"),
            socket_timeout=float(os.getenv("REDIS_SOCKET_TIMEOUT", "5.0")),
            socket_connect_timeout=float(os.getenv("REDIS_CONNECT_TIMEOUT", "5.0")),
            embedding_ttl=int(os.getenv("CACHE_EMBEDDING_TTL", str(3600 * 24 * 7))),
            query_expansion_ttl=int(os.getenv("CACHE_QUERY_EXPANSION_TTL", str(3600 * 24))),
            search_results_ttl=int(os.getenv("CACHE_SEARCH_RESULTS_TTL", "3600")),
            max_connections=int(os.getenv("REDIS_MAX_CONNECTIONS", "20"))
        )


class CacheManager:
    """Redis-based cache manager with fallback to no-op when Redis unavailable."""
    
    def __init__(self, config: Optional[CacheConfig] = None):
        """Initialize cache manager.
        
        Args:
            config: Cache configuration. If None, loads from environment.
        """
        self.config = config or CacheConfig.from_env()
        self._redis_client: Optional[redis.Redis] = None
        self._connection_pool: Optional[redis.ConnectionPool] = None
        self._is_connected = False
        self._circuit_breaker: Optional[CircuitBreaker] = None
        
        if self.config.enabled and REDIS_AVAILABLE and self.config.host:
            self._initialize_redis()
        else:
            if not REDIS_AVAILABLE:
                logger.warning("Redis not available. Caching disabled.")
            elif not self.config.enabled:
                logger.info("Caching disabled via configuration.")
            elif not self.config.host:
                logger.warning("Redis host not configured. Caching disabled.")
    
    def _initialize_redis(self) -> None:
        """Initialize Redis connection with connection pooling."""
        try:
            # Create connection pool for better performance
            self._connection_pool = redis.ConnectionPool(
                host=self.config.host,
                port=self.config.port,
                db=self.config.db,
                password=self.config.password,
                socket_timeout=self.config.socket_timeout,
                socket_connect_timeout=self.config.socket_connect_timeout,
                max_connections=self.config.max_connections,
                decode_responses=False  # We handle binary data
            )
            
            self._redis_client = redis.Redis(
                connection_pool=self._connection_pool,
                decode_responses=False
            )
            
            # Initialize circuit breaker for connection testing
            self._circuit_breaker = self._get_circuit_breaker()
            
            # Test connection through circuit breaker
            self._circuit_breaker.call(lambda: self._redis_client.ping())
            self._is_connected = True
            logger.info(f"Connected to Redis at {self.config.host}:{self.config.port}")
            
        except (ConnectionError, TimeoutError) as e:
            logger.warning(f"Failed to connect to Redis - connection/timeout error: {e}. Caching disabled.")
            self._is_connected = False
            self._redis_client = None
            self._connection_pool = None
            self._circuit_breaker = None
        except (ValueError, TypeError) as e:
            logger.warning(f"Failed to connect to Redis - configuration error: {e}. Caching disabled.")
            self._is_connected = False
            self._redis_client = None
            self._connection_pool = None
            self._circuit_breaker = None
        except MemoryError as e:
            logger.warning(f"Failed to connect to Redis - insufficient memory: {e}. Caching disabled.")
            self._is_connected = False
            self._redis_client = None
            self._connection_pool = None
            self._circuit_breaker = None
        except Exception as e:
            # Defensive catch-all for Redis connection - caching should never break the app
            error_type = type(e).__name__
            logger.warning(f"Failed to connect to Redis - unexpected {error_type}: {e}. Caching disabled.")
            self._is_connected = False
            self._redis_client = None
            self._connection_pool = None
            self._circuit_breaker = None
    
    def _get_circuit_breaker(self) -> CircuitBreaker:
        """Get or create circuit breaker for Redis operations."""
        circuit_config = CircuitBreakerConfig(
            failure_threshold=CircuitBreakerDefaults.REDIS_FAILURE_THRESHOLD,
            success_threshold=CircuitBreakerDefaults.REDIS_SUCCESS_THRESHOLD,
            timeout_seconds=CircuitBreakerDefaults.REDIS_TIMEOUT_SECONDS,
            half_open_max_requests=CircuitBreakerDefaults.REDIS_HALF_OPEN_MAX_REQUESTS,
            failure_window_seconds=CircuitBreakerDefaults.REDIS_FAILURE_WINDOW_SECONDS,
            service_name="redis"
        )
        return CircuitBreaker(circuit_config)
    
    def is_available(self) -> bool:
        """Check if cache is available."""
        return (self._is_connected and 
                self._redis_client is not None and 
                self._circuit_breaker is not None and 
                self._circuit_breaker.state != CircuitState.OPEN)
    
    @property
    def circuit_breaker(self) -> Optional[CircuitBreaker]:
        """Get the circuit breaker instance for testing/monitoring."""
        return self._circuit_breaker
    
    def _generate_key(self, namespace: str, identifier: str) -> str:
        """Generate cache key with namespace."""
        # Hash long identifiers to ensure key length limits
        if len(identifier) > 200:
            identifier = hashlib.sha256(identifier.encode()).hexdigest()
        return f"slack_kb:{namespace}:{identifier}"
    
    def get_embedding(self, text: str, model_name: str) -> Optional[np.ndarray]:
        """Get cached embedding for text."""
        if not self.is_available():
            return None
        
        key = self._generate_key("embedding", f"{model_name}:{text}")
        try:
            cached_data = self._circuit_breaker.call(lambda: self._redis_client.get(key))
            if cached_data:
                embedding = self._deserialize_embedding(cached_data)
                logger.debug(f"Cache hit for embedding: {text[:50]}...")
                return embedding
        except (ValueError, TypeError) as e:
            logger.warning(f"Failed to deserialize embedding from cache: {e}")
            # Remove corrupted cache entry
            try:
                self._circuit_breaker.call(lambda: self._redis_client.delete(key))
            except (ConnectionError, TimeoutError) as delete_error:
                logger.warning(f"Failed to delete corrupted cache entry {key} - connection issue: {delete_error}")
            except Exception as delete_error:
                logger.warning(f"Failed to delete corrupted cache entry {key} - unexpected error: {delete_error}")
        except (ConnectionError, TimeoutError) as e:
            logger.warning(f"Failed to get embedding from cache - connection issue: {e}")
        except MemoryError as e:
            logger.warning(f"Failed to get embedding from cache - memory issue: {e}")
        except Exception as e:
            # Defensive catch-all - cache failures should never break the application
            error_type = type(e).__name__
            logger.warning(f"Failed to get embedding from cache - unexpected {error_type}: {e}")
        
        return None
    
    def set_embedding(self, text: str, model_name: str, embedding: np.ndarray) -> None:
        """Cache embedding for text."""
        if not self.is_available():
            return
        
        key = self._generate_key("embedding", f"{model_name}:{text}")
        try:
            serialized = self._serialize_embedding(embedding)
            self._circuit_breaker.call(lambda: self._redis_client.setex(key, self.config.embedding_ttl, serialized))
            logger.debug(f"Cached embedding for: {text[:50]}...")
        except (TypeError, ValueError) as e:
            logger.warning(f"Failed to serialize embedding: {e}")
        except (ConnectionError, TimeoutError) as e:
            logger.warning(f"Failed to cache embedding - connection issue: {e}")
        except MemoryError as e:
            logger.warning(f"Failed to cache embedding - memory issue: {e}")
        except Exception as e:
            # Defensive catch-all - cache failures should never break the application
            error_type = type(e).__name__
            logger.warning(f"Failed to cache embedding - unexpected {error_type}: {e}")
    
    def get_query_expansion(self, query: str, expansion_type: str) -> Optional[List[str]]:
        """Get cached query expansion."""
        if not self.is_available():
            return None
        
        key = self._generate_key("query_expansion", f"{expansion_type}:{query}")
        try:
            cached_data = self._circuit_breaker.call(lambda: self._redis_client.get(key))
            if cached_data:
                expansion = json.loads(cached_data.decode())
                logger.debug(f"Cache hit for query expansion: {query}")
                return expansion
        except Exception as e:
            logger.warning(f"Failed to get query expansion from cache: {e}")
        
        return None
    
    def set_query_expansion(self, query: str, expansion_type: str, expansion: List[str]) -> None:
        """Cache query expansion."""
        if not self.is_available():
            return
        
        key = self._generate_key("query_expansion", f"{expansion_type}:{query}")
        try:
            serialized = json.dumps(expansion)
            self._circuit_breaker.call(lambda: self._redis_client.setex(key, self.config.query_expansion_ttl, serialized))
            logger.debug(f"Cached query expansion for: {query}")
        except Exception as e:
            logger.warning(f"Failed to cache query expansion: {e}")
    
    def get_search_results(self, query_hash: str) -> Optional[List[Dict[str, Any]]]:
        """Get cached search results."""
        if not self.is_available():
            return None
        
        key = self._generate_key("search_results", query_hash)
        try:
            cached_data = self._circuit_breaker.call(lambda: self._redis_client.get(key))
            if cached_data:
                results = json.loads(cached_data.decode())
                logger.debug(f"Cache hit for search results: {query_hash[:10]}...")
                return results
        except Exception as e:
            logger.warning(f"Failed to get search results from cache: {e}")
        
        return None
    
    def set_search_results(self, query_hash: str, results: List[Dict[str, Any]]) -> None:
        """Cache search results."""
        if not self.is_available():
            return
        
        key = self._generate_key("search_results", query_hash)
        try:
            serialized = json.dumps(results)
            self._circuit_breaker.call(lambda: self._redis_client.setex(key, self.config.search_results_ttl, serialized))
            logger.debug(f"Cached search results for: {query_hash[:10]}...")
        except Exception as e:
            logger.warning(f"Failed to cache search results: {e}")
    
    def generate_search_hash(self, query: str, params: Dict[str, Any]) -> str:
        """Generate hash for search query with parameters."""
        # Include query and relevant search parameters
        search_key = {
            "query": query,
            "params": sorted(params.items())  # Sort for consistent hashing
        }
        serialized = json.dumps(search_key, sort_keys=True)
        return hashlib.sha256(serialized.encode()).hexdigest()
    
    def invalidate_search_cache(self, pattern: str = "search_results:*") -> int:
        """Invalidate search results cache (e.g., when documents are updated)."""
        if not self.is_available():
            return 0
        
        try:
            full_pattern = f"slack_kb:{pattern}"
            keys = self._circuit_breaker.call(lambda: self._redis_client.keys(full_pattern))
            if keys:
                deleted = self._circuit_breaker.call(lambda: self._redis_client.delete(*keys))
                logger.info(f"Invalidated {deleted} search cache entries")
                return deleted
        except Exception as e:
            logger.warning(f"Failed to invalidate search cache: {e}")
        
        return 0
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if not self.is_available():
            return {"status": "unavailable", "reason": "Redis not connected"}
        
        try:
            info = self._circuit_breaker.call(lambda: self._redis_client.info())
            
            # Count keys by namespace
            key_counts = {}
            for namespace in ["embedding", "query_expansion", "search_results"]:
                pattern = f"slack_kb:{namespace}:*"
                keys = self._circuit_breaker.call(lambda: self._redis_client.keys(pattern))
                key_counts[namespace] = len(keys)
            
            return {
                "status": "connected",
                "redis_version": info.get("redis_version", "unknown"),
                "used_memory": info.get("used_memory_human", "unknown"),
                "connected_clients": info.get("connected_clients", 0),
                "total_commands_processed": info.get("total_commands_processed", 0),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
                "hit_rate": self._calculate_hit_rate(info),
                "key_counts": key_counts,
                "config": {
                    "host": self.config.host,
                    "port": self.config.port,
                    "db": self.config.db,
                    "embedding_ttl": self.config.embedding_ttl,
                    "query_expansion_ttl": self.config.query_expansion_ttl,
                    "search_results_ttl": self.config.search_results_ttl
                }
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def _calculate_hit_rate(self, info: Dict[str, Any]) -> float:
        """Calculate cache hit rate."""
        hits = info.get("keyspace_hits", 0)
        misses = info.get("keyspace_misses", 0)
        total = hits + misses
        return (hits / total * 100) if total > 0 else 0.0
    
    def _serialize_embedding(self, embedding: np.ndarray) -> bytes:
        """Safely serialize numpy array embedding to bytes using JSON + base64.
        
        This replaces the unsafe pickle serialization with a secure alternative.
        The format stores array metadata in JSON and the data as base64.
        """
        if not isinstance(embedding, np.ndarray):
            raise TypeError("Embedding must be a numpy array")
        
        # Create metadata dictionary
        metadata = {
            "dtype": str(embedding.dtype),
            "shape": embedding.shape,
            "data": base64.b64encode(embedding.tobytes()).decode('ascii')
        }
        
        # Serialize to JSON bytes
        return json.dumps(metadata).encode('utf-8')
    
    def _deserialize_embedding(self, data: bytes) -> np.ndarray:
        """Safely deserialize numpy array embedding from bytes.
        
        This replaces the unsafe pickle deserialization with a secure alternative.
        """
        try:
            # Parse JSON metadata
            metadata = json.loads(data.decode('utf-8'))
            
            # Validate required fields
            required_fields = ["dtype", "shape", "data"]
            for field in required_fields:
                if field not in metadata:
                    raise ValueError(f"Missing required field: {field}")
            
            # Reconstruct numpy array
            array_data = base64.b64decode(metadata["data"])
            embedding = np.frombuffer(array_data, dtype=metadata["dtype"])
            embedding = embedding.reshape(metadata["shape"])
            
            return embedding
            
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            raise ValueError(f"Invalid embedding cache data: {e}") from e
    
    def flush_cache(self, namespace: Optional[str] = None) -> int:
        """Flush cache entries. If namespace is provided, only flush that namespace."""
        if not self.is_available():
            return 0
        
        try:
            if namespace:
                pattern = f"slack_kb:{namespace}:*"
                keys = self._circuit_breaker.call(lambda: self._redis_client.keys(pattern))
                if keys:
                    deleted = self._circuit_breaker.call(lambda: self._redis_client.delete(*keys))
                    logger.info(f"Flushed {deleted} entries from {namespace} cache")
                    return deleted
            else:
                # Flush all slack_kb keys
                pattern = "slack_kb:*"
                keys = self._circuit_breaker.call(lambda: self._redis_client.keys(pattern))
                if keys:
                    deleted = self._circuit_breaker.call(lambda: self._redis_client.delete(*keys))
                    logger.info(f"Flushed {deleted} total cache entries")
                    return deleted
        except Exception as e:
            logger.warning(f"Failed to flush cache: {e}")
        
        return 0
    
    def close(self) -> None:
        """Close Redis connection."""
        if self._connection_pool:
            self._connection_pool.disconnect()
            logger.info("Closed Redis connection pool")


# Global cache instance (initialized lazily)
_cache_manager: Optional[CacheManager] = None


def _cleanup_cache_resources() -> None:
    """Clean up global cache resources on application shutdown."""
    global _cache_manager
    
    if _cache_manager is not None:
        try:
            _cache_manager.close()
            logger.info("Redis connection pool closed during cleanup")
        except Exception as e:
            logger.warning(f"Error closing cache manager during cleanup: {e}")
        finally:
            _cache_manager = None


# Register cleanup function to run on application exit
atexit.register(_cleanup_cache_resources)


def get_cache_manager() -> CacheManager:
    """Get global cache manager instance."""
    global _cache_manager
    
    if _cache_manager is None:
        _cache_manager = CacheManager()
    
    return _cache_manager


def is_cache_available() -> bool:
    """Check if caching is available."""
    return REDIS_AVAILABLE and get_cache_manager().is_available()