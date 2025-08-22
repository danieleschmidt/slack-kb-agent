"""Advanced caching system for Neural Architecture Search results.

This module implements multi-tier caching with intelligent eviction,
result prediction, and cache warming for optimal NAS performance.
"""

import asyncio
import hashlib
import json
import logging
import pickle
import time
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np
from collections import OrderedDict, defaultdict

from .neural_architecture_search import ArchitectureCandidate
from .advanced_cache import AdvancedLRUCache

logger = logging.getLogger(__name__)


class CacheLevel(Enum):
    """Cache levels in the hierarchy."""
    L1_MEMORY = "l1_memory"
    L2_REDIS = "l2_redis"
    L3_DISK = "l3_disk"
    L4_DISTRIBUTED = "l4_distributed"


class CacheStrategy(Enum):
    """Caching strategies for different scenarios."""
    LRU = "lru"
    LFU = "lfu"
    TTL = "ttl"
    INTELLIGENT = "intelligent"
    PREDICTIVE = "predictive"


class EvictionReason(Enum):
    """Reasons for cache eviction."""
    CAPACITY_EXCEEDED = "capacity_exceeded"
    TTL_EXPIRED = "ttl_expired"
    RELEVANCE_LOW = "relevance_low"
    MANUAL_INVALIDATION = "manual_invalidation"
    CACHE_WARMING = "cache_warming"


@dataclass
class CacheEntry:
    """Represents a cache entry with metadata."""
    key: str
    value: Any
    created_at: float
    last_accessed: float
    access_count: int
    ttl_seconds: Optional[float]
    size_bytes: int
    relevance_score: float
    prediction_accuracy: Optional[float] = None
    tags: Set[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = set()
    
    @property
    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        if self.ttl_seconds is None:
            return False
        return time.time() - self.created_at > self.ttl_seconds
    
    @property
    def age_seconds(self) -> float:
        """Get age of cache entry in seconds."""
        return time.time() - self.created_at


@dataclass
class CacheStats:
    """Cache performance statistics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size_bytes: int = 0
    entry_count: int = 0
    hit_rate: float = 0.0
    average_access_time_ms: float = 0.0
    memory_utilization: float = 0.0


class IntelligentCacheManager:
    """Intelligent cache manager with predictive capabilities."""
    
    def __init__(
        self,
        max_size_bytes: int = 1024 * 1024 * 1024,  # 1GB
        max_entries: int = 100000,
        default_ttl: float = 3600.0,  # 1 hour
        cache_strategy: CacheStrategy = CacheStrategy.INTELLIGENT
    ):
        self.max_size_bytes = max_size_bytes
        self.max_entries = max_entries
        self.default_ttl = default_ttl
        self.cache_strategy = cache_strategy
        
        # Multi-level cache storage
        self.l1_cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.l2_cache: Dict[str, CacheEntry] = {}  # Simulated Redis
        self.l3_cache: Dict[str, str] = {}  # Disk paths
        
        # Cache metadata
        self.cache_stats = CacheStats()
        self.access_patterns: Dict[str, List[float]] = defaultdict(list)
        self.similarity_index: Dict[str, List[str]] = defaultdict(list)
        
        # Predictive caching
        self.prediction_model = self._initialize_prediction_model()
        self.cache_warming_queue: List[str] = []
        
        # Background tasks
        self._background_tasks: List[asyncio.Task] = []
        self._start_background_tasks()
        
        logger.info(f"Initialized intelligent cache manager with {cache_strategy.value} strategy")
    
    def _initialize_prediction_model(self) -> Dict[str, Any]:
        """Initialize simple prediction model for cache behavior."""
        return {
            "access_frequency_weights": np.ones(24),  # Hourly weights
            "similarity_threshold": 0.8,
            "prefetch_probability_threshold": 0.6,
            "learning_rate": 0.1
        }
    
    def _start_background_tasks(self):
        """Start background maintenance tasks."""
        self._background_tasks.extend([
            asyncio.create_task(self._cache_maintenance_loop()),
            asyncio.create_task(self._predictive_prefetch_loop()),
            asyncio.create_task(self._statistics_update_loop())
        ])
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache with intelligent lookup."""
        start_time = time.time()
        
        try:
            # L1 cache lookup
            if key in self.l1_cache:
                entry = self.l1_cache[key]
                if not entry.is_expired:
                    self._update_access_stats(entry)
                    self.cache_stats.hits += 1
                    self._record_access_pattern(key)
                    return entry.value
                else:
                    # Remove expired entry
                    self._evict_entry(key, EvictionReason.TTL_EXPIRED)
            
            # L2 cache lookup (Redis simulation)
            if key in self.l2_cache:
                entry = self.l2_cache[key]
                if not entry.is_expired:
                    # Promote to L1
                    await self._promote_to_l1(key, entry)
                    self.cache_stats.hits += 1
                    self._record_access_pattern(key)
                    return entry.value
                else:
                    del self.l2_cache[key]
            
            # L3 cache lookup (disk simulation)
            if key in self.l3_cache:
                value = await self._load_from_disk(key)
                if value is not None:
                    # Create cache entry and promote to L1
                    entry = CacheEntry(
                        key=key,
                        value=value,
                        created_at=time.time(),
                        last_accessed=time.time(),
                        access_count=1,
                        ttl_seconds=self.default_ttl,
                        size_bytes=self._estimate_size(value),
                        relevance_score=0.5
                    )
                    await self._promote_to_l1(key, entry)
                    self.cache_stats.hits += 1
                    self._record_access_pattern(key)
                    return value
            
            # Cache miss
            self.cache_stats.misses += 1
            return None
            
        finally:
            access_time = (time.time() - start_time) * 1000
            self._update_access_time_stats(access_time)
    
    async def put(
        self,
        key: str,
        value: Any,
        ttl: Optional[float] = None,
        tags: Optional[Set[str]] = None,
        relevance_score: float = 1.0
    ) -> bool:
        """Put value in cache with intelligent placement."""
        try:
            # Create cache entry
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=time.time(),
                last_accessed=time.time(),
                access_count=1,
                ttl_seconds=ttl or self.default_ttl,
                size_bytes=self._estimate_size(value),
                relevance_score=relevance_score,
                tags=tags or set()
            )
            
            # Check capacity and evict if necessary
            await self._ensure_capacity(entry.size_bytes)
            
            # Store in L1 cache
            self.l1_cache[key] = entry
            self.l1_cache.move_to_end(key)  # Mark as most recently used
            
            # Update statistics
            self.cache_stats.entry_count += 1
            self.cache_stats.size_bytes += entry.size_bytes
            
            # Update similarity index
            await self._update_similarity_index(key, value)
            
            # Schedule background promotion to lower levels
            asyncio.create_task(self._background_promote_to_lower_levels(key, entry))
            
            return True
            
        except Exception as e:
            logger.error(f"Error putting cache entry {key}: {e}")
            return False
    
    async def invalidate(self, key: str) -> bool:
        """Invalidate cache entry across all levels."""
        invalidated = False
        
        # Remove from L1
        if key in self.l1_cache:
            self._evict_entry(key, EvictionReason.MANUAL_INVALIDATION)
            invalidated = True
        
        # Remove from L2
        if key in self.l2_cache:
            del self.l2_cache[key]
            invalidated = True
        
        # Remove from L3
        if key in self.l3_cache:
            await self._remove_from_disk(key)
            del self.l3_cache[key]
            invalidated = True
        
        return invalidated
    
    async def invalidate_by_tags(self, tags: Set[str]) -> int:
        """Invalidate all cache entries matching any of the given tags."""
        invalidated_count = 0
        
        # Collect keys to invalidate
        keys_to_invalidate = []
        
        for key, entry in list(self.l1_cache.items()):
            if entry.tags.intersection(tags):
                keys_to_invalidate.append(key)
        
        for key, entry in list(self.l2_cache.items()):
            if entry.tags.intersection(tags):
                keys_to_invalidate.append(key)
        
        # Invalidate collected keys
        for key in keys_to_invalidate:
            if await self.invalidate(key):
                invalidated_count += 1
        
        return invalidated_count
    
    def generate_cache_key(self, config: Dict[str, Any], prefix: str = "nas") -> str:
        """Generate deterministic cache key from configuration."""
        # Sort config for consistent key generation
        sorted_config = json.dumps(config, sort_keys=True)
        config_hash = hashlib.sha256(sorted_config.encode()).hexdigest()
        return f"{prefix}:{config_hash[:16]}"
    
    async def cache_architecture_prediction(
        self,
        config: Dict[str, Any],
        prediction_result: Dict[str, float],
        ttl: float = 7200.0  # 2 hours
    ) -> str:
        """Cache architecture prediction result."""
        key = self.generate_cache_key(config, "prediction")
        
        await self.put(
            key=key,
            value=prediction_result,
            ttl=ttl,
            tags={"prediction", "architecture", config.get("architecture_type", "unknown")},
            relevance_score=prediction_result.get("confidence", 0.5)
        )
        
        return key
    
    async def get_cached_prediction(self, config: Dict[str, Any]) -> Optional[Dict[str, float]]:
        """Get cached architecture prediction."""
        key = self.generate_cache_key(config, "prediction")
        return await self.get(key)
    
    async def cache_optimization_result(
        self,
        original_config: Dict[str, Any],
        optimization_result: Dict[str, Any],
        ttl: float = 10800.0  # 3 hours
    ) -> str:
        """Cache architecture optimization result."""
        key = self.generate_cache_key(original_config, "optimization")
        
        await self.put(
            key=key,
            value=optimization_result,
            ttl=ttl,
            tags={"optimization", "tpu", original_config.get("architecture_type", "unknown")},
            relevance_score=optimization_result.get("performance_gain", 1.0)
        )
        
        return key
    
    async def get_cached_optimization(self, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get cached architecture optimization result."""
        key = self.generate_cache_key(config, "optimization")
        return await self.get(key)
    
    async def warm_cache_for_search(
        self,
        search_configs: List[Dict[str, Any]],
        search_space: Dict[str, List[Any]]
    ):
        """Warm cache with predicted useful entries for upcoming search."""
        logger.info("Starting cache warming for search")
        
        # Generate similar configurations
        similar_configs = self._generate_similar_configs(search_configs, search_space)
        
        # Add to warming queue
        for config in similar_configs:
            key = self.generate_cache_key(config)
            if key not in self.cache_warming_queue and not await self.get(key):
                self.cache_warming_queue.append(key)
        
        logger.info(f"Added {len(similar_configs)} configurations to cache warming queue")
    
    def _generate_similar_configs(
        self,
        base_configs: List[Dict[str, Any]],
        search_space: Dict[str, List[Any]],
        num_variations: int = 50
    ) -> List[Dict[str, Any]]:
        """Generate similar configurations for cache warming."""
        similar_configs = []
        
        for base_config in base_configs[:5]:  # Limit to 5 base configs
            for _ in range(num_variations // len(base_configs)):
                config = base_config.copy()
                
                # Apply small variations
                for param, values in search_space.items():
                    if param in config and isinstance(values, list):
                        # Small random variation
                        current_idx = values.index(config[param]) if config[param] in values else 0
                        variation_range = max(1, len(values) // 10)
                        
                        new_idx = np.clip(
                            current_idx + np.random.randint(-variation_range, variation_range + 1),
                            0, len(values) - 1
                        )
                        config[param] = values[new_idx]
                
                similar_configs.append(config)
        
        return similar_configs
    
    async def _ensure_capacity(self, new_entry_size: int):
        """Ensure cache has enough capacity for new entry."""
        while (self.cache_stats.size_bytes + new_entry_size > self.max_size_bytes or
               self.cache_stats.entry_count >= self.max_entries):
            
            if not self.l1_cache:
                break
            
            # Select victim for eviction based on strategy
            victim_key = self._select_eviction_victim()
            if victim_key:
                self._evict_entry(victim_key, EvictionReason.CAPACITY_EXCEEDED)
            else:
                break
    
    def _select_eviction_victim(self) -> Optional[str]:
        """Select cache entry for eviction based on current strategy."""
        if not self.l1_cache:
            return None
        
        if self.cache_strategy == CacheStrategy.LRU:
            return next(iter(self.l1_cache))  # Least recently used
        
        elif self.cache_strategy == CacheStrategy.LFU:
            return min(self.l1_cache.keys(), key=lambda k: self.l1_cache[k].access_count)
        
        elif self.cache_strategy == CacheStrategy.TTL:
            # Find entry closest to expiration
            return min(
                self.l1_cache.keys(),
                key=lambda k: (self.l1_cache[k].created_at + (self.l1_cache[k].ttl_seconds or float('inf')))
            )
        
        elif self.cache_strategy == CacheStrategy.INTELLIGENT:
            return self._intelligent_eviction_selection()
        
        elif self.cache_strategy == CacheStrategy.PREDICTIVE:
            return self._predictive_eviction_selection()
        
        return next(iter(self.l1_cache))  # Fallback to LRU
    
    def _intelligent_eviction_selection(self) -> Optional[str]:
        """Select eviction victim using intelligent scoring."""
        if not self.l1_cache:
            return None
        
        def eviction_score(key: str) -> float:
            entry = self.l1_cache[key]
            current_time = time.time()
            
            # Age factor (older entries more likely to evict)
            age_factor = (current_time - entry.last_accessed) / 3600.0  # Hours since last access
            
            # Frequency factor (less frequently used more likely to evict)
            frequency_factor = 1.0 / max(entry.access_count, 1)
            
            # Relevance factor (less relevant more likely to evict)
            relevance_factor = 1.0 - entry.relevance_score
            
            # Size factor (larger entries slightly more likely to evict)
            size_factor = entry.size_bytes / (1024 * 1024)  # MB
            
            # TTL factor (closer to expiration more likely to evict)
            ttl_factor = 0.0
            if entry.ttl_seconds:
                time_to_expiration = (entry.created_at + entry.ttl_seconds) - current_time
                ttl_factor = 1.0 / max(time_to_expiration / 3600.0, 0.1)  # Hours to expiration
            
            return age_factor * 0.4 + frequency_factor * 0.2 + relevance_factor * 0.2 + size_factor * 0.1 + ttl_factor * 0.1
        
        return max(self.l1_cache.keys(), key=eviction_score)
    
    def _predictive_eviction_selection(self) -> Optional[str]:
        """Select eviction victim using predictive model."""
        if not self.l1_cache:
            return None
        
        # Predict future access probability for each entry
        def future_access_probability(key: str) -> float:
            entry = self.l1_cache[key]
            access_history = self.access_patterns.get(key, [])
            
            if not access_history:
                return 0.1  # Low probability if no history
            
            # Simple prediction based on recent access pattern
            recent_accesses = [t for t in access_history if time.time() - t < 3600]  # Last hour
            hourly_rate = len(recent_accesses)
            
            # Adjust based on overall frequency
            total_frequency = len(access_history) / max(entry.age_seconds / 3600.0, 1.0)
            
            return min(hourly_rate * 0.7 + total_frequency * 0.3, 1.0)
        
        # Select entry with lowest future access probability
        return min(self.l1_cache.keys(), key=future_access_probability)
    
    def _evict_entry(self, key: str, reason: EvictionReason):
        """Evict entry from L1 cache."""
        if key not in self.l1_cache:
            return
        
        entry = self.l1_cache[key]
        
        # Update statistics
        self.cache_stats.evictions += 1
        self.cache_stats.entry_count -= 1
        self.cache_stats.size_bytes -= entry.size_bytes
        
        # Remove from L1
        del self.l1_cache[key]
        
        # Optionally promote to L2 if it's valuable
        if reason == EvictionReason.CAPACITY_EXCEEDED and entry.relevance_score > 0.6:
            self.l2_cache[key] = entry
        
        logger.debug(f"Evicted cache entry {key} due to {reason.value}")
    
    async def _promote_to_l1(self, key: str, entry: CacheEntry):
        """Promote cache entry from lower level to L1."""
        # Ensure capacity
        await self._ensure_capacity(entry.size_bytes)
        
        # Update access stats
        entry.last_accessed = time.time()
        entry.access_count += 1
        
        # Add to L1
        self.l1_cache[key] = entry
        self.l1_cache.move_to_end(key)
        
        # Update statistics
        self.cache_stats.entry_count += 1
        self.cache_stats.size_bytes += entry.size_bytes
    
    async def _background_promote_to_lower_levels(self, key: str, entry: CacheEntry):
        """Promote valuable entries to lower cache levels in background."""
        try:
            # Promote to L2 after some time if frequently accessed
            await asyncio.sleep(300)  # 5 minutes
            
            if key in self.l1_cache and self.l1_cache[key].access_count > 2:
                self.l2_cache[key] = entry
            
            # Promote to L3 after more time if very frequently accessed
            await asyncio.sleep(1800)  # 30 minutes
            
            if key in self.l2_cache and self.l2_cache[key].access_count > 10:
                await self._save_to_disk(key, entry.value)
                self.l3_cache[key] = f"disk:{key}"
                
        except Exception as e:
            logger.warning(f"Background promotion failed for {key}: {e}")
    
    async def _save_to_disk(self, key: str, value: Any) -> str:
        """Save cache entry to disk (simulated)."""
        # In real implementation, would save to actual disk
        return f"disk_path_for_{key}"
    
    async def _load_from_disk(self, key: str) -> Optional[Any]:
        """Load cache entry from disk (simulated)."""
        # In real implementation, would load from actual disk
        if key in self.l3_cache:
            return {"simulated": "disk_value", "key": key}
        return None
    
    async def _remove_from_disk(self, key: str):
        """Remove cache entry from disk (simulated)."""
        # In real implementation, would remove from actual disk
        pass
    
    def _update_access_stats(self, entry: CacheEntry):
        """Update access statistics for cache entry."""
        entry.last_accessed = time.time()
        entry.access_count += 1
    
    def _record_access_pattern(self, key: str):
        """Record access pattern for predictive caching."""
        current_time = time.time()
        self.access_patterns[key].append(current_time)
        
        # Keep only recent history (last 24 hours)
        cutoff_time = current_time - 86400
        self.access_patterns[key] = [t for t in self.access_patterns[key] if t >= cutoff_time]
    
    def _update_access_time_stats(self, access_time_ms: float):
        """Update average access time statistics."""
        if self.cache_stats.hits + self.cache_stats.misses > 0:
            total_ops = self.cache_stats.hits + self.cache_stats.misses
            self.cache_stats.average_access_time_ms = (
                (self.cache_stats.average_access_time_ms * (total_ops - 1) + access_time_ms) / total_ops
            )
    
    def _estimate_size(self, value: Any) -> int:
        """Estimate size of value in bytes."""
        try:
            return len(pickle.dumps(value))
        except Exception:
            # Fallback estimation
            if isinstance(value, str):
                return len(value.encode('utf-8'))
            elif isinstance(value, (int, float)):
                return 8
            elif isinstance(value, dict):
                return sum(self._estimate_size(k) + self._estimate_size(v) for k, v in value.items())
            elif isinstance(value, list):
                return sum(self._estimate_size(item) for item in value)
            else:
                return 1024  # Default 1KB estimate
    
    async def _update_similarity_index(self, key: str, value: Any):
        """Update similarity index for cache warming."""
        # Simple similarity based on config parameters
        if isinstance(value, dict) and "config" in str(value):
            # Extract architecture characteristics
            characteristics = self._extract_characteristics(value)
            
            # Find similar entries
            for other_key in list(self.similarity_index.keys())[-100:]:  # Limit comparison
                if other_key != key:
                    try:
                        other_entry = await self.get(other_key)
                        if other_entry:
                            other_characteristics = self._extract_characteristics(other_entry)
                            similarity = self._calculate_similarity(characteristics, other_characteristics)
                            
                            if similarity > self.prediction_model["similarity_threshold"]:
                                self.similarity_index[key].append(other_key)
                                self.similarity_index[other_key].append(key)
                    except Exception:
                        continue
    
    def _extract_characteristics(self, value: Any) -> Dict[str, Any]:
        """Extract key characteristics from cached value for similarity calculation."""
        characteristics = {}
        
        if isinstance(value, dict):
            # Extract numerical features
            for key, val in value.items():
                if isinstance(val, (int, float)):
                    characteristics[key] = val
                elif isinstance(val, str) and len(val) < 50:
                    characteristics[key] = val
        
        return characteristics
    
    def _calculate_similarity(self, chars1: Dict[str, Any], chars2: Dict[str, Any]) -> float:
        """Calculate similarity between two characteristic sets."""
        if not chars1 or not chars2:
            return 0.0
        
        common_keys = set(chars1.keys()) & set(chars2.keys())
        if not common_keys:
            return 0.0
        
        similarity_sum = 0.0
        for key in common_keys:
            val1, val2 = chars1[key], chars2[key]
            
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                # Numerical similarity
                if val1 == 0 and val2 == 0:
                    similarity_sum += 1.0
                else:
                    max_val = max(abs(val1), abs(val2))
                    if max_val > 0:
                        similarity_sum += 1.0 - abs(val1 - val2) / max_val
            
            elif val1 == val2:
                similarity_sum += 1.0
        
        return similarity_sum / len(common_keys)
    
    # Background maintenance tasks
    async def _cache_maintenance_loop(self):
        """Background cache maintenance."""
        while True:
            try:
                await asyncio.sleep(300)  # 5 minutes
                
                # Remove expired entries
                expired_keys = []
                for key, entry in list(self.l1_cache.items()):
                    if entry.is_expired:
                        expired_keys.append(key)
                
                for key in expired_keys:
                    self._evict_entry(key, EvictionReason.TTL_EXPIRED)
                
                # Update hit rate
                total_ops = self.cache_stats.hits + self.cache_stats.misses
                if total_ops > 0:
                    self.cache_stats.hit_rate = self.cache_stats.hits / total_ops
                
                # Update memory utilization
                self.cache_stats.memory_utilization = self.cache_stats.size_bytes / self.max_size_bytes
                
                logger.debug(f"Cache maintenance: {len(expired_keys)} expired entries removed")
                
            except Exception as e:
                logger.error(f"Cache maintenance error: {e}")
    
    async def _predictive_prefetch_loop(self):
        """Background predictive prefetching."""
        while True:
            try:
                await asyncio.sleep(60)  # 1 minute
                
                # Process cache warming queue
                if self.cache_warming_queue:
                    key = self.cache_warming_queue.pop(0)
                    # In real implementation, would generate and cache predicted values
                    logger.debug(f"Would prefetch for key: {key}")
                
            except Exception as e:
                logger.error(f"Predictive prefetch error: {e}")
    
    async def _statistics_update_loop(self):
        """Background statistics updates."""
        while True:
            try:
                await asyncio.sleep(10)  # 10 seconds
                
                # Update real-time statistics
                total_ops = self.cache_stats.hits + self.cache_stats.misses
                if total_ops > 0:
                    self.cache_stats.hit_rate = self.cache_stats.hits / total_ops
                
                self.cache_stats.memory_utilization = self.cache_stats.size_bytes / self.max_size_bytes
                
            except Exception as e:
                logger.error(f"Statistics update error: {e}")
    
    def get_cache_stats(self) -> CacheStats:
        """Get current cache statistics."""
        return self.cache_stats
    
    def get_detailed_stats(self) -> Dict[str, Any]:
        """Get detailed cache statistics."""
        l1_entries = len(self.l1_cache)
        l2_entries = len(self.l2_cache)
        l3_entries = len(self.l3_cache)
        
        return {
            "basic_stats": asdict(self.cache_stats),
            "level_distribution": {
                "l1_entries": l1_entries,
                "l2_entries": l2_entries,
                "l3_entries": l3_entries,
                "total_entries": l1_entries + l2_entries + l3_entries
            },
            "cache_strategy": self.cache_strategy.value,
            "capacity_usage": {
                "size_bytes": self.cache_stats.size_bytes,
                "max_size_bytes": self.max_size_bytes,
                "utilization_percent": (self.cache_stats.size_bytes / self.max_size_bytes) * 100
            },
            "access_patterns": {
                "tracked_keys": len(self.access_patterns),
                "warming_queue_size": len(self.cache_warming_queue)
            }
        }
    
    async def cleanup(self):
        """Cleanup cache manager resources."""
        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()
        
        # Clear caches
        self.l1_cache.clear()
        self.l2_cache.clear()
        self.l3_cache.clear()
        
        logger.info("Cache manager cleanup completed")


# Factory function
def create_nas_cache_manager(
    max_size_mb: int = 1024,
    cache_strategy: CacheStrategy = CacheStrategy.INTELLIGENT
) -> IntelligentCacheManager:
    """Create NAS cache manager with specified configuration."""
    return IntelligentCacheManager(
        max_size_bytes=max_size_mb * 1024 * 1024,
        cache_strategy=cache_strategy
    )


# Demo usage
async def demo_nas_caching():
    """Demonstrate NAS caching capabilities."""
    print("💾 NAS Caching Demo")
    
    # Create cache manager
    cache_manager = create_nas_cache_manager(
        max_size_mb=100,
        cache_strategy=CacheStrategy.INTELLIGENT
    )
    
    # Test architecture prediction caching
    print("\n🔍 Testing prediction caching...")
    
    test_config = {
        "architecture_type": "transformer",
        "num_layers": 12,
        "hidden_size": 768,
        "num_heads": 12
    }
    
    prediction_result = {
        "accuracy": 0.92,
        "latency": 15.5,
        "efficiency": 0.85,
        "confidence": 0.9
    }
    
    # Cache prediction
    key = await cache_manager.cache_architecture_prediction(test_config, prediction_result)
    print(f"Cached prediction with key: {key[:16]}...")
    
    # Retrieve cached prediction
    cached_result = await cache_manager.get_cached_prediction(test_config)
    print(f"Retrieved cached result: accuracy={cached_result['accuracy']}")
    
    # Test cache warming
    print("\n🔥 Testing cache warming...")
    
    search_configs = [test_config]
    search_space = {
        "num_layers": [6, 12, 24, 48],
        "hidden_size": [384, 512, 768, 1024, 1536],
        "num_heads": [6, 8, 12, 16, 24]
    }
    
    await cache_manager.warm_cache_for_search(search_configs, search_space)
    
    # Show cache statistics
    stats = cache_manager.get_detailed_stats()
    print(f"\n📊 Cache Statistics:")
    print(f"  Hit rate: {stats['basic_stats']['hit_rate']:.2%}")
    print(f"  Total entries: {stats['level_distribution']['total_entries']}")
    print(f"  Memory utilization: {stats['capacity_usage']['utilization_percent']:.1f}%")
    
    # Cleanup
    await cache_manager.cleanup()


if __name__ == "__main__":
    asyncio.run(demo_nas_caching())