"""Advanced caching strategies with intelligent cache management."""

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from enum import Enum
import hashlib

from .cache import CacheManager
from .exceptions import CacheError


logger = logging.getLogger(__name__)


class CacheStrategy(Enum):
    """Cache eviction strategies."""
    LRU = "lru"              # Least Recently Used
    LFU = "lfu"              # Least Frequently Used
    TTL = "ttl"              # Time To Live
    ADAPTIVE = "adaptive"     # Adaptive based on access patterns
    SMART = "smart"          # Smart prediction-based


@dataclass
class CacheEntry:
    """Enhanced cache entry with metadata."""
    
    key: str
    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    size_bytes: int = 0
    ttl_seconds: Optional[int] = None
    priority: int = 1  # 1=low, 5=high priority
    tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.ttl_seconds is None:
            return False
        return datetime.utcnow() > self.created_at + timedelta(seconds=self.ttl_seconds)
    
    @property
    def age_seconds(self) -> float:
        """Get age of entry in seconds."""
        return (datetime.utcnow() - self.created_at).total_seconds()
    
    @property
    def idle_seconds(self) -> float:
        """Get idle time since last access in seconds."""
        return (datetime.utcnow() - self.last_accessed).total_seconds()


class CacheStats:
    """Cache statistics for monitoring and optimization."""
    
    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.expired_removals = 0
        self.size_evictions = 0
        self.total_gets = 0
        self.total_sets = 0
        self.total_deletes = 0
        self.start_time = datetime.utcnow()
        
        # Access pattern tracking
        self.access_patterns: Dict[str, List[datetime]] = defaultdict(list)
        self.popular_keys: Dict[str, int] = defaultdict(int)
        
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    @property
    def miss_rate(self) -> float:
        """Calculate cache miss rate."""
        return 1.0 - self.hit_rate
    
    def record_hit(self, key: str) -> None:
        """Record a cache hit."""
        self.hits += 1
        self.total_gets += 1
        self.popular_keys[key] += 1
        self.access_patterns[key].append(datetime.utcnow())
        
        # Keep only recent access times (last 100 accesses)
        if len(self.access_patterns[key]) > 100:
            self.access_patterns[key] = self.access_patterns[key][-100:]
    
    def record_miss(self, key: str) -> None:
        """Record a cache miss."""
        self.misses += 1
        self.total_gets += 1
    
    def record_set(self) -> None:
        """Record a cache set operation."""
        self.total_sets += 1
    
    def record_eviction(self, reason: str = "size") -> None:
        """Record a cache eviction."""
        self.evictions += 1
        if reason == "size":
            self.size_evictions += 1
        elif reason == "expired":
            self.expired_removals += 1
    
    def get_summary(self) -> Dict[str, Any]:
        """Get cache statistics summary."""
        uptime = (datetime.utcnow() - self.start_time).total_seconds()
        
        return {
            'hit_rate': self.hit_rate,
            'miss_rate': self.miss_rate,
            'total_operations': self.total_gets + self.total_sets + self.total_deletes,
            'hits': self.hits,
            'misses': self.misses,
            'evictions': self.evictions,
            'expired_removals': self.expired_removals,
            'size_evictions': self.size_evictions,
            'uptime_seconds': uptime,
            'operations_per_second': (self.total_gets + self.total_sets) / uptime if uptime > 0 else 0,
            'popular_keys': dict(sorted(self.popular_keys.items(), key=lambda x: x[1], reverse=True)[:10])
        }


class SmartCachePredictor:
    """Predicts cache access patterns for proactive caching."""
    
    def __init__(self, learning_window_hours: int = 24):
        self.access_history: Dict[str, List[datetime]] = defaultdict(list)
        self.pattern_cache: Dict[str, Dict[str, Any]] = {}
        self.learning_window = timedelta(hours=learning_window_hours)
        
    def record_access(self, key: str, timestamp: datetime = None) -> None:
        """Record an access for pattern learning."""
        if timestamp is None:
            timestamp = datetime.utcnow()
            
        self.access_history[key].append(timestamp)
        
        # Clean old history
        cutoff = datetime.utcnow() - self.learning_window
        self.access_history[key] = [
            ts for ts in self.access_history[key] if ts > cutoff
        ]
        
        # Update pattern cache
        if len(self.access_history[key]) >= 5:  # Need minimum data for patterns
            self._update_pattern_cache(key)
    
    def predict_next_access(self, key: str) -> Optional[datetime]:
        """Predict when a key will next be accessed."""
        if key not in self.pattern_cache:
            return None
        
        pattern = self.pattern_cache[key]
        if pattern['confidence'] < 0.3:  # Low confidence
            return None
        
        # Simple prediction based on average interval
        last_access = max(self.access_history[key]) if self.access_history[key] else None
        if not last_access:
            return None
        
        avg_interval = pattern['avg_interval_seconds']
        return last_access + timedelta(seconds=avg_interval)
    
    def get_prefetch_candidates(self, max_candidates: int = 10) -> List[Tuple[str, float]]:
        """Get keys that should be prefetched with confidence scores."""
        candidates = []
        now = datetime.utcnow()
        
        for key, pattern in self.pattern_cache.items():
            if pattern['confidence'] < 0.5:  # Skip low confidence patterns
                continue
            
            predicted_access = self.predict_next_access(key)
            if not predicted_access:
                continue
            
            # Check if predicted access is soon (within next hour)
            time_until_access = (predicted_access - now).total_seconds()
            if 0 < time_until_access < 3600:  # Next hour
                confidence = pattern['confidence']
                urgency = max(0, 1 - (time_until_access / 3600))  # Higher urgency = sooner access
                score = confidence * urgency
                candidates.append((key, score))
        
        # Sort by score and return top candidates
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:max_candidates]
    
    def _update_pattern_cache(self, key: str) -> None:
        """Update pattern analysis for a key."""
        history = self.access_history[key]
        if len(history) < 2:
            return
        
        # Calculate intervals between accesses
        intervals = []
        for i in range(1, len(history)):
            interval = (history[i] - history[i-1]).total_seconds()
            intervals.append(interval)
        
        # Calculate pattern metrics
        avg_interval = sum(intervals) / len(intervals)
        interval_variance = sum((x - avg_interval) ** 2 for x in intervals) / len(intervals)
        interval_stddev = interval_variance ** 0.5
        
        # Calculate confidence based on consistency
        if avg_interval > 0:
            coefficient_of_variation = interval_stddev / avg_interval
            confidence = max(0, 1 - min(coefficient_of_variation, 1))  # More consistent = higher confidence
        else:
            confidence = 0
        
        # Detect time-based patterns (daily, weekly, etc.)
        daily_pattern = self._detect_daily_pattern(history)
        weekly_pattern = self._detect_weekly_pattern(history)
        
        self.pattern_cache[key] = {
            'avg_interval_seconds': avg_interval,
            'interval_stddev': interval_stddev,
            'confidence': confidence,
            'access_count': len(history),
            'daily_pattern': daily_pattern,
            'weekly_pattern': weekly_pattern,
            'last_updated': datetime.utcnow()
        }
    
    def _detect_daily_pattern(self, history: List[datetime]) -> Dict[str, Any]:
        """Detect daily access patterns."""
        if len(history) < 7:  # Need at least a week of data
            return {'detected': False}
        
        # Group by hour of day
        hour_counts = defaultdict(int)
        for ts in history:
            hour_counts[ts.hour] += 1
        
        if not hour_counts:
            return {'detected': False}
        
        # Find peak hours
        max_count = max(hour_counts.values())
        peak_hours = [hour for hour, count in hour_counts.items() if count >= max_count * 0.8]
        
        return {
            'detected': True,
            'peak_hours': sorted(peak_hours),
            'hour_distribution': dict(hour_counts)
        }
    
    def _detect_weekly_pattern(self, history: List[datetime]) -> Dict[str, Any]:
        """Detect weekly access patterns."""
        if len(history) < 14:  # Need at least two weeks of data
            return {'detected': False}
        
        # Group by day of week (0=Monday, 6=Sunday)
        day_counts = defaultdict(int)
        for ts in history:
            day_counts[ts.weekday()] += 1
        
        if not day_counts:
            return {'detected': False}
        
        # Find peak days
        max_count = max(day_counts.values())
        peak_days = [day for day, count in day_counts.items() if count >= max_count * 0.8]
        
        return {
            'detected': True,
            'peak_days': sorted(peak_days),
            'day_distribution': dict(day_counts)
        }


class AdaptiveCacheManager:
    """Advanced cache manager with multiple strategies and intelligence."""
    
    def __init__(
        self,
        max_size: int = 10000,
        max_memory_mb: int = 512,
        default_ttl: int = 3600,
        strategy: CacheStrategy = CacheStrategy.ADAPTIVE,
        enable_prediction: bool = True
    ):
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.default_ttl = default_ttl
        self.strategy = strategy
        
        # Storage
        self.entries: Dict[str, CacheEntry] = {}
        self.access_order = OrderedDict()  # For LRU
        self.frequency_counter: Dict[str, int] = defaultdict(int)  # For LFU
        
        # Statistics and monitoring
        self.stats = CacheStats()
        self.current_memory_bytes = 0
        
        # Intelligence components
        self.predictor = SmartCachePredictor() if enable_prediction else None
        self._last_cleanup = datetime.utcnow()
        self._cleanup_interval = timedelta(minutes=5)
        
        # Strategy-specific data
        self._adaptive_weights = {
            'frequency': 0.3,
            'recency': 0.3,
            'size': 0.2,
            'priority': 0.2
        }
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache with intelligent tracking."""
        entry = self.entries.get(key)
        
        if entry is None:
            self.stats.record_miss(key)
            return None
        
        # Check expiration
        if entry.is_expired:
            await self._remove_entry(key, "expired")
            self.stats.record_miss(key)
            return None
        
        # Update access metadata
        entry.last_accessed = datetime.utcnow()
        entry.access_count += 1
        self.frequency_counter[key] += 1
        
        # Update access order for LRU
        if key in self.access_order:
            self.access_order.move_to_end(key)
        else:
            self.access_order[key] = True
        
        # Record for prediction
        if self.predictor:
            self.predictor.record_access(key)
        
        self.stats.record_hit(key)
        return entry.value
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        priority: int = 1,
        tags: Optional[Set[str]] = None
    ) -> bool:
        """Set value in cache with intelligent management."""
        try:
            # Calculate size
            size_bytes = self._calculate_size(value)
            
            # Check if we need to make space
            await self._ensure_space(size_bytes)
            
            # Create entry
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=datetime.utcnow(),
                last_accessed=datetime.utcnow(),
                size_bytes=size_bytes,
                ttl_seconds=ttl or self.default_ttl,
                priority=priority,
                tags=tags or set()
            )
            
            # Remove old entry if exists
            if key in self.entries:
                await self._remove_entry(key)
            
            # Add new entry
            self.entries[key] = entry
            self.access_order[key] = True
            self.current_memory_bytes += size_bytes
            
            self.stats.record_set()
            return True
            
        except Exception as e:
            logger.error(f"Failed to set cache entry {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete entry from cache."""
        if key in self.entries:
            await self._remove_entry(key)
            return True
        return False
    
    async def clear_by_tags(self, tags: Set[str]) -> int:
        """Clear all entries with any of the specified tags."""
        keys_to_remove = []
        
        for key, entry in self.entries.items():
            if entry.tags.intersection(tags):
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            await self._remove_entry(key)
        
        return len(keys_to_remove)
    
    async def cleanup_expired(self) -> int:
        """Clean up expired entries."""
        expired_keys = []
        
        for key, entry in self.entries.items():
            if entry.is_expired:
                expired_keys.append(key)
        
        for key in expired_keys:
            await self._remove_entry(key, "expired")
        
        return len(expired_keys)
    
    async def prefetch_predicted(self, prefetch_func) -> int:
        """Prefetch entries based on access predictions."""
        if not self.predictor:
            return 0
        
        candidates = self.predictor.get_prefetch_candidates()
        prefetched = 0
        
        for key, confidence in candidates:
            if key not in self.entries:  # Only prefetch if not already cached
                try:
                    value = await prefetch_func(key)
                    if value is not None:
                        await self.set(key, value, priority=2)  # Medium priority for prefetched
                        prefetched += 1
                except Exception as e:
                    logger.warning(f"Failed to prefetch {key}: {e}")
        
        return prefetched
    
    async def optimize_cache(self) -> Dict[str, Any]:
        """Perform cache optimization and return results."""
        start_time = time.time()
        
        # Cleanup expired entries
        expired_cleaned = await self.cleanup_expired()
        
        # Adjust adaptive weights based on access patterns
        if self.strategy == CacheStrategy.ADAPTIVE:
            self._adjust_adaptive_weights()
        
        # Evict low-value entries if over memory limit
        memory_evicted = 0
        if self.current_memory_bytes > self.max_memory_bytes:
            memory_evicted = await self._evict_by_memory_pressure()
        
        optimization_time = time.time() - start_time
        
        return {
            'expired_cleaned': expired_cleaned,
            'memory_evicted': memory_evicted,
            'optimization_time_ms': optimization_time * 1000,
            'current_size': len(self.entries),
            'current_memory_mb': self.current_memory_bytes / (1024 * 1024),
            'cache_stats': self.stats.get_summary()
        }
    
    async def _ensure_space(self, needed_bytes: int) -> None:
        """Ensure enough space for new entry."""
        # Check size limit
        while len(self.entries) >= self.max_size:
            await self._evict_one_entry()
        
        # Check memory limit
        while (self.current_memory_bytes + needed_bytes) > self.max_memory_bytes:
            await self._evict_one_entry()
    
    async def _evict_one_entry(self) -> None:
        """Evict one entry based on current strategy."""
        if not self.entries:
            return
        
        if self.strategy == CacheStrategy.LRU:
            key = next(iter(self.access_order))
        elif self.strategy == CacheStrategy.LFU:
            key = min(self.frequency_counter.items(), key=lambda x: x[1])[0]
        elif self.strategy == CacheStrategy.TTL:
            # Evict entry that will expire soonest
            key = min(
                self.entries.items(),
                key=lambda x: x[1].created_at + timedelta(seconds=x[1].ttl_seconds or 0)
            )[0]
        elif self.strategy == CacheStrategy.ADAPTIVE:
            key = self._adaptive_eviction()
        else:  # SMART
            key = self._smart_eviction()
        
        await self._remove_entry(key, "size")
    
    def _adaptive_eviction(self) -> str:
        """Adaptive eviction based on multiple factors."""
        if not self.entries:
            return ""
        
        scores = {}
        for key, entry in self.entries.items():
            # Calculate composite score (lower = more likely to evict)
            frequency_score = self.frequency_counter.get(key, 0)
            recency_score = 1.0 / (entry.idle_seconds + 1)  # More recent = higher score
            size_score = 1.0 / (entry.size_bytes + 1)  # Smaller = higher score
            priority_score = entry.priority
            
            composite_score = (
                frequency_score * self._adaptive_weights['frequency'] +
                recency_score * self._adaptive_weights['recency'] +
                size_score * self._adaptive_weights['size'] +
                priority_score * self._adaptive_weights['priority']
            )
            
            scores[key] = composite_score
        
        # Return key with lowest score
        return min(scores.items(), key=lambda x: x[1])[0]
    
    def _smart_eviction(self) -> str:
        """Smart eviction considering predictions."""
        if not self.entries or not self.predictor:
            return self._adaptive_eviction()
        
        scores = {}
        for key, entry in self.entries.items():
            base_score = self.frequency_counter.get(key, 0) * (1.0 / (entry.idle_seconds + 1))
            
            # Check if key is predicted to be accessed soon
            predicted_access = self.predictor.predict_next_access(key)
            if predicted_access:
                time_until_access = (predicted_access - datetime.utcnow()).total_seconds()
                if time_until_access > 0:
                    # Boost score if predicted to be accessed soon
                    prediction_boost = max(0, 1 - (time_until_access / 3600))  # Within next hour
                    base_score *= (1 + prediction_boost)
            
            scores[key] = base_score
        
        return min(scores.items(), key=lambda x: x[1])[0]
    
    async def _evict_by_memory_pressure(self) -> int:
        """Evict entries to relieve memory pressure."""
        evicted = 0
        target_memory = self.max_memory_bytes * 0.8  # Target 80% of max
        
        while self.current_memory_bytes > target_memory and self.entries:
            await self._evict_one_entry()
            evicted += 1
        
        return evicted
    
    def _adjust_adaptive_weights(self) -> None:
        """Adjust adaptive weights based on access patterns."""
        # Simple adjustment based on hit rate
        hit_rate = self.stats.hit_rate
        
        if hit_rate < 0.5:  # Low hit rate - favor frequency and recency more
            self._adaptive_weights['frequency'] = 0.4
            self._adaptive_weights['recency'] = 0.4
            self._adaptive_weights['size'] = 0.1
            self._adaptive_weights['priority'] = 0.1
        elif hit_rate > 0.8:  # High hit rate - consider size more
            self._adaptive_weights['frequency'] = 0.2
            self._adaptive_weights['recency'] = 0.2
            self._adaptive_weights['size'] = 0.4
            self._adaptive_weights['priority'] = 0.2
    
    async def _remove_entry(self, key: str, reason: str = "manual") -> None:
        """Remove entry and update tracking."""
        if key not in self.entries:
            return
        
        entry = self.entries[key]
        self.current_memory_bytes -= entry.size_bytes
        
        del self.entries[key]
        self.access_order.pop(key, None)
        self.frequency_counter.pop(key, None)
        
        self.stats.record_eviction(reason)
    
    def _calculate_size(self, value: Any) -> int:
        """Estimate size of value in bytes."""
        try:
            if isinstance(value, str):
                return len(value.encode('utf-8'))
            elif isinstance(value, (dict, list)):
                return len(json.dumps(value, ensure_ascii=False).encode('utf-8'))
            elif isinstance(value, (int, float)):
                return 8  # Approximate
            elif isinstance(value, bool):
                return 1
            else:
                # For other types, use string representation
                return len(str(value).encode('utf-8'))
        except Exception:
            return 1024  # Default estimate
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get comprehensive cache information."""
        return {
            'strategy': self.strategy.value,
            'max_size': self.max_size,
            'current_size': len(self.entries),
            'max_memory_mb': self.max_memory_bytes / (1024 * 1024),
            'current_memory_mb': self.current_memory_bytes / (1024 * 1024),
            'memory_usage_percent': (self.current_memory_bytes / self.max_memory_bytes) * 100,
            'stats': self.stats.get_summary(),
            'adaptive_weights': self._adaptive_weights if self.strategy == CacheStrategy.ADAPTIVE else None,
            'prediction_enabled': self.predictor is not None
        }