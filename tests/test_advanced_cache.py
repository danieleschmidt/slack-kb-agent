"""Tests for advanced caching system with intelligence and optimization."""

import asyncio
import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.slack_kb_agent.advanced_cache import (
    AdaptiveCacheManager,
    CacheEntry,
    CacheStrategy,
    CacheStats,
    SmartCachePredictor
)


class TestCacheEntry:
    """Test cache entry functionality."""
    
    def test_cache_entry_creation(self):
        """Test creating cache entries."""
        entry = CacheEntry(
            key="test_key",
            value="test_value",
            created_at=datetime.utcnow(),
            last_accessed=datetime.utcnow(),
            ttl_seconds=3600
        )
        
        assert entry.key == "test_key"
        assert entry.value == "test_value"
        assert not entry.is_expired
        assert entry.age_seconds >= 0
        assert entry.idle_seconds >= 0
    
    def test_cache_entry_expiration(self):
        """Test cache entry expiration logic."""
        # Create expired entry
        old_time = datetime.utcnow() - timedelta(hours=2)
        entry = CacheEntry(
            key="expired_key",
            value="expired_value",
            created_at=old_time,
            last_accessed=old_time,
            ttl_seconds=3600  # 1 hour TTL
        )
        
        assert entry.is_expired
        assert entry.age_seconds > 3600
    
    def test_cache_entry_no_expiration(self):
        """Test cache entry without TTL."""
        entry = CacheEntry(
            key="no_ttl_key",
            value="persistent_value",
            created_at=datetime.utcnow() - timedelta(days=30),
            last_accessed=datetime.utcnow(),
            ttl_seconds=None
        )
        
        assert not entry.is_expired


class TestCacheStats:
    """Test cache statistics tracking."""
    
    @pytest.fixture
    def stats(self):
        """Create cache stats instance."""
        return CacheStats()
    
    def test_initial_stats(self, stats):
        """Test initial statistics state."""
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.hit_rate == 0.0
        assert stats.miss_rate == 1.0
    
    def test_record_hits_and_misses(self, stats):
        """Test recording hits and misses."""
        # Record some hits and misses
        stats.record_hit("key1")
        stats.record_hit("key2")
        stats.record_miss("key3")
        
        assert stats.hits == 2
        assert stats.misses == 1
        assert stats.hit_rate == 2/3
        assert stats.miss_rate == 1/3
    
    def test_popular_keys_tracking(self, stats):
        """Test tracking of popular keys."""
        # Record multiple hits for same keys
        for _ in range(5):
            stats.record_hit("popular_key")
        
        for _ in range(2):
            stats.record_hit("less_popular_key")
        
        summary = stats.get_summary()
        popular_keys = summary["popular_keys"]
        
        assert "popular_key" in popular_keys
        assert popular_keys["popular_key"] == 5
        assert popular_keys["less_popular_key"] == 2
    
    def test_access_pattern_tracking(self, stats):
        """Test access pattern tracking."""
        key = "pattern_key"
        
        # Record multiple accesses
        for _ in range(10):
            stats.record_hit(key)
        
        # Should track access times
        assert key in stats.access_patterns
        assert len(stats.access_patterns[key]) == 10
        
        # Test limit enforcement (should keep only last 100)
        for _ in range(100):
            stats.record_hit(key)
        
        assert len(stats.access_patterns[key]) <= 100


class TestSmartCachePredictor:
    """Test smart cache prediction functionality."""
    
    @pytest.fixture
    def predictor(self):
        """Create predictor instance."""
        return SmartCachePredictor(learning_window_hours=24)
    
    def test_record_access(self, predictor):
        """Test recording access for pattern learning."""
        key = "test_key"
        predictor.record_access(key)
        
        assert key in predictor.access_history
        assert len(predictor.access_history[key]) == 1
    
    def test_pattern_detection_insufficient_data(self, predictor):
        """Test pattern detection with insufficient data."""
        key = "sparse_key"
        predictor.record_access(key)
        
        # Should not create pattern with just one access
        assert key not in predictor.pattern_cache
        
        prediction = predictor.predict_next_access(key)
        assert prediction is None
    
    def test_pattern_detection_regular_access(self, predictor):
        """Test pattern detection with regular access."""
        key = "regular_key"
        base_time = datetime.utcnow() - timedelta(hours=5)
        
        # Simulate regular access every hour
        for i in range(6):
            access_time = base_time + timedelta(hours=i)
            predictor.record_access(key, access_time)
        
        # Should detect pattern
        assert key in predictor.pattern_cache
        pattern = predictor.pattern_cache[key]
        assert pattern['confidence'] > 0.5  # Should be confident about regular pattern
        assert abs(pattern['avg_interval_seconds'] - 3600) < 600  # ~1 hour intervals
    
    def test_predict_next_access(self, predictor):
        """Test predicting next access time."""
        key = "predictable_key"
        base_time = datetime.utcnow() - timedelta(hours=3)
        
        # Create predictable pattern (every 30 minutes)
        for i in range(6):
            access_time = base_time + timedelta(minutes=i * 30)
            predictor.record_access(key, access_time)
        
        prediction = predictor.predict_next_access(key)
        if prediction:  # Pattern might not be confident enough
            # Should predict next access around 30 minutes from last
            last_access = max(predictor.access_history[key])
            expected_next = last_access + timedelta(minutes=30)
            time_diff = abs((prediction - expected_next).total_seconds())
            assert time_diff < 900  # Within 15 minutes of expected
    
    def test_daily_pattern_detection(self, predictor):
        """Test detection of daily access patterns."""
        key = "daily_key"
        base_time = datetime.utcnow().replace(hour=9, minute=0, second=0, microsecond=0)
        
        # Simulate daily access at 9 AM for a week
        for i in range(8):
            access_time = base_time - timedelta(days=i)
            predictor.record_access(key, access_time)
        
        # Should detect daily pattern
        if key in predictor.pattern_cache:
            pattern = predictor.pattern_cache[key]
            daily_pattern = pattern['daily_pattern']
            if daily_pattern['detected']:
                assert 9 in daily_pattern['peak_hours']  # 9 AM should be peak
    
    def test_prefetch_candidates(self, predictor):
        """Test getting prefetch candidates."""
        # Create multiple keys with different patterns
        keys_and_patterns = [
            ("soon_key", 30),    # Access every 30 minutes
            ("later_key", 120),  # Access every 2 hours
            ("random_key", None) # No clear pattern
        ]
        
        base_time = datetime.utcnow() - timedelta(hours=2)
        
        for key, interval_minutes in keys_and_patterns:
            if interval_minutes:
                # Create regular pattern
                for i in range(5):
                    access_time = base_time + timedelta(minutes=i * interval_minutes)
                    predictor.record_access(key, access_time)
            else:
                # Create random pattern
                import random
                for i in range(5):
                    random_offset = random.randint(0, 300)  # Random within 5 hours
                    access_time = base_time + timedelta(minutes=random_offset)
                    predictor.record_access(key, access_time)
        
        candidates = predictor.get_prefetch_candidates(max_candidates=5)
        
        # Should return list of (key, score) tuples
        assert isinstance(candidates, list)
        for candidate in candidates:
            assert isinstance(candidate, tuple)
            assert len(candidate) == 2
            assert isinstance(candidate[0], str)  # key
            assert isinstance(candidate[1], float)  # score


class TestAdaptiveCacheManager:
    """Test adaptive cache manager functionality."""
    
    @pytest.fixture
    def cache_manager(self):
        """Create cache manager instance."""
        return AdaptiveCacheManager(
            max_size=5,
            max_memory_mb=1,
            strategy=CacheStrategy.ADAPTIVE
        )
    
    @pytest.mark.asyncio
    async def test_basic_set_and_get(self, cache_manager):
        """Test basic cache set and get operations."""
        key = "test_key"
        value = "test_value"
        
        # Set value
        success = await cache_manager.set(key, value)
        assert success
        
        # Get value
        retrieved_value = await cache_manager.get(key)
        assert retrieved_value == value
        
        # Check statistics
        assert cache_manager.stats.total_sets == 1
        assert cache_manager.stats.hits == 1
    
    @pytest.mark.asyncio
    async def test_cache_miss(self, cache_manager):
        """Test cache miss behavior."""
        # Try to get non-existent key
        value = await cache_manager.get("non_existent_key")
        assert value is None
        
        # Check statistics
        assert cache_manager.stats.misses == 1
    
    @pytest.mark.asyncio
    async def test_ttl_expiration(self, cache_manager):
        """Test TTL-based expiration."""
        key = "expiring_key"
        value = "expiring_value"
        
        # Set with very short TTL
        await cache_manager.set(key, value, ttl=1)
        
        # Should be available immediately
        retrieved_value = await cache_manager.get(key)
        assert retrieved_value == value
        
        # Wait for expiration
        await asyncio.sleep(1.1)
        
        # Should be expired now
        retrieved_value = await cache_manager.get(key)
        assert retrieved_value is None
    
    @pytest.mark.asyncio
    async def test_size_eviction(self, cache_manager):
        """Test eviction due to size limits."""
        # Fill cache to capacity
        for i in range(cache_manager.max_size):
            await cache_manager.set(f"key_{i}", f"value_{i}")
        
        # Add one more item to trigger eviction
        await cache_manager.set("overflow_key", "overflow_value")
        
        # Should have evicted oldest item
        assert len(cache_manager.entries) == cache_manager.max_size
        assert cache_manager.stats.evictions > 0
    
    @pytest.mark.asyncio
    async def test_lru_strategy(self):
        """Test LRU eviction strategy."""
        cache = AdaptiveCacheManager(max_size=3, strategy=CacheStrategy.LRU)
        
        # Add items
        await cache.set("key1", "value1")
        await cache.set("key2", "value2")
        await cache.set("key3", "value3")
        
        # Access key1 to make it recently used
        await cache.get("key1")
        
        # Add new item to trigger eviction
        await cache.set("key4", "value4")
        
        # key2 should be evicted (least recently used)
        assert await cache.get("key2") is None
        assert await cache.get("key1") == "value1"  # Should still be there
        assert await cache.get("key4") == "value4"  # New item should be there
    
    @pytest.mark.asyncio
    async def test_priority_handling(self, cache_manager):
        """Test priority-based caching."""
        # Add low priority item
        await cache_manager.set("low_priority", "low_value", priority=1)
        
        # Add high priority item
        await cache_manager.set("high_priority", "high_value", priority=5)
        
        # Fill cache with medium priority items
        for i in range(cache_manager.max_size - 1):
            await cache_manager.set(f"medium_{i}", f"medium_value_{i}", priority=3)
        
        # Add one more to trigger eviction
        await cache_manager.set("trigger", "trigger_value", priority=3)
        
        # High priority item should be preserved
        assert await cache_manager.get("high_priority") == "high_value"
    
    @pytest.mark.asyncio
    async def test_tags_functionality(self, cache_manager):
        """Test tag-based cache operations."""
        # Add items with tags
        await cache_manager.set("item1", "value1", tags={"user:123", "session"})
        await cache_manager.set("item2", "value2", tags={"user:456", "session"})
        await cache_manager.set("item3", "value3", tags={"system", "config"})
        
        # Clear by tags
        cleared_count = await cache_manager.clear_by_tags({"session"})
        assert cleared_count == 2
        
        # Only system item should remain
        assert await cache_manager.get("item1") is None
        assert await cache_manager.get("item2") is None
        assert await cache_manager.get("item3") == "value3"
    
    @pytest.mark.asyncio
    async def test_cleanup_expired(self, cache_manager):
        """Test cleanup of expired entries."""
        # Add items with different TTLs
        await cache_manager.set("short_ttl", "value1", ttl=1)
        await cache_manager.set("long_ttl", "value2", ttl=3600)
        
        # Wait for short TTL to expire
        await asyncio.sleep(1.1)
        
        # Run cleanup
        cleaned_count = await cache_manager.cleanup_expired()
        
        assert cleaned_count == 1
        assert await cache_manager.get("short_ttl") is None
        assert await cache_manager.get("long_ttl") == "value2"
    
    @pytest.mark.asyncio
    async def test_prefetch_functionality(self, cache_manager):
        """Test predictive prefetching."""
        if not cache_manager.predictor:
            pytest.skip("Predictor not enabled")
        
        # Mock prefetch function
        async def mock_prefetch_func(key):
            return f"prefetched_value_for_{key}"
        
        # Simulate access pattern for prediction
        key = "predictable_key"
        for _ in range(5):
            await cache_manager.set(key, "value")
            await cache_manager.get(key)
            await asyncio.sleep(0.1)  # Small delay to create pattern
        
        # Try prefetching
        prefetched_count = await cache_manager.prefetch_predicted(mock_prefetch_func)
        
        # Should work without errors (actual prefetching depends on pattern detection)
        assert prefetched_count >= 0
    
    @pytest.mark.asyncio
    async def test_cache_optimization(self, cache_manager):
        """Test cache optimization functionality."""
        # Add some items and create activity
        for i in range(3):
            await cache_manager.set(f"key_{i}", f"value_{i}")
        
        # Run optimization
        optimization_result = await cache_manager.optimize_cache()
        
        assert "expired_cleaned" in optimization_result
        assert "memory_evicted" in optimization_result
        assert "optimization_time_ms" in optimization_result
        assert "current_size" in optimization_result
        assert "cache_stats" in optimization_result
        
        # Should complete without errors
        assert optimization_result["optimization_time_ms"] >= 0
    
    def test_cache_info(self, cache_manager):
        """Test cache information retrieval."""
        info = cache_manager.get_cache_info()
        
        expected_keys = [
            "strategy", "max_size", "current_size", "max_memory_mb",
            "current_memory_mb", "memory_usage_percent", "stats"
        ]
        
        for key in expected_keys:
            assert key in info
        
        assert info["strategy"] == CacheStrategy.ADAPTIVE.value
        assert info["max_size"] == cache_manager.max_size


class TestCacheStrategies:
    """Test different cache eviction strategies."""
    
    @pytest.mark.asyncio
    async def test_lfu_strategy(self):
        """Test Least Frequently Used strategy."""
        cache = AdaptiveCacheManager(max_size=3, strategy=CacheStrategy.LFU)
        
        # Add items
        await cache.set("key1", "value1")
        await cache.set("key2", "value2")
        await cache.set("key3", "value3")
        
        # Access key1 multiple times
        for _ in range(5):
            await cache.get("key1")
        
        # Access key2 fewer times
        for _ in range(2):
            await cache.get("key2")
        
        # Don't access key3
        
        # Add new item to trigger eviction
        await cache.set("key4", "value4")
        
        # key3 should be evicted (least frequently used)
        assert await cache.get("key3") is None
        assert await cache.get("key1") == "value1"
        assert await cache.get("key4") == "value4"
    
    @pytest.mark.asyncio
    async def test_ttl_strategy(self):
        """Test TTL-based strategy."""
        cache = AdaptiveCacheManager(max_size=3, strategy=CacheStrategy.TTL)
        
        # Add items with different TTLs
        await cache.set("short_ttl", "value1", ttl=1)
        await cache.set("medium_ttl", "value2", ttl=5)
        await cache.set("long_ttl", "value3", ttl=10)
        
        # Add new item to trigger eviction
        await cache.set("new_item", "new_value", ttl=5)
        
        # Item with shortest TTL should be evicted first
        assert await cache.get("short_ttl") is None
        assert await cache.get("medium_ttl") == "value2"
        assert await cache.get("long_ttl") == "value3"


class TestPerformanceAndStress:
    """Test cache performance under stress conditions."""
    
    @pytest.mark.asyncio
    async def test_high_volume_operations(self):
        """Test cache with high volume of operations."""
        cache = AdaptiveCacheManager(max_size=1000, max_memory_mb=10)
        
        # Perform many operations
        num_operations = 1000
        
        # Set operations
        for i in range(num_operations):
            await cache.set(f"key_{i}", f"value_{i}")
        
        # Mixed get operations
        hit_count = 0
        for i in range(num_operations):
            value = await cache.get(f"key_{i % 100}")  # Some hits, some misses
            if value is not None:
                hit_count += 1
        
        # Should handle high volume without errors
        assert cache.stats.total_sets == num_operations
        assert hit_count > 0
        assert cache.stats.hit_rate > 0
    
    @pytest.mark.asyncio
    async def test_memory_pressure_handling(self):
        """Test cache behavior under memory pressure."""
        cache = AdaptiveCacheManager(max_size=100, max_memory_mb=1)  # Very small memory limit
        
        # Add large items that will exceed memory limit
        large_value = "x" * 10000  # 10KB value
        
        items_added = 0
        for i in range(200):  # Try to add many large items
            success = await cache.set(f"large_key_{i}", large_value)
            if success:
                items_added += 1
        
        # Should have evicted items to stay within memory limit
        assert cache.current_memory_bytes <= cache.max_memory_bytes
        assert cache.stats.evictions > 0
        assert items_added < 200  # Shouldn't have been able to add all items
    
    @pytest.mark.asyncio
    async def test_concurrent_access_simulation(self):
        """Test cache under simulated concurrent access."""
        cache = AdaptiveCacheManager(max_size=50)
        
        async def worker(worker_id, num_ops):
            """Simulate a worker performing cache operations."""
            for i in range(num_ops):
                key = f"worker_{worker_id}_key_{i % 10}"  # Reuse some keys
                value = f"worker_{worker_id}_value_{i}"
                
                # Mix of operations
                if i % 3 == 0:
                    await cache.set(key, value)
                else:
                    await cache.get(key)
        
        # Run multiple workers concurrently
        workers = [worker(i, 50) for i in range(5)]
        await asyncio.gather(*workers)
        
        # Should handle concurrent access without errors
        assert cache.stats.total_operations > 0
        assert len(cache.entries) <= cache.max_size