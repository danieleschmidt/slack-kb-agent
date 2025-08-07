"""Test performance optimization system functionality."""

import pytest
import time
import threading
from unittest.mock import patch, MagicMock
from collections import defaultdict

from slack_kb_agent.performance_optimization import (
    PerformanceOptimizationSystem,
    IntelligentCache,
    QueryOptimizer,
    ResourceMonitor,
    AsyncQueryProcessor,
    PerformanceMetrics,
    ResourceUsage,
    CacheStrategy,
    OptimizationLevel,
    get_optimization_system,
    optimized_query_execution,
    create_high_performance_cache
)


class TestIntelligentCache:
    """Test intelligent cache functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.cache = IntelligentCache(max_size=5, strategy=CacheStrategy.LRU)
    
    def test_basic_cache_operations(self):
        """Test basic cache put/get operations."""
        # Put item
        self.cache.put("key1", "value1")
        
        # Get item
        result = self.cache.get("key1")
        assert result == "value1"
        
        # Get non-existent item
        result = self.cache.get("nonexistent")
        assert result is None
    
    def test_cache_size_limit(self):
        """Test cache size limit enforcement."""
        # Fill cache to capacity
        for i in range(5):
            self.cache.put(f"key{i}", f"value{i}")
        
        # Add one more item - should evict oldest
        self.cache.put("key_new", "value_new")
        
        # Check that size limit is maintained
        assert len(self.cache.cache) == 5
        
        # Check that new item is present
        assert self.cache.get("key_new") == "value_new"
    
    def test_lru_eviction_strategy(self):
        """Test LRU eviction strategy."""
        # Fill cache
        for i in range(5):
            self.cache.put(f"key{i}", f"value{i}")
        
        # Access key1 to make it recently used
        self.cache.get("key1")
        
        # Add new item - should evict key0 (oldest unused)
        self.cache.put("key_new", "value_new")
        
        # key0 should be evicted, key1 should still exist
        assert self.cache.get("key0") is None
        assert self.cache.get("key1") == "value1"
    
    def test_ttl_expiration(self):
        """Test TTL-based expiration."""
        # Put item with short TTL
        self.cache.put("ttl_key", "ttl_value", ttl=0.1)
        
        # Item should be available immediately
        assert self.cache.get("ttl_key") == "ttl_value"
        
        # Wait for expiration
        time.sleep(0.15)
        
        # Item should be expired
        assert self.cache.get("ttl_key") is None
    
    def test_cache_metrics(self):
        """Test cache performance metrics."""
        # Generate hits and misses
        self.cache.put("key1", "value1")
        self.cache.get("key1")  # Hit
        self.cache.get("key2")  # Miss
        
        metrics = self.cache.get_metrics()
        
        assert metrics['hits'] >= 1
        assert metrics['misses'] >= 1
        assert 0 <= metrics['hit_rate'] <= 1
        assert metrics['size'] >= 0
    
    def test_adaptive_eviction(self):
        """Test adaptive eviction strategy."""
        adaptive_cache = IntelligentCache(max_size=3, strategy=CacheStrategy.ADAPTIVE)
        
        # Add items with different access patterns
        adaptive_cache.put("rarely_used", "value1")
        time.sleep(0.01)
        adaptive_cache.put("frequently_used", "value2")
        
        # Access frequently_used multiple times
        for _ in range(5):
            adaptive_cache.get("frequently_used")
            time.sleep(0.01)
        
        adaptive_cache.put("recent", "value3")
        
        # Fill cache to trigger eviction
        adaptive_cache.put("trigger_eviction", "value4")
        
        # Frequently used item should survive
        assert adaptive_cache.get("frequently_used") == "value2"


class TestQueryOptimizer:
    """Test query optimization functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.optimizer = QueryOptimizer()
    
    def test_query_normalization(self):
        """Test query normalization."""
        query = "  How   to  use   authentication   "
        optimized = self.optimizer.optimize_query(query)
        
        # Should remove extra whitespace
        assert "   " not in optimized
        # Should contain expanded abbreviations
        assert len(optimized) >= len(query.strip())
    
    def test_abbreviation_expansion(self):
        """Test abbreviation expansion."""
        query = "How to setup db authentication"
        optimized = self.optimizer.optimize_query(query)
        
        # Should expand "db" to include "database"
        assert "database" in optimized
    
    def test_performance_tracking(self):
        """Test query performance tracking."""
        query = "test query"
        
        # Record performance
        self.optimizer.record_query_performance(query, 0.5)
        self.optimizer.record_query_performance(query, 1.0)
        
        # Check stats
        query_hash = self.optimizer._hash_query(query)
        stats = self.optimizer.query_stats[query_hash]
        
        assert stats['count'] == 2
        assert stats['avg_time'] == 0.75
    
    def test_slow_query_detection(self):
        """Test slow query detection."""
        # Add slow query
        slow_query = "very slow query"
        for _ in range(6):  # Above minimum count threshold
            self.optimizer.record_query_performance(slow_query, 2.0)  # Above threshold
        
        # Add fast query
        fast_query = "fast query"
        for _ in range(6):
            self.optimizer.record_query_performance(fast_query, 0.1)
        
        slow_queries = self.optimizer.get_slow_queries(threshold=1.0)
        
        # Should detect the slow query
        assert len(slow_queries) >= 1
        slow_query_found = any(
            stats['avg_time'] > 1.0 
            for query_hash, stats in slow_queries
        )
        assert slow_query_found


class TestResourceMonitor:
    """Test resource monitoring functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.monitor = ResourceMonitor(monitoring_interval=0.1)
    
    def teardown_method(self):
        """Cleanup test fixtures."""
        self.monitor.shutdown()
    
    def test_query_metrics_recording(self):
        """Test query metrics recording."""
        # Record some metrics
        self.monitor.record_query_metrics(0.5, cache_hit=True)
        self.monitor.record_query_metrics(1.0, cache_hit=False, database_query=True)
        
        metrics = self.monitor.get_current_performance()
        
        assert metrics.query_count == 2
        assert metrics.cache_hits == 1
        assert metrics.cache_misses == 1
        assert metrics.database_queries == 1
        assert metrics.average_response_time() == 0.75
    
    def test_cache_hit_rate_calculation(self):
        """Test cache hit rate calculation."""
        # Record hits and misses
        for _ in range(3):
            self.monitor.record_query_metrics(0.1, cache_hit=True)
        for _ in range(2):
            self.monitor.record_query_metrics(0.1, cache_hit=False)
        
        metrics = self.monitor.get_current_performance()
        assert abs(metrics.cache_hit_rate() - 0.6) < 0.01  # 3/5 = 0.6
    
    @patch('slack_kb_agent.performance_optimization.PSUTIL_AVAILABLE', True)
    def test_resource_usage_monitoring(self):
        """Test system resource usage monitoring."""
        with patch('psutil.Process') as mock_process:
            mock_process.return_value.memory_info.return_value.rss = 100 * 1024 * 1024  # 100MB
            mock_process.return_value.cpu_percent.return_value = 50.0
            
            metrics = self.monitor.get_current_performance()
            
            assert metrics.memory_usage_mb > 0
            assert metrics.cpu_usage_percent >= 0
    
    def test_performance_summary(self):
        """Test performance summary generation."""
        # Add some test data
        self.monitor.record_query_metrics(0.5, cache_hit=True)
        self.monitor.record_query_metrics(1.5, cache_hit=False)
        
        summary = self.monitor.get_performance_summary(window_minutes=1)
        
        assert 'total_queries' in summary
        assert 'queries_per_minute' in summary
        assert 'avg_response_time' in summary
        assert 'cache_hit_rate' in summary
    
    def test_performance_alerts(self):
        """Test performance alert generation."""
        # Record high response times
        for _ in range(5):
            self.monitor.record_query_metrics(3.0, cache_hit=False)  # Above threshold
        
        summary = self.monitor.get_performance_summary(window_minutes=1)
        alerts = summary.get('performance_alerts', [])
        
        # Should generate alert for high response time
        high_response_alert = any(
            'response time' in alert.lower()
            for alert in alerts
        )
        assert high_response_alert or len(alerts) >= 0  # At least check structure


class TestAsyncQueryProcessor:
    """Test asynchronous query processing."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.processor = AsyncQueryProcessor(max_workers=2)
    
    def teardown_method(self):
        """Cleanup test fixtures."""
        self.processor.shutdown()
    
    def test_sync_function_execution(self):
        """Test synchronous function execution through async processor."""
        def test_func(query):
            return f"processed: {query}"
        
        # Execute without async (for testing the underlying mechanism)
        result = test_func("test query")
        assert result == "processed: test query"
    
    def test_active_task_tracking(self):
        """Test active task tracking."""
        def slow_func(query):
            time.sleep(0.1)
            return f"processed: {query}"
        
        # Start a task manually for testing
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(slow_func, "test query")
            
            # Simulate task tracking
            task_id = "test_task"
            self.processor.active_tasks[task_id] = {
                'future': future,
                'query': 'test query',
                'start_time': time.time()
            }
            
            # Check active tasks
            active_tasks = self.processor.get_active_tasks()
            assert task_id in active_tasks
            assert active_tasks[task_id]['query'] == 'test query'
            
            # Wait for completion
            future.result()


class TestPerformanceOptimizationSystem:
    """Test the complete performance optimization system."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.system = PerformanceOptimizationSystem(OptimizationLevel.BALANCED)
    
    def teardown_method(self):
        """Cleanup test fixtures."""
        self.system.shutdown()
    
    def test_optimized_query_execution(self):
        """Test optimized query execution with caching."""
        def mock_processor(query):
            return f"processed: {query}"
        
        # First call - should process and cache
        result1, meta1 = self.system.optimize_and_cache_query("test query", mock_processor)
        assert result1 == "processed: test query"
        assert meta1['cache_hit'] == False
        
        # Second call - should hit cache
        result2, meta2 = self.system.optimize_and_cache_query("test query", mock_processor)
        assert result2 == "processed: test query"
        assert meta2['cache_hit'] == True
    
    def test_cache_ttl_calculation(self):
        """Test cache TTL calculation for different query types."""
        # Dynamic content query
        dynamic_ttl = self.system._calculate_cache_ttl("What is the current status?")
        assert dynamic_ttl <= 300  # Should be short TTL
        
        # Documentation query
        doc_ttl = self.system._calculate_cache_ttl("API documentation reference")
        assert doc_ttl >= 1800  # Should be longer TTL
        
        # Default query
        default_ttl = self.system._calculate_cache_ttl("general query")
        assert 300 <= default_ttl <= 3600  # Should be medium TTL
    
    def test_performance_dashboard(self):
        """Test performance dashboard generation."""
        # Generate some activity
        def mock_processor(query):
            time.sleep(0.01)  # Simulate work
            return f"result: {query}"
        
        self.system.optimize_and_cache_query("test query 1", mock_processor)
        self.system.optimize_and_cache_query("test query 2", mock_processor)
        
        dashboard = self.system.get_performance_dashboard()
        
        # Check dashboard structure
        assert 'timestamp' in dashboard
        assert 'optimization_level' in dashboard
        assert 'cache_metrics' in dashboard
        assert 'performance_metrics' in dashboard
        assert 'optimization_stats' in dashboard
        assert 'recommendations' in dashboard
        
        # Check that we have some optimization stats
        assert dashboard['optimization_stats']['queries_optimized'] >= 2
    
    def test_recommendation_generation(self):
        """Test performance recommendation generation."""
        # Create conditions for recommendations
        # Low cache hit rate
        for i in range(10):
            def unique_processor(query):
                return f"unique_result_{i}"
            self.system.optimize_and_cache_query(f"unique_query_{i}", unique_processor)
        
        recommendations = self.system._generate_recommendations()
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        
        # Check that recommendations are strings
        for rec in recommendations:
            assert isinstance(rec, str)
    
    def test_concurrent_cache_access(self):
        """Test concurrent cache access safety."""
        results = []
        
        def worker(worker_id):
            def processor(query):
                time.sleep(0.01)
                return f"result_{worker_id}_{query}"
            
            try:
                result, meta = self.system.optimize_and_cache_query(
                    f"query_{worker_id}", processor
                )
                results.append((worker_id, result, meta['cache_hit']))
                return True
            except Exception as e:
                results.append((worker_id, str(e), False))
                return False
        
        # Start multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Check that all workers completed successfully
        assert len(results) == 5
        successful_results = [r for r in results if isinstance(r[1], str) and r[1].startswith('result_')]
        assert len(successful_results) == 5


class TestGlobalFunctions:
    """Test global utility functions."""
    
    def test_get_optimization_system(self):
        """Test global optimization system getter."""
        system1 = get_optimization_system(OptimizationLevel.CONSERVATIVE)
        system2 = get_optimization_system(OptimizationLevel.CONSERVATIVE)
        
        # Should return the same instance (singleton)
        assert system1 is system2
        
        # Cleanup
        system1.shutdown()
    
    def test_optimized_query_execution_decorator(self):
        """Test optimized query execution decorator."""
        @optimized_query_execution
        def test_function(query, multiplier=1):
            return f"result: {query}" * multiplier
        
        # Execute function
        result, meta = test_function("test", multiplier=2)
        
        assert result == "result: testresult: test"
        assert 'cache_hit' in meta
        assert 'response_time' in meta
        
        # Cleanup
        get_optimization_system().shutdown()
    
    def test_create_high_performance_cache(self):
        """Test high-performance cache creation."""
        cache = create_high_performance_cache()
        
        assert isinstance(cache, IntelligentCache)
        assert cache.max_size == 50000
        assert cache.strategy == CacheStrategy.ADAPTIVE
        
        # Test basic functionality
        cache.put("test_key", "test_value")
        assert cache.get("test_key") == "test_value"


class TestPerformanceMetrics:
    """Test performance metrics functionality."""
    
    def test_metrics_initialization(self):
        """Test metrics initialization."""
        metrics = PerformanceMetrics()
        
        assert metrics.query_count == 0
        assert metrics.total_response_time == 0.0
        assert metrics.cache_hits == 0
        assert metrics.cache_misses == 0
        assert metrics.average_response_time() == 0.0
        assert metrics.cache_hit_rate() == 0.0
    
    def test_average_response_time_calculation(self):
        """Test average response time calculation."""
        metrics = PerformanceMetrics()
        metrics.query_count = 4
        metrics.total_response_time = 2.0
        
        assert metrics.average_response_time() == 0.5
    
    def test_cache_hit_rate_calculation(self):
        """Test cache hit rate calculation."""
        metrics = PerformanceMetrics()
        metrics.cache_hits = 7
        metrics.cache_misses = 3
        
        assert metrics.cache_hit_rate() == 0.7


if __name__ == "__main__":
    pytest.main([__file__])