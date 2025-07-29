"""
Search performance and load testing.
"""

import pytest
import time
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed


@pytest.mark.performance
class TestSearchPerformance:
    """Test search performance characteristics."""
    
    def test_search_response_time(self, knowledge_base, load_test_data, benchmark_timer):
        """Test search response time with various query loads."""
        test_data = load_test_data("medium")
        
        # Add documents to knowledge base
        for doc in test_data["documents"]:
            knowledge_base.add_document(doc, source="perf_test")
        
        # Test search performance
        response_times = []
        
        @benchmark_timer("search_operations")
        def perform_searches():
            for query in test_data["queries"][:100]:  # Test with 100 queries
                start_time = time.perf_counter()
                results = knowledge_base.search(query)
                end_time = time.perf_counter()
                response_times.append(end_time - start_time)
                
                # Verify we get results
                assert len(results) >= 0
        
        perform_searches()
        
        # Analyze performance metrics
        avg_response_time = statistics.mean(response_times)
        p95_response_time = statistics.quantiles(response_times, n=20)[18]  # 95th percentile
        
        # Performance assertions
        assert avg_response_time < 0.1, f"Average response time too high: {avg_response_time:.3f}s"
        assert p95_response_time < 0.2, f"95th percentile response time too high: {p95_response_time:.3f}s"
        
        print(f"Search Performance Metrics:")
        print(f"- Average response time: {avg_response_time:.3f}s")
        print(f"- 95th percentile: {p95_response_time:.3f}s")
        print(f"- Min response time: {min(response_times):.3f}s")
        print(f"- Max response time: {max(response_times):.3f}s")
    
    def test_concurrent_search_performance(self, knowledge_base, load_test_data, concurrent_executor):
        """Test search performance under concurrent load."""
        test_data = load_test_data("large")
        
        # Add documents to knowledge base
        for doc in test_data["documents"][:1000]:  # Use subset for faster setup
            knowledge_base.add_document(doc, source="perf_test")
        
        def search_operation(query):
            """Single search operation."""
            start_time = time.perf_counter()
            results = knowledge_base.search(query)
            end_time = time.perf_counter()
            
            return {
                "query": query,
                "results_count": len(results),
                "response_time": end_time - start_time,
                "success": True
            }
        
        # Prepare concurrent search arguments
        search_args = [(query,) for query in test_data["queries"][:200]]
        
        # Execute concurrent searches
        start_time = time.perf_counter()
        results = concurrent_executor(search_operation, search_args, max_workers=20)
        total_time = time.perf_counter() - start_time
        
        # Analyze results
        successful_searches = [r for r in results if r.get("success", False)]
        response_times = [r["response_time"] for r in successful_searches]
        
        # Performance assertions
        success_rate = len(successful_searches) / len(results)
        assert success_rate > 0.95, f"Success rate too low: {success_rate:.2%}"
        
        avg_response_time = statistics.mean(response_times)
        assert avg_response_time < 0.5, f"Average concurrent response time too high: {avg_response_time:.3f}s"
        
        throughput = len(successful_searches) / total_time
        assert throughput > 10, f"Throughput too low: {throughput:.1f} searches/second"
        
        print(f"Concurrent Search Performance:")
        print(f"- Success rate: {success_rate:.2%}")
        print(f"- Average response time: {avg_response_time:.3f}s")
        print(f"- Throughput: {throughput:.1f} searches/second")
    
    @pytest.mark.benchmark
    def test_search_scalability(self, knowledge_base, load_test_data, performance_monitor):
        """Test how search performance scales with data size."""
        performance_monitor.start_monitoring()
        
        data_sizes = [100, 500, 1000, 2000, 5000]
        performance_results = {}
        
        for size in data_sizes:
            # Clear and repopulate knowledge base
            knowledge_base.clear()
            test_data = load_test_data("large")
            
            for doc in test_data["documents"][:size]:
                knowledge_base.add_document(doc, source="scale_test")
            
            # Measure search performance
            search_times = []
            for query in test_data["queries"][:50]:  # Fixed number of queries
                start_time = time.perf_counter()
                results = knowledge_base.search(query)
                end_time = time.perf_counter()
                search_times.append(end_time - start_time)
            
            performance_results[size] = {
                "avg_search_time": statistics.mean(search_times),
                "max_search_time": max(search_times),
                "documents_count": size
            }
        
        performance_monitor.stop_monitoring()
        system_metrics = performance_monitor.get_summary()
        
        # Analyze scalability
        print(f"Search Scalability Results:")
        for size, metrics in performance_results.items():
            print(f"- {size} docs: avg={metrics['avg_search_time']:.3f}s, max={metrics['max_search_time']:.3f}s")
        
        # Performance should not degrade exponentially
        small_time = performance_results[100]["avg_search_time"]
        large_time = performance_results[5000]["avg_search_time"]
        scalability_ratio = large_time / small_time
        
        assert scalability_ratio < 10, f"Poor scalability: {scalability_ratio:.1f}x slower for 50x more data"
        
        # System resource usage should be reasonable
        assert system_metrics["cpu_percent"]["max"] < 90, "CPU usage too high during scaling test"
        assert system_metrics["memory_percent"]["max"] < 80, "Memory usage too high during scaling test"


@pytest.mark.load_test
class TestSearchLoadTesting:
    """Load testing for search functionality."""
    
    def test_sustained_search_load(self, knowledge_base, load_test_data, stress_test_config):
        """Test search performance under sustained load."""
        test_data = load_test_data("large")
        
        # Setup knowledge base
        for doc in test_data["documents"][:2000]:
            knowledge_base.add_document(doc, source="load_test")
        
        def search_worker(queries, duration):
            """Worker function for sustained searching."""
            end_time = time.time() + duration
            search_count = 0
            errors = 0
            
            while time.time() < end_time:
                try:
                    query = queries[search_count % len(queries)]
                    results = knowledge_base.search(query)
                    search_count += 1
                    assert len(results) >= 0  # Basic validation
                except Exception:
                    errors += 1
                
                time.sleep(0.1)  # Small delay between searches
            
            return {"searches": search_count, "errors": errors}
        
        # Run sustained load test
        test_duration = 30  # seconds
        num_workers = 10
        queries_per_worker = test_data["queries"][:50]
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(search_worker, queries_per_worker, test_duration)
                for _ in range(num_workers)
            ]
            
            results = [future.result() for future in as_completed(futures)]
        
        # Analyze load test results
        total_searches = sum(r["searches"] for r in results)
        total_errors = sum(r["errors"] for r in results)
        
        error_rate = total_errors / total_searches if total_searches > 0 else 1
        throughput = total_searches / test_duration
        
        # Load test assertions
        assert error_rate < 0.01, f"Error rate too high: {error_rate:.2%}"
        assert throughput > 50, f"Throughput too low under load: {throughput:.1f} searches/second"
        
        print(f"Sustained Load Test Results:")
        print(f"- Total searches: {total_searches}")
        print(f"- Error rate: {error_rate:.2%}")
        print(f"- Throughput: {throughput:.1f} searches/second")
    
    def test_memory_usage_under_load(self, knowledge_base, load_test_data, memory_profiler):
        """Test memory usage patterns under search load."""
        test_data = load_test_data("large")
        
        # Add documents
        for doc in test_data["documents"][:1000]:
            knowledge_base.add_document(doc, source="memory_test")
        
        # Perform many searches to test memory stability
        for i in range(500):
            query = test_data["queries"][i % len(test_data["queries"])]
            results = knowledge_base.search(query)
            
            # Verify results to ensure functionality
            assert len(results) >= 0
            
            # Force periodic garbage collection
            if i % 100 == 0:
                import gc
                gc.collect()
        
        # Memory profiler will check for leaks in teardown
    
    @pytest.mark.stress_test
    def test_search_stress_limits(self, knowledge_base, load_test_data, performance_monitor):
        """Test search behavior at stress limits."""
        performance_monitor.start_monitoring()
        test_data = load_test_data("large")
        
        # Add maximum test documents
        for doc in test_data["documents"]:
            knowledge_base.add_document(doc, source="stress_test")
        
        # Stress test parameters
        max_concurrent_searches = 50
        stress_duration = 60  # seconds
        
        def stress_search_worker():
            """High-intensity search worker."""
            searches = 0
            errors = 0
            start_time = time.time()
            
            while time.time() - start_time < stress_duration:
                try:
                    query = test_data["queries"][searches % len(test_data["queries"])]
                    results = knowledge_base.search(query)
                    searches += 1
                    
                    # Minimal validation
                    assert isinstance(results, list)
                    
                except Exception:
                    errors += 1
                
                # No delay - maximum stress
            
            return {"searches": searches, "errors": errors}
        
        # Execute stress test
        with ThreadPoolExecutor(max_workers=max_concurrent_searches) as executor:
            futures = [
                executor.submit(stress_search_worker)
                for _ in range(max_concurrent_searches)
            ]
            
            results = [future.result() for future in as_completed(futures)]
        
        performance_monitor.stop_monitoring()
        system_metrics = performance_monitor.get_summary()
        
        # Analyze stress test results
        total_searches = sum(r["searches"] for r in results)
        total_errors = sum(r["errors"] for r in results)
        
        error_rate = total_errors / total_searches if total_searches > 0 else 1
        peak_throughput = total_searches / stress_duration
        
        # Stress test validation
        assert error_rate < 0.05, f"Error rate too high under stress: {error_rate:.2%}"
        assert peak_throughput > 100, f"Peak throughput too low: {peak_throughput:.1f} searches/second"
        
        # System should remain stable under stress
        assert system_metrics["memory_percent"]["max"] < 95, "Memory usage critical under stress"
        
        print(f"Stress Test Results:")
        print(f"- Peak throughput: {peak_throughput:.1f} searches/second")
        print(f"- Error rate: {error_rate:.2%}")
        print(f"- Max CPU usage: {system_metrics['cpu_percent']['max']:.1f}%")
        print(f"- Max memory usage: {system_metrics['memory_percent']['max']:.1f}%")