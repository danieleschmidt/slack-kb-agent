"""
Performance testing configuration and fixtures.
"""

import pytest
import time
from typing import Dict, Any, List
import psutil
import threading
from unittest.mock import Mock


class PerformanceMonitor:
    """Monitor system performance during tests."""
    
    def __init__(self):
        self.metrics: Dict[str, List[float]] = {
            'cpu_percent': [],
            'memory_percent': [],
            'disk_io_read': [],
            'disk_io_write': [],
            'network_sent': [],
            'network_recv': []
        }
        self.monitoring = False
        self.monitor_thread = None
    
    def start_monitoring(self, interval: float = 0.1):
        """Start performance monitoring."""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, args=(interval,))
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop performance monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
    
    def _monitor_loop(self, interval: float):
        """Monitor system performance metrics."""
        while self.monitoring:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=None)
                self.metrics['cpu_percent'].append(cpu_percent)
                
                # Memory usage
                memory = psutil.virtual_memory()
                self.metrics['memory_percent'].append(memory.percent)
                
                # Disk I/O
                disk_io = psutil.disk_io_counters()
                if disk_io:
                    self.metrics['disk_io_read'].append(disk_io.read_bytes)
                    self.metrics['disk_io_write'].append(disk_io.write_bytes)
                
                # Network I/O
                network_io = psutil.net_io_counters()
                if network_io:
                    self.metrics['network_sent'].append(network_io.bytes_sent)
                    self.metrics['network_recv'].append(network_io.bytes_recv)
                
            except Exception:
                pass  # Ignore monitoring errors
            
            time.sleep(interval)
    
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Get performance summary statistics."""
        summary = {}
        
        for metric, values in self.metrics.items():
            if values:
                summary[metric] = {
                    'min': min(values),
                    'max': max(values),
                    'avg': sum(values) / len(values),
                    'count': len(values)
                }
            else:
                summary[metric] = {'min': 0, 'max': 0, 'avg': 0, 'count': 0}
        
        return summary


@pytest.fixture
def performance_monitor():
    """Provide performance monitoring capability."""
    monitor = PerformanceMonitor()
    yield monitor
    monitor.stop_monitoring()


@pytest.fixture
def benchmark_timer():
    """Time execution of test operations."""
    times = {}
    
    def timer(operation_name: str):
        def decorator(func):
            start_time = time.perf_counter()
            result = func()
            end_time = time.perf_counter()
            times[operation_name] = end_time - start_time
            return result
        return decorator
    
    timer.get_times = lambda: times.copy()
    return timer


@pytest.fixture
def memory_profiler():
    """Profile memory usage during tests."""
    import gc
    import sys
    
    initial_objects = len(gc.get_objects())
    initial_memory = psutil.Process().memory_info().rss
    
    yield
    
    # Force garbage collection
    gc.collect()
    
    final_objects = len(gc.get_objects())
    final_memory = psutil.Process().memory_info().rss
    
    # Check for memory leaks
    object_diff = final_objects - initial_objects
    memory_diff = final_memory - initial_memory
    
    # Warning thresholds
    if object_diff > 1000:
        pytest.warns(UserWarning, f"Potential object leak: {object_diff} objects created")
    
    if memory_diff > 50 * 1024 * 1024:  # 50MB
        pytest.warns(UserWarning, f"Potential memory leak: {memory_diff / 1024 / 1024:.2f}MB")


@pytest.fixture
def load_test_data():
    """Generate test data for load testing."""
    def generate_data(size: str = "small") -> Dict[str, Any]:
        if size == "small":
            return {
                "documents": [f"Test document {i}" for i in range(100)],
                "queries": [f"Test query {i}" for i in range(50)],
                "users": [f"user_{i}" for i in range(10)]
            }
        elif size == "medium":
            return {
                "documents": [f"Test document {i}" for i in range(1000)],
                "queries": [f"Test query {i}" for i in range(500)],
                "users": [f"user_{i}" for i in range(100)]
            }
        elif size == "large":
            return {
                "documents": [f"Test document {i}" for i in range(10000)],
                "queries": [f"Test query {i}" for i in range(5000)],
                "users": [f"user_{i}" for i in range(1000)]
            }
        else:
            raise ValueError(f"Unknown size: {size}")
    
    return generate_data


@pytest.fixture
def concurrent_executor():
    """Execute operations concurrently for load testing."""
    import concurrent.futures
    
    def execute_concurrent(func, args_list, max_workers=10):
        """Execute function concurrently with different arguments."""
        results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(func, *args) for args in args_list]
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    results.append({"error": str(e)})
        
        return results
    
    return execute_concurrent


@pytest.fixture
def stress_test_config():
    """Configuration for stress testing."""
    return {
        "concurrent_users": [10, 50, 100, 200],
        "test_duration": 60,  # seconds
        "ramp_up_time": 10,   # seconds
        "operations_per_second": [1, 5, 10, 20, 50],
        "memory_limit_mb": 500,
        "cpu_limit_percent": 80
    }


@pytest.fixture
def mock_heavy_operation():
    """Mock heavy operations for performance testing."""
    def create_mock(duration: float = 0.1, memory_mb: int = 10):
        """Create mock that simulates heavy operation."""
        mock = Mock()
        
        def side_effect(*args, **kwargs):
            # Simulate CPU usage
            time.sleep(duration)
            
            # Simulate memory usage
            if memory_mb > 0:
                dummy_data = bytearray(memory_mb * 1024 * 1024)
                time.sleep(0.01)  # Brief pause
                del dummy_data
            
            return f"Mock result after {duration}s"
        
        mock.side_effect = side_effect
        return mock
    
    return create_mock


@pytest.fixture
def database_performance_setup(test_database):
    """Setup for database performance testing."""
    from slack_kb_agent.database import DatabaseManager
    
    db_manager = DatabaseManager(test_database)
    
    # Pre-populate with test data
    test_data = []
    for i in range(1000):
        test_data.append({
            'content': f'Performance test document {i}',
            'source': f'perf_test_{i % 10}',
            'metadata': {'test_id': i, 'category': f'cat_{i % 5}'}
        })
    
    # Insert test data in batches
    batch_size = 100
    for i in range(0, len(test_data), batch_size):
        batch = test_data[i:i + batch_size]
        db_manager.bulk_insert_documents(batch)
    
    yield db_manager
    
    # Cleanup
    db_manager.clear_test_data()


# Performance test markers
def pytest_configure(config):
    """Configure custom markers for performance tests."""
    config.addinivalue_line(
        "markers", "performance: mark test as performance test"
    )
    config.addinivalue_line(
        "markers", "load_test: mark test as load test"
    )
    config.addinivalue_line(
        "markers", "stress_test: mark test as stress test"
    )
    config.addinivalue_line(
        "markers", "benchmark: mark test as benchmark"
    )


def pytest_runtest_setup(item):
    """Setup for performance tests."""
    if "performance" in item.keywords:
        # Ensure clean environment for performance tests
        import gc
        gc.collect()


def pytest_runtest_teardown(item, nextitem):
    """Cleanup after performance tests."""
    if "performance" in item.keywords:
        # Force cleanup
        import gc
        gc.collect()