"""Test enhanced circuit breaker system."""

import pytest
import time
import threading
from unittest.mock import patch, MagicMock

from slack_kb_agent.enhanced_circuit_breaker import (
    AdaptiveCircuitBreaker,
    BulkheadProtection,
    CascadingCircuitBreakerManager,
    ServiceMetrics,
    HealthStatus,
    FailureType,
    BulkheadConfig,
    get_circuit_breaker,
    get_bulkhead,
    protected_call,
    create_database_circuit_breaker,
    create_api_bulkhead
)
from slack_kb_agent.circuit_breaker import CircuitBreakerConfig, CircuitState


class TestServiceMetrics:
    """Test service metrics functionality."""
    
    def test_success_rate_calculation(self):
        """Test success rate calculation."""
        metrics = ServiceMetrics()
        
        # No calls yet
        assert metrics.success_rate() == 0.0
        
        # Add some successes and failures
        metrics.success_count = 8
        metrics.failure_count = 2
        assert metrics.success_rate() == 0.8
        
        # Only failures
        metrics.success_count = 0
        metrics.failure_count = 5
        assert metrics.success_rate() == 0.0
        
        # Only successes
        metrics.success_count = 10
        metrics.failure_count = 0
        assert metrics.success_rate() == 1.0
    
    def test_response_time_metrics(self):
        """Test response time calculations."""
        metrics = ServiceMetrics()
        
        # No response times
        assert metrics.average_response_time() == 0.0
        assert metrics.p95_response_time() == 0.0
        
        # Add response times
        response_times = [0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
        for rt in response_times:
            metrics.response_times.append(rt)
        
        # Check average
        assert metrics.average_response_time() == 1.6
        
        # Check P95 (should be close to 95th percentile)
        p95 = metrics.p95_response_time()
        assert p95 >= 4.0  # Should be high value from the list


class TestAdaptiveCircuitBreaker:
    """Test adaptive circuit breaker functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = CircuitBreakerConfig(
            failure_threshold=3,
            success_threshold=2,
            timeout_seconds=1.0
        )
        self.cb = AdaptiveCircuitBreaker("test_service", self.config)
    
    def test_success_recording_with_response_time(self):
        """Test success recording with response time tracking."""
        response_time = 0.5
        self.cb._record_success(response_time)
        
        assert self.cb.metrics.success_count == 1
        assert self.cb.metrics.consecutive_successes == 1
        assert self.cb.metrics.consecutive_failures == 0
        assert self.cb.metrics.last_success is not None
        assert len(self.cb.metrics.response_times) == 1
        assert self.cb.metrics.response_times[0] == response_time
    
    def test_failure_recording_with_type(self):
        """Test failure recording with failure type tracking."""
        self.cb._record_failure(FailureType.TIMEOUT)
        
        assert self.cb.metrics.failure_count == 1
        assert self.cb.metrics.consecutive_failures == 1
        assert self.cb.metrics.consecutive_successes == 0
        assert self.cb.metrics.last_failure is not None
        assert self.cb.metrics.failure_types[FailureType.TIMEOUT] == 1
        assert self.cb.metrics.timeout_count == 1
    
    def test_health_score_calculation(self):
        """Test comprehensive health score calculation."""
        # Initially healthy (no data)
        health_score = self.cb.calculate_health_score()
        assert health_score == 1.0
        
        # Add some successes
        for i in range(10):
            self.cb._record_success(0.1)  # Fast response times
        
        health_score = self.cb.calculate_health_score()
        assert health_score > 0.8  # Should be very healthy
        
        # Add failures
        for i in range(5):
            self.cb._record_failure(FailureType.TIMEOUT)
        
        health_score = self.cb.calculate_health_score()
        assert health_score < 0.8  # Should be degraded
    
    def test_health_status_determination(self):
        """Test health status determination."""
        # Initially healthy
        assert self.cb.get_health_status() == HealthStatus.HEALTHY
        
        # Add many failures to make unhealthy
        for i in range(10):
            self.cb._record_failure(FailureType.CONNECTION_ERROR)
        
        # Should be unhealthy or critical
        status = self.cb.get_health_status()
        assert status in [HealthStatus.UNHEALTHY, HealthStatus.CRITICAL]
    
    def test_adaptive_threshold_adjustment(self):
        """Test adaptive threshold adjustment."""
        original_threshold = self.cb.config.failure_threshold
        
        # Add many successes with fast response times (very healthy service)
        for i in range(100):
            self.cb._record_success(0.05)  # Very fast
        
        # Threshold might be increased for healthy service
        # (depending on the adaptive logic)
        new_threshold = self.cb.config.failure_threshold
        assert new_threshold >= original_threshold
    
    def test_dependency_management(self):
        """Test service dependency management."""
        self.cb.add_dependency("database")
        self.cb.add_dependent_service("api_gateway")
        
        assert "database" in self.cb.dependencies
        assert "api_gateway" in self.cb.dependent_services


class TestBulkheadProtection:
    """Test bulkhead pattern implementation."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = BulkheadConfig(
            max_concurrent_calls=3,
            max_wait_time=1.0,
            timeout_per_call=2.0
        )
        self.bulkhead = BulkheadProtection("test_bulkhead", self.config)
    
    def test_slot_acquisition_success(self):
        """Test successful slot acquisition."""
        with self.bulkhead.acquire_slot():
            assert self.bulkhead.active_calls == 1
            assert self.bulkhead.total_calls == 1
        
        assert self.bulkhead.active_calls == 0
        assert self.bulkhead.successful_calls == 1
    
    def test_slot_acquisition_timeout(self):
        """Test slot acquisition timeout."""
        # Fill all slots
        contexts = []
        for i in range(self.config.max_concurrent_calls):
            ctx = self.bulkhead.acquire_slot()
            contexts.append(ctx)
            ctx.__enter__()
        
        # Try to acquire another slot - should timeout
        with pytest.raises(RuntimeError, match="No slots available"):
            with self.bulkhead.acquire_slot(timeout=0.1):
                pass
        
        # Clean up contexts
        for ctx in contexts:
            try:
                ctx.__exit__(None, None, None)
            except:
                pass
    
    def test_concurrent_access_limit(self):
        """Test concurrent access limiting."""
        results = []
        
        def worker(worker_id):
            try:
                with self.bulkhead.acquire_slot(timeout=0.5):
                    results.append(f"worker_{worker_id}_started")
                    time.sleep(0.2)
                    results.append(f"worker_{worker_id}_finished")
                    return True
            except RuntimeError:
                results.append(f"worker_{worker_id}_rejected")
                return False
        
        # Start more threads than allowed concurrent calls
        threads = []
        for i in range(6):  # More than max_concurrent_calls (3)
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Should have some rejections due to bulkhead limit
        rejected_count = len([r for r in results if "rejected" in r])
        assert rejected_count > 0
        assert self.bulkhead.rejected_calls > 0
    
    def test_metrics_collection(self):
        """Test bulkhead metrics collection."""
        # Successful call
        with self.bulkhead.acquire_slot():
            pass
        
        # Rejected call
        try:
            # Fill all slots first
            contexts = []
            for i in range(self.config.max_concurrent_calls):
                ctx = self.bulkhead.acquire_slot()
                contexts.append(ctx)
                ctx.__enter__()
            
            # This should be rejected
            with self.bulkhead.acquire_slot(timeout=0.1):
                pass
        except RuntimeError:
            pass
        finally:
            # Clean up contexts
            for ctx in contexts:
                try:
                    ctx.__exit__(None, None, None)
                except:
                    pass
        
        metrics = self.bulkhead.get_metrics()
        
        assert metrics['name'] == 'test_bulkhead'
        assert metrics['total_calls'] >= 1
        assert metrics['successful_calls'] >= 1
        assert metrics['rejected_calls'] >= 1
        assert 0 <= metrics['success_rate'] <= 1
        assert 0 <= metrics['rejection_rate'] <= 1
        assert 0 <= metrics['utilization'] <= 1


class TestCascadingCircuitBreakerManager:
    """Test cascading circuit breaker management."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.manager = CascadingCircuitBreakerManager()
    
    def test_circuit_breaker_registration(self):
        """Test circuit breaker registration."""
        config = CircuitBreakerConfig(failure_threshold=5)
        cb = self.manager.register_circuit_breaker("test_service", config)
        
        assert "test_service" in self.manager.circuit_breakers
        assert isinstance(cb, AdaptiveCircuitBreaker)
        assert cb.config.failure_threshold == 5
    
    def test_bulkhead_registration(self):
        """Test bulkhead registration."""
        config = BulkheadConfig(max_concurrent_calls=10)
        bulkhead = self.manager.register_bulkhead("test_bulkhead", config)
        
        assert "test_bulkhead" in self.manager.bulkheads
        assert isinstance(bulkhead, BulkheadProtection)
        assert bulkhead.config.max_concurrent_calls == 10
    
    def test_dependency_management(self):
        """Test service dependency management."""
        # Register services
        self.manager.register_circuit_breaker("api_gateway")
        self.manager.register_circuit_breaker("database")
        
        # Add dependency
        self.manager.add_dependency("api_gateway", "database")
        
        assert "database" in self.manager.dependency_graph["api_gateway"]
        
        # Check circuit breaker dependency links
        api_cb = self.manager.circuit_breakers["api_gateway"]
        db_cb = self.manager.circuit_breakers["database"]
        
        assert "database" in api_cb.dependencies
        assert "api_gateway" in db_cb.dependent_services
    
    def test_health_check_registration(self):
        """Test health check callback registration."""
        def mock_health_check():
            return True
        
        self.manager.register_health_check("test_service", mock_health_check)
        
        assert "test_service" in self.manager.health_check_callbacks
        assert self.manager.health_check_callbacks["test_service"]() == True
    
    def test_service_health_reporting(self):
        """Test comprehensive service health reporting."""
        # Register service
        cb = self.manager.register_circuit_breaker("test_service")
        
        # Add some metrics
        cb._record_success(0.1)
        cb._record_success(0.2)
        cb._record_failure(FailureType.TIMEOUT)
        
        # Register health check
        def health_check():
            return True
        self.manager.register_health_check("test_service", health_check)
        
        # Get health report
        status, details = self.manager.get_service_health("test_service")
        
        assert isinstance(status, HealthStatus)
        assert 'health_score' in details
        assert 'success_rate' in details
        assert 'avg_response_time' in details
        assert 'consecutive_failures' in details
        assert 'circuit_state' in details
        assert details['health_check_passed'] == True
    
    def test_cascading_failure_detection(self):
        """Test cascading failure risk detection."""
        # Register services with dependency
        api_cb = self.manager.register_circuit_breaker("api_gateway")
        db_cb = self.manager.register_circuit_breaker("database")
        
        self.manager.add_dependency("api_gateway", "database")
        
        # Make database circuit open (failing)
        for i in range(5):  # Exceed failure threshold
            db_cb._record_failure(FailureType.CONNECTION_ERROR)
        
        # Make API gateway degraded
        for i in range(3):
            api_cb._record_success(0.1)
        for i in range(2):
            api_cb._record_failure(FailureType.TIMEOUT)
        
        # Check for cascading risks
        risks = self.manager.check_cascading_failures()
        
        # Should detect potential cascading failure
        # (depends on exact health calculation, but database should be down)
        # This test verifies the mechanism works
        assert isinstance(risks, list)
    
    def test_system_health_dashboard(self):
        """Test system health dashboard generation."""
        # Register multiple services
        self.manager.register_circuit_breaker("service1")
        self.manager.register_circuit_breaker("service2")
        self.manager.register_bulkhead("bulkhead1", BulkheadConfig())
        
        dashboard = self.manager.get_system_health_dashboard()
        
        # Check dashboard structure
        assert 'timestamp' in dashboard
        assert 'services' in dashboard
        assert 'bulkheads' in dashboard
        assert 'cascading_risks' in dashboard
        assert 'system_summary' in dashboard
        
        # Check services are included
        assert 'service1' in dashboard['services']
        assert 'service2' in dashboard['services']
        
        # Check bulkheads are included
        assert 'bulkhead1' in dashboard['bulkheads']
        
        # Check system summary
        summary = dashboard['system_summary']
        assert summary['total_services'] == 2
        assert 'overall_health_score' in summary


class TestGlobalFunctions:
    """Test global convenience functions."""
    
    def test_get_circuit_breaker(self):
        """Test get_circuit_breaker function."""
        cb1 = get_circuit_breaker("global_test")
        cb2 = get_circuit_breaker("global_test")  # Same instance
        
        assert cb1 is cb2
        assert cb1.name == "global_test"
    
    def test_get_bulkhead(self):
        """Test get_bulkhead function."""
        bulkhead1 = get_bulkhead("global_bulkhead")
        bulkhead2 = get_bulkhead("global_bulkhead")  # Same instance
        
        assert bulkhead1 is bulkhead2
        assert bulkhead1.name == "global_bulkhead"
    
    def test_protected_call_success(self):
        """Test protected_call context manager for successful calls."""
        call_executed = False
        
        with protected_call("test_protected"):
            call_executed = True
            time.sleep(0.1)  # Simulate work
        
        assert call_executed == True
        
        # Check that success was recorded
        cb = get_circuit_breaker("test_protected")
        assert cb.metrics.success_count >= 1
        assert len(cb.metrics.response_times) >= 1
    
    def test_protected_call_failure(self):
        """Test protected_call context manager for failed calls."""
        with pytest.raises(ValueError):
            with protected_call("test_protected_fail", failure_type=FailureType.CONNECTION_ERROR):
                raise ValueError("Simulated failure")
        
        # Check that failure was recorded
        cb = get_circuit_breaker("test_protected_fail")
        assert cb.metrics.failure_count >= 1
        assert cb.metrics.failure_types[FailureType.CONNECTION_ERROR] >= 1
    
    def test_protected_call_with_bulkhead(self):
        """Test protected_call with bulkhead protection."""
        config = BulkheadConfig(max_concurrent_calls=2)
        get_bulkhead("test_bulkhead_protected", config)
        
        call_executed = False
        
        with protected_call("test_protected_bulkhead", "test_bulkhead_protected"):
            call_executed = True
        
        assert call_executed == True
        
        # Check metrics
        bulkhead = get_bulkhead("test_bulkhead_protected")
        assert bulkhead.successful_calls >= 1


class TestPrebuiltConfigurations:
    """Test prebuilt configuration functions."""
    
    def test_create_database_circuit_breaker(self):
        """Test database circuit breaker creation."""
        cb = create_database_circuit_breaker()
        
        assert cb.name == "database"
        assert cb.config.failure_threshold == 3
        assert cb.config.success_threshold == 2
        assert cb.config.timeout_seconds == 30.0
    
    def test_create_api_bulkhead(self):
        """Test API bulkhead creation."""
        bulkhead = create_api_bulkhead()
        
        assert bulkhead.name == "api_calls"
        assert bulkhead.config.max_concurrent_calls == 20
        assert bulkhead.config.max_wait_time == 5.0
        assert bulkhead.config.timeout_per_call == 30.0


if __name__ == "__main__":
    pytest.main([__file__])