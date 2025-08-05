"""Tests for quantum task planner."""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

from slack_kb_agent.quantum_task_planner import (
    QuantumTask, QuantumTaskPlanner, TaskState, TaskPriority,
    get_quantum_planner, create_simple_task, create_dependent_tasks, create_entangled_task_pair
)


class TestQuantumTask:
    """Test quantum task functionality."""
    
    def test_task_creation(self):
        """Test quantum task creation."""
        task = QuantumTask(
            name="test_task",
            description="Test task description",
            priority=TaskPriority.HIGH,
            superposition_states=["ready", "running", "blocked"]
        )
        
        assert task.name == "test_task"
        assert task.description == "Test task description"
        assert task.priority == TaskPriority.HIGH
        assert task.state == TaskState.SUPERPOSITION
        assert len(task.superposition_states) == 3
        assert len(task.probability_amplitudes) == 0  # Set during planner creation
    
    def test_state_collapse(self):
        """Test quantum state collapse."""
        task = QuantumTask(
            name="test_task",
            superposition_states=["ready", "running", "blocked"]
        )
        
        # Initially in superposition
        assert task.state == TaskState.SUPERPOSITION
        
        # Collapse state
        task.collapse_state("running")
        
        assert task.state == TaskState.COLLAPSED
        assert task.superposition_states == ["running"]
    
    def test_task_entanglement(self):
        """Test quantum task entanglement."""
        task1 = QuantumTask(name="task1")
        task2 = QuantumTask(name="task2")
        
        # Initially not entangled
        assert len(task1.entangled_tasks) == 0
        assert len(task2.entangled_tasks) == 0
        
        # Create entanglement
        task1.entangle_with(task2)
        
        assert task2.id in task1.entangled_tasks
        assert task1.id in task2.entangled_tasks
        assert task1.state == TaskState.ENTANGLED
        assert task2.state == TaskState.ENTANGLED
    
    def test_execution_probability_calculation(self):
        """Test execution probability calculation."""
        task = QuantumTask(
            name="test_task",
            priority=TaskPriority.HIGH,
            dependencies=["dep1", "dep2"],
            entangled_tasks={"entangled1", "entangled2"}
        )
        
        probability = task.calculate_execution_probability()
        
        # Should be less than base priority due to dependencies and entanglement
        assert 0.0 <= probability <= 1.0
        assert probability < TaskPriority.HIGH.quantum_weight
    
    def test_retry_penalty(self):
        """Test retry penalty in probability calculation."""
        task = QuantumTask(
            name="test_task",
            priority=TaskPriority.HIGH,
            retry_count=2
        )
        
        probability = task.calculate_execution_probability()
        expected_penalty = 0.1 * 2  # 0.1 per retry
        expected_probability = TaskPriority.HIGH.quantum_weight - expected_penalty
        
        assert abs(probability - expected_probability) < 0.001


class TestQuantumTaskPlanner:
    """Test quantum task planner functionality."""
    
    @pytest.fixture
    def planner(self):
        """Create a quantum task planner for testing."""
        return QuantumTaskPlanner(max_concurrent_tasks=3, coherence_time=60)
    
    def test_planner_initialization(self, planner):
        """Test planner initialization."""
        assert planner.max_concurrent_tasks == 3
        assert planner.coherence_time == 60
        assert len(planner.tasks) == 0
        assert len(planner.execution_queue) == 0
    
    def test_task_creation(self, planner):
        """Test task creation through planner."""
        def dummy_executor(task):
            return True
        
        task = planner.create_task(
            "test_task",
            "Test description",
            dummy_executor,
            TaskPriority.MEDIUM,
            superposition_states=["ready", "blocked"]
        )
        
        assert task.id in planner.tasks
        assert task.name == "test_task"
        assert len(task.probability_amplitudes) == 2
        assert abs(sum(task.probability_amplitudes.values()) - 1.0) < 0.001
    
    def test_task_state_measurement(self, planner):
        """Test quantum state measurement."""
        def dummy_executor(task):
            return True
        
        task = planner.create_task(
            "test_task",
            "Test description", 
            dummy_executor,
            superposition_states=["ready", "blocked", "waiting"]
        )
        
        # Measure state - should collapse superposition
        observed_state = planner.measure_task_state(task.id)
        
        assert observed_state in ["ready", "blocked", "waiting"]
        assert task.state == TaskState.COLLAPSED
    
    def test_task_entanglement_creation(self, planner):
        """Test creating task entanglement through planner."""
        def dummy_executor(task):
            return True
        
        task1 = planner.create_task("task1", "Description 1", dummy_executor)
        task2 = planner.create_task("task2", "Description 2", dummy_executor)
        
        success = planner.entangle_tasks(task1.id, task2.id)
        
        assert success
        assert task2.id in task1.entangled_tasks
        assert task1.id in task2.entangled_tasks
    
    def test_task_scheduling_without_dependencies(self, planner):
        """Test scheduling task without dependencies."""
        def dummy_executor(task):
            return True
        
        task = planner.create_task(
            "test_task",
            "Test description",
            dummy_executor,
            TaskPriority.HIGH
        )
        
        success = planner.schedule_task(task.id)
        
        assert success
        assert task in planner.execution_queue
        assert task.state == TaskState.COLLAPSED
    
    def test_task_scheduling_with_dependencies(self, planner):
        """Test scheduling task with unmet dependencies."""
        def dummy_executor(task):
            return True
        
        # Create dependency task that's not completed
        dep_task = planner.create_task("dep_task", "Dependency", dummy_executor)
        
        # Create task with dependency
        task = planner.create_task(
            "test_task",
            "Test description",
            dummy_executor,
            TaskPriority.HIGH,
            dependencies=[dep_task.id]
        )
        
        # Should fail to schedule due to unmet dependency
        success = planner.schedule_task(task.id)
        
        assert not success
        assert task not in planner.execution_queue
    
    def test_task_scheduling_with_met_dependencies(self, planner):
        """Test scheduling task with met dependencies."""
        def dummy_executor(task):
            return True
        
        # Create and complete dependency task
        dep_task = planner.create_task("dep_task", "Dependency", dummy_executor)
        dep_task.state = TaskState.COMPLETED
        
        # Create task with dependency
        task = planner.create_task(
            "test_task",
            "Test description",
            dummy_executor,
            TaskPriority.HIGH,
            dependencies=[dep_task.id]
        )
        
        # Should succeed to schedule
        success = planner.schedule_task(task.id)
        
        assert success
        assert task in planner.execution_queue
    
    @pytest.mark.asyncio
    async def test_task_execution_success(self, planner):
        """Test successful task execution."""
        execution_called = False
        
        def successful_executor(task):
            nonlocal execution_called
            execution_called = True
            return True
        
        task = planner.create_task(
            "test_task",
            "Test description",
            successful_executor
        )
        
        success = await planner.execute_task(task)
        
        assert success
        assert execution_called
        assert task.state == TaskState.COMPLETED
        assert task in planner.completed_tasks
    
    @pytest.mark.asyncio
    async def test_task_execution_failure(self, planner):
        """Test failed task execution."""
        def failing_executor(task):
            raise Exception("Task failed")
        
        task = planner.create_task(
            "test_task",
            "Test description",
            failing_executor,
            max_retries=1
        )
        
        # First execution should fail and retry
        success1 = await planner.execute_task(task)
        assert not success1
        assert task.retry_count == 1
        assert task.state == TaskState.SUPERPOSITION  # Reset for retry
        
        # Second execution should fail permanently
        success2 = await planner.execute_task(task)
        assert not success2
        assert task.state == TaskState.FAILED
        assert task in planner.failed_tasks
    
    @pytest.mark.asyncio
    async def test_async_task_execution(self, planner):
        """Test execution of async tasks."""
        execution_called = False
        
        async def async_executor(task):
            nonlocal execution_called
            execution_called = True
            await asyncio.sleep(0.01)  # Small delay
            return True
        
        task = planner.create_task(
            "async_task",
            "Async test description",
            async_executor
        )
        
        success = await planner.execute_task(task)
        
        assert success
        assert execution_called
        assert task.state == TaskState.COMPLETED
    
    def test_decoherence_check(self, planner):
        """Test quantum decoherence detection."""
        def dummy_executor(task):
            return True
        
        # Create task with old timestamp
        old_time = datetime.now() - timedelta(seconds=planner.coherence_time + 10)
        
        task = planner.create_task("old_task", "Old description", dummy_executor)
        task.created_at = old_time
        
        # Check decoherence
        planner.check_decoherence()
        
        assert task.state == TaskState.DECOHERENT
    
    def test_entangled_task_updates(self, planner):
        """Test entangled task probability updates."""
        def dummy_executor(task):
            return True
        
        task1 = planner.create_task("task1", "Description 1", dummy_executor)
        task2 = planner.create_task("task2", "Description 2", dummy_executor)
        
        planner.entangle_tasks(task1.id, task2.id)
        
        # Get initial probabilities
        initial_prob = sum(task2.probability_amplitudes.values())
        
        # Update task1 to completed state
        planner._update_entangled_tasks(task1.id, TaskState.COMPLETED)
        
        # Task2 probabilities should be updated
        updated_prob = sum(task2.probability_amplitudes.values())
        assert abs(updated_prob - 1.0) < 0.001  # Should still sum to 1
    
    def test_system_metrics(self, planner):
        """Test system metrics collection."""
        def dummy_executor(task):
            return True
        
        # Create various tasks
        task1 = planner.create_task("task1", "Description 1", dummy_executor)
        task2 = planner.create_task("task2", "Description 2", dummy_executor)
        task3 = planner.create_task("task3", "Description 3", dummy_executor)
        
        # Set different states
        task1.state = TaskState.COMPLETED
        task2.state = TaskState.ENTANGLED
        planner.completed_tasks.append(task1)
        planner.execution_queue.append(task2)
        
        metrics = planner.get_system_metrics()
        
        assert metrics["total_tasks"] == 3
        assert metrics["completed_tasks"] == 1
        assert metrics["superposition_tasks"] == 1  # task3
        assert metrics["entangled_tasks"] == 1     # task2
        assert metrics["queue_length"] == 1
        assert "average_execution_time" in metrics
        assert "coherence_time" in metrics


class TestUtilityFunctions:
    """Test utility functions."""
    
    @patch('slack_kb_agent.quantum_task_planner.get_quantum_planner')
    def test_create_simple_task(self, mock_get_planner):
        """Test simple task creation utility."""
        mock_planner = Mock()
        mock_task = Mock()
        mock_planner.create_task.return_value = mock_task
        mock_get_planner.return_value = mock_planner
        
        def dummy_func():
            pass
        
        result = create_simple_task("test_task", dummy_func, TaskPriority.HIGH)
        
        mock_planner.create_task.assert_called_once_with(
            "test_task", 
            "Execute test_task", 
            dummy_func, 
            TaskPriority.HIGH
        )
        assert result == mock_task
    
    @patch('slack_kb_agent.quantum_task_planner.get_quantum_planner')
    def test_create_dependent_tasks(self, mock_get_planner):
        """Test dependent task creation utility."""
        mock_planner = Mock()
        mock_tasks = [Mock(), Mock(), Mock()]
        mock_planner.create_task.side_effect = mock_tasks
        mock_get_planner.return_value = mock_planner
        
        def func1():
            pass
        def func2():
            pass
        def func3():
            pass
        
        tasks_spec = [("task1", func1), ("task2", func2), ("task3", func3)]
        result = create_dependent_tasks(tasks_spec, TaskPriority.MEDIUM)
        
        assert len(result) == 3
        assert mock_planner.create_task.call_count == 3
        
        # Check dependency chain
        calls = mock_planner.create_task.call_args_list
        assert calls[0][1]["dependencies"] == []  # First task has no dependencies
        assert calls[1][1]["dependencies"] == [mock_tasks[0].id]  # Second depends on first
        assert calls[2][1]["dependencies"] == [mock_tasks[1].id]  # Third depends on second
    
    @patch('slack_kb_agent.quantum_task_planner.get_quantum_planner')
    def test_create_entangled_task_pair(self, mock_get_planner):
        """Test entangled task pair creation utility."""
        mock_planner = Mock()
        mock_task1 = Mock()
        mock_task2 = Mock()
        mock_planner.create_task.side_effect = [mock_task1, mock_task2]
        mock_get_planner.return_value = mock_planner
        
        def func1():
            pass
        def func2():
            pass
        
        task1, task2 = create_entangled_task_pair(
            "task1", func1, "task2", func2, TaskPriority.LOW
        )
        
        assert task1 == mock_task1
        assert task2 == mock_task2
        mock_planner.entangle_tasks.assert_called_once_with(mock_task1.id, mock_task2.id)


class TestGlobalPlannerInstance:
    """Test global planner instance management."""
    
    @patch('slack_kb_agent.quantum_task_planner.get_slack_bot_config')
    def test_get_quantum_planner_singleton(self, mock_get_config):
        """Test global planner singleton behavior."""
        mock_config = Mock()
        mock_config.max_concurrent_tasks = 10
        mock_config.quantum_coherence_time = 600
        mock_get_config.return_value = mock_config
        
        # Reset global instance
        import slack_kb_agent.quantum_task_planner as qtp_module
        qtp_module._quantum_planner = None
        
        planner1 = get_quantum_planner()
        planner2 = get_quantum_planner()
        
        assert planner1 is planner2  # Should be same instance
        assert planner1.max_concurrent_tasks == 10
        assert planner1.coherence_time == 600
    
    @patch('slack_kb_agent.quantum_task_planner.get_slack_bot_config')
    def test_get_quantum_planner_defaults(self, mock_get_config):
        """Test planner creation with default values."""
        mock_config = Mock()
        # Don't set max_concurrent_tasks or quantum_coherence_time attributes
        mock_get_config.return_value = mock_config
        
        # Reset global instance
        import slack_kb_agent.quantum_task_planner as qtp_module
        qtp_module._quantum_planner = None
        
        planner = get_quantum_planner()
        
        assert planner.max_concurrent_tasks == 5  # Default value
        assert planner.coherence_time == 300     # Default value