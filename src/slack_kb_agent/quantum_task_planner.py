"""Quantum-inspired task planning and autonomous execution system."""

from __future__ import annotations

import asyncio
import json
import time
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any, Callable, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from datetime import datetime, timedelta
import uuid

from .monitoring import get_global_metrics, StructuredLogger
from .cache import get_cache_manager
from .configuration import get_slack_bot_config

logger = logging.getLogger(__name__)


class TaskState(Enum):
    """Quantum-inspired task states with superposition."""
    SUPERPOSITION = "superposition"  # Task exists in multiple potential states
    COLLAPSED = "collapsed"          # Task state has been observed/determined
    ENTANGLED = "entangled"         # Task depends on other tasks
    DECOHERENT = "decoherent"       # Task has lost quantum properties
    COMPLETED = "completed"         # Task is finished
    FAILED = "failed"               # Task failed execution


class TaskPriority(Enum):
    """Priority levels with quantum weighting."""
    CRITICAL = ("critical", 1.0)
    HIGH = ("high", 0.8)
    MEDIUM = ("medium", 0.6)
    LOW = ("low", 0.4)
    BACKGROUND = ("background", 0.2)
    
    def __init__(self, name: str, weight: float):
        self.priority_name = name
        self.quantum_weight = weight


@dataclass
class QuantumTask:
    """A quantum-inspired task with superposition and entanglement properties."""
    
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    state: TaskState = TaskState.SUPERPOSITION
    priority: TaskPriority = TaskPriority.MEDIUM
    
    # Quantum properties
    probability_amplitudes: Dict[str, float] = field(default_factory=dict)
    entangled_tasks: Set[str] = field(default_factory=set)
    superposition_states: List[str] = field(default_factory=list)
    
    # Execution properties
    executor: Optional[Callable] = None
    dependencies: List[str] = field(default_factory=list)
    artifacts: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    execution_time: timedelta = field(default_factory=timedelta)
    retry_count: int = 0
    max_retries: int = 3
    
    def collapse_state(self, observed_state: str) -> None:
        """Collapse quantum superposition to observed state."""
        if self.state == TaskState.SUPERPOSITION:
            self.state = TaskState.COLLAPSED
            self.superposition_states = [observed_state]
            self.updated_at = datetime.now()
            logger.info(f"Task {self.id} collapsed to state: {observed_state}")
    
    def entangle_with(self, other_task: 'QuantumTask') -> None:
        """Create quantum entanglement between tasks."""
        self.entangled_tasks.add(other_task.id)
        other_task.entangled_tasks.add(self.id)
        self.state = TaskState.ENTANGLED
        other_task.state = TaskState.ENTANGLED
        logger.info(f"Tasks {self.id} and {other_task.id} are now entangled")
    
    def calculate_execution_probability(self) -> float:
        """Calculate probability of successful execution based on quantum state."""
        base_probability = self.priority.quantum_weight
        
        # Adjust for retry count
        retry_penalty = 0.1 * self.retry_count
        
        # Adjust for dependencies
        dependency_factor = 1.0 - (0.05 * len(self.dependencies))
        
        # Adjust for entangled tasks
        entanglement_factor = 1.0 - (0.02 * len(self.entangled_tasks))
        
        probability = base_probability * dependency_factor * entanglement_factor - retry_penalty
        return max(0.0, min(1.0, probability))


class QuantumTaskPlanner:
    """Quantum-inspired task planning and execution engine."""
    
    def __init__(self, max_concurrent_tasks: int = 5, coherence_time: int = 300):
        self.tasks: Dict[str, QuantumTask] = {}
        self.execution_queue: List[QuantumTask] = []
        self.completed_tasks: List[QuantumTask] = []
        self.failed_tasks: List[QuantumTask] = []
        
        self.max_concurrent_tasks = max_concurrent_tasks
        self.coherence_time = coherence_time  # Time before decoherence
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent_tasks)
        
        self.metrics = get_global_metrics()
        self.cache = get_cache_manager()
        self.logger = StructuredLogger("quantum_task_planner")
        
        # Quantum state management
        self._lock = threading.Lock()
        self._running_tasks: Set[str] = set()
        self._last_measurement = time.time()
        
        logger.info(f"Quantum Task Planner initialized with {max_concurrent_tasks} max concurrent tasks")
    
    def create_task(
        self,
        name: str,
        description: str,
        executor: Callable,
        priority: TaskPriority = TaskPriority.MEDIUM,
        dependencies: Optional[List[str]] = None,
        superposition_states: Optional[List[str]] = None
    ) -> QuantumTask:
        """Create a new quantum task in superposition state."""
        task = QuantumTask(
            name=name,
            description=description,
            priority=priority,
            executor=executor,
            dependencies=dependencies or [],
            superposition_states=superposition_states or ["ready", "blocked", "waiting"]
        )
        
        # Initialize probability amplitudes
        num_states = len(task.superposition_states)
        base_amplitude = 1.0 / num_states
        task.probability_amplitudes = {
            state: base_amplitude for state in task.superposition_states
        }
        
        with self._lock:
            self.tasks[task.id] = task
            
        self.logger.log_event("task_created", {
            "task_id": task.id,
            "name": task.name,
            "priority": task.priority.priority_name,
            "states": task.superposition_states
        })
        
        return task
    
    def measure_task_state(self, task_id: str) -> Optional[str]:
        """Measure task state, causing wave function collapse."""
        with self._lock:
            task = self.tasks.get(task_id)
            if not task or task.state != TaskState.SUPERPOSITION:
                return None
            
            # Quantum measurement - collapse to most probable state
            max_amplitude = max(task.probability_amplitudes.values())
            observed_states = [
                state for state, amplitude in task.probability_amplitudes.items()
                if amplitude == max_amplitude
            ]
            
            # If multiple states have same probability, select first one
            observed_state = observed_states[0]
            task.collapse_state(observed_state)
            
            self._last_measurement = time.time()
            return observed_state
    
    def entangle_tasks(self, task1_id: str, task2_id: str) -> bool:
        """Create quantum entanglement between two tasks."""
        with self._lock:
            task1 = self.tasks.get(task1_id)
            task2 = self.tasks.get(task2_id)
            
            if not task1 or not task2:
                return False
            
            task1.entangle_with(task2)
            return True
    
    def schedule_task(self, task_id: str) -> bool:
        """Schedule a task for execution using quantum probability."""
        with self._lock:
            task = self.tasks.get(task_id)
            if not task:
                return False
            
            # Check dependencies
            for dep_id in task.dependencies:
                dep_task = self.tasks.get(dep_id)
                if not dep_task or dep_task.state != TaskState.COMPLETED:
                    logger.debug(f"Task {task_id} blocked by dependency {dep_id}")
                    return False
            
            # Calculate execution probability
            exec_probability = task.calculate_execution_probability()
            
            if exec_probability > 0.5:  # Quantum threshold
                self.execution_queue.append(task)
                task.state = TaskState.COLLAPSED
                self.logger.log_event("task_scheduled", {
                    "task_id": task.id,
                    "probability": exec_probability
                })
                return True
            
            return False
    
    async def execute_task(self, task: QuantumTask) -> bool:
        """Execute a quantum task with error handling and metrics."""
        start_time = time.time()
        
        try:
            with self._lock:
                self._running_tasks.add(task.id)
            
            self.logger.log_event("task_execution_started", {
                "task_id": task.id,
                "name": task.name
            })
            
            # Execute the task
            if asyncio.iscoroutinefunction(task.executor):
                result = await task.executor(task)
            else:
                result = task.executor(task)
            
            # Handle result
            if result:
                task.state = TaskState.COMPLETED
                task.artifacts['result'] = result
                task.execution_time = timedelta(seconds=time.time() - start_time)
                
                with self._lock:
                    self.completed_tasks.append(task)
                    self._running_tasks.discard(task.id)
                
                self.logger.log_event("task_completed", {
                    "task_id": task.id,
                    "execution_time": task.execution_time.total_seconds()
                })
                
                # Update entangled tasks
                self._update_entangled_tasks(task.id, TaskState.COMPLETED)
                return True
            else:
                raise Exception("Task execution returned False")
                
        except Exception as e:
            task.retry_count += 1
            execution_time = time.time() - start_time
            
            if task.retry_count >= task.max_retries:
                task.state = TaskState.FAILED
                with self._lock:
                    self.failed_tasks.append(task)
                    self._running_tasks.discard(task.id)
                
                self.logger.log_event("task_failed", {
                    "task_id": task.id,
                    "error": str(e),
                    "retry_count": task.retry_count,
                    "execution_time": execution_time
                })
                
                self._update_entangled_tasks(task.id, TaskState.FAILED)
                return False
            else:
                # Reset to superposition for retry
                task.state = TaskState.SUPERPOSITION
                with self._lock:
                    self._running_tasks.discard(task.id)
                
                self.logger.log_event("task_retry", {
                    "task_id": task.id,
                    "retry_count": task.retry_count,
                    "error": str(e)
                })
                
                return False
    
    def _update_entangled_tasks(self, task_id: str, new_state: TaskState) -> None:
        """Update quantum entangled tasks when one changes state."""
        task = self.tasks.get(task_id)
        if not task:
            return
        
        for entangled_id in task.entangled_tasks:
            entangled_task = self.tasks.get(entangled_id)
            if entangled_task and entangled_task.state == TaskState.ENTANGLED:
                # Quantum entanglement effect
                if new_state == TaskState.COMPLETED:
                    # Increase probability of success for entangled task
                    for state in entangled_task.probability_amplitudes:
                        if "ready" in state or "success" in state:
                            entangled_task.probability_amplitudes[state] *= 1.2
                elif new_state == TaskState.FAILED:
                    # Decrease probability for entangled task
                    for state in entangled_task.probability_amplitudes:
                        entangled_task.probability_amplitudes[state] *= 0.8
                
                # Normalize probabilities
                total = sum(entangled_task.probability_amplitudes.values())
                if total > 0:
                    for state in entangled_task.probability_amplitudes:
                        entangled_task.probability_amplitudes[state] /= total
    
    def check_decoherence(self) -> None:
        """Check for quantum decoherence and clean up old tasks."""
        current_time = time.time()
        decoherence_threshold = current_time - self.coherence_time
        
        decoherent_tasks = []
        with self._lock:
            for task in self.tasks.values():
                if (task.state in [TaskState.SUPERPOSITION, TaskState.ENTANGLED] and
                    task.created_at.timestamp() < decoherence_threshold):
                    task.state = TaskState.DECOHERENT
                    decoherent_tasks.append(task.id)
        
        if decoherent_tasks:
            self.logger.log_event("quantum_decoherence", {
                "decoherent_tasks": len(decoherent_tasks),
                "task_ids": decoherent_tasks[:10]  # Log first 10
            })
    
    async def run_execution_loop(self) -> None:
        """Main execution loop for quantum task processing."""
        logger.info("Starting quantum task execution loop")
        
        while True:
            try:
                # Check for decoherence
                self.check_decoherence()
                
                # Schedule ready tasks
                for task in list(self.tasks.values()):
                    if (task.state == TaskState.SUPERPOSITION and 
                        len(self._running_tasks) < self.max_concurrent_tasks):
                        self.schedule_task(task.id)
                
                # Execute queued tasks
                execution_futures = []
                while (self.execution_queue and 
                       len(self._running_tasks) < self.max_concurrent_tasks):
                    task = self.execution_queue.pop(0)
                    future = asyncio.create_task(self.execute_task(task))
                    execution_futures.append(future)
                
                # Wait for some tasks to complete
                if execution_futures:
                    done, pending = await asyncio.wait(
                        execution_futures, 
                        timeout=1.0, 
                        return_when=asyncio.FIRST_COMPLETED
                    )
                    
                    # Log completion metrics
                    for future in done:
                        try:
                            await future
                        except Exception as e:
                            logger.error(f"Task execution error: {e}")
                
                await asyncio.sleep(0.1)  # Prevent tight loop
                
            except Exception as e:
                logger.error(f"Error in quantum execution loop: {e}")
                await asyncio.sleep(1.0)
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get quantum system metrics and statistics."""
        with self._lock:
            total_tasks = len(self.tasks)
            running_tasks = len(self._running_tasks)
            completed_tasks = len(self.completed_tasks)
            failed_tasks = len(self.failed_tasks)
            
            superposition_tasks = sum(
                1 for task in self.tasks.values() 
                if task.state == TaskState.SUPERPOSITION
            )
            entangled_tasks = sum(
                1 for task in self.tasks.values() 
                if task.state == TaskState.ENTANGLED
            )
            
            avg_execution_time = 0.0
            if self.completed_tasks:
                total_time = sum(task.execution_time.total_seconds() for task in self.completed_tasks)
                avg_execution_time = total_time / len(self.completed_tasks)
        
        return {
            "total_tasks": total_tasks,
            "running_tasks": running_tasks,
            "completed_tasks": completed_tasks,
            "failed_tasks": failed_tasks,
            "superposition_tasks": superposition_tasks,
            "entangled_tasks": entangled_tasks,
            "queue_length": len(self.execution_queue),
            "average_execution_time": avg_execution_time,
            "coherence_time": self.coherence_time,
            "last_measurement": self._last_measurement
        }
    
    def cleanup(self) -> None:
        """Clean up resources and shutdown executor."""
        logger.info("Shutting down quantum task planner")
        self.executor.shutdown(wait=True)


# Global instance
_quantum_planner: Optional[QuantumTaskPlanner] = None


def get_quantum_planner() -> QuantumTaskPlanner:
    """Get or create global quantum task planner instance."""
    global _quantum_planner
    if _quantum_planner is None:
        config = get_slack_bot_config()
        max_concurrent = getattr(config, 'max_concurrent_tasks', 5)
        coherence_time = getattr(config, 'quantum_coherence_time', 300)
        _quantum_planner = QuantumTaskPlanner(max_concurrent, coherence_time)
    return _quantum_planner


# Utility functions for common task patterns
def create_simple_task(name: str, func: Callable, priority: TaskPriority = TaskPriority.MEDIUM) -> QuantumTask:
    """Create a simple quantum task."""
    planner = get_quantum_planner()
    return planner.create_task(name, f"Execute {name}", func, priority)


def create_dependent_tasks(tasks: List[Tuple[str, Callable]], priority: TaskPriority = TaskPriority.MEDIUM) -> List[QuantumTask]:
    """Create a chain of dependent quantum tasks."""
    planner = get_quantum_planner()
    created_tasks = []
    
    for i, (name, func) in enumerate(tasks):
        dependencies = [created_tasks[i-1].id] if i > 0 else []
        task = planner.create_task(
            name, f"Execute {name}", func, priority, dependencies
        )
        created_tasks.append(task)
    
    return created_tasks


def create_entangled_task_pair(
    name1: str, func1: Callable, 
    name2: str, func2: Callable,
    priority: TaskPriority = TaskPriority.MEDIUM
) -> Tuple[QuantumTask, QuantumTask]:
    """Create two quantum entangled tasks."""
    planner = get_quantum_planner()
    
    task1 = planner.create_task(name1, f"Execute {name1}", func1, priority)
    task2 = planner.create_task(name2, f"Execute {name2}", func2, priority)
    
    planner.entangle_tasks(task1.id, task2.id)
    
    return task1, task2