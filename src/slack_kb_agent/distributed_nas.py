"""Distributed Neural Architecture Search for massive parallel execution.

This module implements distributed NAS capabilities with support for:
- Multi-node distributed search
- Dynamic load balancing
- Fault-tolerant distributed execution
- Auto-scaling based on workload
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import hashlib
import uuid
import numpy as np

from .neural_architecture_search import (
    ArchitectureCandidate, 
    NeuralArchitectureSearchEngine,
    SearchStrategy,
    TPUv6Config
)
from .nas_reliability import NASReliabilityManager

logger = logging.getLogger(__name__)


class WorkerStatus(Enum):
    """Status of distributed workers."""
    IDLE = "idle"
    BUSY = "busy"
    FAILED = "failed"
    OFFLINE = "offline"
    INITIALIZING = "initializing"


class TaskPriority(Enum):
    """Task priority levels for distributed execution."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class LoadBalancingStrategy(Enum):
    """Load balancing strategies for task distribution."""
    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    PERFORMANCE_WEIGHTED = "performance_weighted"
    RESOURCE_AWARE = "resource_aware"
    INTELLIGENT = "intelligent"


@dataclass
class WorkerNode:
    """Represents a distributed worker node."""
    node_id: str
    hostname: str
    port: int
    status: WorkerStatus
    capabilities: Dict[str, Any]
    current_load: float  # 0.0 to 1.0
    performance_score: float  # Historical performance metric
    last_heartbeat: float
    active_tasks: Set[str]
    completed_tasks: int = 0
    failed_tasks: int = 0
    total_runtime: float = 0.0


@dataclass
class DistributedTask:
    """Represents a task for distributed execution."""
    task_id: str
    task_type: str
    priority: TaskPriority
    config: Dict[str, Any]
    estimated_runtime: float
    resource_requirements: Dict[str, float]
    retry_count: int = 0
    max_retries: int = 3
    created_at: float = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    assigned_node: Optional[str] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()


class DistributedNASCoordinator:
    """Coordinates distributed Neural Architecture Search across multiple nodes."""
    
    def __init__(
        self,
        coordinator_id: str = None,
        load_balancing_strategy: LoadBalancingStrategy = LoadBalancingStrategy.INTELLIGENT,
        auto_scaling: bool = True,
        max_nodes: int = 100
    ):
        self.coordinator_id = coordinator_id or str(uuid.uuid4())
        self.load_balancing_strategy = load_balancing_strategy
        self.auto_scaling = auto_scaling
        self.max_nodes = max_nodes
        
        # Worker management
        self.workers: Dict[str, WorkerNode] = {}
        self.task_queue: List[DistributedTask] = []
        self.completed_tasks: Dict[str, DistributedTask] = {}
        self.active_tasks: Dict[str, DistributedTask] = {}
        
        # Performance tracking
        self.global_performance_metrics = {
            "tasks_completed": 0,
            "tasks_failed": 0,
            "total_search_time": 0.0,
            "average_task_time": 0.0,
            "throughput_tasks_per_minute": 0.0,
            "worker_utilization": 0.0
        }
        
        # Auto-scaling parameters
        self.scaling_thresholds = {
            "scale_up_queue_threshold": 10,
            "scale_down_utilization_threshold": 0.3,
            "scale_up_utilization_threshold": 0.8,
            "min_nodes": 1,
            "max_nodes": max_nodes
        }
        
        logger.info(f"Initialized distributed NAS coordinator: {self.coordinator_id}")
    
    async def register_worker(
        self,
        hostname: str,
        port: int,
        capabilities: Dict[str, Any]
    ) -> str:
        """Register a new worker node."""
        node_id = f"{hostname}:{port}_{uuid.uuid4().hex[:8]}"
        
        worker = WorkerNode(
            node_id=node_id,
            hostname=hostname,
            port=port,
            status=WorkerStatus.INITIALIZING,
            capabilities=capabilities,
            current_load=0.0,
            performance_score=1.0,  # Initial neutral score
            last_heartbeat=time.time(),
            active_tasks=set()
        )
        
        self.workers[node_id] = worker
        
        # Initialize worker with health check
        if await self._health_check_worker(worker):
            worker.status = WorkerStatus.IDLE
            logger.info(f"Worker registered successfully: {node_id}")
        else:
            worker.status = WorkerStatus.FAILED
            logger.error(f"Worker registration failed: {node_id}")
        
        return node_id
    
    async def distributed_architecture_search(
        self,
        search_config: Dict[str, Any],
        num_candidates: int = 100,
        max_search_time: int = 3600,  # 1 hour
        search_strategy: SearchStrategy = SearchStrategy.ZERO_SHOT
    ) -> List[ArchitectureCandidate]:
        """Perform distributed architecture search."""
        logger.info(f"Starting distributed NAS with {num_candidates} candidates")
        start_time = time.time()
        
        # Ensure we have active workers
        active_workers = self._get_active_workers()
        if not active_workers:
            logger.error("No active workers available for distributed search")
            raise RuntimeError("No active workers available")
        
        # Create distributed tasks
        tasks = self._create_search_tasks(
            search_config, 
            num_candidates, 
            search_strategy,
            len(active_workers)
        )
        
        # Queue tasks for execution
        for task in tasks:
            self.task_queue.append(task)
        
        logger.info(f"Created {len(tasks)} distributed tasks")
        
        # Execute tasks with load balancing
        results = await self._execute_distributed_tasks(max_search_time)
        
        # Aggregate results
        all_candidates = []
        for task_id, result in results.items():
            if result and "candidates" in result:
                all_candidates.extend(result["candidates"])
        
        # Sort by performance
        all_candidates.sort(
            key=lambda x: x.predicted_accuracy * x.tpu_efficiency_score / max(x.predicted_latency, 1.0),
            reverse=True
        )
        
        search_time = time.time() - start_time
        self._update_performance_metrics(len(tasks), search_time)
        
        logger.info(f"Distributed search completed: {len(all_candidates)} candidates in {search_time:.2f}s")
        
        return all_candidates[:num_candidates]  # Return top candidates
    
    def _create_search_tasks(
        self,
        config: Dict[str, Any],
        num_candidates: int,
        strategy: SearchStrategy,
        num_workers: int
    ) -> List[DistributedTask]:
        """Create distributed search tasks."""
        tasks = []
        
        # Distribute candidates across workers
        candidates_per_worker = max(1, num_candidates // num_workers)
        
        for i in range(num_workers):
            start_idx = i * candidates_per_worker
            end_idx = min((i + 1) * candidates_per_worker, num_candidates)
            
            if start_idx >= num_candidates:
                break
            
            task_config = config.copy()
            task_config.update({
                "search_strategy": strategy.value,
                "num_candidates": end_idx - start_idx,
                "search_id": f"distributed_search_{i}",
                "worker_index": i
            })
            
            task = DistributedTask(
                task_id=f"search_task_{i}_{uuid.uuid4().hex[:8]}",
                task_type="architecture_search",
                priority=TaskPriority.NORMAL,
                config=task_config,
                estimated_runtime=self._estimate_task_runtime(task_config),
                resource_requirements={
                    "cpu_cores": 4,
                    "memory_gb": 8,
                    "gpu_memory_gb": 16
                }
            )
            
            tasks.append(task)
        
        return tasks
    
    async def _execute_distributed_tasks(self, max_time: int) -> Dict[str, Any]:
        """Execute tasks across distributed workers."""
        start_time = time.time()
        results = {}
        
        # Start task scheduler
        scheduler_task = asyncio.create_task(self._task_scheduler())
        
        # Monitor progress
        while (time.time() - start_time) < max_time:
            # Check if all tasks are complete
            if not self.task_queue and not self.active_tasks:
                break
            
            # Auto-scaling check
            if self.auto_scaling:
                await self._check_auto_scaling()
            
            # Health check workers
            await self._health_check_all_workers()
            
            # Update metrics
            self._update_real_time_metrics()
            
            await asyncio.sleep(1.0)  # Check every second
        
        # Cancel scheduler
        scheduler_task.cancel()
        
        # Collect results
        for task_id, task in self.completed_tasks.items():
            if task.result:
                results[task_id] = task.result
        
        return results
    
    async def _task_scheduler(self):
        """Main task scheduling loop."""
        while True:
            try:
                if self.task_queue:
                    # Get next task
                    task = self._get_next_task()
                    if task:
                        # Find best worker
                        worker = self._select_worker(task)
                        if worker:
                            # Assign task
                            await self._assign_task_to_worker(task, worker)
                
                await asyncio.sleep(0.1)  # Small delay to prevent busy loop
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Task scheduler error: {e}")
                await asyncio.sleep(1.0)
    
    def _get_next_task(self) -> Optional[DistributedTask]:
        """Get next task from queue based on priority."""
        if not self.task_queue:
            return None
        
        # Sort by priority and creation time
        priority_order = [TaskPriority.CRITICAL, TaskPriority.HIGH, TaskPriority.NORMAL, TaskPriority.LOW]
        
        for priority in priority_order:
            for task in self.task_queue:
                if task.priority == priority:
                    self.task_queue.remove(task)
                    return task
        
        # Fallback to first task
        return self.task_queue.pop(0)
    
    def _select_worker(self, task: DistributedTask) -> Optional[WorkerNode]:
        """Select best worker for task based on load balancing strategy."""
        available_workers = [
            w for w in self.workers.values() 
            if w.status == WorkerStatus.IDLE and self._worker_can_handle_task(w, task)
        ]
        
        if not available_workers:
            return None
        
        if self.load_balancing_strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return available_workers[len(self.active_tasks) % len(available_workers)]
        
        elif self.load_balancing_strategy == LoadBalancingStrategy.LEAST_LOADED:
            return min(available_workers, key=lambda w: w.current_load)
        
        elif self.load_balancing_strategy == LoadBalancingStrategy.PERFORMANCE_WEIGHTED:
            return max(available_workers, key=lambda w: w.performance_score)
        
        elif self.load_balancing_strategy == LoadBalancingStrategy.RESOURCE_AWARE:
            return self._select_resource_optimal_worker(available_workers, task)
        
        elif self.load_balancing_strategy == LoadBalancingStrategy.INTELLIGENT:
            return self._select_intelligent_worker(available_workers, task)
        
        return available_workers[0]  # Fallback
    
    def _select_resource_optimal_worker(
        self, 
        workers: List[WorkerNode], 
        task: DistributedTask
    ) -> WorkerNode:
        """Select worker with optimal resource match."""
        def resource_score(worker: WorkerNode) -> float:
            caps = worker.capabilities
            reqs = task.resource_requirements
            
            score = 0.0
            for resource, required in reqs.items():
                available = caps.get(resource, 0)
                if available >= required:
                    # Efficiency bonus for not over-provisioning
                    score += 1.0 - (available - required) / max(available, 1.0)
                else:
                    # Penalty for insufficient resources
                    score -= 1.0
            
            return score
        
        return max(workers, key=resource_score)
    
    def _select_intelligent_worker(
        self, 
        workers: List[WorkerNode], 
        task: DistributedTask
    ) -> WorkerNode:
        """Select worker using intelligent multi-factor scoring."""
        def intelligent_score(worker: WorkerNode) -> float:
            # Performance factor (0.4 weight)
            performance_factor = worker.performance_score * 0.4
            
            # Load factor (0.3 weight) - prefer less loaded
            load_factor = (1.0 - worker.current_load) * 0.3
            
            # Success rate factor (0.2 weight)
            total_tasks = worker.completed_tasks + worker.failed_tasks
            success_rate = worker.completed_tasks / max(total_tasks, 1)
            success_factor = success_rate * 0.2
            
            # Resource efficiency factor (0.1 weight)
            caps = worker.capabilities
            reqs = task.resource_requirements
            
            resource_efficiency = 0.0
            for resource, required in reqs.items():
                available = caps.get(resource, 0)
                if available >= required:
                    efficiency = required / available
                    resource_efficiency += efficiency
            
            resource_factor = (resource_efficiency / max(len(reqs), 1)) * 0.1
            
            return performance_factor + load_factor + success_factor + resource_factor
        
        return max(workers, key=intelligent_score)
    
    async def _assign_task_to_worker(self, task: DistributedTask, worker: WorkerNode):
        """Assign task to worker."""
        try:
            # Update task state
            task.assigned_node = worker.node_id
            task.started_at = time.time()
            
            # Update worker state
            worker.status = WorkerStatus.BUSY
            worker.active_tasks.add(task.task_id)
            
            # Move to active tasks
            self.active_tasks[task.task_id] = task
            
            # Send task to worker (simulated)
            success = await self._send_task_to_worker(task, worker)
            
            if success:
                logger.info(f"Task {task.task_id} assigned to worker {worker.node_id}")
                
                # Start monitoring task
                asyncio.create_task(self._monitor_task(task, worker))
            else:
                # Assignment failed, return task to queue
                await self._handle_task_failure(task, worker, "Failed to assign task")
                
        except Exception as e:
            logger.error(f"Error assigning task {task.task_id}: {e}")
            await self._handle_task_failure(task, worker, str(e))
    
    async def _send_task_to_worker(self, task: DistributedTask, worker: WorkerNode) -> bool:
        """Send task to worker (simulated network communication)."""
        try:
            # Simulate network delay
            await asyncio.sleep(0.1)
            
            # Simulate task execution on worker
            if worker.status != WorkerStatus.FAILED:
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to send task to worker {worker.node_id}: {e}")
            return False
    
    async def _monitor_task(self, task: DistributedTask, worker: WorkerNode):
        """Monitor task execution."""
        try:
            # Simulate task execution time
            execution_time = task.estimated_runtime + np.random.normal(0, task.estimated_runtime * 0.2)
            execution_time = max(1.0, execution_time)  # Minimum 1 second
            
            await asyncio.sleep(execution_time)
            
            # Simulate task completion
            success_probability = worker.performance_score * 0.9  # 90% base success rate
            if np.random.random() < success_probability:
                # Task succeeded
                await self._handle_task_completion(task, worker, {"candidates": []})
            else:
                # Task failed
                await self._handle_task_failure(task, worker, "Simulated task failure")
                
        except Exception as e:
            logger.error(f"Error monitoring task {task.task_id}: {e}")
            await self._handle_task_failure(task, worker, str(e))
    
    async def _handle_task_completion(
        self, 
        task: DistributedTask, 
        worker: WorkerNode, 
        result: Any
    ):
        """Handle successful task completion."""
        try:
            # Update task
            task.completed_at = time.time()
            task.result = result
            
            # Update worker
            worker.status = WorkerStatus.IDLE
            worker.active_tasks.discard(task.task_id)
            worker.completed_tasks += 1
            
            # Calculate performance score update
            execution_time = task.completed_at - task.started_at
            expected_time = task.estimated_runtime
            performance_ratio = expected_time / max(execution_time, 0.1)
            
            # Update worker performance score (exponential moving average)
            worker.performance_score = 0.9 * worker.performance_score + 0.1 * performance_ratio
            worker.total_runtime += execution_time
            
            # Move to completed tasks
            self.active_tasks.pop(task.task_id, None)
            self.completed_tasks[task.task_id] = task
            
            logger.info(f"Task {task.task_id} completed successfully on worker {worker.node_id}")
            
        except Exception as e:
            logger.error(f"Error handling task completion: {e}")
    
    async def _handle_task_failure(
        self, 
        task: DistributedTask, 
        worker: WorkerNode, 
        error_message: str
    ):
        """Handle task failure."""
        try:
            # Update task
            task.error = error_message
            task.retry_count += 1
            
            # Update worker
            worker.status = WorkerStatus.IDLE  # Worker might still be functional
            worker.active_tasks.discard(task.task_id)
            worker.failed_tasks += 1
            
            # Reduce worker performance score
            worker.performance_score = max(0.1, worker.performance_score * 0.8)
            
            # Remove from active tasks
            self.active_tasks.pop(task.task_id, None)
            
            # Retry logic
            if task.retry_count <= task.max_retries:
                logger.warning(f"Task {task.task_id} failed, retrying ({task.retry_count}/{task.max_retries})")
                self.task_queue.append(task)  # Re-queue for retry
            else:
                logger.error(f"Task {task.task_id} failed permanently: {error_message}")
                self.completed_tasks[task.task_id] = task  # Mark as completed (failed)
            
        except Exception as e:
            logger.error(f"Error handling task failure: {e}")
    
    def _get_active_workers(self) -> List[WorkerNode]:
        """Get list of active (non-failed) workers."""
        return [w for w in self.workers.values() if w.status != WorkerStatus.OFFLINE]
    
    def _worker_can_handle_task(self, worker: WorkerNode, task: DistributedTask) -> bool:
        """Check if worker can handle the given task."""
        caps = worker.capabilities
        reqs = task.resource_requirements
        
        for resource, required in reqs.items():
            if caps.get(resource, 0) < required:
                return False
        
        return True
    
    def _estimate_task_runtime(self, config: Dict[str, Any]) -> float:
        """Estimate task runtime based on configuration."""
        base_time = 30.0  # 30 seconds base
        num_candidates = config.get("num_candidates", 10)
        
        # Scale with number of candidates
        return base_time * (num_candidates / 10.0)
    
    async def _health_check_worker(self, worker: WorkerNode) -> bool:
        """Perform health check on worker."""
        try:
            # Simulate network health check
            await asyncio.sleep(0.1)
            
            # Update heartbeat
            worker.last_heartbeat = time.time()
            
            # Simple health check logic
            return worker.status != WorkerStatus.FAILED
            
        except Exception as e:
            logger.error(f"Health check failed for worker {worker.node_id}: {e}")
            return False
    
    async def _health_check_all_workers(self):
        """Health check all workers."""
        current_time = time.time()
        
        for worker in self.workers.values():
            # Check heartbeat timeout
            if current_time - worker.last_heartbeat > 60.0:  # 60 second timeout
                if worker.status != WorkerStatus.OFFLINE:
                    logger.warning(f"Worker {worker.node_id} heartbeat timeout")
                    worker.status = WorkerStatus.OFFLINE
            else:
                # Perform actual health check
                if worker.status == WorkerStatus.OFFLINE:
                    if await self._health_check_worker(worker):
                        worker.status = WorkerStatus.IDLE
                        logger.info(f"Worker {worker.node_id} recovered")
    
    async def _check_auto_scaling(self):
        """Check if auto-scaling is needed."""
        if not self.auto_scaling:
            return
        
        active_workers = len(self._get_active_workers())
        queue_size = len(self.task_queue)
        
        # Scale up conditions
        if (queue_size > self.scaling_thresholds["scale_up_queue_threshold"] or
            self._get_average_utilization() > self.scaling_thresholds["scale_up_utilization_threshold"]):
            
            if active_workers < self.scaling_thresholds["max_nodes"]:
                await self._scale_up()
        
        # Scale down conditions
        elif (queue_size == 0 and 
              self._get_average_utilization() < self.scaling_thresholds["scale_down_utilization_threshold"]):
            
            if active_workers > self.scaling_thresholds["min_nodes"]:
                await self._scale_down()
    
    async def _scale_up(self):
        """Scale up by requesting new worker nodes."""
        logger.info("Auto-scaling: Requesting additional worker nodes")
        # In real implementation, this would trigger cloud instance creation
        
        # Simulate new worker registration
        await self.register_worker(
            hostname=f"auto-worker-{len(self.workers)}",
            port=8000 + len(self.workers),
            capabilities={
                "cpu_cores": 8,
                "memory_gb": 16,
                "gpu_memory_gb": 32
            }
        )
    
    async def _scale_down(self):
        """Scale down by removing idle worker nodes."""
        idle_workers = [
            w for w in self.workers.values() 
            if w.status == WorkerStatus.IDLE and not w.active_tasks
        ]
        
        if idle_workers:
            worker_to_remove = idle_workers[0]
            logger.info(f"Auto-scaling: Removing idle worker {worker_to_remove.node_id}")
            worker_to_remove.status = WorkerStatus.OFFLINE
            # In real implementation, would terminate cloud instance
    
    def _get_average_utilization(self) -> float:
        """Calculate average worker utilization."""
        active_workers = self._get_active_workers()
        if not active_workers:
            return 0.0
        
        total_utilization = sum(
            len(w.active_tasks) / max(w.capabilities.get("max_concurrent_tasks", 4), 1)
            for w in active_workers
        )
        
        return total_utilization / len(active_workers)
    
    def _update_performance_metrics(self, num_tasks: int, total_time: float):
        """Update global performance metrics."""
        self.global_performance_metrics["tasks_completed"] += num_tasks
        self.global_performance_metrics["total_search_time"] += total_time
        
        total_tasks = self.global_performance_metrics["tasks_completed"]
        if total_tasks > 0:
            self.global_performance_metrics["average_task_time"] = (
                self.global_performance_metrics["total_search_time"] / total_tasks
            )
            
            self.global_performance_metrics["throughput_tasks_per_minute"] = (
                total_tasks / (self.global_performance_metrics["total_search_time"] / 60.0)
            )
        
        self.global_performance_metrics["worker_utilization"] = self._get_average_utilization()
    
    def _update_real_time_metrics(self):
        """Update real-time performance metrics."""
        self.global_performance_metrics["worker_utilization"] = self._get_average_utilization()
    
    def get_cluster_status(self) -> Dict[str, Any]:
        """Get comprehensive cluster status."""
        active_workers = self._get_active_workers()
        
        return {
            "coordinator_id": self.coordinator_id,
            "total_workers": len(self.workers),
            "active_workers": len(active_workers),
            "idle_workers": len([w for w in active_workers if w.status == WorkerStatus.IDLE]),
            "busy_workers": len([w for w in active_workers if w.status == WorkerStatus.BUSY]),
            "failed_workers": len([w for w in self.workers.values() if w.status == WorkerStatus.FAILED]),
            "queued_tasks": len(self.task_queue),
            "active_tasks": len(self.active_tasks),
            "completed_tasks": len(self.completed_tasks),
            "average_utilization": self._get_average_utilization(),
            "performance_metrics": self.global_performance_metrics,
            "auto_scaling_enabled": self.auto_scaling,
            "load_balancing_strategy": self.load_balancing_strategy.value
        }


# Factory function
def create_distributed_nas_coordinator(
    load_balancing_strategy: LoadBalancingStrategy = LoadBalancingStrategy.INTELLIGENT,
    auto_scaling: bool = True,
    max_nodes: int = 50
) -> DistributedNASCoordinator:
    """Create distributed NAS coordinator."""
    return DistributedNASCoordinator(
        load_balancing_strategy=load_balancing_strategy,
        auto_scaling=auto_scaling,
        max_nodes=max_nodes
    )


# Demo usage
async def demo_distributed_nas():
    """Demonstrate distributed NAS capabilities."""
    print("🌐 Distributed NAS Demo")
    
    # Create coordinator
    coordinator = create_distributed_nas_coordinator(
        load_balancing_strategy=LoadBalancingStrategy.INTELLIGENT,
        auto_scaling=True,
        max_nodes=10
    )
    
    # Register some workers
    worker_configs = [
        {"hostname": "worker1", "port": 8001, "capabilities": {"cpu_cores": 8, "memory_gb": 16, "gpu_memory_gb": 24}},
        {"hostname": "worker2", "port": 8002, "capabilities": {"cpu_cores": 16, "memory_gb": 32, "gpu_memory_gb": 48}},
        {"hostname": "worker3", "port": 8003, "capabilities": {"cpu_cores": 4, "memory_gb": 8, "gpu_memory_gb": 12}},
    ]
    
    print(f"\n📝 Registering {len(worker_configs)} workers...")
    for config in worker_configs:
        worker_id = await coordinator.register_worker(**config)
        print(f"Registered worker: {worker_id}")
    
    # Perform distributed search
    search_config = {
        "architecture_types": ["transformer", "cnn", "hybrid"],
        "max_layers": 24,
        "max_hidden_size": 2048
    }
    
    print(f"\n🔍 Starting distributed architecture search...")
    start_time = time.time()
    
    candidates = await coordinator.distributed_architecture_search(
        search_config=search_config,
        num_candidates=50,
        max_search_time=120,  # 2 minutes
        search_strategy=SearchStrategy.ZERO_SHOT
    )
    
    search_time = time.time() - start_time
    
    print(f"✅ Distributed search completed!")
    print(f"Found {len(candidates)} candidates in {search_time:.2f} seconds")
    
    # Show cluster status
    status = coordinator.get_cluster_status()
    print(f"\n📊 Cluster Status:")
    print(f"  Active workers: {status['active_workers']}/{status['total_workers']}")
    print(f"  Completed tasks: {status['completed_tasks']}")
    print(f"  Average utilization: {status['average_utilization']:.2f}")
    print(f"  Throughput: {status['performance_metrics']['throughput_tasks_per_minute']:.1f} tasks/min")


if __name__ == "__main__":
    asyncio.run(demo_distributed_nas())