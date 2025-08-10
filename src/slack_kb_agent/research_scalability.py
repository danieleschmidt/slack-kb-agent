"""Advanced Research Scalability and Global Distribution System.

This module implements comprehensive scalability mechanisms for research workloads,
including adaptive resource management, intelligent load balancing, and global optimization.
"""

import asyncio
import json
import time
import logging
import hashlib
import numpy as np
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any, Callable, Union
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import threading
import multiprocessing
import psutil
import statistics
import math

logger = logging.getLogger(__name__)


class ScalingStrategy(Enum):
    """Strategies for scaling research workloads."""
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"
    HYBRID = "hybrid"
    ADAPTIVE = "adaptive"


class ResourceType(Enum):
    """Types of computing resources."""
    CPU_CORE = "cpu_core"
    MEMORY_GB = "memory_gb"
    GPU_UNIT = "gpu_unit"
    STORAGE_GB = "storage_gb"
    NETWORK_MBPS = "network_mbps"


class RegionType(Enum):
    """Geographic regions for global distribution."""
    US_EAST = "us_east"
    US_WEST = "us_west"
    EU_WEST = "eu_west"
    AP_SOUTHEAST = "ap_southeast"
    AP_NORTHEAST = "ap_northeast"


@dataclass
class ResourceCapacity:
    """Resource capacity specification."""
    cpu_cores: int = 0
    memory_gb: float = 0.0
    gpu_units: int = 0
    storage_gb: float = 0.0
    network_mbps: float = 0.0
    cost_per_hour: float = 0.0


@dataclass
class WorkloadMetrics:
    """Metrics for research workload performance."""
    throughput_ops_per_sec: float = 0.0
    latency_ms: float = 0.0
    resource_utilization: Dict[ResourceType, float] = field(default_factory=dict)
    queue_length: int = 0
    error_rate: float = 0.0
    cost_per_operation: float = 0.0
    scalability_score: float = 0.0


@dataclass
class RegionConfig:
    """Configuration for a geographic region."""
    region: RegionType
    available_capacity: ResourceCapacity
    latency_ms: float = 0.0
    compliance_zones: List[str] = field(default_factory=list)
    cost_multiplier: float = 1.0
    active: bool = True


class AdaptiveResourceManager:
    """Manages resources adaptively based on workload demands."""
    
    def __init__(self, initial_capacity: ResourceCapacity):
        self.base_capacity = initial_capacity
        self.current_capacity = initial_capacity
        self.utilization_history = deque(maxlen=100)
        self.scaling_history = deque(maxlen=50)
        self.prediction_model = self._initialize_prediction_model()
        self.resource_locks = threading.Lock()
        
    def _initialize_prediction_model(self) -> Dict[str, Any]:
        """Initialize resource demand prediction model."""
        return {
            "weights": np.random.random(6) * 0.1,  # Features: time, utilization, queue, throughput, latency, trend
            "bias": 0.0,
            "learning_rate": 0.01,
            "prediction_horizon": 300,  # 5 minutes
            "accuracy": 0.0
        }
    
    def monitor_resources(self) -> Dict[ResourceType, float]:
        """Monitor current resource utilization."""
        utilization = {}
        
        try:
            # CPU utilization
            cpu_percent = psutil.cpu_percent(interval=1)
            utilization[ResourceType.CPU_CORE] = cpu_percent / 100.0
            
            # Memory utilization
            memory = psutil.virtual_memory()
            utilization[ResourceType.MEMORY_GB] = memory.percent / 100.0
            
            # Storage utilization
            disk = psutil.disk_usage('/')
            utilization[ResourceType.STORAGE_GB] = disk.percent / 100.0
            
            # Network utilization (mock)
            utilization[ResourceType.NETWORK_MBPS] = np.random.random() * 0.5
            
            # GPU utilization (mock)
            utilization[ResourceType.GPU_UNIT] = np.random.random() * 0.7
            
        except Exception as e:
            logger.error(f"Error monitoring resources: {str(e)}")
            # Fallback to mock values
            for resource_type in ResourceType:
                utilization[resource_type] = np.random.random() * 0.6
        
        # Store utilization history
        self.utilization_history.append({
            "timestamp": time.time(),
            "utilization": utilization.copy()
        })
        
        return utilization
    
    def predict_resource_demand(self, horizon_seconds: int = 300) -> Dict[ResourceType, float]:
        """Predict future resource demand."""
        if len(self.utilization_history) < 10:
            # Not enough history, return current utilization
            current_util = self.monitor_resources()
            return current_util
        
        predictions = {}
        recent_history = list(self.utilization_history)[-20:]  # Last 20 data points
        
        for resource_type in ResourceType:
            # Extract time series for this resource
            values = [entry["utilization"].get(resource_type, 0) for entry in recent_history]
            timestamps = [entry["timestamp"] for entry in recent_history]
            
            # Simple trend-based prediction
            prediction = self._predict_with_trend(values, timestamps, horizon_seconds)
            predictions[resource_type] = max(0.0, min(1.0, prediction))
        
        return predictions
    
    def _predict_with_trend(self, values: List[float], timestamps: List[float], horizon: int) -> float:
        """Predict future value using trend analysis."""
        if len(values) < 3:
            return values[-1] if values else 0.0
        
        # Calculate trend using linear regression
        n = len(values)
        x = list(range(n))
        
        sum_x = sum(x)
        sum_y = sum(values)
        sum_xy = sum(xi * yi for xi, yi in zip(x, values))
        sum_x2 = sum(xi * xi for xi in x)
        
        if n * sum_x2 - sum_x * sum_x == 0:
            return values[-1]  # No trend
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        intercept = (sum_y - slope * sum_x) / n
        
        # Project into future
        future_x = n + horizon / 60  # Assuming 1-minute intervals
        prediction = slope * future_x + intercept
        
        # Add some noise/uncertainty
        uncertainty = abs(slope) * 0.1
        prediction += np.random.normal(0, uncertainty)
        
        return prediction
    
    def calculate_optimal_capacity(self, predicted_demand: Dict[ResourceType, float]) -> ResourceCapacity:
        """Calculate optimal resource capacity based on predicted demand."""
        # Add safety margin to predictions
        safety_margin = 0.2  # 20% buffer
        
        optimal_capacity = ResourceCapacity()
        
        # Calculate required capacity for each resource type
        cpu_required = predicted_demand.get(ResourceType.CPU_CORE, 0.5) * (1 + safety_margin)
        memory_required = predicted_demand.get(ResourceType.MEMORY_GB, 0.5) * (1 + safety_margin)
        gpu_required = predicted_demand.get(ResourceType.GPU_UNIT, 0.3) * (1 + safety_margin)
        storage_required = predicted_demand.get(ResourceType.STORAGE_GB, 0.4) * (1 + safety_margin)
        network_required = predicted_demand.get(ResourceType.NETWORK_MBPS, 0.3) * (1 + safety_margin)
        
        # Convert utilization percentages to absolute capacity
        optimal_capacity.cpu_cores = max(1, int(self.base_capacity.cpu_cores * cpu_required))
        optimal_capacity.memory_gb = max(1.0, self.base_capacity.memory_gb * memory_required)
        optimal_capacity.gpu_units = int(self.base_capacity.gpu_units * gpu_required)
        optimal_capacity.storage_gb = max(10.0, self.base_capacity.storage_gb * storage_required)
        optimal_capacity.network_mbps = max(10.0, self.base_capacity.network_mbps * network_required)
        
        # Calculate estimated cost
        optimal_capacity.cost_per_hour = self._calculate_capacity_cost(optimal_capacity)
        
        return optimal_capacity
    
    def _calculate_capacity_cost(self, capacity: ResourceCapacity) -> float:
        """Calculate hourly cost for given capacity."""
        # Mock cost calculation based on resource requirements
        cost = 0.0
        cost += capacity.cpu_cores * 0.05  # $0.05 per CPU core per hour
        cost += capacity.memory_gb * 0.01  # $0.01 per GB RAM per hour
        cost += capacity.gpu_units * 2.50  # $2.50 per GPU per hour
        cost += capacity.storage_gb * 0.0001  # $0.0001 per GB storage per hour
        cost += capacity.network_mbps * 0.001  # $0.001 per Mbps per hour
        
        return cost
    
    def scale_resources(self, target_capacity: ResourceCapacity) -> bool:
        """Scale resources to target capacity."""
        with self.resource_locks:
            scaling_start = time.time()
            
            logger.info(f"Scaling resources from {self.current_capacity} to {target_capacity}")
            
            # Calculate scaling requirements
            scaling_actions = self._calculate_scaling_actions(self.current_capacity, target_capacity)
            
            # Execute scaling actions
            scaling_success = self._execute_scaling_actions(scaling_actions)
            
            if scaling_success:
                self.current_capacity = target_capacity
                scaling_time = time.time() - scaling_start
                
                self.scaling_history.append({
                    "timestamp": time.time(),
                    "from_capacity": self.current_capacity,
                    "to_capacity": target_capacity,
                    "scaling_time": scaling_time,
                    "success": True
                })
                
                logger.info(f"Resource scaling completed in {scaling_time:.2f}s")
                return True
            else:
                logger.error("Resource scaling failed")
                return False
    
    def _calculate_scaling_actions(self, current: ResourceCapacity, target: ResourceCapacity) -> List[Dict[str, Any]]:
        """Calculate required scaling actions."""
        actions = []
        
        if target.cpu_cores != current.cpu_cores:
            actions.append({
                "resource": ResourceType.CPU_CORE,
                "action": "scale_up" if target.cpu_cores > current.cpu_cores else "scale_down",
                "from": current.cpu_cores,
                "to": target.cpu_cores
            })
        
        if target.memory_gb != current.memory_gb:
            actions.append({
                "resource": ResourceType.MEMORY_GB,
                "action": "scale_up" if target.memory_gb > current.memory_gb else "scale_down",
                "from": current.memory_gb,
                "to": target.memory_gb
            })
        
        if target.gpu_units != current.gpu_units:
            actions.append({
                "resource": ResourceType.GPU_UNIT,
                "action": "scale_up" if target.gpu_units > current.gpu_units else "scale_down",
                "from": current.gpu_units,
                "to": target.gpu_units
            })
        
        return actions
    
    def _execute_scaling_actions(self, actions: List[Dict[str, Any]]) -> bool:
        """Execute scaling actions."""
        # Mock scaling implementation
        for action in actions:
            resource_type = action["resource"]
            action_type = action["action"]
            
            logger.info(f"Executing {action_type} for {resource_type.value}")
            
            # Simulate scaling time
            time.sleep(0.1)  # Mock scaling delay
            
            # Mock success rate (95% success)
            if np.random.random() < 0.05:
                logger.error(f"Scaling action failed for {resource_type.value}")
                return False
        
        return True


class IntelligentLoadBalancer:
    """Intelligent load balancer with adaptive algorithms."""
    
    def __init__(self, regions: List[RegionConfig]):
        self.regions = {region.region: region for region in regions}
        self.traffic_history = defaultdict(lambda: deque(maxlen=100))
        self.performance_metrics = defaultdict(lambda: deque(maxlen=50))
        self.routing_algorithm = "adaptive_weighted"
        
    def route_request(self, request_metadata: Dict[str, Any]) -> RegionType:
        """Route request to optimal region."""
        if self.routing_algorithm == "adaptive_weighted":
            return self._adaptive_weighted_routing(request_metadata)
        elif self.routing_algorithm == "latency_based":
            return self._latency_based_routing(request_metadata)
        elif self.routing_algorithm == "load_based":
            return self._load_based_routing(request_metadata)
        else:
            return self._round_robin_routing()
    
    def _adaptive_weighted_routing(self, request_metadata: Dict[str, Any]) -> RegionType:
        """Adaptive weighted routing based on multiple factors."""
        region_scores = {}
        
        for region_type, region_config in self.regions.items():
            if not region_config.active:
                continue
            
            # Base score calculation
            score = 0.0
            
            # Factor 1: Latency (lower is better)
            latency_score = 1.0 - (region_config.latency_ms / 1000.0)  # Normalize to 0-1
            score += latency_score * 0.3
            
            # Factor 2: Current load (lower is better)
            current_load = self._get_current_load(region_type)
            load_score = 1.0 - current_load
            score += load_score * 0.25
            
            # Factor 3: Cost efficiency (lower cost multiplier is better)
            cost_score = 1.0 / region_config.cost_multiplier
            score += cost_score * 0.15
            
            # Factor 4: Compliance requirements
            compliance_score = self._check_compliance_match(request_metadata, region_config)
            score += compliance_score * 0.2
            
            # Factor 5: Historical performance
            perf_score = self._get_historical_performance(region_type)
            score += perf_score * 0.1
            
            region_scores[region_type] = score
        
        # Select region with highest score
        if region_scores:
            best_region = max(region_scores.keys(), key=lambda r: region_scores[r])
            
            # Update traffic history
            self.traffic_history[best_region].append({
                "timestamp": time.time(),
                "request_metadata": request_metadata
            })
            
            return best_region
        
        # Fallback to US_EAST if no regions available
        return RegionType.US_EAST
    
    def _latency_based_routing(self, request_metadata: Dict[str, Any]) -> RegionType:
        """Route based on minimum latency."""
        min_latency = float('inf')
        best_region = RegionType.US_EAST
        
        for region_type, region_config in self.regions.items():
            if region_config.active and region_config.latency_ms < min_latency:
                min_latency = region_config.latency_ms
                best_region = region_type
        
        return best_region
    
    def _load_based_routing(self, request_metadata: Dict[str, Any]) -> RegionType:
        """Route based on current load."""
        min_load = float('inf')
        best_region = RegionType.US_EAST
        
        for region_type, region_config in self.regions.items():
            if region_config.active:
                current_load = self._get_current_load(region_type)
                if current_load < min_load:
                    min_load = current_load
                    best_region = region_type
        
        return best_region
    
    def _round_robin_routing(self) -> RegionType:
        """Simple round-robin routing."""
        active_regions = [r for r, config in self.regions.items() if config.active]
        if not active_regions:
            return RegionType.US_EAST
        
        # Use timestamp to create pseudo-round-robin
        region_index = int(time.time()) % len(active_regions)
        return active_regions[region_index]
    
    def _get_current_load(self, region: RegionType) -> float:
        """Get current load for region (0.0 to 1.0)."""
        recent_traffic = list(self.traffic_history[region])[-10:]  # Last 10 requests
        time_window = 60  # 1 minute window
        
        current_time = time.time()
        recent_requests = [
            req for req in recent_traffic 
            if current_time - req["timestamp"] < time_window
        ]
        
        # Normalize load (assume 10 requests per minute is 100% load)
        load = min(1.0, len(recent_requests) / 10.0)
        return load
    
    def _check_compliance_match(self, request_metadata: Dict[str, Any], region_config: RegionConfig) -> float:
        """Check compliance requirements match."""
        required_compliance = request_metadata.get("compliance_requirements", [])
        
        if not required_compliance:
            return 1.0  # No requirements, all regions are fine
        
        matched_requirements = 0
        for requirement in required_compliance:
            if requirement in region_config.compliance_zones:
                matched_requirements += 1
        
        return matched_requirements / len(required_compliance) if required_compliance else 1.0
    
    def _get_historical_performance(self, region: RegionType) -> float:
        """Get historical performance score for region."""
        recent_metrics = list(self.performance_metrics[region])[-20:]
        
        if not recent_metrics:
            return 0.5  # Neutral score
        
        # Calculate average performance score
        avg_performance = statistics.mean([m.get("performance_score", 0.5) for m in recent_metrics])
        return avg_performance
    
    def update_region_performance(self, region: RegionType, metrics: WorkloadMetrics):
        """Update performance metrics for region."""
        performance_score = self._calculate_performance_score(metrics)
        
        self.performance_metrics[region].append({
            "timestamp": time.time(),
            "metrics": metrics,
            "performance_score": performance_score
        })
    
    def _calculate_performance_score(self, metrics: WorkloadMetrics) -> float:
        """Calculate overall performance score (0.0 to 1.0)."""
        score = 0.0
        
        # Throughput component (higher is better)
        throughput_score = min(1.0, metrics.throughput_ops_per_sec / 1000.0)  # Normalize to max 1000 ops/sec
        score += throughput_score * 0.3
        
        # Latency component (lower is better)
        latency_score = max(0.0, 1.0 - (metrics.latency_ms / 1000.0))  # Normalize to max 1000ms
        score += latency_score * 0.3
        
        # Resource utilization (optimal around 0.7)
        avg_utilization = statistics.mean(metrics.resource_utilization.values()) if metrics.resource_utilization else 0.5
        util_score = 1.0 - abs(avg_utilization - 0.7) / 0.7  # Penalty for being too high or too low
        score += util_score * 0.2
        
        # Error rate component (lower is better)
        error_score = max(0.0, 1.0 - metrics.error_rate)
        score += error_score * 0.2
        
        return max(0.0, min(1.0, score))


class GlobalScaleOptimizer:
    """Global scale optimizer for research workloads."""
    
    def __init__(self):
        self.resource_managers = {}
        self.load_balancer = None
        self.optimization_history = deque(maxlen=100)
        self.global_metrics = {}
        self.scaling_policies = {}
        
    def initialize_regions(self, region_configs: List[RegionConfig]):
        """Initialize regional resource managers."""
        for config in region_configs:
            self.resource_managers[config.region] = AdaptiveResourceManager(
                config.available_capacity
            )
        
        self.load_balancer = IntelligentLoadBalancer(region_configs)
        
        # Initialize scaling policies
        self._initialize_scaling_policies()
    
    def _initialize_scaling_policies(self):
        """Initialize auto-scaling policies."""
        self.scaling_policies = {
            "cpu_threshold_up": 0.8,    # Scale up when CPU > 80%
            "cpu_threshold_down": 0.3,  # Scale down when CPU < 30%
            "memory_threshold_up": 0.85, # Scale up when memory > 85%
            "memory_threshold_down": 0.4, # Scale down when memory < 40%
            "scaling_cooldown": 300,    # 5-minute cooldown between scaling actions
            "min_instances": 1,         # Minimum instances per region
            "max_instances": 20,        # Maximum instances per region
            "scale_up_factor": 1.5,     # Scale up by 50%
            "scale_down_factor": 0.7    # Scale down by 30%
        }
    
    def optimize_global_distribution(self, workload_forecast: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize resource distribution across regions."""
        optimization_start = time.time()
        
        # Collect current metrics from all regions
        global_state = self._collect_global_state()
        
        # Predict demand for each region
        regional_forecasts = self._predict_regional_demand(workload_forecast, global_state)
        
        # Calculate optimal resource allocation
        optimal_allocation = self._calculate_optimal_allocation(regional_forecasts)
        
        # Execute resource reallocation
        reallocation_results = self._execute_resource_reallocation(optimal_allocation)
        
        # Update load balancing weights
        self._update_load_balancing_weights(optimal_allocation)
        
        optimization_result = {
            "timestamp": datetime.now(),
            "optimization_time": time.time() - optimization_start,
            "global_state": global_state,
            "regional_forecasts": regional_forecasts,
            "optimal_allocation": optimal_allocation,
            "reallocation_results": reallocation_results,
            "total_cost": sum(alloc["cost_per_hour"] for alloc in optimal_allocation.values()),
            "predicted_performance": self._predict_global_performance(optimal_allocation)
        }
        
        self.optimization_history.append(optimization_result)
        return optimization_result
    
    def _collect_global_state(self) -> Dict[str, Any]:
        """Collect current state from all regions."""
        global_state = {
            "regions": {},
            "total_capacity": ResourceCapacity(),
            "total_utilization": {},
            "global_load": 0.0,
            "timestamp": time.time()
        }
        
        for region, manager in self.resource_managers.items():
            utilization = manager.monitor_resources()
            
            region_state = {
                "current_capacity": manager.current_capacity,
                "utilization": utilization,
                "load": self.load_balancer._get_current_load(region) if self.load_balancer else 0.5,
                "active": self.load_balancer.regions[region].active if self.load_balancer else True
            }
            
            global_state["regions"][region.value] = region_state
            
            # Aggregate global capacity
            global_state["total_capacity"].cpu_cores += manager.current_capacity.cpu_cores
            global_state["total_capacity"].memory_gb += manager.current_capacity.memory_gb
            global_state["total_capacity"].gpu_units += manager.current_capacity.gpu_units
        
        # Calculate global utilization
        if self.resource_managers:
            for resource_type in ResourceType:
                utilizations = [
                    state["utilization"].get(resource_type, 0.0)
                    for state in global_state["regions"].values()
                ]
                global_state["total_utilization"][resource_type.value] = statistics.mean(utilizations)
        
        return global_state
    
    def _predict_regional_demand(self, workload_forecast: Dict[str, Any], global_state: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """Predict resource demand for each region."""
        regional_forecasts = {}
        
        for region, manager in self.resource_managers.items():
            # Get base demand prediction from regional manager
            base_forecast = manager.predict_resource_demand()
            
            # Apply global workload distribution factors
            distribution_factor = self._calculate_distribution_factor(region, workload_forecast)
            
            # Adjust forecast based on distribution factor
            regional_forecast = {}
            for resource_type, demand in base_forecast.items():
                adjusted_demand = demand * distribution_factor
                regional_forecast[resource_type.value] = adjusted_demand
            
            # Add additional forecast metrics
            regional_forecast["expected_throughput"] = workload_forecast.get("total_throughput", 100) * distribution_factor
            regional_forecast["expected_latency"] = self._predict_latency(adjusted_demand, manager.current_capacity)
            regional_forecast["confidence"] = self._calculate_forecast_confidence(region, global_state)
            
            regional_forecasts[region.value] = regional_forecast
        
        return regional_forecasts
    
    def _calculate_distribution_factor(self, region: RegionType, workload_forecast: Dict[str, Any]) -> float:
        """Calculate how much of global workload should go to this region."""
        # Base factor: equal distribution
        base_factor = 1.0 / len(self.resource_managers)
        
        # Adjust based on region characteristics
        if self.load_balancer:
            region_config = self.load_balancer.regions[region]
            
            # Consider latency (lower latency regions get more traffic)
            latency_factor = 1.0 - (region_config.latency_ms / 1000.0)
            
            # Consider cost (lower cost regions get more traffic for batch workloads)
            cost_factor = 1.0 / region_config.cost_multiplier
            
            # Consider compliance (regions with required compliance get priority)
            compliance_requirements = workload_forecast.get("compliance_requirements", [])
            compliance_factor = 1.0
            if compliance_requirements:
                matched = sum(1 for req in compliance_requirements if req in region_config.compliance_zones)
                compliance_factor = 1.0 + (matched / len(compliance_requirements))
            
            # Weighted combination
            adjustment_factor = (
                latency_factor * 0.3 +
                cost_factor * 0.2 +
                compliance_factor * 0.5
            )
            
            return base_factor * adjustment_factor
        
        return base_factor
    
    def _predict_latency(self, resource_demand: Dict[ResourceType, float], current_capacity: ResourceCapacity) -> float:
        """Predict latency based on resource demand and capacity."""
        # Simple queuing theory model
        cpu_utilization = resource_demand.get(ResourceType.CPU_CORE, 0.5)
        memory_utilization = resource_demand.get(ResourceType.MEMORY_GB, 0.5)
        
        # Base latency increases exponentially with utilization
        cpu_latency = 50.0 / (1.0 - min(0.99, cpu_utilization))  # M/M/1 queue approximation
        memory_latency = 20.0 / (1.0 - min(0.99, memory_utilization))
        
        return cpu_latency + memory_latency
    
    def _calculate_forecast_confidence(self, region: RegionType, global_state: Dict[str, Any]) -> float:
        """Calculate confidence level for regional forecast."""
        # Base confidence
        confidence = 0.7
        
        # Increase confidence with more historical data
        manager = self.resource_managers[region]
        if len(manager.utilization_history) >= 50:
            confidence += 0.2
        
        # Decrease confidence with high volatility
        recent_utilizations = [
            entry["utilization"].get(ResourceType.CPU_CORE, 0.5)
            for entry in list(manager.utilization_history)[-10:]
        ]
        if len(recent_utilizations) > 1:
            volatility = statistics.stdev(recent_utilizations)
            confidence -= min(0.3, volatility)
        
        return max(0.1, min(1.0, confidence))
    
    def _calculate_optimal_allocation(self, regional_forecasts: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, Any]]:
        """Calculate optimal resource allocation across regions."""
        optimal_allocation = {}
        
        for region_name, forecast in regional_forecasts.items():
            region_type = RegionType(region_name)
            manager = self.resource_managers[region_type]
            
            # Convert forecast back to ResourceType dict
            resource_demand = {}
            for resource_name, demand in forecast.items():
                try:
                    resource_type = ResourceType(resource_name)
                    resource_demand[resource_type] = demand
                except ValueError:
                    continue  # Skip non-resource forecast items
            
            # Calculate optimal capacity
            optimal_capacity = manager.calculate_optimal_capacity(resource_demand)
            
            # Apply scaling policies
            scaled_capacity = self._apply_scaling_policies(
                region_type, manager.current_capacity, optimal_capacity
            )
            
            optimal_allocation[region_name] = {
                "current_capacity": manager.current_capacity,
                "optimal_capacity": optimal_capacity,
                "scaled_capacity": scaled_capacity,
                "cost_per_hour": scaled_capacity.cost_per_hour,
                "scaling_required": scaled_capacity != manager.current_capacity,
                "expected_performance": self._predict_regional_performance(scaled_capacity, forecast)
            }
        
        return optimal_allocation
    
    def _apply_scaling_policies(self, region: RegionType, current: ResourceCapacity, optimal: ResourceCapacity) -> ResourceCapacity:
        """Apply scaling policies to optimal capacity."""
        scaled = ResourceCapacity()
        
        # Apply scaling factors
        cpu_scale_factor = optimal.cpu_cores / max(1, current.cpu_cores)
        memory_scale_factor = optimal.memory_gb / max(1.0, current.memory_gb)
        
        # Limit scaling factors based on policies
        max_scale_up = self.scaling_policies["scale_up_factor"]
        max_scale_down = self.scaling_policies["scale_down_factor"]
        
        cpu_scale_factor = max(max_scale_down, min(max_scale_up, cpu_scale_factor))
        memory_scale_factor = max(max_scale_down, min(max_scale_up, memory_scale_factor))
        
        # Apply scaling
        scaled.cpu_cores = max(
            self.scaling_policies["min_instances"],
            min(self.scaling_policies["max_instances"], int(current.cpu_cores * cpu_scale_factor))
        )
        scaled.memory_gb = current.memory_gb * memory_scale_factor
        scaled.gpu_units = optimal.gpu_units  # GPU scaling is more discrete
        scaled.storage_gb = optimal.storage_gb
        scaled.network_mbps = optimal.network_mbps
        
        # Calculate cost
        scaled.cost_per_hour = self.resource_managers[region]._calculate_capacity_cost(scaled)
        
        return scaled
    
    def _predict_regional_performance(self, capacity: ResourceCapacity, forecast: Dict[str, float]) -> Dict[str, float]:
        """Predict performance metrics for given capacity and forecast."""
        return {
            "expected_throughput": forecast.get("expected_throughput", 100),
            "expected_latency": forecast.get("expected_latency", 200),
            "expected_utilization": statistics.mean([
                forecast.get(rt.value, 0.5) for rt in ResourceType
            ]),
            "scalability_score": min(1.0, capacity.cpu_cores / max(1, forecast.get("expected_throughput", 100) / 10))
        }
    
    def _execute_resource_reallocation(self, optimal_allocation: Dict[str, Dict[str, Any]]) -> Dict[str, bool]:
        """Execute resource reallocation across regions."""
        reallocation_results = {}
        
        for region_name, allocation in optimal_allocation.items():
            region_type = RegionType(region_name)
            manager = self.resource_managers[region_type]
            
            if allocation["scaling_required"]:
                scaled_capacity = allocation["scaled_capacity"]
                success = manager.scale_resources(scaled_capacity)
                reallocation_results[region_name] = success
                
                logger.info(f"Resource reallocation {'succeeded' if success else 'failed'} for region {region_name}")
            else:
                reallocation_results[region_name] = True  # No scaling needed
        
        return reallocation_results
    
    def _update_load_balancing_weights(self, optimal_allocation: Dict[str, Dict[str, Any]]):
        """Update load balancer weights based on optimal allocation."""
        if not self.load_balancer:
            return
        
        total_capacity = sum(
            alloc["scaled_capacity"].cpu_cores 
            for alloc in optimal_allocation.values()
        )
        
        for region_name, allocation in optimal_allocation.items():
            region_type = RegionType(region_name)
            region_capacity = allocation["scaled_capacity"].cpu_cores
            
            # Update region configuration with new weight
            if region_type in self.load_balancer.regions:
                weight = region_capacity / total_capacity if total_capacity > 0 else 0.0
                # Store weight for routing decisions (would be used in actual implementation)
                logger.info(f"Updated load balancing weight for {region_name}: {weight:.3f}")
    
    def _predict_global_performance(self, optimal_allocation: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """Predict global system performance."""
        regional_performances = [
            alloc["expected_performance"]
            for alloc in optimal_allocation.values()
        ]
        
        if not regional_performances:
            return {"global_throughput": 0, "global_latency": 1000, "global_utilization": 0}
        
        # Aggregate performance across regions
        global_throughput = sum(perf["expected_throughput"] for perf in regional_performances)
        global_latency = statistics.mean(perf["expected_latency"] for perf in regional_performances)
        global_utilization = statistics.mean(perf["expected_utilization"] for perf in regional_performances)
        
        return {
            "global_throughput": global_throughput,
            "global_latency": global_latency,
            "global_utilization": global_utilization,
            "scalability_score": statistics.mean(perf["scalability_score"] for perf in regional_performances)
        }
    
    def get_scaling_recommendations(self) -> Dict[str, Any]:
        """Get current scaling recommendations."""
        recommendations = {
            "timestamp": datetime.now(),
            "recommendations": [],
            "global_metrics": self._collect_global_state(),
            "cost_optimization_opportunities": [],
            "performance_improvements": []
        }
        
        for region, manager in self.resource_managers.items():
            # Get current utilization
            utilization = manager.monitor_resources()
            
            # Check if scaling is recommended
            cpu_util = utilization.get(ResourceType.CPU_CORE, 0.5)
            memory_util = utilization.get(ResourceType.MEMORY_GB, 0.5)
            
            if cpu_util > self.scaling_policies["cpu_threshold_up"]:
                recommendations["recommendations"].append({
                    "region": region.value,
                    "action": "scale_up",
                    "resource": "cpu",
                    "current_utilization": cpu_util,
                    "reason": "High CPU utilization",
                    "priority": "high" if cpu_util > 0.9 else "medium"
                })
            elif cpu_util < self.scaling_policies["cpu_threshold_down"]:
                recommendations["recommendations"].append({
                    "region": region.value,
                    "action": "scale_down",
                    "resource": "cpu",
                    "current_utilization": cpu_util,
                    "reason": "Low CPU utilization - cost optimization opportunity",
                    "priority": "low"
                })
            
            if memory_util > self.scaling_policies["memory_threshold_up"]:
                recommendations["recommendations"].append({
                    "region": region.value,
                    "action": "scale_up",
                    "resource": "memory",
                    "current_utilization": memory_util,
                    "reason": "High memory utilization",
                    "priority": "high"
                })
        
        return recommendations


# Global scalability system instances
_global_optimizer = None
_default_regions = None


def get_global_optimizer() -> GlobalScaleOptimizer:
    """Get global scale optimizer instance."""
    global _global_optimizer, _default_regions
    
    if _global_optimizer is None:
        _global_optimizer = GlobalScaleOptimizer()
        
        # Initialize with default regions if not already done
        if _default_regions is None:
            _default_regions = [
                RegionConfig(
                    region=RegionType.US_EAST,
                    available_capacity=ResourceCapacity(
                        cpu_cores=16, memory_gb=64, gpu_units=2, 
                        storage_gb=1000, network_mbps=1000
                    ),
                    latency_ms=50,
                    compliance_zones=["SOC2", "HIPAA"],
                    cost_multiplier=1.0
                ),
                RegionConfig(
                    region=RegionType.US_WEST,
                    available_capacity=ResourceCapacity(
                        cpu_cores=12, memory_gb=48, gpu_units=1,
                        storage_gb=800, network_mbps=800
                    ),
                    latency_ms=75,
                    compliance_zones=["SOC2"],
                    cost_multiplier=1.1
                ),
                RegionConfig(
                    region=RegionType.EU_WEST,
                    available_capacity=ResourceCapacity(
                        cpu_cores=20, memory_gb=80, gpu_units=3,
                        storage_gb=1200, network_mbps=1200
                    ),
                    latency_ms=120,
                    compliance_zones=["GDPR", "ISO27001"],
                    cost_multiplier=1.2
                ),
                RegionConfig(
                    region=RegionType.AP_SOUTHEAST,
                    available_capacity=ResourceCapacity(
                        cpu_cores=8, memory_gb=32, gpu_units=1,
                        storage_gb=600, network_mbps=600
                    ),
                    latency_ms=200,
                    compliance_zones=["PDPA"],
                    cost_multiplier=0.8
                )
            ]
        
        _global_optimizer.initialize_regions(_default_regions)
    
    return _global_optimizer


def optimize_research_scalability(workload_forecast: Dict[str, Any]) -> Dict[str, Any]:
    """Optimize research workload scalability across global regions."""
    optimizer = get_global_optimizer()
    return optimizer.optimize_global_distribution(workload_forecast)


def get_scaling_recommendations() -> Dict[str, Any]:
    """Get current scaling recommendations."""
    optimizer = get_global_optimizer()
    return optimizer.get_scaling_recommendations()


def monitor_global_performance() -> Dict[str, Any]:
    """Monitor global system performance."""
    optimizer = get_global_optimizer()
    
    # Collect current state
    global_state = optimizer._collect_global_state()
    
    # Get recent optimization results
    recent_optimizations = list(optimizer.optimization_history)[-5:]
    
    # Calculate performance trends
    performance_trend = "stable"
    if len(recent_optimizations) >= 2:
        latest_perf = recent_optimizations[-1]["predicted_performance"]["global_throughput"]
        previous_perf = recent_optimizations[-2]["predicted_performance"]["global_throughput"]
        
        if latest_perf > previous_perf * 1.1:
            performance_trend = "improving"
        elif latest_perf < previous_perf * 0.9:
            performance_trend = "declining"
    
    return {
        "timestamp": datetime.now(),
        "global_state": global_state,
        "recent_optimizations": len(recent_optimizations),
        "performance_trend": performance_trend,
        "total_regions": len(optimizer.resource_managers),
        "active_regions": sum(1 for region in optimizer.load_balancer.regions.values() if region.active),
        "total_cost_per_hour": global_state.get("total_cost", 0.0),
        "optimization_recommendations": optimizer.get_scaling_recommendations()
    }