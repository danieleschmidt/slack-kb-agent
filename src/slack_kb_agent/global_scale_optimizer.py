"""Global Scale Optimizer with Adaptive Resource Management and Auto-Scaling.

This module implements advanced scaling capabilities including adaptive resource management,
geographic distribution, load balancing, and intelligent auto-scaling for the Slack KB Agent.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

from .adaptive_learning_engine import get_adaptive_learning_engine
from .cache import get_cache_manager
from .monitoring import StructuredLogger
from .predictive_monitoring import get_predictive_monitoring

logger = logging.getLogger(__name__)


class ScalingStrategy(Enum):
    """Different scaling strategies available."""
    REACTIVE = ("reactive", "Scale based on current metrics")
    PREDICTIVE = ("predictive", "Scale based on predicted load")
    ADAPTIVE = ("adaptive", "Learn optimal scaling patterns")
    HYBRID = ("hybrid", "Combine multiple strategies")

    def __init__(self, name: str, description: str):
        self.strategy_name = name
        self.description = description


class ResourceType(Enum):
    """Types of resources that can be scaled."""
    CPU = "cpu"
    MEMORY = "memory"
    STORAGE = "storage"
    NETWORK = "network"
    INSTANCES = "instances"
    CACHE = "cache"
    DATABASE_CONNECTIONS = "database_connections"


@dataclass
class ResourceProfile:
    """Profile defining resource requirements and constraints."""
    resource_type: ResourceType
    current_allocation: float
    target_allocation: float
    min_allocation: float
    max_allocation: float
    scaling_factor: float = 1.5
    cooldown_period: timedelta = field(default_factory=lambda: timedelta(minutes=5))
    last_scaled: Optional[datetime] = None

    def can_scale(self) -> bool:
        """Check if resource can be scaled based on cooldown period."""
        if self.last_scaled is None:
            return True
        return datetime.now() - self.last_scaled > self.cooldown_period

    def calculate_scale_amount(self, target_utilization: float = 0.7) -> float:
        """Calculate how much to scale based on target utilization."""
        if self.current_allocation == 0:
            return self.min_allocation

        current_utilization = min(1.0, self.target_allocation / self.current_allocation)
        if current_utilization > target_utilization:
            scale_factor = min(self.scaling_factor, self.max_allocation / self.current_allocation)
            return self.current_allocation * scale_factor
        elif current_utilization < target_utilization * 0.5:
            scale_factor = max(1.0 / self.scaling_factor, self.min_allocation / self.current_allocation)
            return self.current_allocation * scale_factor

        return self.current_allocation


@dataclass
class GeographicRegion:
    """Represents a geographic region for global distribution."""
    region_id: str
    region_name: str
    datacenter_locations: List[str]
    active_instances: int = 0
    max_instances: int = 10
    current_load: float = 0.0
    latency_to_users: Dict[str, float] = field(default_factory=dict)
    compliance_requirements: List[str] = field(default_factory=list)

    def get_capacity_utilization(self) -> float:
        """Get current capacity utilization."""
        return self.active_instances / self.max_instances if self.max_instances > 0 else 0

    def can_handle_additional_load(self, additional_load: float) -> bool:
        """Check if region can handle additional load."""
        projected_load = self.current_load + additional_load
        return projected_load <= self.get_capacity_utilization() * 1.2  # 20% buffer


@dataclass
class ScalingEvent:
    """Records a scaling event for analysis and learning."""
    event_id: str
    timestamp: datetime
    resource_type: ResourceType
    scaling_action: str  # scale_up, scale_down, scale_out, scale_in
    previous_allocation: float
    new_allocation: float
    trigger_metrics: Dict[str, float]
    strategy_used: ScalingStrategy
    success: bool = True
    duration: Optional[timedelta] = None
    impact_metrics: Dict[str, float] = field(default_factory=dict)


class GlobalScaleOptimizer:
    """Advanced global scaling and resource optimization system."""

    def __init__(self):
        self.logger = StructuredLogger("global_scale_optimizer")
        self.cache = get_cache_manager()
        self.predictive_monitoring = get_predictive_monitoring()
        self.learning_engine = get_adaptive_learning_engine()

        # Resource management
        self.resource_profiles: Dict[ResourceType, ResourceProfile] = self._initialize_resource_profiles()
        self.geographic_regions: Dict[str, GeographicRegion] = self._initialize_regions()
        self.scaling_events: deque = deque(maxlen=1000)

        # Scaling configuration
        self.scaling_strategy = ScalingStrategy.HYBRID
        self.auto_scaling_enabled = True
        self.global_distribution_enabled = True

        # Performance targets
        self.performance_targets = {
            'response_time_ms': 200,
            'cpu_utilization': 0.7,
            'memory_utilization': 0.75,
            'error_rate': 0.01,
            'availability': 0.9999
        }

        # Load balancing weights
        self.load_balancing_weights = {
            'latency': 0.4,
            'capacity': 0.3,
            'cost': 0.2,
            'compliance': 0.1
        }

        # Scaling thresholds
        self.scale_up_thresholds = {
            ResourceType.CPU: 0.8,
            ResourceType.MEMORY: 0.85,
            ResourceType.INSTANCES: 0.9
        }

        self.scale_down_thresholds = {
            ResourceType.CPU: 0.3,
            ResourceType.MEMORY: 0.4,
            ResourceType.INSTANCES: 0.4
        }

        # Thread safety
        self._scaling_lock = threading.Lock()

        logger.info("Global Scale Optimizer initialized with hybrid scaling strategy")

    def _initialize_resource_profiles(self) -> Dict[ResourceType, ResourceProfile]:
        """Initialize resource profiles with default configurations."""
        return {
            ResourceType.CPU: ResourceProfile(
                resource_type=ResourceType.CPU,
                current_allocation=2.0,  # 2 CPU cores
                target_allocation=2.0,
                min_allocation=1.0,
                max_allocation=16.0,
                scaling_factor=1.5
            ),
            ResourceType.MEMORY: ResourceProfile(
                resource_type=ResourceType.MEMORY,
                current_allocation=4.0,  # 4 GB RAM
                target_allocation=4.0,
                min_allocation=2.0,
                max_allocation=32.0,
                scaling_factor=1.5
            ),
            ResourceType.INSTANCES: ResourceProfile(
                resource_type=ResourceType.INSTANCES,
                current_allocation=2,  # 2 instances
                target_allocation=2,
                min_allocation=1,
                max_allocation=20,
                scaling_factor=2.0,
                cooldown_period=timedelta(minutes=10)
            ),
            ResourceType.CACHE: ResourceProfile(
                resource_type=ResourceType.CACHE,
                current_allocation=1.0,  # 1 GB cache
                target_allocation=1.0,
                min_allocation=0.5,
                max_allocation=8.0,
                scaling_factor=2.0
            )
        }

    def _initialize_regions(self) -> Dict[str, GeographicRegion]:
        """Initialize geographic regions for global distribution."""
        return {
            'us-east-1': GeographicRegion(
                region_id='us-east-1',
                region_name='US East (Virginia)',
                datacenter_locations=['us-east-1a', 'us-east-1b', 'us-east-1c'],
                max_instances=10,
                compliance_requirements=['SOC2', 'GDPR']
            ),
            'us-west-2': GeographicRegion(
                region_id='us-west-2',
                region_name='US West (Oregon)',
                datacenter_locations=['us-west-2a', 'us-west-2b', 'us-west-2c'],
                max_instances=8,
                compliance_requirements=['SOC2']
            ),
            'eu-west-1': GeographicRegion(
                region_id='eu-west-1',
                region_name='Europe (Ireland)',
                datacenter_locations=['eu-west-1a', 'eu-west-1b', 'eu-west-1c'],
                max_instances=6,
                compliance_requirements=['GDPR', 'ISO27001']
            ),
            'ap-southeast-1': GeographicRegion(
                region_id='ap-southeast-1',
                region_name='Asia Pacific (Singapore)',
                datacenter_locations=['ap-southeast-1a', 'ap-southeast-1b'],
                max_instances=4,
                compliance_requirements=['SOC2']
            )
        }

    async def start_auto_scaling(self) -> None:
        """Start the auto-scaling monitoring and execution loop."""
        self.logger.info("Starting auto-scaling system")

        scaling_tasks = [
            self._continuous_resource_monitoring(),
            self._continuous_scaling_decisions(),
            self._continuous_load_balancing(),
            self._continuous_optimization_learning()
        ]

        await asyncio.gather(*scaling_tasks, return_exceptions=True)

    async def _continuous_resource_monitoring(self) -> None:
        """Continuously monitor resource utilization across all regions."""
        while True:
            try:
                if self.auto_scaling_enabled:
                    # Collect resource metrics from all regions
                    for region_id, region in self.geographic_regions.items():
                        metrics = await self._collect_region_metrics(region_id)
                        await self._update_region_state(region_id, metrics)

                    # Update resource profiles based on current utilization
                    await self._update_resource_profiles()

                await asyncio.sleep(30)  # Monitor every 30 seconds

            except Exception as e:
                logger.error(f"Error in resource monitoring: {e}")
                await asyncio.sleep(10)

    async def _continuous_scaling_decisions(self) -> None:
        """Continuously evaluate and execute scaling decisions."""
        while True:
            try:
                if self.auto_scaling_enabled:
                    # Evaluate scaling needs for each resource type
                    scaling_decisions = await self._evaluate_scaling_needs()

                    # Execute scaling decisions
                    for decision in scaling_decisions:
                        await self._execute_scaling_decision(decision)

                await asyncio.sleep(60)  # Evaluate every minute

            except Exception as e:
                logger.error(f"Error in scaling decisions: {e}")
                await asyncio.sleep(20)

    async def _continuous_load_balancing(self) -> None:
        """Continuously optimize load balancing across regions."""
        while True:
            try:
                if self.global_distribution_enabled:
                    # Analyze current load distribution
                    load_distribution = await self._analyze_load_distribution()

                    # Optimize load balancing weights
                    new_weights = await self._optimize_load_balancing(load_distribution)

                    if new_weights:
                        await self._apply_load_balancing_changes(new_weights)

                await asyncio.sleep(300)  # Optimize every 5 minutes

            except Exception as e:
                logger.error(f"Error in load balancing: {e}")
                await asyncio.sleep(60)

    async def _continuous_optimization_learning(self) -> None:
        """Continuously learn from scaling events to optimize future decisions."""
        while True:
            try:
                # Analyze recent scaling events for patterns
                optimization_insights = await self._analyze_scaling_patterns()

                # Update scaling parameters based on learnings
                if optimization_insights:
                    await self._apply_optimization_insights(optimization_insights)

                # Generate scaling recommendations
                recommendations = await self._generate_scaling_recommendations()
                if recommendations:
                    self.logger.info(f"Generated {len(recommendations)} scaling recommendations")

                await asyncio.sleep(1800)  # Learn every 30 minutes

            except Exception as e:
                logger.error(f"Error in optimization learning: {e}")
                await asyncio.sleep(300)

    async def _collect_region_metrics(self, region_id: str) -> Dict[str, float]:
        """Collect metrics for a specific region."""
        try:
            # Simulate region-specific metrics collection
            base_metrics = {
                'cpu_utilization': max(0, min(1, np.random.normal(0.6, 0.2))),
                'memory_utilization': max(0, min(1, np.random.normal(0.7, 0.15))),
                'network_utilization': max(0, min(1, np.random.normal(0.4, 0.1))),
                'request_rate': max(0, np.random.normal(100, 25)),
                'response_time': max(50, np.random.normal(300, 100)),
                'error_rate': max(0, min(1, np.random.normal(0.01, 0.005))),
                'active_connections': max(0, int(np.random.normal(50, 15)))
            }

            # Add region-specific variations
            region_factors = {
                'us-east-1': 1.0,  # Baseline
                'us-west-2': 0.8,  # Lower load
                'eu-west-1': 1.2,  # Higher load
                'ap-southeast-1': 0.6  # Much lower load
            }

            factor = region_factors.get(region_id, 1.0)
            for metric in ['cpu_utilization', 'memory_utilization', 'request_rate']:
                base_metrics[metric] *= factor

            return base_metrics

        except Exception as e:
            logger.error(f"Error collecting metrics for region {region_id}: {e}")
            return {}

    async def _update_region_state(self, region_id: str, metrics: Dict[str, float]) -> None:
        """Update region state based on collected metrics."""
        try:
            region = self.geographic_regions.get(region_id)
            if not region:
                return

            # Update region load and performance metrics
            region.current_load = metrics.get('cpu_utilization', 0.5)

            # Simulate latency measurements to different user populations
            region.latency_to_users = {
                'north_america': 20 if 'us-' in region_id else 150,
                'europe': 150 if 'us-' in region_id else (20 if 'eu-' in region_id else 200),
                'asia_pacific': 200 if 'ap-' not in region_id else 30
            }

            self.logger.debug(f"Updated region {region_id} state: load={region.current_load:.2f}")

        except Exception as e:
            logger.error(f"Error updating region {region_id} state: {e}")

    async def _update_resource_profiles(self) -> None:
        """Update resource profiles based on current utilization."""
        try:
            # Calculate aggregate metrics across all regions
            aggregate_metrics = await self._calculate_aggregate_metrics()

            # Update target allocations for each resource type
            for resource_type, profile in self.resource_profiles.items():
                current_utilization = aggregate_metrics.get(f'{resource_type.value}_utilization', 0.5)

                if current_utilization > self.scale_up_thresholds.get(resource_type, 0.8):
                    profile.target_allocation = min(
                        profile.max_allocation,
                        profile.current_allocation * profile.scaling_factor
                    )
                elif current_utilization < self.scale_down_thresholds.get(resource_type, 0.3):
                    profile.target_allocation = max(
                        profile.min_allocation,
                        profile.current_allocation / profile.scaling_factor
                    )

        except Exception as e:
            logger.error(f"Error updating resource profiles: {e}")

    async def _calculate_aggregate_metrics(self) -> Dict[str, float]:
        """Calculate aggregate metrics across all regions."""
        try:
            total_cpu = 0.0
            total_memory = 0.0
            total_requests = 0.0
            total_instances = 0

            for region_id, region in self.geographic_regions.items():
                region_metrics = await self._collect_region_metrics(region_id)

                # Weight by region capacity
                weight = region.active_instances if region.active_instances > 0 else 1

                total_cpu += region_metrics.get('cpu_utilization', 0) * weight
                total_memory += region_metrics.get('memory_utilization', 0) * weight
                total_requests += region_metrics.get('request_rate', 0)
                total_instances += region.active_instances

            total_weight = sum(r.active_instances if r.active_instances > 0 else 1
                             for r in self.geographic_regions.values())

            return {
                'cpu_utilization': total_cpu / total_weight if total_weight > 0 else 0,
                'memory_utilization': total_memory / total_weight if total_weight > 0 else 0,
                'instances_utilization': total_instances / sum(r.max_instances for r in self.geographic_regions.values()),
                'total_request_rate': total_requests
            }

        except Exception as e:
            logger.error(f"Error calculating aggregate metrics: {e}")
            return {}

    async def _evaluate_scaling_needs(self) -> List[Dict[str, Any]]:
        """Evaluate scaling needs based on current metrics and strategy."""
        scaling_decisions = []

        try:
            aggregate_metrics = await self._calculate_aggregate_metrics()

            for resource_type, profile in self.resource_profiles.items():
                if not profile.can_scale():
                    continue

                current_utilization = aggregate_metrics.get(f'{resource_type.value}_utilization', 0.5)

                # Determine if scaling is needed
                scale_decision = await self._make_scaling_decision(
                    resource_type, profile, current_utilization, aggregate_metrics
                )

                if scale_decision:
                    scaling_decisions.append(scale_decision)

        except Exception as e:
            logger.error(f"Error evaluating scaling needs: {e}")

        return scaling_decisions

    async def _make_scaling_decision(self, resource_type: ResourceType,
                                   profile: ResourceProfile, current_utilization: float,
                                   aggregate_metrics: Dict[str, float]) -> Optional[Dict[str, Any]]:
        """Make a scaling decision for a specific resource."""
        try:
            # Get thresholds for this resource type
            scale_up_threshold = self.scale_up_thresholds.get(resource_type, 0.8)
            scale_down_threshold = self.scale_down_thresholds.get(resource_type, 0.3)

            if current_utilization > scale_up_threshold:
                # Scale up decision
                new_allocation = profile.calculate_scale_amount(target_utilization=0.7)

                if new_allocation > profile.current_allocation:
                    return {
                        'resource_type': resource_type,
                        'action': 'scale_up',
                        'current_allocation': profile.current_allocation,
                        'target_allocation': new_allocation,
                        'trigger_utilization': current_utilization,
                        'trigger_metrics': aggregate_metrics,
                        'strategy': self.scaling_strategy,
                        'confidence': self._calculate_decision_confidence(current_utilization, scale_up_threshold)
                    }

            elif current_utilization < scale_down_threshold:
                # Scale down decision
                new_allocation = profile.calculate_scale_amount(target_utilization=0.5)

                if new_allocation < profile.current_allocation:
                    return {
                        'resource_type': resource_type,
                        'action': 'scale_down',
                        'current_allocation': profile.current_allocation,
                        'target_allocation': new_allocation,
                        'trigger_utilization': current_utilization,
                        'trigger_metrics': aggregate_metrics,
                        'strategy': self.scaling_strategy,
                        'confidence': self._calculate_decision_confidence(scale_down_threshold, current_utilization)
                    }

        except Exception as e:
            logger.error(f"Error making scaling decision for {resource_type}: {e}")

        return None

    def _calculate_decision_confidence(self, threshold: float, actual: float) -> float:
        """Calculate confidence in scaling decision based on how far past threshold."""
        distance = abs(actual - threshold)
        return min(1.0, distance * 2)  # Linear confidence based on distance from threshold

    async def _execute_scaling_decision(self, decision: Dict[str, Any]) -> None:
        """Execute a scaling decision."""
        try:
            with self._scaling_lock:
                resource_type = decision['resource_type']
                action = decision['action']
                target_allocation = decision['target_allocation']

                self.logger.info(f"Executing {action} for {resource_type.value}: {target_allocation}")

                # Create scaling event record
                event = ScalingEvent(
                    event_id=f"{resource_type.value}_{action}_{int(time.time())}",
                    timestamp=datetime.now(),
                    resource_type=resource_type,
                    scaling_action=action,
                    previous_allocation=decision['current_allocation'],
                    new_allocation=target_allocation,
                    trigger_metrics=decision['trigger_metrics'],
                    strategy_used=decision['strategy']
                )

                # Execute the actual scaling (simulated)
                success = await self._perform_resource_scaling(resource_type, target_allocation)

                if success:
                    # Update resource profile
                    profile = self.resource_profiles[resource_type]
                    profile.current_allocation = target_allocation
                    profile.last_scaled = datetime.now()

                    self.logger.info(f"Successfully scaled {resource_type.value} to {target_allocation}")

                    # Record successful event
                    event.success = True
                    self.scaling_events.append(event)

                    # Learn from this scaling event
                    await self.learning_engine.learn_from_performance(
                        {'scaling_success': 1.0, 'resource_efficiency': 0.85},
                        {
                            'resource_type': resource_type.value,
                            'scaling_action': action,
                            'allocation_change': target_allocation - decision['current_allocation']
                        }
                    )

                else:
                    self.logger.error(f"Failed to scale {resource_type.value}")
                    event.success = False
                    self.scaling_events.append(event)

        except Exception as e:
            logger.error(f"Error executing scaling decision: {e}")

    async def _perform_resource_scaling(self, resource_type: ResourceType, target_allocation: float) -> bool:
        """Perform the actual resource scaling operation."""
        try:
            # Simulate scaling operations with different success rates and delays
            scaling_delays = {
                ResourceType.CPU: 5,
                ResourceType.MEMORY: 3,
                ResourceType.INSTANCES: 30,
                ResourceType.CACHE: 2
            }

            scaling_success_rates = {
                ResourceType.CPU: 0.98,
                ResourceType.MEMORY: 0.95,
                ResourceType.INSTANCES: 0.90,
                ResourceType.CACHE: 0.99
            }

            # Simulate scaling delay
            delay = scaling_delays.get(resource_type, 10)
            await asyncio.sleep(min(delay, 5))  # Cap delay for demo

            # Simulate success/failure
            success_rate = scaling_success_rates.get(resource_type, 0.95)
            success = np.random.random() < success_rate

            if success:
                self.logger.debug(f"Resource scaling simulation successful for {resource_type.value}")
            else:
                self.logger.warning(f"Resource scaling simulation failed for {resource_type.value}")

            return success

        except Exception as e:
            logger.error(f"Error performing resource scaling for {resource_type}: {e}")
            return False

    async def _analyze_load_distribution(self) -> Dict[str, Any]:
        """Analyze current load distribution across regions."""
        try:
            distribution_analysis = {
                'region_loads': {},
                'total_capacity': 0,
                'utilized_capacity': 0,
                'load_imbalance_score': 0.0,
                'optimization_opportunities': []
            }

            for region_id, region in self.geographic_regions.items():
                region_metrics = await self._collect_region_metrics(region_id)

                distribution_analysis['region_loads'][region_id] = {
                    'current_load': region.current_load,
                    'capacity_utilization': region.get_capacity_utilization(),
                    'active_instances': region.active_instances,
                    'max_instances': region.max_instances,
                    'request_rate': region_metrics.get('request_rate', 0),
                    'response_time': region_metrics.get('response_time', 300)
                }

                distribution_analysis['total_capacity'] += region.max_instances
                distribution_analysis['utilized_capacity'] += region.active_instances

            # Calculate load imbalance
            loads = [region.current_load for region in self.geographic_regions.values()]
            if loads:
                load_variance = np.var(loads)
                distribution_analysis['load_imbalance_score'] = load_variance

            return distribution_analysis

        except Exception as e:
            logger.error(f"Error analyzing load distribution: {e}")
            return {}

    async def _optimize_load_balancing(self, load_distribution: Dict[str, Any]) -> Optional[Dict[str, float]]:
        """Optimize load balancing weights based on current distribution."""
        try:
            if not load_distribution.get('region_loads'):
                return None

            # Calculate optimal weights based on performance and capacity
            optimal_weights = {}
            total_score = 0

            for region_id, region_data in load_distribution['region_loads'].items():
                # Calculate region score based on multiple factors
                capacity_score = 1.0 - region_data['capacity_utilization']
                latency_score = 1.0 / (region_data['response_time'] / 200.0)  # Normalize to 200ms baseline
                load_score = 1.0 - region_data['current_load']

                # Weighted composite score
                composite_score = (
                    capacity_score * self.load_balancing_weights['capacity'] +
                    latency_score * self.load_balancing_weights['latency'] +
                    load_score * 0.3  # Additional weight for current load
                )

                optimal_weights[region_id] = max(0.1, composite_score)  # Minimum weight of 0.1
                total_score += optimal_weights[region_id]

            # Normalize weights to sum to 1.0
            if total_score > 0:
                for region_id in optimal_weights:
                    optimal_weights[region_id] /= total_score

                return optimal_weights

        except Exception as e:
            logger.error(f"Error optimizing load balancing: {e}")

        return None

    async def _apply_load_balancing_changes(self, new_weights: Dict[str, float]) -> None:
        """Apply new load balancing weights."""
        try:
            self.logger.info(f"Applying new load balancing weights: {new_weights}")

            # In production, this would update load balancer configuration
            # For demo, we just log the changes
            for region_id, weight in new_weights.items():
                self.logger.info(f"Region {region_id}: weight = {weight:.3f}")

            # Store the weights for future use
            self.cache.set('load_balancing_weights', new_weights, ttl=3600)

        except Exception as e:
            logger.error(f"Error applying load balancing changes: {e}")

    async def _analyze_scaling_patterns(self) -> Dict[str, Any]:
        """Analyze recent scaling events to identify patterns and optimization opportunities."""
        try:
            if len(self.scaling_events) < 5:
                return {}

            recent_events = list(self.scaling_events)[-50:]  # Analyze last 50 events

            analysis = {
                'success_rate': sum(1 for event in recent_events if event.success) / len(recent_events),
                'most_scaled_resources': defaultdict(int),
                'scaling_frequency': defaultdict(int),
                'optimization_insights': []
            }

            for event in recent_events:
                analysis['most_scaled_resources'][event.resource_type.value] += 1
                analysis['scaling_frequency'][event.scaling_action] += 1

            # Generate optimization insights
            if analysis['success_rate'] < 0.9:
                analysis['optimization_insights'].append({
                    'insight': 'Low scaling success rate detected',
                    'recommendation': 'Review scaling thresholds and resource constraints',
                    'priority': 'high'
                })

            # Check for oscillating behavior
            scale_up_count = analysis['scaling_frequency']['scale_up']
            scale_down_count = analysis['scaling_frequency']['scale_down']

            if abs(scale_up_count - scale_down_count) / max(scale_up_count + scale_down_count, 1) < 0.2:
                analysis['optimization_insights'].append({
                    'insight': 'Potential scaling oscillation detected',
                    'recommendation': 'Increase scaling cooldown periods',
                    'priority': 'medium'
                })

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing scaling patterns: {e}")
            return {}

    async def _apply_optimization_insights(self, insights: Dict[str, Any]) -> None:
        """Apply optimization insights to improve scaling behavior."""
        try:
            for insight in insights.get('optimization_insights', []):
                if insight.get('priority') == 'high':
                    if 'scaling thresholds' in insight.get('recommendation', ''):
                        # Adjust scaling thresholds
                        for resource_type in self.scale_up_thresholds:
                            self.scale_up_thresholds[resource_type] *= 1.1  # Make slightly more conservative

                        self.logger.info("Applied optimization: Adjusted scaling thresholds")

                elif insight.get('priority') == 'medium':
                    if 'cooldown periods' in insight.get('recommendation', ''):
                        # Increase cooldown periods
                        for profile in self.resource_profiles.values():
                            profile.cooldown_period = timedelta(
                                seconds=profile.cooldown_period.total_seconds() * 1.2
                            )

                        self.logger.info("Applied optimization: Increased cooldown periods")

        except Exception as e:
            logger.error(f"Error applying optimization insights: {e}")

    async def _generate_scaling_recommendations(self) -> List[Dict[str, Any]]:
        """Generate recommendations for scaling optimization."""
        recommendations = []

        try:
            aggregate_metrics = await self._calculate_aggregate_metrics()
            load_distribution = await self._analyze_load_distribution()

            # Resource utilization recommendations
            for resource_type, profile in self.resource_profiles.items():
                current_util = aggregate_metrics.get(f'{resource_type.value}_utilization', 0.5)

                if current_util > 0.9:
                    recommendations.append({
                        'type': 'capacity_planning',
                        'priority': 'high',
                        'resource': resource_type.value,
                        'recommendation': f'Consider increasing max capacity for {resource_type.value}',
                        'current_utilization': current_util,
                        'suggested_action': 'increase_max_capacity'
                    })

                elif current_util < 0.2:
                    recommendations.append({
                        'type': 'cost_optimization',
                        'priority': 'medium',
                        'resource': resource_type.value,
                        'recommendation': f'Consider reducing min capacity for {resource_type.value}',
                        'current_utilization': current_util,
                        'suggested_action': 'reduce_min_capacity'
                    })

            # Load distribution recommendations
            if load_distribution.get('load_imbalance_score', 0) > 0.1:
                recommendations.append({
                    'type': 'load_balancing',
                    'priority': 'medium',
                    'recommendation': 'Load imbalance detected across regions',
                    'suggested_action': 'optimize_load_balancing',
                    'imbalance_score': load_distribution['load_imbalance_score']
                })

        except Exception as e:
            logger.error(f"Error generating scaling recommendations: {e}")

        return recommendations

    def get_scaling_status(self) -> Dict[str, Any]:
        """Get current scaling system status."""
        return {
            'auto_scaling_enabled': self.auto_scaling_enabled,
            'global_distribution_enabled': self.global_distribution_enabled,
            'scaling_strategy': self.scaling_strategy.strategy_name,
            'resource_profiles': {
                resource_type.value: {
                    'current_allocation': profile.current_allocation,
                    'target_allocation': profile.target_allocation,
                    'utilization': profile.target_allocation / profile.current_allocation if profile.current_allocation > 0 else 0,
                    'can_scale': profile.can_scale()
                } for resource_type, profile in self.resource_profiles.items()
            },
            'geographic_regions': {
                region_id: {
                    'active_instances': region.active_instances,
                    'max_instances': region.max_instances,
                    'capacity_utilization': region.get_capacity_utilization(),
                    'current_load': region.current_load
                } for region_id, region in self.geographic_regions.items()
            },
            'recent_scaling_events': len(self.scaling_events),
            'performance_targets': self.performance_targets,
            'last_update': datetime.now().isoformat()
        }

    def get_scaling_metrics(self) -> Dict[str, Any]:
        """Get scaling performance metrics."""
        recent_events = list(self.scaling_events)[-20:] if self.scaling_events else []

        return {
            'total_scaling_events': len(self.scaling_events),
            'recent_success_rate': sum(1 for event in recent_events if event.success) / len(recent_events) if recent_events else 0,
            'scaling_frequency': {
                'scale_up': sum(1 for event in recent_events if event.scaling_action == 'scale_up'),
                'scale_down': sum(1 for event in recent_events if event.scaling_action == 'scale_down')
            },
            'resource_scaling_counts': {
                resource_type.value: sum(1 for event in recent_events if event.resource_type == resource_type)
                for resource_type in ResourceType
            },
            'average_scaling_impact': self._calculate_average_scaling_impact(recent_events),
            'optimization_effectiveness': self._calculate_optimization_effectiveness(),
            'cost_efficiency_score': self._calculate_cost_efficiency(),
            'metrics_timestamp': datetime.now().isoformat()
        }

    def _calculate_average_scaling_impact(self, events: List[ScalingEvent]) -> Dict[str, float]:
        """Calculate average impact of scaling events."""
        if not events:
            return {}

        impact_metrics = defaultdict(list)
        for event in events:
            for metric, value in event.impact_metrics.items():
                impact_metrics[metric].append(value)

        return {
            metric: sum(values) / len(values)
            for metric, values in impact_metrics.items()
        }

    def _calculate_optimization_effectiveness(self) -> float:
        """Calculate how effective the optimization has been."""
        # Simplified calculation - in production would use more sophisticated metrics
        recent_events = list(self.scaling_events)[-10:] if len(self.scaling_events) >= 10 else []
        if not recent_events:
            return 0.5

        success_rate = sum(1 for event in recent_events if event.success) / len(recent_events)
        return success_rate

    def _calculate_cost_efficiency(self) -> float:
        """Calculate cost efficiency score."""
        # Simplified calculation based on resource utilization
        total_utilization = 0
        resource_count = len(self.resource_profiles)

        for profile in self.resource_profiles.values():
            utilization = profile.target_allocation / profile.max_allocation
            total_utilization += utilization

        return total_utilization / resource_count if resource_count > 0 else 0.5


# Global instance management
_global_scale_optimizer_instance: Optional[GlobalScaleOptimizer] = None


def get_global_scale_optimizer() -> GlobalScaleOptimizer:
    """Get or create the global scale optimizer instance."""
    global _global_scale_optimizer_instance
    if _global_scale_optimizer_instance is None:
        _global_scale_optimizer_instance = GlobalScaleOptimizer()
    return _global_scale_optimizer_instance


async def demonstrate_global_scaling() -> Dict[str, Any]:
    """Demonstrate global scaling capabilities."""
    optimizer = get_global_scale_optimizer()

    # Simulate some load and scaling decisions
    for _ in range(3):
        scaling_decisions = await optimizer._evaluate_scaling_needs()
        for decision in scaling_decisions[:2]:  # Execute first 2 decisions
            await optimizer._execute_scaling_decision(decision)
        await asyncio.sleep(0.5)

    # Analyze load distribution
    load_distribution = await optimizer._analyze_load_distribution()

    # Generate recommendations
    recommendations = await optimizer._generate_scaling_recommendations()

    # Get status and metrics
    status = optimizer.get_scaling_status()
    metrics = optimizer.get_scaling_metrics()

    return {
        'scaling_status': status,
        'scaling_metrics': metrics,
        'load_distribution': load_distribution,
        'recommendations': recommendations,
        'demonstration_timestamp': datetime.now().isoformat()
    }


if __name__ == "__main__":
    # Demo execution
    async def main():
        results = await demonstrate_global_scaling()
        print(json.dumps(results, indent=2, default=str))

    asyncio.run(main())
