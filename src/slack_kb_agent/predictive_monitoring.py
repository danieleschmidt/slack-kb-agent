"""Predictive Monitoring and Self-Healing System.

This module implements advanced monitoring capabilities with predictive analytics,
anomaly detection, and automated self-healing mechanisms for the Slack KB Agent.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
import warnings
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import numpy as np

# Suppress numpy warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning)

from .adaptive_learning_engine import get_adaptive_learning_engine
from .cache import get_cache_manager
from .monitoring import StructuredLogger, get_global_metrics
from .resilience import get_circuit_breaker

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    CRITICAL = ("critical", 1.0)
    HIGH = ("high", 0.8)
    MEDIUM = ("medium", 0.6)
    LOW = ("low", 0.4)
    INFO = ("info", 0.2)

    def __init__(self, name: str, weight: float):
        self.severity_name = name
        self.weight = weight


class AnomalyType(Enum):
    """Types of anomalies that can be detected."""
    PERFORMANCE_DEGRADATION = "performance_degradation"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    ERROR_SPIKE = "error_spike"
    UNUSUAL_TRAFFIC = "unusual_traffic"
    SERVICE_DISRUPTION = "service_disruption"
    DATA_QUALITY_ISSUE = "data_quality_issue"
    SECURITY_ANOMALY = "security_anomaly"


@dataclass
class Anomaly:
    """Represents a detected anomaly."""
    anomaly_id: str
    anomaly_type: AnomalyType
    severity: AlertSeverity
    confidence_score: float
    detected_at: datetime
    description: str
    affected_components: List[str]
    metrics: Dict[str, float]
    predicted_impact: Dict[str, float]
    recommended_actions: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'anomaly_id': self.anomaly_id,
            'type': self.anomaly_type.value,
            'severity': self.severity.severity_name,
            'confidence': self.confidence_score,
            'detected_at': self.detected_at.isoformat(),
            'description': self.description,
            'affected_components': self.affected_components,
            'metrics': self.metrics,
            'predicted_impact': self.predicted_impact,
            'recommended_actions': self.recommended_actions
        }


@dataclass
class HealthPrediction:
    """Health prediction for system components."""
    component: str
    current_health: float  # 0.0 to 1.0
    predicted_health_1h: float
    predicted_health_24h: float
    confidence: float
    risk_factors: List[str]
    recommendations: List[str]
    predicted_at: datetime


class PredictiveMonitoring:
    """Advanced monitoring with predictive analytics and self-healing."""

    def __init__(self, prediction_window_hours: int = 24):
        self.prediction_window_hours = prediction_window_hours
        self.logger = StructuredLogger("predictive_monitoring")
        self.cache = get_cache_manager()
        self.learning_engine = get_adaptive_learning_engine()

        # Monitoring data storage
        self.metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.anomalies: Dict[str, Anomaly] = {}
        self.health_predictions: Dict[str, HealthPrediction] = {}

        # Anomaly detection models (simplified)
        self.baseline_metrics: Dict[str, float] = {}
        self.metric_thresholds: Dict[str, Dict[str, float]] = {
            'response_time': {'warning': 1000, 'critical': 2000},  # ms
            'error_rate': {'warning': 0.05, 'critical': 0.1},     # 5%, 10%
            'memory_usage': {'warning': 0.8, 'critical': 0.9},    # 80%, 90%
            'cpu_usage': {'warning': 0.7, 'critical': 0.85},      # 70%, 85%
            'disk_usage': {'warning': 0.8, 'critical': 0.9}       # 80%, 90%
        }

        # Self-healing actions
        self.healing_actions: Dict[str, Callable] = {
            'restart_service': self._restart_service_action,
            'clear_cache': self._clear_cache_action,
            'scale_resources': self._scale_resources_action,
            'circuit_breaker_open': self._circuit_breaker_action,
            'reduce_load': self._reduce_load_action
        }

        # Monitoring configuration
        self.monitoring_interval = 60  # seconds
        self.anomaly_detection_enabled = True
        self.self_healing_enabled = True

        # Thread safety
        self._monitoring_lock = threading.Lock()

        logger.info(f"Predictive Monitoring initialized with {prediction_window_hours}h prediction window")

    async def start_monitoring(self) -> None:
        """Start the continuous monitoring process."""
        self.logger.info("Starting predictive monitoring")

        monitoring_tasks = [
            self._continuous_metrics_collection(),
            self._continuous_anomaly_detection(),
            self._continuous_health_prediction(),
            self._continuous_self_healing()
        ]

        await asyncio.gather(*monitoring_tasks, return_exceptions=True)

    async def _continuous_metrics_collection(self) -> None:
        """Continuously collect system metrics."""
        while True:
            try:
                metrics = await self._collect_system_metrics()
                await self._store_metrics(metrics)
                await asyncio.sleep(self.monitoring_interval)

            except Exception as e:
                logger.error(f"Error in metrics collection: {e}")
                await asyncio.sleep(5)  # Brief pause on error

    async def _continuous_anomaly_detection(self) -> None:
        """Continuously detect anomalies in system behavior."""
        while True:
            try:
                if self.anomaly_detection_enabled:
                    anomalies = await self._detect_anomalies()
                    for anomaly in anomalies:
                        await self._handle_anomaly(anomaly)

                await asyncio.sleep(self.monitoring_interval)

            except Exception as e:
                logger.error(f"Error in anomaly detection: {e}")
                await asyncio.sleep(10)

    async def _continuous_health_prediction(self) -> None:
        """Continuously predict system health."""
        while True:
            try:
                predictions = await self._predict_system_health()
                self.health_predictions.update(predictions)

                # Log predictions for critical components
                for component, prediction in predictions.items():
                    if prediction.predicted_health_24h < 0.5:
                        self.logger.warning(f"Health prediction for {component}: {prediction.predicted_health_24h:.2f}")

                await asyncio.sleep(300)  # Update predictions every 5 minutes

            except Exception as e:
                logger.error(f"Error in health prediction: {e}")
                await asyncio.sleep(30)

    async def _continuous_self_healing(self) -> None:
        """Continuously perform self-healing actions."""
        while True:
            try:
                if self.self_healing_enabled:
                    await self._execute_self_healing_actions()

                await asyncio.sleep(30)  # Check for healing opportunities frequently

            except Exception as e:
                logger.error(f"Error in self-healing: {e}")
                await asyncio.sleep(15)

    async def _collect_system_metrics(self) -> Dict[str, float]:
        """Collect comprehensive system metrics."""
        try:
            # Get metrics from global metrics collector
            global_metrics = get_global_metrics()

            # Simulate additional metrics collection
            current_time = time.time()
            metrics = {
                'timestamp': current_time,
                'response_time': np.random.normal(800, 200),  # Simulated response time
                'error_rate': max(0, np.random.normal(0.02, 0.01)),  # Simulated error rate
                'memory_usage': min(1.0, max(0, np.random.normal(0.6, 0.1))),  # Simulated memory
                'cpu_usage': min(1.0, max(0, np.random.normal(0.5, 0.15))),  # Simulated CPU
                'disk_usage': min(1.0, max(0, np.random.normal(0.4, 0.05))),  # Simulated disk
                'active_connections': max(0, int(np.random.normal(50, 15))),  # Active connections
                'cache_hit_rate': min(1.0, max(0, np.random.normal(0.85, 0.1))),  # Cache performance
                'query_volume': max(0, int(np.random.normal(100, 25))),  # Queries per minute
                'data_quality_score': min(1.0, max(0, np.random.normal(0.95, 0.05)))  # Data quality
            }

            return metrics

        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            return {'timestamp': time.time(), 'error': str(e)}

    async def _store_metrics(self, metrics: Dict[str, float]) -> None:
        """Store metrics in historical data structure."""
        with self._monitoring_lock:
            timestamp = metrics.pop('timestamp', time.time())

            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)) and not np.isnan(value):
                    self.metrics_history[metric_name].append({
                        'timestamp': timestamp,
                        'value': value
                    })

                    # Update baseline metrics
                    if metric_name in self.baseline_metrics:
                        alpha = 0.05  # Smoothing factor
                        self.baseline_metrics[metric_name] = (
                            (1 - alpha) * self.baseline_metrics[metric_name] + alpha * value
                        )
                    else:
                        self.baseline_metrics[metric_name] = value

    async def _detect_anomalies(self) -> List[Anomaly]:
        """Detect anomalies in system metrics using multiple techniques."""
        anomalies = []

        try:
            # Statistical anomaly detection
            statistical_anomalies = await self._detect_statistical_anomalies()
            anomalies.extend(statistical_anomalies)

            # Threshold-based anomaly detection
            threshold_anomalies = await self._detect_threshold_anomalies()
            anomalies.extend(threshold_anomalies)

            # Pattern-based anomaly detection
            pattern_anomalies = await self._detect_pattern_anomalies()
            anomalies.extend(pattern_anomalies)

        except Exception as e:
            logger.error(f"Error in anomaly detection: {e}")

        return anomalies

    async def _detect_statistical_anomalies(self) -> List[Anomaly]:
        """Detect statistical anomalies using Z-score and other methods."""
        anomalies = []

        for metric_name, history in self.metrics_history.items():
            if len(history) < 10:  # Need sufficient data
                continue

            try:
                # Get recent values
                recent_values = [entry['value'] for entry in list(history)[-100:]]
                current_value = recent_values[-1]

                # Calculate Z-score
                mean_value = np.mean(recent_values[:-10])  # Baseline excluding recent values
                std_value = np.std(recent_values[:-10])

                if std_value > 0:
                    z_score = abs((current_value - mean_value) / std_value)

                    if z_score > 3.0:  # Strong anomaly
                        anomaly = Anomaly(
                            anomaly_id=f"statistical_{metric_name}_{int(time.time())}",
                            anomaly_type=self._classify_anomaly_type(metric_name, current_value),
                            severity=AlertSeverity.HIGH if z_score > 4.0 else AlertSeverity.MEDIUM,
                            confidence_score=min(1.0, z_score / 5.0),
                            detected_at=datetime.now(),
                            description=f"Statistical anomaly in {metric_name}: {current_value:.2f} (Z-score: {z_score:.2f})",
                            affected_components=[metric_name],
                            metrics={metric_name: current_value, 'z_score': z_score},
                            predicted_impact=await self._predict_anomaly_impact(metric_name, current_value),
                            recommended_actions=await self._get_anomaly_recommendations(metric_name)
                        )
                        anomalies.append(anomaly)

            except Exception as e:
                logger.error(f"Error in statistical anomaly detection for {metric_name}: {e}")

        return anomalies

    async def _detect_threshold_anomalies(self) -> List[Anomaly]:
        """Detect anomalies based on predefined thresholds."""
        anomalies = []

        for metric_name, thresholds in self.metric_thresholds.items():
            if metric_name not in self.metrics_history:
                continue

            try:
                history = self.metrics_history[metric_name]
                if not history:
                    continue

                current_value = history[-1]['value']

                # Check critical threshold
                if current_value > thresholds.get('critical', float('inf')):
                    anomaly = Anomaly(
                        anomaly_id=f"threshold_critical_{metric_name}_{int(time.time())}",
                        anomaly_type=self._classify_anomaly_type(metric_name, current_value),
                        severity=AlertSeverity.CRITICAL,
                        confidence_score=1.0,
                        detected_at=datetime.now(),
                        description=f"Critical threshold exceeded for {metric_name}: {current_value:.2f}",
                        affected_components=[metric_name],
                        metrics={metric_name: current_value, 'threshold': thresholds['critical']},
                        predicted_impact=await self._predict_anomaly_impact(metric_name, current_value),
                        recommended_actions=await self._get_anomaly_recommendations(metric_name)
                    )
                    anomalies.append(anomaly)

                # Check warning threshold
                elif current_value > thresholds.get('warning', float('inf')):
                    anomaly = Anomaly(
                        anomaly_id=f"threshold_warning_{metric_name}_{int(time.time())}",
                        anomaly_type=self._classify_anomaly_type(metric_name, current_value),
                        severity=AlertSeverity.MEDIUM,
                        confidence_score=0.8,
                        detected_at=datetime.now(),
                        description=f"Warning threshold exceeded for {metric_name}: {current_value:.2f}",
                        affected_components=[metric_name],
                        metrics={metric_name: current_value, 'threshold': thresholds['warning']},
                        predicted_impact=await self._predict_anomaly_impact(metric_name, current_value),
                        recommended_actions=await self._get_anomaly_recommendations(metric_name)
                    )
                    anomalies.append(anomaly)

            except Exception as e:
                logger.error(f"Error in threshold anomaly detection for {metric_name}: {e}")

        return anomalies

    async def _detect_pattern_anomalies(self) -> List[Anomaly]:
        """Detect anomalies in patterns and trends."""
        anomalies = []

        try:
            # Detect unusual trends
            for metric_name, history in self.metrics_history.items():
                if len(history) < 20:  # Need sufficient data for trend analysis
                    continue

                # Get recent trend
                recent_values = [entry['value'] for entry in list(history)[-20:]]

                # Simple trend detection using linear regression
                x = np.arange(len(recent_values))
                slope = np.polyfit(x, recent_values, 1)[0]

                # Detect rapid degradation
                if metric_name in ['response_time', 'error_rate', 'memory_usage'] and slope > 0.1:
                    anomaly = Anomaly(
                        anomaly_id=f"pattern_degradation_{metric_name}_{int(time.time())}",
                        anomaly_type=AnomalyType.PERFORMANCE_DEGRADATION,
                        severity=AlertSeverity.HIGH,
                        confidence_score=0.7,
                        detected_at=datetime.now(),
                        description=f"Rapid degradation trend detected in {metric_name}",
                        affected_components=[metric_name],
                        metrics={metric_name: recent_values[-1], 'trend_slope': slope},
                        predicted_impact={'performance_impact': 0.8},
                        recommended_actions=[{'action': 'investigate_trend', 'priority': 'high'}]
                    )
                    anomalies.append(anomaly)

        except Exception as e:
            logger.error(f"Error in pattern anomaly detection: {e}")

        return anomalies

    def _classify_anomaly_type(self, metric_name: str, value: float) -> AnomalyType:
        """Classify the type of anomaly based on metric and value."""
        if metric_name in ['response_time'] and value > 1000:
            return AnomalyType.PERFORMANCE_DEGRADATION
        elif metric_name in ['memory_usage', 'cpu_usage', 'disk_usage'] and value > 0.8:
            return AnomalyType.RESOURCE_EXHAUSTION
        elif metric_name in ['error_rate'] and value > 0.05:
            return AnomalyType.ERROR_SPIKE
        elif metric_name in ['query_volume', 'active_connections']:
            return AnomalyType.UNUSUAL_TRAFFIC
        else:
            return AnomalyType.SERVICE_DISRUPTION

    async def _predict_anomaly_impact(self, metric_name: str, value: float) -> Dict[str, float]:
        """Predict the impact of an anomaly."""
        impact = {
            'user_experience': 0.5,
            'system_stability': 0.5,
            'performance_degradation': 0.5
        }

        if metric_name == 'response_time':
            impact['user_experience'] = min(1.0, value / 2000.0)
            impact['performance_degradation'] = min(1.0, value / 1500.0)
        elif metric_name == 'error_rate':
            impact['user_experience'] = min(1.0, value * 10)
            impact['system_stability'] = min(1.0, value * 8)
        elif metric_name in ['memory_usage', 'cpu_usage']:
            impact['system_stability'] = min(1.0, value)
            impact['performance_degradation'] = min(1.0, value * 1.2)

        return impact

    async def _get_anomaly_recommendations(self, metric_name: str) -> List[Dict[str, Any]]:
        """Get recommendations for handling specific anomalies."""
        recommendations = []

        if metric_name == 'response_time':
            recommendations.extend([
                {'action': 'enable_aggressive_caching', 'priority': 'high'},
                {'action': 'check_database_performance', 'priority': 'medium'},
                {'action': 'scale_compute_resources', 'priority': 'medium'}
            ])
        elif metric_name == 'memory_usage':
            recommendations.extend([
                {'action': 'clear_cache', 'priority': 'high'},
                {'action': 'restart_service', 'priority': 'medium'},
                {'action': 'investigate_memory_leak', 'priority': 'high'}
            ])
        elif metric_name == 'error_rate':
            recommendations.extend([
                {'action': 'check_error_logs', 'priority': 'critical'},
                {'action': 'circuit_breaker_open', 'priority': 'high'},
                {'action': 'rollback_recent_changes', 'priority': 'medium'}
            ])

        return recommendations

    async def _predict_system_health(self) -> Dict[str, HealthPrediction]:
        """Predict system health for various components."""
        predictions = {}

        components = ['api_service', 'database', 'cache', 'search_engine', 'monitoring']

        for component in components:
            try:
                current_health = await self._calculate_current_health(component)
                health_1h = await self._predict_health(component, hours=1)
                health_24h = await self._predict_health(component, hours=24)

                predictions[component] = HealthPrediction(
                    component=component,
                    current_health=current_health,
                    predicted_health_1h=health_1h,
                    predicted_health_24h=health_24h,
                    confidence=0.75,
                    risk_factors=await self._identify_risk_factors(component),
                    recommendations=await self._get_health_recommendations(component, health_24h),
                    predicted_at=datetime.now()
                )

            except Exception as e:
                logger.error(f"Error predicting health for {component}: {e}")

        return predictions

    async def _calculate_current_health(self, component: str) -> float:
        """Calculate current health score for a component."""
        # Simplified health calculation based on relevant metrics
        health_factors = {
            'api_service': ['response_time', 'error_rate', 'cpu_usage'],
            'database': ['response_time', 'memory_usage', 'error_rate'],
            'cache': ['cache_hit_rate', 'memory_usage'],
            'search_engine': ['response_time', 'data_quality_score'],
            'monitoring': ['cpu_usage', 'memory_usage']
        }

        factors = health_factors.get(component, ['cpu_usage', 'memory_usage'])
        health_scores = []

        for factor in factors:
            if factor in self.metrics_history and self.metrics_history[factor]:
                current_value = self.metrics_history[factor][-1]['value']

                # Convert metric to health score (0-1, higher is better)
                if factor in ['response_time', 'error_rate', 'cpu_usage', 'memory_usage']:
                    # Lower is better for these metrics
                    threshold = self.metric_thresholds.get(factor, {}).get('warning', 1.0)
                    health_score = max(0, 1.0 - (current_value / threshold))
                else:
                    # Higher is better for these metrics
                    health_score = min(1.0, current_value)

                health_scores.append(health_score)

        return sum(health_scores) / len(health_scores) if health_scores else 0.5

    async def _predict_health(self, component: str, hours: int) -> float:
        """Predict future health of a component."""
        current_health = await self._calculate_current_health(component)

        # Simple prediction based on trends (in production, use ML models)
        trend_factor = np.random.normal(0, 0.1)  # Random trend simulation
        predicted_health = max(0, min(1.0, current_health + trend_factor))

        return predicted_health

    async def _identify_risk_factors(self, component: str) -> List[str]:
        """Identify risk factors for a component."""
        risk_factors = []

        # Check for trending issues
        for metric_name in ['response_time', 'error_rate', 'memory_usage']:
            if metric_name in self.metrics_history and len(self.metrics_history[metric_name]) > 5:
                recent_values = [entry['value'] for entry in list(self.metrics_history[metric_name])[-5:]]
                if len(recent_values) >= 2 and recent_values[-1] > recent_values[0] * 1.2:
                    risk_factors.append(f"Increasing {metric_name}")

        return risk_factors

    async def _get_health_recommendations(self, component: str, predicted_health: float) -> List[str]:
        """Get health recommendations for a component."""
        recommendations = []

        if predicted_health < 0.3:
            recommendations.extend([
                f"Immediate attention required for {component}",
                "Consider emergency maintenance window",
                "Prepare rollback procedures"
            ])
        elif predicted_health < 0.6:
            recommendations.extend([
                f"Monitor {component} closely",
                "Schedule preventive maintenance",
                "Review recent changes"
            ])

        return recommendations

    async def _handle_anomaly(self, anomaly: Anomaly) -> None:
        """Handle a detected anomaly."""
        try:
            # Store the anomaly
            self.anomalies[anomaly.anomaly_id] = anomaly

            # Log the anomaly
            self.logger.warning(f"Anomaly detected: {anomaly.description}")

            # Learn from the anomaly
            await self.learning_engine.learn_from_errors(
                anomaly.anomaly_type.value,
                {
                    'severity': anomaly.severity.severity_name,
                    'affected_components': anomaly.affected_components,
                    'metrics': anomaly.metrics
                },
                recovery_success=False  # Will be updated if recovery succeeds
            )

            # Execute recommended actions if self-healing is enabled
            if self.self_healing_enabled and anomaly.severity.weight > 0.6:
                await self._execute_anomaly_actions(anomaly)

        except Exception as e:
            logger.error(f"Error handling anomaly {anomaly.anomaly_id}: {e}")

    async def _execute_anomaly_actions(self, anomaly: Anomaly) -> None:
        """Execute recommended actions for an anomaly."""
        for action in anomaly.recommended_actions:
            try:
                action_name = action.get('action')
                if action_name in self.healing_actions:
                    self.logger.info(f"Executing healing action: {action_name}")
                    success = await self.healing_actions[action_name](anomaly)

                    if success:
                        self.logger.info(f"Successfully executed {action_name}")
                        # Update learning engine with successful recovery
                        await self.learning_engine.learn_from_errors(
                            anomaly.anomaly_type.value,
                            {'action_taken': action_name},
                            recovery_success=True
                        )
                    else:
                        self.logger.warning(f"Failed to execute {action_name}")

            except Exception as e:
                logger.error(f"Error executing action {action.get('action')}: {e}")

    async def _execute_self_healing_actions(self) -> None:
        """Execute proactive self-healing actions based on predictions."""
        try:
            # Check health predictions for proactive actions
            for component, prediction in self.health_predictions.items():
                if prediction.predicted_health_24h < 0.4:
                    self.logger.info(f"Proactive healing for {component} (predicted health: {prediction.predicted_health_24h:.2f})")

                    # Execute preventive actions
                    if component == 'cache' and 'clear_cache' in self.healing_actions:
                        await self.healing_actions['clear_cache'](None)
                    elif 'memory_usage' in prediction.risk_factors and 'restart_service' in self.healing_actions:
                        await self.healing_actions['restart_service'](None)

        except Exception as e:
            logger.error(f"Error in proactive self-healing: {e}")

    # Self-healing action implementations
    async def _restart_service_action(self, anomaly: Optional[Anomaly]) -> bool:
        """Simulate service restart action."""
        try:
            self.logger.info("Simulating service restart for healing")
            # In production, this would restart the actual service
            await asyncio.sleep(1)  # Simulate restart time
            return True
        except Exception as e:
            logger.error(f"Error in restart service action: {e}")
            return False

    async def _clear_cache_action(self, anomaly: Optional[Anomaly]) -> bool:
        """Clear cache action for memory pressure relief."""
        try:
            self.logger.info("Clearing cache for healing")
            cache_manager = self.cache
            if hasattr(cache_manager, 'clear'):
                cache_manager.clear()
            return True
        except Exception as e:
            logger.error(f"Error in clear cache action: {e}")
            return False

    async def _scale_resources_action(self, anomaly: Optional[Anomaly]) -> bool:
        """Simulate resource scaling action."""
        try:
            self.logger.info("Simulating resource scaling for healing")
            # In production, this would trigger auto-scaling
            await asyncio.sleep(0.5)
            return True
        except Exception as e:
            logger.error(f"Error in scale resources action: {e}")
            return False

    async def _circuit_breaker_action(self, anomaly: Optional[Anomaly]) -> bool:
        """Open circuit breaker to prevent cascading failures."""
        try:
            self.logger.info("Opening circuit breaker for healing")
            circuit_breaker = get_circuit_breaker()
            if hasattr(circuit_breaker, 'open'):
                circuit_breaker.open()
            return True
        except Exception as e:
            logger.error(f"Error in circuit breaker action: {e}")
            return False

    async def _reduce_load_action(self, anomaly: Optional[Anomaly]) -> bool:
        """Simulate load reduction action."""
        try:
            self.logger.info("Simulating load reduction for healing")
            # In production, this would implement load shedding
            await asyncio.sleep(0.2)
            return True
        except Exception as e:
            logger.error(f"Error in reduce load action: {e}")
            return False

    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get current monitoring system status."""
        return {
            'monitoring_enabled': True,
            'anomaly_detection_enabled': self.anomaly_detection_enabled,
            'self_healing_enabled': self.self_healing_enabled,
            'metrics_tracked': len(self.metrics_history),
            'active_anomalies': len(self.anomalies),
            'health_predictions': len(self.health_predictions),
            'baseline_metrics': len(self.baseline_metrics),
            'monitoring_interval': self.monitoring_interval,
            'prediction_window_hours': self.prediction_window_hours,
            'last_update': datetime.now().isoformat()
        }

    def get_anomaly_report(self) -> Dict[str, Any]:
        """Get comprehensive anomaly report."""
        return {
            'total_anomalies': len(self.anomalies),
            'anomalies_by_severity': {
                severity.severity_name: len([
                    a for a in self.anomalies.values()
                    if a.severity == severity
                ]) for severity in AlertSeverity
            },
            'anomalies_by_type': {
                anomaly_type.value: len([
                    a for a in self.anomalies.values()
                    if a.anomaly_type == anomaly_type
                ]) for anomaly_type in AnomalyType
            },
            'recent_anomalies': [
                anomaly.to_dict() for anomaly in
                sorted(self.anomalies.values(),
                      key=lambda x: x.detected_at, reverse=True)[:5]
            ],
            'report_generated': datetime.now().isoformat()
        }


# Global instance management
_predictive_monitoring_instance: Optional[PredictiveMonitoring] = None


def get_predictive_monitoring(prediction_window_hours: int = 24) -> PredictiveMonitoring:
    """Get or create the global predictive monitoring instance."""
    global _predictive_monitoring_instance
    if _predictive_monitoring_instance is None:
        _predictive_monitoring_instance = PredictiveMonitoring(prediction_window_hours)
    return _predictive_monitoring_instance


async def demonstrate_predictive_monitoring() -> Dict[str, Any]:
    """Demonstrate predictive monitoring capabilities."""
    monitoring = get_predictive_monitoring()

    # Collect some sample metrics
    for _ in range(5):
        metrics = await monitoring._collect_system_metrics()
        await monitoring._store_metrics(metrics)
        await asyncio.sleep(0.1)

    # Detect anomalies
    anomalies = await monitoring._detect_anomalies()

    # Get health predictions
    health_predictions = await monitoring._predict_system_health()

    # Get status reports
    status = monitoring.get_monitoring_status()
    anomaly_report = monitoring.get_anomaly_report()

    return {
        'monitoring_status': status,
        'detected_anomalies': len(anomalies),
        'health_predictions': {
            component: {
                'current_health': prediction.current_health,
                'predicted_health_24h': prediction.predicted_health_24h,
                'risk_factors': prediction.risk_factors
            } for component, prediction in health_predictions.items()
        },
        'anomaly_report': anomaly_report,
        'timestamp': datetime.now().isoformat()
    }


if __name__ == "__main__":
    # Demo execution
    async def main():
        results = await demonstrate_predictive_monitoring()
        print(json.dumps(results, indent=2, default=str))

    asyncio.run(main())
