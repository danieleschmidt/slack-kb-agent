"""Adaptive Learning Engine for Continuous System Improvement.

This module implements machine learning-driven adaptation and optimization
for the Slack KB Agent, enabling continuous learning from usage patterns,
performance metrics, and user feedback.
"""

from __future__ import annotations

import asyncio
import json
import time
import logging
import numpy as np
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Set
from datetime import datetime, timedelta
import threading
from pathlib import Path

from .monitoring import get_global_metrics, StructuredLogger
from .cache import get_cache_manager
from .analytics import UsageAnalytics

logger = logging.getLogger(__name__)


@dataclass
class LearningPattern:
    """Represents a learned pattern from system behavior."""
    pattern_id: str
    pattern_type: str  # query, performance, error, user_behavior
    features: Dict[str, float]
    outcomes: Dict[str, float]
    confidence_score: float
    occurrence_count: int
    last_seen: datetime
    success_rate: float = 0.0
    
    def update_confidence(self, new_outcome: bool) -> None:
        """Update confidence score based on new outcome."""
        self.occurrence_count += 1
        if new_outcome:
            self.success_rate = (self.success_rate * (self.occurrence_count - 1) + 1.0) / self.occurrence_count
        else:
            self.success_rate = (self.success_rate * (self.occurrence_count - 1)) / self.occurrence_count
        
        # Confidence increases with more data points and higher success rate
        self.confidence_score = min(1.0, (self.occurrence_count / 100.0) * self.success_rate)
        self.last_seen = datetime.now()


@dataclass
class AdaptationRule:
    """Represents an adaptation rule learned from patterns."""
    rule_id: str
    condition: Dict[str, Any]  # Conditions that trigger this rule
    action: Dict[str, Any]     # Actions to take when triggered
    effectiveness_score: float
    application_count: int
    created_at: datetime
    
    def apply_rule(self, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Apply adaptation rule if conditions are met."""
        if self._matches_condition(context):
            self.application_count += 1
            return self.action
        return None
    
    def _matches_condition(self, context: Dict[str, Any]) -> bool:
        """Check if context matches rule conditions."""
        for key, expected_value in self.condition.items():
            if key not in context:
                return False
            
            context_value = context[key]
            if isinstance(expected_value, dict):
                # Range or comparison conditions
                if 'min' in expected_value and context_value < expected_value['min']:
                    return False
                if 'max' in expected_value and context_value > expected_value['max']:
                    return False
            else:
                # Exact match
                if context_value != expected_value:
                    return False
        
        return True


class AdaptiveLearningEngine:
    """Advanced learning engine for continuous system improvement."""
    
    def __init__(self, learning_rate: float = 0.1):
        self.learning_rate = learning_rate
        self.logger = StructuredLogger("adaptive_learning")
        self.cache = get_cache_manager()
        
        # Learning components
        self.patterns: Dict[str, LearningPattern] = {}
        self.adaptation_rules: Dict[str, AdaptationRule] = {}
        self.feature_importance: Dict[str, float] = defaultdict(float)
        
        # Learning history and metrics
        self.learning_history: deque = deque(maxlen=10000)
        self.adaptation_history: deque = deque(maxlen=1000)
        self.performance_baseline: Dict[str, float] = {}
        
        # Continuous learning settings
        self.min_pattern_confidence = 0.7
        self.max_patterns = 1000
        self.learning_window_hours = 24
        
        # Learning locks for thread safety
        self._learning_lock = threading.Lock()
        self._pattern_lock = threading.Lock()
        
        logger.info(f"Adaptive Learning Engine initialized with learning rate: {learning_rate}")
    
    async def learn_from_query(self, query: str, response: str, 
                              performance_metrics: Dict[str, float],
                              user_feedback: Optional[float] = None) -> None:
        """Learn from query-response interactions."""
        try:
            features = await self._extract_query_features(query, response)
            outcomes = {
                'response_time': performance_metrics.get('response_time', 0),
                'relevance_score': performance_metrics.get('relevance_score', 0),
                'user_satisfaction': user_feedback or 0.5
            }
            
            pattern_id = f"query_{hash(query) % 10000}"
            await self._update_or_create_pattern(pattern_id, "query", features, outcomes)
            
            # Learn adaptation rules if pattern is strong enough
            pattern = self.patterns.get(pattern_id)
            if pattern and pattern.confidence_score > self.min_pattern_confidence:
                await self._derive_adaptation_rules(pattern)
            
            self.logger.debug(f"Learned from query pattern {pattern_id}")
            
        except Exception as e:
            logger.error(f"Error learning from query: {e}")
    
    async def learn_from_performance(self, metrics: Dict[str, float],
                                   context: Dict[str, Any]) -> None:
        """Learn from performance metrics and system context."""
        try:
            features = await self._extract_performance_features(metrics, context)
            outcomes = metrics
            
            pattern_id = f"performance_{int(time.time()) // 3600}"  # Hourly patterns
            await self._update_or_create_pattern(pattern_id, "performance", features, outcomes)
            
            # Update performance baseline
            await self._update_performance_baseline(metrics)
            
            self.logger.debug(f"Learned from performance pattern {pattern_id}")
            
        except Exception as e:
            logger.error(f"Error learning from performance: {e}")
    
    async def learn_from_errors(self, error_type: str, error_context: Dict[str, Any],
                               recovery_success: bool) -> None:
        """Learn from error patterns and recovery strategies."""
        try:
            features = await self._extract_error_features(error_type, error_context)
            outcomes = {
                'recovery_success': 1.0 if recovery_success else 0.0,
                'error_severity': error_context.get('severity', 0.5)
            }
            
            pattern_id = f"error_{error_type}_{hash(str(error_context)) % 1000}"
            await self._update_or_create_pattern(pattern_id, "error", features, outcomes)
            
            self.logger.info(f"Learned from error pattern {pattern_id}, recovery: {recovery_success}")
            
        except Exception as e:
            logger.error(f"Error learning from errors: {e}")
    
    async def get_adaptive_recommendations(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get adaptive recommendations based on learned patterns."""
        recommendations = []
        
        try:
            with self._pattern_lock:
                for rule in self.adaptation_rules.values():
                    recommendation = rule.apply_rule(context)
                    if recommendation:
                        recommendation['rule_id'] = rule.rule_id
                        recommendation['confidence'] = rule.effectiveness_score
                        recommendations.append(recommendation)
            
            # Sort by confidence/effectiveness
            recommendations.sort(key=lambda x: x.get('confidence', 0), reverse=True)
            
            self.logger.debug(f"Generated {len(recommendations)} adaptive recommendations")
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
        
        return recommendations[:10]  # Return top 10 recommendations
    
    async def optimize_system_parameters(self) -> Dict[str, Any]:
        """Optimize system parameters based on learned patterns."""
        try:
            optimizations = {
                'cache_settings': await self._optimize_cache_settings(),
                'search_parameters': await self._optimize_search_parameters(),
                'response_formatting': await self._optimize_response_formatting(),
                'resource_allocation': await self._optimize_resource_allocation()
            }
            
            # Apply optimizations if they meet confidence thresholds
            applied_optimizations = []
            for category, optimization in optimizations.items():
                if optimization.get('confidence', 0) > 0.8:
                    await self._apply_optimization(category, optimization)
                    applied_optimizations.append(category)
            
            self.logger.info(f"Applied optimizations: {applied_optimizations}")
            
            return {
                'optimizations': optimizations,
                'applied': applied_optimizations,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error optimizing system parameters: {e}")
            return {'error': str(e)}
    
    async def _extract_query_features(self, query: str, response: str) -> Dict[str, float]:
        """Extract features from query-response pairs."""
        return {
            'query_length': len(query),
            'query_complexity': len(query.split()) / 10.0,  # Normalized word count
            'response_length': len(response),
            'query_sentiment': 0.5,  # Placeholder for sentiment analysis
            'technical_terms': len([w for w in query.split() if len(w) > 8]) / len(query.split()) if query.split() else 0
        }
    
    async def _extract_performance_features(self, metrics: Dict[str, float], 
                                          context: Dict[str, Any]) -> Dict[str, float]:
        """Extract features from performance metrics."""
        return {
            'response_time': metrics.get('response_time', 0),
            'memory_usage': metrics.get('memory_usage', 0),
            'cpu_usage': metrics.get('cpu_usage', 0),
            'active_users': context.get('active_users', 0),
            'time_of_day': datetime.now().hour / 24.0,
            'day_of_week': datetime.now().weekday() / 7.0
        }
    
    async def _extract_error_features(self, error_type: str, 
                                     error_context: Dict[str, Any]) -> Dict[str, float]:
        """Extract features from error patterns."""
        return {
            'error_frequency': error_context.get('frequency', 1),
            'system_load': error_context.get('system_load', 0.5),
            'user_load': error_context.get('user_load', 0.5),
            'memory_pressure': error_context.get('memory_pressure', 0),
            'network_latency': error_context.get('network_latency', 0)
        }
    
    async def _update_or_create_pattern(self, pattern_id: str, pattern_type: str,
                                       features: Dict[str, float], 
                                       outcomes: Dict[str, float]) -> None:
        """Update existing pattern or create new one."""
        with self._pattern_lock:
            if pattern_id in self.patterns:
                pattern = self.patterns[pattern_id]
                pattern.occurrence_count += 1
                
                # Update features with exponential moving average
                for key, value in features.items():
                    if key in pattern.features:
                        pattern.features[key] = (1 - self.learning_rate) * pattern.features[key] + \
                                              self.learning_rate * value
                    else:
                        pattern.features[key] = value
                
                # Update outcomes similarly
                for key, value in outcomes.items():
                    if key in pattern.outcomes:
                        pattern.outcomes[key] = (1 - self.learning_rate) * pattern.outcomes[key] + \
                                              self.learning_rate * value
                    else:
                        pattern.outcomes[key] = value
                
                pattern.last_seen = datetime.now()
                
            else:
                # Create new pattern
                self.patterns[pattern_id] = LearningPattern(
                    pattern_id=pattern_id,
                    pattern_type=pattern_type,
                    features=features.copy(),
                    outcomes=outcomes.copy(),
                    confidence_score=0.1,  # Start with low confidence
                    occurrence_count=1,
                    last_seen=datetime.now()
                )
            
            # Manage pattern memory - remove old patterns if too many
            if len(self.patterns) > self.max_patterns:
                await self._cleanup_old_patterns()
    
    async def _derive_adaptation_rules(self, pattern: LearningPattern) -> None:
        """Derive adaptation rules from strong patterns."""
        try:
            # Example: If response time is consistently high, suggest caching
            if (pattern.pattern_type == "performance" and 
                pattern.outcomes.get('response_time', 0) > 1000 and  # > 1 second
                pattern.confidence_score > 0.8):
                
                rule_id = f"cache_rule_{pattern.pattern_id}"
                condition = {
                    'response_time': {'min': 800},  # Trigger when response time > 800ms
                    'query_type': pattern.features.get('query_type', 'general')
                }
                action = {
                    'action_type': 'enable_aggressive_caching',
                    'parameters': {'cache_ttl': 3600, 'cache_size_mb': 100}
                }
                
                self.adaptation_rules[rule_id] = AdaptationRule(
                    rule_id=rule_id,
                    condition=condition,
                    action=action,
                    effectiveness_score=pattern.confidence_score,
                    application_count=0,
                    created_at=datetime.now()
                )
                
                self.logger.info(f"Derived adaptation rule: {rule_id}")
        
        except Exception as e:
            logger.error(f"Error deriving adaptation rules: {e}")
    
    async def _optimize_cache_settings(self) -> Dict[str, Any]:
        """Optimize cache settings based on usage patterns."""
        cache_patterns = [p for p in self.patterns.values() if 'cache_hit_rate' in p.outcomes]
        
        if not cache_patterns:
            return {'confidence': 0, 'reason': 'insufficient_data'}
        
        avg_hit_rate = sum(p.outcomes['cache_hit_rate'] for p in cache_patterns) / len(cache_patterns)
        
        if avg_hit_rate < 0.7:  # Low hit rate
            return {
                'action': 'increase_cache_size',
                'parameters': {'size_multiplier': 1.5},
                'confidence': 0.85,
                'reasoning': f'Current hit rate {avg_hit_rate:.2f} below target 0.7'
            }
        
        return {'confidence': 0.5, 'action': 'no_change'}
    
    async def _optimize_search_parameters(self) -> Dict[str, Any]:
        """Optimize search parameters based on relevance feedback."""
        search_patterns = [p for p in self.patterns.values() if p.pattern_type == 'query']
        
        if not search_patterns:
            return {'confidence': 0, 'reason': 'insufficient_data'}
        
        avg_relevance = sum(p.outcomes.get('relevance_score', 0.5) for p in search_patterns) / len(search_patterns)
        
        if avg_relevance < 0.8:  # Low relevance
            return {
                'action': 'adjust_search_weights',
                'parameters': {
                    'semantic_weight': 0.7,
                    'keyword_weight': 0.3,
                    'similarity_threshold': 0.6
                },
                'confidence': 0.9,
                'reasoning': f'Current relevance {avg_relevance:.2f} below target 0.8'
            }
        
        return {'confidence': 0.5, 'action': 'no_change'}
    
    async def _optimize_response_formatting(self) -> Dict[str, Any]:
        """Optimize response formatting based on user feedback."""
        return {
            'action': 'maintain_current_format',
            'confidence': 0.6,
            'reasoning': 'Response format optimization requires more user feedback data'
        }
    
    async def _optimize_resource_allocation(self) -> Dict[str, Any]:
        """Optimize resource allocation based on usage patterns."""
        performance_patterns = [p for p in self.patterns.values() if p.pattern_type == 'performance']
        
        if not performance_patterns:
            return {'confidence': 0, 'reason': 'insufficient_data'}
        
        avg_cpu = sum(p.features.get('cpu_usage', 0.5) for p in performance_patterns) / len(performance_patterns)
        avg_memory = sum(p.features.get('memory_usage', 0.5) for p in performance_patterns) / len(performance_patterns)
        
        optimizations = []
        
        if avg_cpu > 0.8:  # High CPU usage
            optimizations.append({
                'resource': 'cpu',
                'action': 'scale_up',
                'recommendation': 'Consider adding more CPU cores or instances'
            })
        
        if avg_memory > 0.8:  # High memory usage
            optimizations.append({
                'resource': 'memory',
                'action': 'optimize_memory',
                'recommendation': 'Implement memory pooling and garbage collection optimization'
            })
        
        return {
            'optimizations': optimizations,
            'confidence': 0.85 if optimizations else 0.3,
            'current_utilization': {'cpu': avg_cpu, 'memory': avg_memory}
        }
    
    async def _apply_optimization(self, category: str, optimization: Dict[str, Any]) -> None:
        """Apply an optimization to the system."""
        try:
            # In a real implementation, this would apply the optimization
            # For now, we just log the intended optimization
            self.adaptation_history.append({
                'timestamp': datetime.now(),
                'category': category,
                'optimization': optimization,
                'applied': True
            })
            
            self.logger.info(f"Applied optimization for {category}: {optimization.get('action', 'unknown')}")
            
        except Exception as e:
            logger.error(f"Error applying optimization {category}: {e}")
    
    async def _update_performance_baseline(self, metrics: Dict[str, float]) -> None:
        """Update performance baseline with exponential moving average."""
        alpha = 0.1  # Smoothing factor
        
        for metric, value in metrics.items():
            if metric in self.performance_baseline:
                self.performance_baseline[metric] = (1 - alpha) * self.performance_baseline[metric] + alpha * value
            else:
                self.performance_baseline[metric] = value
    
    async def _cleanup_old_patterns(self) -> None:
        """Remove old patterns to manage memory usage."""
        cutoff_time = datetime.now() - timedelta(hours=self.learning_window_hours)
        
        patterns_to_remove = [
            pattern_id for pattern_id, pattern in self.patterns.items()
            if pattern.last_seen < cutoff_time and pattern.confidence_score < 0.5
        ]
        
        for pattern_id in patterns_to_remove:
            del self.patterns[pattern_id]
        
        if patterns_to_remove:
            self.logger.debug(f"Cleaned up {len(patterns_to_remove)} old patterns")
    
    def get_learning_statistics(self) -> Dict[str, Any]:
        """Get current learning statistics."""
        return {
            'total_patterns': len(self.patterns),
            'pattern_types': {
                ptype: len([p for p in self.patterns.values() if p.pattern_type == ptype])
                for ptype in ['query', 'performance', 'error']
            },
            'adaptation_rules': len(self.adaptation_rules),
            'learning_rate': self.learning_rate,
            'performance_baseline': self.performance_baseline.copy(),
            'recent_adaptations': len(self.adaptation_history),
            'high_confidence_patterns': len([
                p for p in self.patterns.values() if p.confidence_score > self.min_pattern_confidence
            ])
        }


# Global instance management
_adaptive_learning_engine_instance: Optional[AdaptiveLearningEngine] = None


def get_adaptive_learning_engine(learning_rate: float = 0.1) -> AdaptiveLearningEngine:
    """Get or create the global adaptive learning engine instance."""
    global _adaptive_learning_engine_instance
    if _adaptive_learning_engine_instance is None:
        _adaptive_learning_engine_instance = AdaptiveLearningEngine(learning_rate)
    return _adaptive_learning_engine_instance


async def demonstrate_adaptive_learning() -> Dict[str, Any]:
    """Demonstrate adaptive learning capabilities."""
    engine = get_adaptive_learning_engine()
    
    # Simulate some learning scenarios
    await engine.learn_from_query(
        "How do I deploy the application?",
        "To deploy the application, follow these steps...",
        {'response_time': 1200, 'relevance_score': 0.9},
        user_feedback=0.8
    )
    
    await engine.learn_from_performance(
        {'response_time': 850, 'memory_usage': 0.75, 'cpu_usage': 0.6},
        {'active_users': 25, 'time_of_day': 14}
    )
    
    # Get recommendations and optimizations
    recommendations = await engine.get_adaptive_recommendations({
        'response_time': 1100,
        'query_type': 'technical',
        'user_count': 30
    })
    
    optimizations = await engine.optimize_system_parameters()
    statistics = engine.get_learning_statistics()
    
    return {
        'recommendations': recommendations,
        'optimizations': optimizations,
        'statistics': statistics,
        'timestamp': datetime.now().isoformat()
    }


if __name__ == "__main__":
    # Demo execution
    async def main():
        results = await demonstrate_adaptive_learning()
        print(json.dumps(results, indent=2, default=str))
    
    asyncio.run(main())
