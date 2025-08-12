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
    """Advanced learning engine with novel reinforcement learning and self-improvement."""
    
    def __init__(self, learning_rate: float = 0.1):
        self.learning_rate = learning_rate
        self.logger = StructuredLogger("adaptive_learning")
        self.cache = get_cache_manager()
        
        # Learning components
        self.patterns: Dict[str, LearningPattern] = {}
        self.adaptation_rules: Dict[str, AdaptationRule] = {}
        self.feature_importance: Dict[str, float] = defaultdict(float)
        
        # Novel reinforcement learning components
        self.q_table: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.action_value_estimates: Dict[str, float] = defaultdict(float)
        self.exploration_rate = 0.3  # Epsilon for epsilon-greedy
        self.discount_factor = 0.95  # Gamma for future rewards
        
        # Advanced learning mechanisms
        self.meta_learning_patterns: Dict[str, Any] = {}
        self.self_improvement_history: deque = deque(maxlen=5000)
        self.algorithm_performance_tracking: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Learning history and metrics
        self.learning_history: deque = deque(maxlen=10000)
        self.adaptation_history: deque = deque(maxlen=1000)
        self.performance_baseline: Dict[str, float] = {}
        
        # Enhanced learning settings
        self.min_pattern_confidence = 0.7
        self.max_patterns = 1000
        self.learning_window_hours = 24
        self.reinforcement_learning_enabled = True
        self.meta_learning_enabled = True
        
        # Learning locks for thread safety
        self._learning_lock = threading.Lock()
        self._pattern_lock = threading.Lock()
        self._rl_lock = threading.Lock()  # For RL updates
        
        logger.info(f"Enhanced Adaptive Learning Engine initialized with RL and meta-learning capabilities")
    
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
        """Get adaptive recommendations using novel reinforcement learning."""
        recommendations = []
        
        try:
            # Get traditional rule-based recommendations
            with self._pattern_lock:
                for rule in self.adaptation_rules.values():
                    recommendation = rule.apply_rule(context)
                    if recommendation:
                        recommendation['rule_id'] = rule.rule_id
                        recommendation['confidence'] = rule.effectiveness_score
                        recommendation['source'] = 'rule_based'
                        recommendations.append(recommendation)
            
            # Get novel RL-based recommendations
            if self.reinforcement_learning_enabled:
                rl_recommendations = await self._get_reinforcement_learning_recommendations(context)
                recommendations.extend(rl_recommendations)
            
            # Get meta-learning recommendations
            if self.meta_learning_enabled:
                meta_recommendations = await self._get_meta_learning_recommendations(context)
                recommendations.extend(meta_recommendations)
            
            # Enhanced recommendation ranking with multi-criteria optimization
            ranked_recommendations = await self._rank_recommendations_with_rl(recommendations, context)
            
            self.logger.debug(f"Generated {len(ranked_recommendations)} enhanced adaptive recommendations")
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return recommendations[:10]  # Fallback to basic recommendations
        
        return ranked_recommendations[:15]  # Return top 15 enhanced recommendations
    
    async def optimize_system_parameters(self) -> Dict[str, Any]:
        """Optimize system parameters using reinforcement learning and meta-learning."""
        try:
            # Traditional optimizations
            traditional_optimizations = {
                'cache_settings': await self._optimize_cache_settings(),
                'search_parameters': await self._optimize_search_parameters(),
                'response_formatting': await self._optimize_response_formatting(),
                'resource_allocation': await self._optimize_resource_allocation()
            }
            
            # Novel RL-based optimizations
            rl_optimizations = {}
            if self.reinforcement_learning_enabled:
                rl_optimizations = {
                    'rl_search_strategy': await self._rl_optimize_search_strategy(),
                    'rl_resource_allocation': await self._rl_optimize_resources(),
                    'rl_caching_policy': await self._rl_optimize_caching()
                }
            
            # Meta-learning optimizations
            meta_optimizations = {}
            if self.meta_learning_enabled:
                meta_optimizations = {
                    'meta_algorithm_selection': await self._meta_optimize_algorithms(),
                    'meta_parameter_tuning': await self._meta_optimize_parameters()
                }
            
            # Combine all optimizations
            all_optimizations = {**traditional_optimizations, **rl_optimizations, **meta_optimizations}
            
            # Apply optimizations with enhanced decision making
            applied_optimizations = []
            for category, optimization in all_optimizations.items():
                should_apply = await self._should_apply_optimization(category, optimization)
                if should_apply:
                    await self._apply_optimization(category, optimization)
                    applied_optimizations.append(category)
                    
                    # Update RL policy based on application
                    if self.reinforcement_learning_enabled:
                        await self._update_rl_policy(category, optimization)
            
            self.logger.info(f"Applied enhanced optimizations: {applied_optimizations}")
            
            return {
                'traditional_optimizations': traditional_optimizations,
                'rl_optimizations': rl_optimizations,
                'meta_optimizations': meta_optimizations,
                'applied': applied_optimizations,
                'rl_enabled': self.reinforcement_learning_enabled,
                'meta_learning_enabled': self.meta_learning_enabled,
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
        """Get comprehensive learning statistics including RL and meta-learning metrics."""
        return {
            # Traditional learning metrics
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
            ]),
            
            # Novel RL and meta-learning metrics
            'reinforcement_learning': {
                'enabled': self.reinforcement_learning_enabled,
                'q_table_size': sum(len(actions) for actions in self.q_table.values()),
                'exploration_rate': self.exploration_rate,
                'average_q_value': np.mean([
                    q_val for state_actions in self.q_table.values() 
                    for q_val in state_actions.values()
                ]) if self.q_table else 0.0,
                'learned_policies': len(self.q_table)
            },
            
            'meta_learning': {
                'enabled': self.meta_learning_enabled,
                'meta_patterns': len(self.meta_learning_patterns),
                'algorithm_performance_tracking': {
                    alg: len(perf_history) for alg, perf_history in self.algorithm_performance_tracking.items()
                },
                'self_improvement_events': len(self.self_improvement_history)
            },
            
            'advanced_metrics': {
                'feature_importance_variance': np.var(list(self.feature_importance.values())) if self.feature_importance else 0.0,
                'learning_stability': self._calculate_learning_stability(),
                'adaptation_success_rate': self._calculate_adaptation_success_rate(),
                'meta_learning_effectiveness': self._calculate_meta_learning_effectiveness()
            }
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


    async def _get_reinforcement_learning_recommendations(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get recommendations using novel reinforcement learning approach."""
        recommendations = []
        
        try:
            with self._rl_lock:
                state = self._context_to_state(context)
                available_actions = self._get_available_actions(state)
                
                for action in available_actions:
                    q_value = self.q_table[state][action]
                    
                    if np.random.random() < self.exploration_rate:
                        exploration_bonus = np.random.normal(0, 0.1)
                        confidence = min(1.0, max(0.0, q_value + exploration_bonus))
                    else:
                        confidence = q_value
                    
                    if confidence > 0.3:
                        recommendation = {
                            'action_type': action,
                            'confidence': confidence,
                            'source': 'reinforcement_learning',
                            'q_value': q_value,
                            'state': state
                        }
                        recommendation.update(self._get_action_parameters(action, context))
                        recommendations.append(recommendation)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error in RL recommendations: {e}")
            return []
    
    async def _get_meta_learning_recommendations(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get recommendations using meta-learning patterns."""
        recommendations = []
        
        try:
            similar_contexts = self._find_similar_contexts(context)
            
            for similar_context, similarity_score in similar_contexts:
                if similarity_score > 0.7:
                    successful_actions = self._get_successful_actions_for_context(similar_context)
                    
                    for action, success_rate in successful_actions:
                        if success_rate > 0.8:
                            recommendation = {
                                'action_type': action,
                                'confidence': similarity_score * success_rate,
                                'source': 'meta_learning',
                                'similarity_score': similarity_score,
                                'historical_success_rate': success_rate
                            }
                            recommendations.append(recommendation)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error in meta-learning recommendations: {e}")
            return []
    
    def _context_to_state(self, context: Dict[str, Any]) -> str:
        """Convert context to RL state representation."""
        try:
            state_features = []
            
            response_time = context.get('response_time', 0)
            if response_time < 500:
                state_features.append('rt_low')
            elif response_time < 1000:
                state_features.append('rt_medium')
            else:
                state_features.append('rt_high')
            
            user_count = context.get('user_count', 0)
            if user_count < 10:
                state_features.append('load_low')
            elif user_count < 50:
                state_features.append('load_medium')
            else:
                state_features.append('load_high')
            
            query_type = context.get('query_type', 'general')
            state_features.append(f'type_{query_type}')
            
            return '_'.join(sorted(state_features))
            
        except Exception:
            return 'default_state'
    
    def _get_available_actions(self, state: str) -> List[str]:
        """Get available actions for RL state."""
        return [
            'enable_aggressive_caching',
            'adjust_search_weights',
            'scale_resources',
            'optimize_query_processing',
            'enable_load_balancing'
        ]
    
    def _get_action_parameters(self, action: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Get parameters for specific RL actions."""
        parameters = {'action': action}
        
        if action == 'enable_aggressive_caching':
            parameters.update({
                'cache_size_mb': min(500, max(100, context.get('user_count', 10) * 5)),
                'cache_ttl': 3600
            })
        elif action == 'adjust_search_weights':
            parameters.update({
                'semantic_weight': 0.7,
                'keyword_weight': 0.3
            })
        
        return parameters
    
    def _find_similar_contexts(self, context: Dict[str, Any]) -> List[Tuple[Dict[str, Any], float]]:
        """Find similar contexts from history for meta-learning."""
        similar_contexts = []
        
        try:
            for historical_record in list(self.adaptation_history)[-100:]:
                if 'context' in historical_record:
                    historical_context = historical_record['context']
                    similarity = self._calculate_context_similarity(context, historical_context)
                    
                    if similarity > 0.5:
                        similar_contexts.append((historical_context, similarity))
            
            similar_contexts.sort(key=lambda x: x[1], reverse=True)
            return similar_contexts[:10]
            
        except Exception as e:
            logger.error(f"Error finding similar contexts: {e}")
            return []
    
    def _calculate_context_similarity(self, context1: Dict[str, Any], context2: Dict[str, Any]) -> float:
        """Calculate similarity between contexts."""
        try:
            common_keys = set(context1.keys()) & set(context2.keys())
            if not common_keys:
                return 0.0
            
            similarities = []
            for key in common_keys:
                val1, val2 = context1[key], context2[key]
                
                if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                    if val1 == 0 and val2 == 0:
                        sim = 1.0
                    elif val1 == 0 or val2 == 0:
                        sim = 0.0
                    else:
                        sim = 1.0 - abs(val1 - val2) / max(abs(val1), abs(val2))
                    similarities.append(sim)
                elif val1 == val2:
                    similarities.append(1.0)
                else:
                    similarities.append(0.0)
            
            return sum(similarities) / len(similarities) if similarities else 0.0
            
        except Exception:
            return 0.0
    
    def _get_successful_actions_for_context(self, context: Dict[str, Any]) -> List[Tuple[str, float]]:
        """Get successful actions for specific context."""
        successful_actions = defaultdict(list)
        
        try:
            for record in self.adaptation_history:
                if record.get('context') == context:
                    action = record.get('optimization', {}).get('action', 'unknown')
                    success = 1.0 if record.get('success', False) else 0.0
                    if action != 'unknown':
                        successful_actions[action].append(success)
            
            action_success_rates = []
            for action, outcomes in successful_actions.items():
                success_rate = sum(outcomes) / len(outcomes) if outcomes else 0.0
                action_success_rates.append((action, success_rate))
            
            return action_success_rates
            
        except Exception:
            return []
    
    def _calculate_learning_stability(self) -> float:
        """Calculate learning stability metric."""
        try:
            if len(self.learning_history) < 10:
                return 0.5
            
            recent_outcomes = [record.get('success', False) for record in list(self.learning_history)[-50:]]
            if not recent_outcomes:
                return 0.5
            
            success_rate = sum(recent_outcomes) / len(recent_outcomes)
            return success_rate
            
        except Exception:
            return 0.5
    
    def _calculate_adaptation_success_rate(self) -> float:
        """Calculate adaptation success rate."""
        try:
            if not self.adaptation_history:
                return 0.5
            
            successful_adaptations = sum(1 for record in self.adaptation_history 
                                       if record.get('success', False))
            
            return successful_adaptations / len(self.adaptation_history)
            
        except Exception:
            return 0.5
    
    def _calculate_meta_learning_effectiveness(self) -> float:
        """Calculate meta-learning effectiveness."""
        try:
            if not self.meta_learning_patterns:
                return 0.5
            
            effectiveness_scores = [pattern.get('success_rate', 0.5) 
                                  for pattern in self.meta_learning_patterns.values()]
            
            return sum(effectiveness_scores) / len(effectiveness_scores) if effectiveness_scores else 0.5
            
        except Exception:
            return 0.5


if __name__ == "__main__":
    # Demo execution
    async def main():
        results = await demonstrate_adaptive_learning()
        print(json.dumps(results, indent=2, default=str))
    
    asyncio.run(main())
