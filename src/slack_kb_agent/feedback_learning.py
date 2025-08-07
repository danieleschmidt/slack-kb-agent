"""Advanced feedback learning system for continuous improvement.

This module implements machine learning capabilities that learn from user feedback
to improve query understanding, response ranking, and content curation over time.
"""

from __future__ import annotations

import json
import time
import logging
import hashlib
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any, Set
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    logger.warning("NumPy not available, using basic learning algorithms")


class FeedbackType(Enum):
    """Types of feedback that can be collected."""
    POSITIVE = "positive"          # User found response helpful
    NEGATIVE = "negative"          # User found response unhelpful  
    ESCALATION = "escalation"      # User escalated to human
    REFINEMENT = "refinement"      # User asked follow-up question
    RATING = "rating"              # Explicit rating (1-5 stars)
    CLICK_THROUGH = "click_through" # User clicked on suggested link
    TIME_SPENT = "time_spent"      # How long user engaged with response


class LearningDomain(Enum):
    """Areas where learning can be applied."""
    QUERY_UNDERSTANDING = "query_understanding"
    RESPONSE_RANKING = "response_ranking"
    CONTENT_RELEVANCE = "content_relevance"
    FOLLOW_UP_SUGGESTIONS = "follow_up_suggestions"
    ESCALATION_TRIGGERS = "escalation_triggers"


@dataclass
class FeedbackEvent:
    """Individual feedback event from user interaction."""
    event_id: str
    query: str
    response: str
    feedback_type: FeedbackType
    user_id: Optional[str]
    timestamp: float
    context: Dict[str, Any]
    rating: Optional[int] = None    # 1-5 for explicit ratings
    duration: Optional[float] = None  # Time spent with response
    documents_used: List[str] = None  # Document IDs that contributed
    follow_up_query: Optional[str] = None  # If user refined query
    
    def __post_init__(self):
        if self.documents_used is None:
            self.documents_used = []


@dataclass
class LearningMetrics:
    """Metrics tracking learning system performance."""
    total_feedback_events: int
    positive_feedback_rate: float
    escalation_rate: float
    average_rating: float
    learned_patterns: int
    improvement_trends: Dict[str, float]


class PatternLearner:
    """Learn patterns from user feedback to improve responses."""
    
    def __init__(self, max_patterns: int = 1000):
        self.max_patterns = max_patterns
        self.query_patterns: Dict[str, Dict[str, float]] = {}
        self.response_patterns: Dict[str, Dict[str, float]] = {}
        self.escalation_patterns: Set[str] = set()
        self.high_value_patterns: Set[str] = set()
        
    def learn_from_feedback(self, event: FeedbackEvent):
        """Learn patterns from a feedback event."""
        query_tokens = self._tokenize_query(event.query)
        query_signature = self._generate_signature(query_tokens)
        
        # Learn query patterns
        if query_signature not in self.query_patterns:
            self.query_patterns[query_signature] = defaultdict(float)
        
        pattern_data = self.query_patterns[query_signature]
        
        # Update pattern weights based on feedback
        if event.feedback_type == FeedbackType.POSITIVE:
            pattern_data['positive_weight'] += 1.0
            pattern_data['total_interactions'] += 1.0
            
        elif event.feedback_type == FeedbackType.NEGATIVE:
            pattern_data['negative_weight'] += 1.0
            pattern_data['total_interactions'] += 1.0
            
        elif event.feedback_type == FeedbackType.ESCALATION:
            pattern_data['escalation_weight'] += 1.0
            pattern_data['total_interactions'] += 1.0
            self.escalation_patterns.add(query_signature)
            
        elif event.feedback_type == FeedbackType.RATING and event.rating:
            pattern_data['rating_sum'] += event.rating
            pattern_data['rating_count'] += 1.0
            pattern_data['total_interactions'] += 1.0
            
            # High ratings indicate high-value patterns
            if event.rating >= 4:
                self.high_value_patterns.add(query_signature)
        
        # Learn response effectiveness patterns
        self._learn_response_patterns(event)
        
        # Maintain pattern limits
        self._prune_patterns()
    
    def _tokenize_query(self, query: str) -> List[str]:
        """Extract meaningful tokens from query."""
        import re
        
        # Convert to lowercase and extract words
        tokens = re.findall(r'\w+', query.lower())
        
        # Filter out common stop words
        stop_words = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'how', 'what', 'where', 'when', 'why'
        }
        
        meaningful_tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
        return meaningful_tokens
    
    def _generate_signature(self, tokens: List[str]) -> str:
        """Generate a signature for a set of tokens."""
        # Sort tokens to create consistent signatures
        sorted_tokens = sorted(tokens)
        signature = '|'.join(sorted_tokens[:5])  # Use top 5 tokens
        return hashlib.md5(signature.encode()).hexdigest()[:8]
    
    def _learn_response_patterns(self, event: FeedbackEvent):
        """Learn which types of responses work best."""
        response_features = self._extract_response_features(event.response)
        response_signature = self._generate_signature(response_features)
        
        if response_signature not in self.response_patterns:
            self.response_patterns[response_signature] = defaultdict(float)
        
        pattern_data = self.response_patterns[response_signature]
        
        if event.feedback_type == FeedbackType.POSITIVE:
            pattern_data['effectiveness_score'] += 1.0
        elif event.feedback_type == FeedbackType.NEGATIVE:
            pattern_data['effectiveness_score'] -= 0.5
        elif event.feedback_type == FeedbackType.RATING and event.rating:
            pattern_data['effectiveness_score'] += (event.rating - 3) * 0.5  # Center on 3
        
        pattern_data['usage_count'] += 1.0
    
    def _extract_response_features(self, response: str) -> List[str]:
        """Extract features from response text."""
        features = []
        
        # Length-based features
        if len(response) < 100:
            features.append('short_response')
        elif len(response) > 500:
            features.append('long_response')
        else:
            features.append('medium_response')
        
        # Content-based features
        if 'code' in response.lower() or '```' in response:
            features.append('includes_code')
        if 'http' in response.lower():
            features.append('includes_links')
        if response.count('\n') > 3:
            features.append('structured_format')
        if '1.' in response or '•' in response or '-' in response:
            features.append('includes_list')
        
        return features
    
    def _prune_patterns(self):
        """Remove least useful patterns to maintain memory limits."""
        if len(self.query_patterns) > self.max_patterns:
            # Sort patterns by usefulness score
            pattern_scores = []
            for signature, data in self.query_patterns.items():
                score = self._calculate_pattern_usefulness(data)
                pattern_scores.append((signature, score))
            
            # Keep only the most useful patterns
            pattern_scores.sort(key=lambda x: x[1], reverse=True)
            self.query_patterns = {
                sig: self.query_patterns[sig] 
                for sig, _ in pattern_scores[:self.max_patterns]
            }
    
    def _calculate_pattern_usefulness(self, pattern_data: Dict[str, float]) -> float:
        """Calculate how useful a learned pattern is."""
        total = pattern_data.get('total_interactions', 0)
        if total == 0:
            return 0.0
        
        positive = pattern_data.get('positive_weight', 0)
        negative = pattern_data.get('negative_weight', 0)
        
        # Calculate success rate with smoothing
        success_rate = (positive + 1) / (total + 2)  # Laplace smoothing
        
        # Weight by frequency (popular patterns are more valuable)
        frequency_weight = min(total / 10.0, 1.0)
        
        return success_rate * frequency_weight
    
    def predict_query_success(self, query: str) -> float:
        """Predict likelihood of successful response for a query."""
        tokens = self._tokenize_query(query)
        signature = self._generate_signature(tokens)
        
        if signature in self.query_patterns:
            pattern_data = self.query_patterns[signature]
            return self._calculate_pattern_usefulness(pattern_data)
        
        return 0.5  # Default neutral prediction
    
    def should_escalate(self, query: str) -> bool:
        """Determine if query should be escalated based on learned patterns."""
        tokens = self._tokenize_query(query)
        signature = self._generate_signature(tokens)
        
        if signature in self.escalation_patterns:
            pattern_data = self.query_patterns.get(signature, {})
            escalation_weight = pattern_data.get('escalation_weight', 0)
            total_interactions = pattern_data.get('total_interactions', 1)
            
            # Escalate if escalation rate > 50%
            return (escalation_weight / total_interactions) > 0.5
        
        return False
    
    def get_learned_patterns_count(self) -> int:
        """Get number of learned patterns."""
        return len(self.query_patterns) + len(self.response_patterns)


class FeedbackCollector:
    """Collect and manage user feedback events."""
    
    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = Path(storage_path) if storage_path else Path("feedback_data.json")
        self.feedback_buffer: deque = deque(maxlen=1000)  # In-memory buffer
        self.user_feedback_history: Dict[str, List[FeedbackEvent]] = defaultdict(list)
        
        # Load existing feedback data
        self._load_feedback_data()
    
    def collect_feedback(self, event: FeedbackEvent):
        """Collect a feedback event."""
        # Add to buffer
        self.feedback_buffer.append(event)
        
        # Add to user history
        if event.user_id:
            self.user_feedback_history[event.user_id].append(event)
            
            # Limit per-user history
            if len(self.user_feedback_history[event.user_id]) > 100:
                self.user_feedback_history[event.user_id] = \
                    self.user_feedback_history[event.user_id][-100:]
        
        # Persist to storage periodically
        if len(self.feedback_buffer) % 10 == 0:
            self._persist_feedback_data()
    
    def get_feedback_for_query(self, query_hash: str) -> List[FeedbackEvent]:
        """Get all feedback events for a specific query pattern."""
        matching_events = []
        for event in self.feedback_buffer:
            if self._hash_query(event.query) == query_hash:
                matching_events.append(event)
        return matching_events
    
    def get_user_feedback_history(self, user_id: str, limit: int = 50) -> List[FeedbackEvent]:
        """Get feedback history for a specific user."""
        return self.user_feedback_history.get(user_id, [])[-limit:]
    
    def calculate_user_satisfaction(self, user_id: str) -> float:
        """Calculate overall satisfaction score for a user."""
        history = self.user_feedback_history.get(user_id, [])
        if not history:
            return 0.5
        
        positive_count = sum(1 for event in history if event.feedback_type == FeedbackType.POSITIVE)
        negative_count = sum(1 for event in history if event.feedback_type == FeedbackType.NEGATIVE)
        rating_events = [event for event in history if event.feedback_type == FeedbackType.RATING and event.rating]
        
        if rating_events:
            avg_rating = sum(event.rating for event in rating_events) / len(rating_events)
            return avg_rating / 5.0  # Normalize to 0-1
        
        total_feedback = positive_count + negative_count
        if total_feedback == 0:
            return 0.5
        
        return positive_count / total_feedback
    
    def _hash_query(self, query: str) -> str:
        """Generate hash for query to group similar queries."""
        normalized = query.lower().strip()
        return hashlib.md5(normalized.encode()).hexdigest()[:8]
    
    def _load_feedback_data(self):
        """Load feedback data from storage."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                    
                # Restore feedback buffer
                for event_data in data.get('feedback_buffer', []):
                    event = FeedbackEvent(**event_data)
                    self.feedback_buffer.append(event)
                
                # Restore user histories
                for user_id, events_data in data.get('user_histories', {}).items():
                    events = [FeedbackEvent(**event_data) for event_data in events_data]
                    self.user_feedback_history[user_id] = events
                    
            except Exception as e:
                logger.warning(f"Failed to load feedback data: {e}")
    
    def _persist_feedback_data(self):
        """Persist feedback data to storage."""
        try:
            data = {
                'feedback_buffer': [asdict(event) for event in list(self.feedback_buffer)],
                'user_histories': {
                    user_id: [asdict(event) for event in events]
                    for user_id, events in self.user_feedback_history.items()
                }
            }
            
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Failed to persist feedback data: {e}")


class AdaptiveResponseRanker:
    """Rank and select responses based on learned feedback patterns."""
    
    def __init__(self):
        self.document_scores: Dict[str, float] = defaultdict(float)
        self.query_document_associations: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        
    def learn_from_feedback(self, event: FeedbackEvent):
        """Update ranking based on feedback."""
        if not event.documents_used:
            return
        
        query_signature = self._get_query_signature(event.query)
        
        # Update document scores based on feedback
        score_delta = self._calculate_score_delta(event)
        
        for doc_id in event.documents_used:
            self.document_scores[doc_id] += score_delta
            self.query_document_associations[query_signature][doc_id] += score_delta
    
    def rank_documents(self, documents: List[str], query: str) -> List[Tuple[str, float]]:
        """Rank documents based on learned preferences."""
        query_signature = self._get_query_signature(query)
        
        scored_docs = []
        for doc_id in documents:
            # Base score from general document quality
            base_score = self.document_scores.get(doc_id, 0.0)
            
            # Query-specific association score
            query_score = self.query_document_associations[query_signature].get(doc_id, 0.0)
            
            # Combined score with query preference weighting
            total_score = base_score * 0.3 + query_score * 0.7
            scored_docs.append((doc_id, total_score))
        
        # Sort by score descending
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return scored_docs
    
    def _get_query_signature(self, query: str) -> str:
        """Generate signature for query type."""
        # Extract key terms and normalize
        import re
        terms = re.findall(r'\w+', query.lower())
        key_terms = [t for t in terms if len(t) > 3][:3]  # Top 3 key terms
        return '|'.join(sorted(key_terms))
    
    def _calculate_score_delta(self, event: FeedbackEvent) -> float:
        """Calculate how much to adjust document scores."""
        if event.feedback_type == FeedbackType.POSITIVE:
            return 1.0
        elif event.feedback_type == FeedbackType.NEGATIVE:
            return -0.5
        elif event.feedback_type == FeedbackType.ESCALATION:
            return -1.0
        elif event.feedback_type == FeedbackType.RATING and event.rating:
            return (event.rating - 3) * 0.5  # Center on rating of 3
        else:
            return 0.0


class FeedbackLearningSystem:
    """Main feedback learning system orchestrating all components."""
    
    def __init__(self, storage_path: Optional[str] = None):
        self.feedback_collector = FeedbackCollector(storage_path)
        self.pattern_learner = PatternLearner()
        self.response_ranker = AdaptiveResponseRanker()
        
        # Load existing patterns
        self._initialize_from_existing_data()
    
    def record_feedback(self, query: str, response: str, feedback_type: FeedbackType,
                       user_id: Optional[str] = None, rating: Optional[int] = None,
                       documents_used: Optional[List[str]] = None,
                       context: Optional[Dict[str, Any]] = None) -> str:
        """Record a feedback event and trigger learning."""
        event_id = f"{int(time.time())}_{hashlib.md5(query.encode()).hexdigest()[:8]}"
        
        event = FeedbackEvent(
            event_id=event_id,
            query=query,
            response=response,
            feedback_type=feedback_type,
            user_id=user_id,
            timestamp=time.time(),
            context=context or {},
            rating=rating,
            documents_used=documents_used or []
        )
        
        # Collect feedback
        self.feedback_collector.collect_feedback(event)
        
        # Learn from feedback
        self.pattern_learner.learn_from_feedback(event)
        self.response_ranker.learn_from_feedback(event)
        
        logger.info(f"Recorded feedback event: {event_id}")
        return event_id
    
    def should_escalate_query(self, query: str) -> bool:
        """Determine if query should be escalated based on learned patterns."""
        return self.pattern_learner.should_escalate(query)
    
    def predict_query_success_rate(self, query: str) -> float:
        """Predict success rate for a query."""
        return self.pattern_learner.predict_query_success(query)
    
    def rank_response_documents(self, documents: List[str], query: str) -> List[Tuple[str, float]]:
        """Rank documents for response based on learned preferences."""
        return self.response_ranker.rank_documents(documents, query)
    
    def get_user_satisfaction_score(self, user_id: str) -> float:
        """Get satisfaction score for a user."""
        return self.feedback_collector.calculate_user_satisfaction(user_id)
    
    def get_learning_metrics(self) -> LearningMetrics:
        """Get metrics about the learning system performance."""
        total_events = len(self.feedback_collector.feedback_buffer)
        
        if total_events == 0:
            return LearningMetrics(
                total_feedback_events=0,
                positive_feedback_rate=0.0,
                escalation_rate=0.0,
                average_rating=0.0,
                learned_patterns=0,
                improvement_trends={}
            )
        
        # Calculate metrics from feedback buffer
        positive_count = sum(1 for event in self.feedback_collector.feedback_buffer 
                           if event.feedback_type == FeedbackType.POSITIVE)
        escalation_count = sum(1 for event in self.feedback_collector.feedback_buffer 
                             if event.feedback_type == FeedbackType.ESCALATION)
        rating_events = [event for event in self.feedback_collector.feedback_buffer 
                        if event.feedback_type == FeedbackType.RATING and event.rating]
        
        return LearningMetrics(
            total_feedback_events=total_events,
            positive_feedback_rate=positive_count / total_events,
            escalation_rate=escalation_count / total_events,
            average_rating=sum(e.rating for e in rating_events) / len(rating_events) if rating_events else 0.0,
            learned_patterns=self.pattern_learner.get_learned_patterns_count(),
            improvement_trends=self._calculate_improvement_trends()
        )
    
    def _initialize_from_existing_data(self):
        """Initialize learning components from existing feedback data."""
        for event in self.feedback_collector.feedback_buffer:
            self.pattern_learner.learn_from_feedback(event)
            self.response_ranker.learn_from_feedback(event)
    
    def _calculate_improvement_trends(self) -> Dict[str, float]:
        """Calculate improvement trends over time."""
        # Simple implementation - could be enhanced with time-series analysis
        events = list(self.feedback_collector.feedback_buffer)
        if len(events) < 10:
            return {"insufficient_data": 0.0}
        
        # Compare first half vs second half
        midpoint = len(events) // 2
        first_half = events[:midpoint]
        second_half = events[midpoint:]
        
        def calculate_success_rate(event_list):
            if not event_list:
                return 0.0
            positive = sum(1 for e in event_list if e.feedback_type == FeedbackType.POSITIVE)
            return positive / len(event_list)
        
        first_success = calculate_success_rate(first_half)
        second_success = calculate_success_rate(second_half)
        
        return {
            "success_rate_trend": second_success - first_success,
            "learning_velocity": len(events) / max(1, len(self.pattern_learner.query_patterns))
        }


# Convenience functions for integration
def create_feedback_system(storage_path: Optional[str] = None) -> FeedbackLearningSystem:
    """Create and return a feedback learning system."""
    return FeedbackLearningSystem(storage_path)


def record_positive_feedback(system: FeedbackLearningSystem, query: str, response: str,
                           user_id: Optional[str] = None, documents: Optional[List[str]] = None) -> str:
    """Convenience function to record positive feedback."""
    return system.record_feedback(
        query=query,
        response=response,
        feedback_type=FeedbackType.POSITIVE,
        user_id=user_id,
        documents_used=documents
    )


def record_negative_feedback(system: FeedbackLearningSystem, query: str, response: str,
                           user_id: Optional[str] = None, documents: Optional[List[str]] = None) -> str:
    """Convenience function to record negative feedback."""
    return system.record_feedback(
        query=query,
        response=response,
        feedback_type=FeedbackType.NEGATIVE,
        user_id=user_id,
        documents_used=documents
    )