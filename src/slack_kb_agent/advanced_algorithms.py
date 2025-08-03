"""Advanced algorithms for intelligent knowledge processing and optimization."""

import asyncio
import logging
import numpy as np
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Set, Any
from enum import Enum

from .models import Document, DocumentType, SourceType
from .exceptions import KnowledgeBaseError


logger = logging.getLogger(__name__)


class QueryComplexity(Enum):
    """Query complexity classification."""
    SIMPLE = "simple"          # Single concept, direct lookup
    MODERATE = "moderate"      # Multiple concepts, needs reasoning  
    COMPLEX = "complex"        # Multi-step reasoning, context required
    EXPERT = "expert"          # Domain expertise needed


@dataclass
class KnowledgeGap:
    """Represents an identified gap in knowledge base."""
    
    topic: str
    frequency: int
    last_query: datetime
    suggested_sources: List[str]
    priority_score: float


@dataclass
class QueryInsight:
    """Analytics insight about query patterns."""
    
    query_type: str
    trend: str  # "increasing", "stable", "decreasing"
    success_rate: float
    avg_response_time: float
    recommended_optimization: str


class IntelligentQueryRouter:
    """Advanced query routing with machine learning-like capabilities."""
    
    def __init__(self, learning_window_days: int = 30):
        self.query_patterns: Dict[str, List[Tuple[str, float, datetime]]] = defaultdict(list)
        self.success_patterns: Dict[str, List[bool]] = defaultdict(list)
        self.response_times: Dict[str, List[float]] = defaultdict(list)
        self.learning_window = timedelta(days=learning_window_days)
        
    def classify_query_complexity(self, query: str) -> QueryComplexity:
        """Classify query complexity using heuristics."""
        query_lower = query.lower()
        word_count = len(query.split())
        
        # Simple patterns
        simple_indicators = [
            'what is', 'where is', 'when is', 'who is',
            'define', 'definition', 'meaning',
            'list', 'show me'
        ]
        
        # Complex patterns
        complex_indicators = [
            'compare', 'analyze', 'explain why', 'how does',
            'relationship between', 'impact of', 'pros and cons',
            'best practice', 'recommend', 'strategy'
        ]
        
        # Expert patterns
        expert_indicators = [
            'architecture', 'implement', 'design pattern',
            'performance optimization', 'security implications',
            'scalability', 'troubleshoot', 'debug'
        ]
        
        # Score based on indicators
        simple_score = sum(1 for pattern in simple_indicators if pattern in query_lower)
        complex_score = sum(1 for pattern in complex_indicators if pattern in query_lower)
        expert_score = sum(1 for pattern in expert_indicators if pattern in query_lower)
        
        # Complexity based on word count and indicators
        if expert_score > 0 or word_count > 20:
            return QueryComplexity.EXPERT
        elif complex_score > 0 or word_count > 10:
            return QueryComplexity.COMPLEX
        elif simple_score > 0 and word_count <= 5:
            return QueryComplexity.SIMPLE
        else:
            return QueryComplexity.MODERATE
    
    def route_query(self, query: str, user_context: Dict[str, Any]) -> Dict[str, Any]:
        """Route query based on complexity and user context."""
        complexity = self.classify_query_complexity(query)
        user_expertise = user_context.get('expertise_level', 'intermediate')
        
        routing_decision = {
            'complexity': complexity.value,
            'suggested_search_strategy': self._get_search_strategy(complexity),
            'context_requirements': self._get_context_requirements(complexity),
            'response_style': self._get_response_style(complexity, user_expertise),
            'escalation_needed': self._should_escalate(complexity, user_context)
        }
        
        # Log for learning
        self._record_routing_decision(query, routing_decision)
        
        return routing_decision
    
    def _get_search_strategy(self, complexity: QueryComplexity) -> str:
        """Determine optimal search strategy based on complexity."""
        strategies = {
            QueryComplexity.SIMPLE: "keyword_exact",
            QueryComplexity.MODERATE: "hybrid_weighted",
            QueryComplexity.COMPLEX: "semantic_deep",
            QueryComplexity.EXPERT: "multi_step_reasoning"
        }
        return strategies[complexity]
    
    def _get_context_requirements(self, complexity: QueryComplexity) -> List[str]:
        """Determine what context is needed for the query."""
        context_map = {
            QueryComplexity.SIMPLE: ["definition", "basic_info"],
            QueryComplexity.MODERATE: ["related_docs", "examples"],
            QueryComplexity.COMPLEX: ["background", "prerequisites", "related_topics"],
            QueryComplexity.EXPERT: ["full_context", "architectural_overview", "alternatives"]
        }
        return context_map[complexity]
    
    def _get_response_style(self, complexity: QueryComplexity, user_expertise: str) -> str:
        """Determine response style based on complexity and user expertise."""
        if complexity == QueryComplexity.SIMPLE:
            return "concise"
        elif complexity == QueryComplexity.EXPERT and user_expertise == "expert":
            return "technical_detailed"
        elif user_expertise == "beginner":
            return "explanatory_with_examples"
        else:
            return "structured_comprehensive"
    
    def _should_escalate(self, complexity: QueryComplexity, user_context: Dict[str, Any]) -> bool:
        """Determine if query should be escalated to human expert."""
        if complexity == QueryComplexity.EXPERT:
            # Check if we have low confidence in this topic
            topic_confidence = user_context.get('topic_confidence', 0.8)
            return topic_confidence < 0.6
        return False
    
    def _record_routing_decision(self, query: str, decision: Dict[str, Any]) -> None:
        """Record routing decision for learning."""
        topic = self._extract_main_topic(query)
        timestamp = datetime.utcnow()
        
        self.query_patterns[topic].append((query, 0.0, timestamp))  # Success score TBD
        
        # Clean old entries
        cutoff = datetime.utcnow() - self.learning_window
        for topic_key in list(self.query_patterns.keys()):
            self.query_patterns[topic_key] = [
                entry for entry in self.query_patterns[topic_key]
                if entry[2] > cutoff
            ]
    
    def _extract_main_topic(self, query: str) -> str:
        """Extract main topic from query for categorization."""
        # Simple keyword extraction - in production, use more sophisticated NLP
        keywords = query.lower().split()
        
        # Technical topic keywords
        tech_topics = {
            'database': ['database', 'sql', 'postgres', 'mysql'],
            'api': ['api', 'endpoint', 'rest', 'graphql'],
            'deployment': ['deploy', 'deployment', 'docker', 'kubernetes'],
            'auth': ['auth', 'authentication', 'login', 'token'],
            'monitoring': ['monitor', 'metrics', 'logging', 'observability']
        }
        
        for topic, topic_keywords in tech_topics.items():
            if any(keyword in keywords for keyword in topic_keywords):
                return topic
        
        return "general"


class KnowledgeGapAnalyzer:
    """Analyzes knowledge gaps and suggests improvements."""
    
    def __init__(self, min_frequency_threshold: int = 3):
        self.unanswered_queries: deque = deque(maxlen=1000)
        self.low_confidence_responses: deque = deque(maxlen=500)
        self.min_frequency = min_frequency_threshold
        
    def record_unanswered_query(self, query: str, timestamp: datetime = None) -> None:
        """Record a query that couldn't be answered satisfactorily."""
        if timestamp is None:
            timestamp = datetime.utcnow()
            
        self.unanswered_queries.append({
            'query': query,
            'timestamp': timestamp,
            'topic': self._extract_topic(query)
        })
    
    def record_low_confidence_response(
        self, 
        query: str, 
        confidence: float, 
        timestamp: datetime = None
    ) -> None:
        """Record a response with low confidence score."""
        if timestamp is None:
            timestamp = datetime.utcnow()
            
        self.low_confidence_responses.append({
            'query': query,
            'confidence': confidence,
            'timestamp': timestamp,
            'topic': self._extract_topic(query)
        })
    
    def identify_knowledge_gaps(self, days_window: int = 7) -> List[KnowledgeGap]:
        """Identify knowledge gaps from recent query patterns."""
        cutoff = datetime.utcnow() - timedelta(days=days_window)
        
        # Count topics from unanswered queries
        topic_counts = defaultdict(int)
        topic_last_query = {}
        
        for entry in self.unanswered_queries:
            if entry['timestamp'] > cutoff:
                topic = entry['topic']
                topic_counts[topic] += 1
                if topic not in topic_last_query or entry['timestamp'] > topic_last_query[topic]:
                    topic_last_query[topic] = entry['timestamp']
        
        # Also consider low confidence responses
        for entry in self.low_confidence_responses:
            if entry['timestamp'] > cutoff and entry['confidence'] < 0.5:
                topic = entry['topic']
                topic_counts[topic] += 0.5  # Weight lower than unanswered
                if topic not in topic_last_query or entry['timestamp'] > topic_last_query[topic]:
                    topic_last_query[topic] = entry['timestamp']
        
        # Create knowledge gaps for topics above threshold
        gaps = []
        for topic, frequency in topic_counts.items():
            if frequency >= self.min_frequency:
                priority_score = self._calculate_priority_score(topic, frequency, topic_last_query[topic])
                
                gaps.append(KnowledgeGap(
                    topic=topic,
                    frequency=int(frequency),
                    last_query=topic_last_query[topic],
                    suggested_sources=self._suggest_sources_for_topic(topic),
                    priority_score=priority_score
                ))
        
        # Sort by priority score descending
        return sorted(gaps, key=lambda gap: gap.priority_score, reverse=True)
    
    def _extract_topic(self, query: str) -> str:
        """Extract topic from query - simplified implementation."""
        # In production, use more sophisticated NLP
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['deploy', 'deployment', 'docker', 'k8s']):
            return 'deployment'
        elif any(word in query_lower for word in ['database', 'sql', 'postgres', 'migration']):
            return 'database'
        elif any(word in query_lower for word in ['api', 'endpoint', 'rest', 'graphql']):
            return 'api'
        elif any(word in query_lower for word in ['auth', 'login', 'token', 'permission']):
            return 'authentication'
        elif any(word in query_lower for word in ['test', 'testing', 'pytest', 'unit']):
            return 'testing'
        elif any(word in query_lower for word in ['error', 'bug', 'issue', 'problem']):
            return 'troubleshooting'
        else:
            return 'general'
    
    def _calculate_priority_score(self, topic: str, frequency: float, last_query: datetime) -> float:
        """Calculate priority score for knowledge gap."""
        # Base score from frequency
        frequency_score = min(frequency / 10.0, 1.0)  # Normalize to 0-1
        
        # Recency score (more recent = higher priority)
        days_ago = (datetime.utcnow() - last_query).days
        recency_score = max(0, 1.0 - (days_ago / 30.0))  # Decay over 30 days
        
        # Topic importance weights
        topic_weights = {
            'authentication': 1.2,
            'database': 1.1,
            'deployment': 1.15,
            'api': 1.0,
            'troubleshooting': 1.3,
            'testing': 0.9,
            'general': 0.8
        }
        
        topic_weight = topic_weights.get(topic, 1.0)
        
        return (frequency_score * 0.4 + recency_score * 0.6) * topic_weight
    
    def _suggest_sources_for_topic(self, topic: str) -> List[str]:
        """Suggest potential sources to fill knowledge gap."""
        source_suggestions = {
            'deployment': [
                'DevOps documentation',
                'Deployment runbooks',
                'Docker/Kubernetes guides',
                'CI/CD pipeline docs'
            ],
            'database': [
                'Database schema documentation',
                'Migration guides',
                'SQL reference materials',
                'Database administration docs'
            ],
            'api': [
                'API documentation',
                'OpenAPI/Swagger specs',
                'Integration guides',
                'API testing documentation'
            ],
            'authentication': [
                'Security documentation',
                'Authentication flow diagrams',
                'OAuth/JWT guides',
                'Permission matrices'
            ],
            'testing': [
                'Test documentation',
                'Testing best practices',
                'Test automation guides',
                'Quality assurance procedures'
            ],
            'troubleshooting': [
                'Troubleshooting guides',
                'Error documentation',
                'Support tickets/issues',
                'Incident reports'
            ]
        }
        
        return source_suggestions.get(topic, [
            'General documentation',
            'Team knowledge',
            'External resources'
        ])


class ContentQualityOptimizer:
    """Optimizes content quality and relevance scoring."""
    
    def __init__(self):
        self.document_scores: Dict[str, float] = {}
        self.user_feedback: Dict[str, List[Tuple[str, bool, datetime]]] = defaultdict(list)
        
    def calculate_document_relevance_score(
        self, 
        document: Document, 
        query: str,
        user_context: Dict[str, Any]
    ) -> float:
        """Calculate comprehensive relevance score for document."""
        
        # Base text similarity score (would use more sophisticated matching in production)
        text_score = self._calculate_text_similarity(document.content, query)
        
        # Document metadata scoring
        metadata_score = self._score_document_metadata(document, user_context)
        
        # Historical performance score
        historical_score = self._get_historical_performance_score(document.source)
        
        # Freshness score
        freshness_score = self._calculate_freshness_score(document)
        
        # Authority score (based on source type and document type)
        authority_score = self._calculate_authority_score(document)
        
        # Weighted combination
        relevance_score = (
            text_score * 0.35 +
            metadata_score * 0.15 +
            historical_score * 0.2 +
            freshness_score * 0.15 +
            authority_score * 0.15
        )
        
        return min(relevance_score, 1.0)
    
    def _calculate_text_similarity(self, content: str, query: str) -> float:
        """Simple text similarity calculation - use sentence transformers in production."""
        content_words = set(content.lower().split())
        query_words = set(query.lower().split())
        
        if not query_words:
            return 0.0
            
        intersection = content_words.intersection(query_words)
        union = content_words.union(query_words)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _score_document_metadata(self, document: Document, user_context: Dict[str, Any]) -> float:
        """Score document based on metadata relevance."""
        score = 0.5  # Base score
        
        # Priority boost
        if document.priority > 3:
            score += 0.2
        
        # Tags relevance (simplified)
        user_interests = user_context.get('interests', [])
        if any(tag in user_interests for tag in document.tags):
            score += 0.3
        
        return min(score, 1.0)
    
    def _get_historical_performance_score(self, source: str) -> float:
        """Get historical performance score for document source."""
        if source in self.document_scores:
            return self.document_scores[source]
        return 0.5  # Default score for new sources
    
    def _calculate_freshness_score(self, document: Document) -> float:
        """Calculate freshness score based on document age."""
        if not document.updated_at and not document.created_at:
            return 0.3  # Low score for documents without timestamps
        
        last_update = document.updated_at or document.created_at
        days_old = (datetime.utcnow() - last_update).days
        
        # Decay function: newer content gets higher scores
        if days_old <= 7:
            return 1.0
        elif days_old <= 30:
            return 0.8
        elif days_old <= 90:
            return 0.6
        elif days_old <= 365:
            return 0.4
        else:
            return 0.2
    
    def _calculate_authority_score(self, document: Document) -> float:
        """Calculate authority score based on source and document type."""
        
        # Source type weights
        source_weights = {
            SourceType.GITHUB: 0.9,      # High authority for code/issues
            SourceType.API_IMPORT: 0.8,   # High for structured data
            SourceType.FILE_SYSTEM: 0.7, # Medium for local docs
            SourceType.WEB_CRAWL: 0.6,   # Medium for web content
            SourceType.SLACK: 0.5,       # Lower for chat messages
            SourceType.MANUAL_ENTRY: 0.4  # Lowest for manual entries
        }
        
        # Document type weights
        doc_type_weights = {
            DocumentType.API_DOCUMENTATION: 0.9,
            DocumentType.CODE: 0.8,
            DocumentType.MARKDOWN: 0.7,
            DocumentType.ISSUE: 0.6,
            DocumentType.PULL_REQUEST: 0.6,
            DocumentType.TEXT: 0.5,
            DocumentType.SLACK_MESSAGE: 0.4,
            DocumentType.WEB_CONTENT: 0.5
        }
        
        source_score = source_weights.get(document.source_type, 0.5)
        doc_type_score = doc_type_weights.get(document.doc_type, 0.5)
        
        return (source_score + doc_type_score) / 2
    
    def record_user_feedback(
        self, 
        document_source: str, 
        query: str, 
        was_helpful: bool
    ) -> None:
        """Record user feedback on document helpfulness."""
        self.user_feedback[document_source].append((
            query, 
            was_helpful, 
            datetime.utcnow()
        ))
        
        # Update document score based on feedback
        self._update_document_score(document_source)
    
    def _update_document_score(self, source: str) -> None:
        """Update document score based on accumulated feedback."""
        feedback_list = self.user_feedback[source]
        
        if not feedback_list:
            return
        
        # Consider only recent feedback (last 30 days)
        cutoff = datetime.utcnow() - timedelta(days=30)
        recent_feedback = [
            fb for fb in feedback_list 
            if fb[2] > cutoff
        ]
        
        if not recent_feedback:
            return
        
        # Calculate score based on positive feedback ratio
        positive_count = sum(1 for fb in recent_feedback if fb[1])
        total_count = len(recent_feedback)
        
        # Weighted by recency and volume
        base_score = positive_count / total_count
        volume_weight = min(total_count / 10.0, 1.0)  # More feedback = more reliable
        
        self.document_scores[source] = base_score * volume_weight + 0.5 * (1 - volume_weight)


class PerformanceOptimizer:
    """Optimizes system performance based on usage patterns."""
    
    def __init__(self):
        self.query_response_times: deque = deque(maxlen=1000)
        self.cache_hit_rates: deque = deque(maxlen=100)
        self.resource_usage: deque = deque(maxlen=100)
        
    def analyze_performance_patterns(self) -> Dict[str, Any]:
        """Analyze performance patterns and suggest optimizations."""
        if not self.query_response_times:
            return {"status": "insufficient_data"}
        
        # Response time analysis
        response_times = [entry['response_time'] for entry in self.query_response_times]
        avg_response_time = np.mean(response_times)
        p95_response_time = np.percentile(response_times, 95)
        
        # Cache performance
        cache_hit_rate = np.mean(self.cache_hit_rates) if self.cache_hit_rates else 0.0
        
        # Identify slow query patterns
        slow_queries = [
            entry for entry in self.query_response_times 
            if entry['response_time'] > avg_response_time * 2
        ]
        
        recommendations = []
        
        # Performance recommendations
        if avg_response_time > 2.0:  # Seconds
            recommendations.append("Consider query optimization or indexing improvements")
        
        if cache_hit_rate < 0.3:
            recommendations.append("Improve caching strategy for frequently accessed content")
        
        if len(slow_queries) > len(response_times) * 0.1:  # More than 10% slow queries
            recommendations.append("Investigate and optimize slow query patterns")
        
        return {
            "avg_response_time": avg_response_time,
            "p95_response_time": p95_response_time,
            "cache_hit_rate": cache_hit_rate,
            "slow_query_count": len(slow_queries),
            "recommendations": recommendations,
            "performance_score": self._calculate_performance_score(
                avg_response_time, cache_hit_rate, len(slow_queries) / len(response_times)
            )
        }
    
    def _calculate_performance_score(
        self, 
        avg_response_time: float, 
        cache_hit_rate: float, 
        slow_query_ratio: float
    ) -> float:
        """Calculate overall performance score (0-1, higher is better)."""
        
        # Response time score (inverse relationship)
        time_score = max(0, 1.0 - (avg_response_time - 0.5) / 2.0)
        
        # Cache score (direct relationship)
        cache_score = cache_hit_rate
        
        # Reliability score (inverse of slow query ratio)
        reliability_score = max(0, 1.0 - slow_query_ratio * 2)
        
        return (time_score * 0.4 + cache_score * 0.3 + reliability_score * 0.3)
    
    def record_query_performance(
        self, 
        query: str, 
        response_time: float, 
        cache_hit: bool,
        resource_usage: Dict[str, float]
    ) -> None:
        """Record query performance metrics."""
        self.query_response_times.append({
            'query': query,
            'response_time': response_time,
            'timestamp': datetime.utcnow(),
            'cache_hit': cache_hit
        })
        
        self.cache_hit_rates.append(1.0 if cache_hit else 0.0)
        self.resource_usage.append(resource_usage)