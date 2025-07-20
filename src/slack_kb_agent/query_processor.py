"""Query processing and contextual question answering."""

from __future__ import annotations

import re
import time
import logging
from collections import OrderedDict
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

from .analytics import UsageAnalytics
from .knowledge_base import KnowledgeBase
from .models import Document
from .smart_routing import RoutingEngine, TeamMember
from .llm import get_response_generator, LLMResponse
from .monitoring import get_global_metrics, StructuredLogger

logger = logging.getLogger(__name__)


class QueryIntent(Enum):
    """Types of query intents."""
    QUESTION = "question"  # How-to, what-is questions
    COMMAND = "command"    # Direct action requests
    TROUBLESHOOTING = "troubleshooting"  # Problem/error resolution
    DEFINITION = "definition"  # What is X?
    SEARCH = "search"      # Show me, find X
    CONVERSATIONAL = "conversational"  # Thanks, hello, etc.
    
    @classmethod
    def classify(cls, query: str) -> 'QueryIntent':
        """Classify query intent using pattern matching."""
        query_lower = query.lower().strip()
        
        # Question patterns
        question_patterns = [
            r'^(how|what|where|when|why|which)\b',
            r'\?$',
            r'\b(can i|should i|is it|are there)\b'
        ]
        
        # Command patterns
        command_patterns = [
            r'^(deploy|run|start|stop|create|delete|update)\b',
            r'\b(please|now|immediately)\b',
            r'^(show me|give me|list)\b'
        ]
        
        # Troubleshooting patterns
        trouble_patterns = [
            r'\b(error|fail|broken|not work|not working|issue|problem)\b',
            r'\b(can\'t|cannot|doesn\'t|won\'t|not starting|not running)\b',
            r'\b(debug|fix|solve|resolve)\b'
        ]
        
        # Definition patterns
        definition_patterns = [
            r'^what (is|are)\b',
            r'\bdefinition of\b',
            r'\bexplain\b'
        ]
        
        # Search patterns
        search_patterns = [
            r'^(show|find|search|locate|get)\b',
            r'\bdocumentation\b',
            r'\bexamples?\b'
        ]
        
        # Conversational patterns
        conversational_patterns = [
            r'^(hi|hello|hey|thanks|thank you)\b',
            r'^(ok|okay|cool|great)\b',
            r'\b(please|sorry)\b'
        ]
        
        # Check patterns in order of specificity
        if any(re.search(p, query_lower) for p in definition_patterns):
            return cls.DEFINITION
        elif any(re.search(p, query_lower) for p in trouble_patterns):
            return cls.TROUBLESHOOTING
        elif any(re.search(p, query_lower) for p in search_patterns):
            return cls.SEARCH
        elif any(re.search(p, query_lower) for p in command_patterns):
            return cls.COMMAND
        elif any(re.search(p, query_lower) for p in question_patterns):
            return cls.QUESTION
        elif any(re.search(p, query_lower) for p in conversational_patterns):
            return cls.CONVERSATIONAL
        else:
            return cls.QUESTION  # Default to question
    
    @classmethod
    def classify_with_confidence(cls, query: str) -> Dict[str, Any]:
        """Classify with confidence score."""
        intent = cls.classify(query)
        # Simple confidence based on pattern strength
        confidence = 0.8 if len(query.split()) > 2 else 0.6
        return {"intent": intent, "confidence": confidence}


@dataclass
class QueryResult:
    """Enhanced query result with metadata."""
    original_query: str
    normalized_query: str
    intent: QueryIntent
    expanded_terms: List[str]
    documents: List[Document]
    suggestions: Optional[List[str]] = None
    context_used: bool = False
    processing_time: Optional[float] = None
    error_message: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)


class QueryExpansion:
    """Handles query expansion and term enhancement."""
    
    def __init__(self):
        # Common technical synonyms
        self.synonym_map = {
            "deploy": ["deployment", "release", "publish", "launch", "staging"],
            "api": ["endpoint", "service", "interface", "rest"],
            "auth": ["authentication", "authorization", "login", "oauth"],
            "ci": ["continuous integration", "pipeline", "build"],
            "cd": ["continuous deployment", "continuous delivery"],
            "docker": ["container", "containerization"],
            "k8s": ["kubernetes", "orchestration"],
            "db": ["database", "storage", "persistence"],
            "config": ["configuration", "settings", "environment"],
            "log": ["logging", "logs", "monitoring"]
        }
    
    def expand_synonyms(self, query: str) -> List[str]:
        """Expand query with synonyms."""
        expanded = [query]
        words = query.lower().split()
        
        # Add individual words that have synonyms
        for word in words:
            if word in self.synonym_map:
                expanded.append(word)  # Add the original word too
                for synonym in self.synonym_map[word]:
                    expanded.append(synonym)
        
        # Return unique list preserving original query at the start
        seen = set()
        result = []
        for item in expanded:
            if item not in seen:
                seen.add(item)
                result.append(item)
        return result
    
    def expand_technical_terms(self, query: str) -> List[str]:
        """Expand technical abbreviations and acronyms."""
        expansions = {
            "ci/cd": ["continuous integration", "continuous deployment", "pipeline"],
            "api": ["application programming interface", "endpoint", "service"],
            "oauth": ["authentication", "authorization", "login"],
            "docker": ["container", "containerization", "deployment"],
            "k8s": ["kubernetes", "container orchestration"],
            "sql": ["database", "query", "structured query language"],
            "json": ["javascript object notation", "data format"],
            "rest": ["representational state transfer", "api", "web service"],
            "crud": ["create read update delete", "database operations"]
        }
        
        query_lower = query.lower()
        expanded = [query]
        
        for term, expansion in expansions.items():
            if term in query_lower:
                expanded.extend(expansion)
        
        return expanded
    
    def expand_with_llm(self, query: str, fallback_to_synonyms: bool = True) -> List[str]:
        """Expand query using LLM for semantic understanding."""
        response_generator = get_response_generator()
        
        if not response_generator.is_available():
            if fallback_to_synonyms:
                return self.expand_synonyms(query)
            return [query]
        
        try:
            # Create expansion prompt
            prompt = f"""For the query "{query}", suggest 3-5 related search terms that would help find relevant documentation. 
            
Focus on:
- Technical synonyms and variations
- Related concepts and tools
- Common terminology users might search for

Return only the terms, separated by commas."""
            
            response = response_generator.generate_response(
                query=prompt,
                context_documents=[],
                user_id=None
            )
            
            if response.success and response.content:
                # Parse comma-separated terms from response, handling various formats
                content = response.content.strip()
                
                # Handle different response formats
                if ':' in content:
                    # "Related terms: term1, term2" or similar formats
                    content = content.split(':', 1)[1].strip()
                
                if ',' in content:
                    # Comma-separated format
                    terms = [term.strip() for term in content.split(',')]
                elif '\n' in content:
                    # Newline-separated format
                    terms = [term.strip() for term in content.split('\n')]
                else:
                    # Single line or space-separated format
                    terms = content.split()
                
                # Filter out empty terms and combine with original
                valid_terms = []
                for term in terms:
                    term = term.strip().lower()
                    # Skip very short terms and exact duplicates
                    if (term and len(term) > 2 and term != query.lower()):
                        valid_terms.append(term)
                
                return [query] + valid_terms[:5]  # Limit to 5 additional terms
                
        except Exception as e:
            logger.warning(f"LLM query expansion failed: {e}")
        
        # Fallback to synonym expansion
        if fallback_to_synonyms:
            return self.expand_synonyms(query)
        
        return [query]


class QueryContext:
    """Manages conversation context for follow-up queries."""
    
    def __init__(self, user_id: str, max_history: int = 5):
        self.user_id = user_id
        self.max_history = max_history
        self.history: List[Dict[str, Any]] = []
    
    def add_query(self, query: str, documents: List[str], timestamp: Optional[float] = None):
        """Add query to conversation history."""
        if timestamp is None:
            timestamp = time.time()
        
        self.history.append({
            "query": query,
            "documents": documents,
            "timestamp": timestamp,
            "topics": self._extract_topics(query)
        })
        
        # Limit history size
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
    
    def get_context_for_query(self, query: str) -> Dict[str, Any]:
        """Get relevant context for current query."""
        if not self.history:
            return {"previous_topics": [], "context_relevance": 0.0}
        
        recent_topics = []
        for entry in self.history[-3:]:  # Last 3 queries
            recent_topics.extend(entry["topics"])
        
        return {
            "previous_topics": list(set(recent_topics)),
            "context_relevance": self.calculate_relevance(query)
        }
    
    def calculate_relevance(self, query: str) -> float:
        """Calculate relevance of current query to conversation history."""
        if not self.history:
            return 0.0
        
        query_topics = self._extract_topics(query)
        if not query_topics:
            return 0.0
        
        # Check overlap with recent queries
        recent_topics = []
        for entry in self.history[-2:]:  # Last 2 queries
            recent_topics.extend(entry["topics"])
        
        if not recent_topics:
            return 0.0
        
        # Calculate relevance as Jaccard similarity
        query_set = set(query_topics)
        recent_set = set(recent_topics)
        
        intersection = len(query_set & recent_set)
        union = len(query_set | recent_set)
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def _extract_topics(self, query: str) -> List[str]:
        """Extract key topics from query."""
        # Simple topic extraction - could be enhanced with NLP
        stop_words = {"how", "what", "where", "when", "why", "is", "the", "a", "an", "and", "or", "but", "to", "of", "in", "on", "at", "for", "with", "by", "about", "do", "i"}
        words = [w.lower() for w in re.findall(r'\w+', query) if len(w) > 2]
        topics = [w for w in words if w not in stop_words]
        
        # Add stemmed/related forms for better matching
        enhanced_topics = topics.copy()
        for topic in topics:
            if topic.endswith('ment'):  # deployment -> deploy
                stem = topic[:-4]
                if len(stem) > 2:
                    enhanced_topics.append(stem)
            elif topic.endswith('ing'):  # staging -> stage
                stem = topic[:-3]
                if len(stem) > 2:
                    enhanced_topics.append(stem)
            else:  # deploy -> deployment, staging
                if len(topic) > 3:
                    enhanced_topics.append(topic + 'ment')
                    enhanced_topics.append(topic + 'ing')
        
        # Add deployment-related semantic connections
        deployment_terms = {'deploy', 'deployment', 'staging', 'stage', 'release', 'publish'}
        if any(term in enhanced_topics for term in deployment_terms):
            enhanced_topics.extend(deployment_terms)
        
        return list(set(enhanced_topics))
    
    def cleanup_expired(self, max_age_seconds: int = 3600):
        """Remove expired context entries."""
        current_time = time.time()
        self.history = [
            entry for entry in self.history 
            if current_time - entry["timestamp"] < max_age_seconds
        ]


@dataclass
class Query:
    """Represents a user query with optional metadata."""

    text: str
    user: Optional[str] = None
    channel: Optional[str] = None
    metadata: Dict[str, str] = field(default_factory=dict)


class QueryProcessor:
    """Handle search queries with context, escalation, and analytics."""

    def __init__(
        self,
        kb: KnowledgeBase,
        terminology: Optional[Dict[str, str]] = None,
        *,
        routing: Optional["RoutingEngine"] = None,
        enable_escalation: bool = True,
        analytics: Optional[UsageAnalytics] = None,
    ) -> None:
        """Create a processor with optional routing and analytics."""
        self.kb = kb
        # terminology maps slang or abbreviations to canonical terms
        self.terminology = {
            k.lower(): v.lower() for k, v in (terminology or {}).items()
        }
        self.routing = routing
        self.enable_escalation = enable_escalation
        self.analytics = analytics

    def normalize(self, text: str) -> str:
        """Expand known terminology and return a normalized query string."""
        tokens = []
        for token in text.split():
            key = token.lower()
            tokens.append(self.terminology.get(key, token))
        return " ".join(tokens)

    def process_query(self, query: Query | str) -> List[Document]:
        """Return documents matching the normalized query."""
        if isinstance(query, Query):
            text = query.text
            user = query.user
            channel = query.channel
        else:
            text = query
            user = None
            channel = None
        normalized = self.normalize(text)
        if self.analytics is not None:
            self.analytics.record_query(normalized, user=user, channel=channel)
        return self.kb.search(normalized)

    def search_and_route(
        self, query: Query | str
    ) -> Tuple[List[Document], List["TeamMember"]]:
        """Search the knowledge base and route if no results are found."""

        results = self.process_query(query)
        if results:
            return results, []

        if self.enable_escalation and self.routing is not None:
            text = query.text if isinstance(query, Query) else str(query)
            experts = self.routing.route(text)
            return [], experts

        return [], []


class EnhancedQueryProcessor(QueryProcessor):
    """Enhanced query processor with LLM integration and advanced features."""
    
    def __init__(self, kb: KnowledgeBase, max_user_contexts: Optional[int] = 1000, **kwargs):
        super().__init__(kb, **kwargs)
        self.query_expansion = QueryExpansion()
        self.user_contexts: OrderedDict[str, QueryContext] = OrderedDict()
        self.max_user_contexts = max_user_contexts
        self.metrics = get_global_metrics()
        self.structured_logger = StructuredLogger("enhanced_query_processor")
    
    def process_query(self, query: Query | str, user_id: Optional[str] = None) -> QueryResult:
        """Enhanced query processing with intent classification and expansion."""
        start_time = time.time()
        
        # Extract query text and metadata
        if isinstance(query, Query):
            query_text = query.text
            user_id = user_id or query.user
        else:
            query_text = query
        
        try:
            # Track query processing
            self.metrics.increment_counter("enhanced_queries_total")
            
            # Classify intent
            intent = QueryIntent.classify(query_text)
            self.metrics.increment_counter(f"query_intent_{intent.value}_total")
            self.structured_logger.debug(f"Classified query as {intent.value}", 
                                       query=query_text, intent=intent.value, user_id=user_id)
            
            # Normalize query
            normalized_query = self.normalize(query_text)
            
            # Get user context if available
            context_used = False
            if user_id:
                user_context = self._get_user_context(user_id)
                context_info = user_context.get_context_for_query(query_text)
                context_used = context_info["context_relevance"] > 0.2
                
                # Enhance query with context if relevant
                if context_used:
                    previous_topics = context_info["previous_topics"]
                    if previous_topics:
                        normalized_query += " " + " ".join(previous_topics[:3])
                        logger.debug(f"Enhanced query with context: {previous_topics[:3]}")
            
            # Expand query terms
            expanded_terms = self._expand_query(normalized_query, intent)
            
            # Perform search with expanded terms
            documents = self._enhanced_search(normalized_query, expanded_terms, intent)
            
            # Update user context
            if user_id and documents:
                user_context = self._get_user_context(user_id)
                doc_sources = [doc.source for doc in documents[:5]]
                user_context.add_query(query_text, doc_sources)
            
            # Generate suggestions if no results
            suggestions = None
            if not documents:
                suggestions = self._generate_suggestions(query_text, intent)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            self.metrics.record_histogram("enhanced_query_duration_seconds", processing_time)
            
            # Log successful query processing
            self.structured_logger.info(
                "Query processed successfully",
                query=query_text,
                intent=intent.value,
                documents_found=len(documents),
                processing_time=processing_time,
                context_used=context_used,
                user_id=user_id
            )
            
            # Create metrics
            response_generator = get_response_generator()
            metrics = {
                "intent": intent.value,
                "expanded_terms_count": len(expanded_terms),
                "documents_found": len(documents),
                "context_used": context_used,
                "llm_available": response_generator.is_available()
            }
            
            return QueryResult(
                original_query=query_text,
                normalized_query=normalized_query,
                intent=intent,
                expanded_terms=expanded_terms,
                documents=documents,
                suggestions=suggestions,
                context_used=context_used,
                processing_time=processing_time,
                metrics=metrics
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.metrics.increment_counter("enhanced_query_errors_total")
            self.metrics.record_histogram("enhanced_query_duration_seconds", processing_time)
            
            self.structured_logger.error(
                "Enhanced query processing failed",
                query=query_text,
                error=str(e),
                processing_time=processing_time,
                user_id=user_id
            )
            
            # Fallback to basic processing
            try:
                basic_docs = super().process_query(query)
                return QueryResult(
                    original_query=query_text,
                    normalized_query=query_text,
                    intent=QueryIntent.QUESTION,
                    expanded_terms=[query_text],
                    documents=basic_docs,
                    processing_time=processing_time,
                    error_message=f"Enhanced processing failed, using fallback: {e}"
                )
            except Exception as fallback_error:
                logger.error(f"Fallback processing also failed: {fallback_error}")
                return QueryResult(
                    original_query=query_text,
                    normalized_query=query_text,
                    intent=QueryIntent.QUESTION,
                    expanded_terms=[],
                    documents=[],
                    processing_time=processing_time,
                    error_message=f"All processing failed: {e}"
                )
    
    def _get_user_context(self, user_id: str) -> QueryContext:
        """Get or create user context with LRU eviction."""
        # If user exists, move to end (mark as recently used)
        if user_id in self.user_contexts:
            context = self.user_contexts.pop(user_id)
            self.user_contexts[user_id] = context
        else:
            # Create new context
            context = QueryContext(user_id)
            self.user_contexts[user_id] = context
            
            # Enforce max_user_contexts limit with LRU eviction
            if self.max_user_contexts is not None and len(self.user_contexts) > self.max_user_contexts:
                # Remove least recently used context (first item)
                oldest_user, oldest_context = self.user_contexts.popitem(last=False)
                logger.info(f"Evicted user context for {oldest_user} due to LRU limit of {self.max_user_contexts}")
        
        # Update memory metrics
        self._update_context_metrics()
        
        # Cleanup expired contexts periodically
        context.cleanup_expired()
        
        return context

    def _update_context_metrics(self) -> None:
        """Update user context memory metrics."""
        try:
            self.metrics.set_gauge("query_processor_user_contexts_count", len(self.user_contexts))
            if self.max_user_contexts:
                self.metrics.set_gauge("query_processor_user_contexts_limit", self.max_user_contexts)
                usage_percent = (len(self.user_contexts) / self.max_user_contexts) * 100
                self.metrics.set_gauge("query_processor_user_contexts_usage_percent", usage_percent)
            
            # Count total history entries across all contexts
            total_history_entries = sum(len(ctx.history) for ctx in self.user_contexts.values())
            self.metrics.set_gauge("query_processor_total_history_entries", total_history_entries)
            
        except Exception:
            # Don't let metrics collection crash the application
            pass

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics for the query processor."""
        total_history_entries = sum(len(ctx.history) for ctx in self.user_contexts.values())
        
        stats = {
            "user_contexts_count": len(self.user_contexts),
            "max_user_contexts": self.max_user_contexts,
            "total_history_entries": total_history_entries,
        }
        
        if self.max_user_contexts:
            stats["user_contexts_usage_percent"] = (len(self.user_contexts) / self.max_user_contexts) * 100
        
        return stats
    
    def _expand_query(self, query: str, intent: QueryIntent) -> List[str]:
        """Expand query based on intent and available methods."""
        expanded_terms = [query]
        
        # Different expansion strategies based on intent
        if intent in [QueryIntent.QUESTION, QueryIntent.DEFINITION]:
            # Use LLM for semantic expansion
            llm_expanded = self.query_expansion.expand_with_llm(query)
            expanded_terms.extend(llm_expanded[1:])  # Skip original query
        
        if intent in [QueryIntent.COMMAND, QueryIntent.TROUBLESHOOTING]:
            # Use synonym expansion for action-oriented queries
            synonym_expanded = self.query_expansion.expand_synonyms(query)
            expanded_terms.extend(synonym_expanded[1:])  # Skip original query
        
        # Always try technical term expansion
        tech_expanded = self.query_expansion.expand_technical_terms(query)
        expanded_terms.extend(tech_expanded[1:])  # Skip original query
        
        # Remove duplicates while preserving order
        seen = set()
        result = []
        for term in expanded_terms:
            if term not in seen:
                seen.add(term)
                result.append(term)
        
        return result
    
    def _enhanced_search(self, query: str, expanded_terms: List[str], intent: QueryIntent) -> List[Document]:
        """Perform enhanced search using multiple strategies."""
        all_documents = []
        
        # Primary search with original query
        try:
            primary_docs = self.kb.search(query)
            all_documents.extend(primary_docs)
        except Exception as e:
            logger.warning(f"Primary search failed: {e}")
        
        # Secondary search with expanded terms
        for term in expanded_terms[1:6]:  # Limit to top 5 expansions
            try:
                expanded_docs = self.kb.search(term)
                all_documents.extend(expanded_docs)
            except Exception as e:
                logger.debug(f"Expanded search for '{term}' failed: {e}")
        
        # Semantic search if available
        if hasattr(self.kb, 'search_semantic'):
            try:
                semantic_results = self.kb.search_semantic(query, threshold=0.7)
                if isinstance(semantic_results, list) and semantic_results:
                    # Handle both (doc, score) tuples and plain documents
                    semantic_docs = []
                    for result in semantic_results:
                        if isinstance(result, tuple):
                            doc, score = result
                            semantic_docs.append(doc)
                        else:
                            semantic_docs.append(result)
                    # Add semantic results at the beginning for higher priority
                    all_documents = semantic_docs + all_documents
            except Exception as e:
                logger.debug(f"Semantic search failed: {e}")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_documents = []
        for doc in all_documents:
            doc_key = (doc.content, doc.source)
            if doc_key not in seen:
                seen.add(doc_key)
                unique_documents.append(doc)
        
        # Limit results and prioritize by relevance
        return unique_documents[:10]  # Return top 10 most relevant
    
    def _generate_suggestions(self, query: str, intent: QueryIntent) -> Optional[List[str]]:
        """Generate query suggestions when no results found."""
        response_generator = get_response_generator()
        if not response_generator.is_available():
            return self._generate_basic_suggestions(query, intent)
        
        try:
            prompt = f"""The user searched for "{query}" but no results were found. 
            
Suggest 3-4 alternative search queries that might help them find what they're looking for. 
Focus on:
- Different terminology for the same concept
- Broader or more specific search terms
- Common related topics

Return only the suggested queries, one per line."""
            
            response = response_generator.generate_response(
                query=prompt,
                context_documents=[],
                user_id=None
            )
            
            if response.success and response.content:
                content = response.content.strip()
                
                # Handle different suggestion formats
                if ':' in content:
                    # "Try searching for: suggestion1, suggestion2" format
                    content = content.split(':', 1)[1].strip()
                
                if ',' in content:
                    # Comma-separated suggestions
                    suggestions = [s.strip() for s in content.split(',') if s.strip()]
                elif '\n' in content:
                    # Newline-separated suggestions
                    suggestions = [s.strip() for s in content.split('\n') if s.strip()]
                else:
                    # Single suggestion or space-separated
                    suggestions = [content]
                
                return suggestions[:4]  # Limit to 4 suggestions
                
        except Exception as e:
            logger.warning(f"LLM suggestion generation failed: {e}")
        
        return self._generate_basic_suggestions(query, intent)
    
    def _generate_basic_suggestions(self, query: str, intent: QueryIntent) -> List[str]:
        """Generate basic suggestions without LLM."""
        suggestions = []
        
        # Add expanded terms as suggestions
        expanded = self.query_expansion.expand_synonyms(query)
        suggestions.extend(expanded[1:4])  # Skip original query
        
        # Add intent-specific suggestions
        if intent == QueryIntent.TROUBLESHOOTING:
            suggestions.extend([
                f"{query} error",
                f"{query} not working",
                f"fix {query}"
            ])
        elif intent == QueryIntent.DEFINITION:
            suggestions.extend([
                f"{query} tutorial",
                f"{query} guide",
                f"{query} documentation"
            ])
        
        return suggestions[:4]
