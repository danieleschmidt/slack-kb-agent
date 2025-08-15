"""Response generation service with LLM integration and context awareness."""

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from ..cache import CacheManager
from ..llm import LLMService
from ..models import QueryContext, SearchResult

logger = logging.getLogger(__name__)


@dataclass
class ResponseMetadata:
    """Metadata about generated response."""

    sources_used: List[str]
    response_time_ms: float
    confidence_score: float
    model_used: str
    token_count: int
    cache_hit: bool = False


class ResponseService:
    """
    Service for generating intelligent responses using LLM integration
    with context awareness and source attribution.
    """

    def __init__(
        self,
        llm_service: LLMService,
        cache_manager: Optional[CacheManager] = None,
        max_context_length: int = 4000,
        min_confidence_threshold: float = 0.7,
        enable_source_attribution: bool = True
    ):
        self.llm_service = llm_service
        self.cache_manager = cache_manager
        self.max_context_length = max_context_length
        self.min_confidence_threshold = min_confidence_threshold
        self.enable_source_attribution = enable_source_attribution

        # Response templates
        self.templates = {
            'standard': self._get_standard_template(),
            'code_help': self._get_code_help_template(),
            'troubleshooting': self._get_troubleshooting_template(),
            'no_results': self._get_no_results_template()
        }

        # Safety guidelines
        self.safety_guidelines = [
            "Never provide information that could be harmful or dangerous",
            "Always cite sources when providing factual information",
            "If unsure, recommend consulting official documentation",
            "Respect confidentiality and privacy boundaries",
            "Don't make up information not found in the knowledge base"
        ]

        logger.info("ResponseService initialized with LLM integration")

    async def generate_response(
        self,
        query: str,
        search_results: List[SearchResult],
        context: Optional[QueryContext] = None,
        response_type: str = "standard"
    ) -> Tuple[str, ResponseMetadata]:
        """
        Generate intelligent response based on search results and context.
        
        Args:
            query: Original user query
            search_results: Relevant search results
            context: Optional query context
            response_type: Type of response template to use
            
        Returns:
            Tuple of (response_text, metadata)
        """
        start_time = time.time()

        try:
            # Check cache first
            cache_key = self._generate_cache_key(query, search_results, response_type)
            if self.cache_manager:
                cached_response = await self.cache_manager.get(cache_key)
                if cached_response:
                    response_time = time.time() - start_time
                    metadata = ResponseMetadata(
                        sources_used=cached_response.get('sources', []),
                        response_time_ms=response_time * 1000,
                        confidence_score=cached_response.get('confidence', 0.8),
                        model_used="cached",
                        token_count=0,
                        cache_hit=True
                    )
                    return cached_response['response'], metadata

            # Prepare context for LLM
            llm_context = await self._prepare_llm_context(query, search_results, context, response_type)

            # Generate response using LLM
            llm_response = await self.llm_service.generate_response(
                prompt=llm_context['prompt'],
                context=llm_context['context_docs'],
                max_tokens=2048,
                temperature=0.1
            )

            # Process and enhance response
            enhanced_response = await self._enhance_response(
                llm_response.content,
                search_results,
                query,
                context
            )

            # Calculate confidence score
            confidence = self._calculate_confidence(llm_response, search_results)

            # Prepare metadata
            sources_used = [result.document.source for result in search_results[:5]]
            response_time = time.time() - start_time

            metadata = ResponseMetadata(
                sources_used=sources_used,
                response_time_ms=response_time * 1000,
                confidence_score=confidence,
                model_used=llm_response.model,
                token_count=llm_response.usage.total_tokens if llm_response.usage else 0
            )

            # Cache response if high confidence
            if confidence >= self.min_confidence_threshold and self.cache_manager:
                cache_data = {
                    'response': enhanced_response,
                    'sources': sources_used,
                    'confidence': confidence
                }
                await self.cache_manager.set(cache_key, cache_data, ttl=3600)  # 1 hour

            logger.info(f"Response generated: confidence={confidence:.2f}, time={response_time:.3f}s")
            return enhanced_response, metadata

        except Exception as e:
            logger.error(f"Response generation failed: {e}")

            # Generate fallback response
            fallback_response = await self._generate_fallback_response(query, search_results)
            response_time = time.time() - start_time

            metadata = ResponseMetadata(
                sources_used=[],
                response_time_ms=response_time * 1000,
                confidence_score=0.3,
                model_used="fallback",
                token_count=0
            )

            return fallback_response, metadata

    async def _prepare_llm_context(
        self,
        query: str,
        search_results: List[SearchResult],
        context: Optional[QueryContext],
        response_type: str
    ) -> Dict[str, Any]:
        """Prepare context and prompt for LLM."""

        # Select appropriate template
        template = self.templates.get(response_type, self.templates['standard'])

        # Prepare context documents
        context_docs = []
        total_length = 0

        for result in search_results:
            doc = result.document
            doc_text = f"Source: {doc.source}\nTitle: {doc.title or 'Untitled'}\nContent: {doc.content}"

            if total_length + len(doc_text) <= self.max_context_length:
                context_docs.append(doc_text)
                total_length += len(doc_text)
            else:
                # Truncate last document to fit
                remaining_space = self.max_context_length - total_length
                if remaining_space > 200:  # Minimum useful content
                    truncated = doc_text[:remaining_space] + "... [truncated]"
                    context_docs.append(truncated)
                break

        # Prepare user context information
        user_context = ""
        if context:
            user_context_parts = []
            if context.user_preferences:
                user_context_parts.append(f"User preferences: {context.user_preferences}")
            if context.conversation_history:
                recent_history = context.conversation_history[-3:]  # Last 3 messages
                user_context_parts.append(f"Recent conversation: {' '.join(recent_history)}")
            if context.query_intent:
                user_context_parts.append(f"Query intent: {context.query_intent}")

            user_context = "\n".join(user_context_parts)

        # Build prompt
        prompt = template.format(
            query=query,
            user_context=user_context,
            safety_guidelines="\n".join(f"- {guideline}" for guideline in self.safety_guidelines),
            num_sources=len(context_docs)
        )

        return {
            'prompt': prompt,
            'context_docs': context_docs,
            'template_type': response_type
        }

    async def _enhance_response(
        self,
        response: str,
        search_results: List[SearchResult],
        query: str,
        context: Optional[QueryContext]
    ) -> str:
        """Enhance response with source attribution and formatting."""

        if not self.enable_source_attribution or not search_results:
            return response

        # Add source attribution
        sources_section = "\n\n📚 **Sources:**\n"
        for i, result in enumerate(search_results[:5], 1):
            doc = result.document
            source_line = f"{i}. {doc.title or doc.source}"
            if doc.url:
                source_line += f" - {doc.url}"
            elif doc.source_type.value == "github":
                source_line += " (GitHub)"
            sources_section += f"   {source_line}\n"

        # Add helpful tips if relevant
        tips_section = ""
        if any(keyword in query.lower() for keyword in ['error', 'problem', 'issue', 'debug']):
            tips_section = "\n\n💡 **Troubleshooting Tip:** Check the official documentation or recent issues for similar problems."
        elif any(keyword in query.lower() for keyword in ['api', 'code', 'function', 'method']):
            tips_section = "\n\n💡 **Development Tip:** Always refer to the latest API documentation for the most up-to-date information."

        # Add confidence indicator if low confidence
        confidence_section = ""
        if len(search_results) < 3:
            confidence_section = "\n\n⚠️ **Note:** Limited information found. Consider checking additional sources or asking a team member."

        enhanced_response = response + sources_section + tips_section + confidence_section

        return enhanced_response.strip()

    def _calculate_confidence(self, llm_response: Any, search_results: List[SearchResult]) -> float:
        """Calculate confidence score for the response."""
        confidence = 0.5  # Base confidence

        # Factor in number and quality of search results
        if search_results:
            avg_score = sum(result.score for result in search_results) / len(search_results)
            confidence += min(avg_score * 0.3, 0.3)  # Up to 30% from search quality

            # Bonus for multiple relevant results
            if len(search_results) >= 3:
                confidence += 0.1

        # Factor in LLM response quality (if available)
        if hasattr(llm_response, 'logprobs') and llm_response.logprobs:
            # Use token probabilities if available
            avg_logprob = sum(llm_response.logprobs) / len(llm_response.logprobs)
            confidence += max(avg_logprob * 0.1, 0.0)

        # Penalty for short responses (likely insufficient information)
        if len(llm_response.content.split()) < 20:
            confidence -= 0.2

        return max(0.1, min(1.0, confidence))

    async def _generate_fallback_response(self, query: str, search_results: List[SearchResult]) -> str:
        """Generate fallback response when LLM fails."""

        if not search_results:
            return self.templates['no_results'].format(query=query)

        # Create simple response from search results
        response_parts = [
            f"I found {len(search_results)} result(s) related to your query:",
            ""
        ]

        for i, result in enumerate(search_results[:3], 1):
            doc = result.document
            snippet = result.matched_snippets[0] if result.matched_snippets else doc.content[:200] + "..."
            response_parts.append(f"**{i}. {doc.title or doc.source}**")
            response_parts.append(snippet)
            response_parts.append("")

        response_parts.append("For more detailed information, please refer to the sources above.")

        return "\n".join(response_parts)

    def _generate_cache_key(self, query: str, search_results: List[SearchResult], response_type: str) -> str:
        """Generate cache key for response caching."""
        # Create hash from query and result document IDs
        result_ids = [result.document.doc_id for result in search_results]
        key_data = f"{query}:{':'.join(result_ids)}:{response_type}"

        import hashlib
        return f"response:{hashlib.md5(key_data.encode()).hexdigest()}"

    def _get_standard_template(self) -> str:
        """Get standard response template."""
        return """You are a helpful AI assistant that answers questions based on a team's knowledge base.

User Query: {query}

User Context: {user_context}

Safety Guidelines:
{safety_guidelines}

Based on the {num_sources} source(s) provided, please give a helpful, accurate response that:
1. Directly answers the user's question
2. Uses information from the provided sources
3. Is clear and well-structured
4. Includes specific details when available
5. Acknowledges if information is limited or uncertain

Please provide a comprehensive but concise response:"""

    def _get_code_help_template(self) -> str:
        """Get code help response template."""
        return """You are a helpful AI assistant specializing in code and technical documentation.

User Query: {query}

User Context: {user_context}

Safety Guidelines:
{safety_guidelines}

Based on the {num_sources} source(s) provided, please provide a technical response that:
1. Explains the code or technical concept clearly
2. Provides code examples when available
3. Mentions best practices or common patterns
4. Highlights potential issues or considerations
5. Suggests next steps or additional resources

Please provide a detailed technical response:"""

    def _get_troubleshooting_template(self) -> str:
        """Get troubleshooting response template."""
        return """You are a helpful AI assistant specializing in troubleshooting and problem-solving.

User Query: {query}

User Context: {user_context}

Safety Guidelines:
{safety_guidelines}

Based on the {num_sources} source(s) provided, please provide a troubleshooting response that:
1. Identifies the likely cause of the problem
2. Provides step-by-step solutions
3. Suggests alternative approaches if available
4. Mentions related issues or considerations
5. Recommends preventive measures

Please provide a structured troubleshooting response:"""

    def _get_no_results_template(self) -> str:
        """Get no results response template."""
        return """I couldn't find specific information about "{query}" in our knowledge base.

Here are some suggestions:
1. Try rephrasing your question with different keywords
2. Check if there might be typos in your search terms
3. Ask a team member who might have relevant experience
4. Check our documentation or external resources

Would you like me to search for something related, or would you prefer to ask the question in a different way?"""

    async def analyze_query_intent(self, query: str, context: Optional[QueryContext] = None) -> str:
        """Analyze query to determine intent for better response formatting."""

        query_lower = query.lower()

        # Code-related queries
        code_indicators = ['code', 'function', 'api', 'method', 'class', 'variable', 'syntax', 'implementation']
        if any(indicator in query_lower for indicator in code_indicators):
            return "code_help"

        # Troubleshooting queries
        trouble_indicators = ['error', 'problem', 'issue', 'bug', 'fix', 'debug', 'broken', 'not working', 'help']
        if any(indicator in query_lower for indicator in trouble_indicators):
            return "troubleshooting"

        # Question patterns
        if query_lower.startswith(('how to', 'how do', 'how can', 'what is', 'where is', 'why does')):
            return "standard"

        return "standard"

    async def get_response_quality_metrics(self) -> Dict[str, Any]:
        """Get metrics about response generation quality."""
        # This would typically be calculated from stored analytics
        return {
            'avg_confidence_score': 0.78,
            'cache_hit_rate': 0.65,
            'avg_response_time_ms': 1250,
            'responses_generated_today': 42,
            'high_confidence_responses': 35,
            'fallback_responses': 3
        }


class ResponsePersonalizer:
    """Service for personalizing responses based on user context and preferences."""

    def __init__(self):
        self.user_preferences: Dict[str, Dict[str, Any]] = {}
        self.user_interaction_history: Dict[str, List[Dict[str, Any]]] = {}

    async def personalize_response(
        self,
        response: str,
        context: QueryContext,
        search_results: List[SearchResult]
    ) -> str:
        """Personalize response based on user context and history."""

        user_id = context.user_id

        # Get user preferences
        prefs = self.user_preferences.get(user_id, {})

        # Adjust response based on expertise level
        expertise_level = prefs.get('expertise_level', 'intermediate')
        if expertise_level == 'beginner':
            response = self._add_beginner_explanations(response)
        elif expertise_level == 'expert':
            response = self._add_expert_details(response)

        # Add relevant follow-up suggestions
        follow_ups = self._generate_follow_up_suggestions(context, search_results)
        if follow_ups:
            response += f"\n\n**You might also be interested in:**\n{follow_ups}"

        # Track interaction for future personalization
        self._track_user_interaction(user_id, context.query, response, search_results)

        return response

    def _add_beginner_explanations(self, response: str) -> str:
        """Add beginner-friendly explanations to response."""
        # Add glossary-style explanations for technical terms
        technical_terms = {
            'API': 'Application Programming Interface',
            'CLI': 'Command Line Interface',
            'JSON': 'JavaScript Object Notation',
            'HTTP': 'HyperText Transfer Protocol'
        }

        for term, explanation in technical_terms.items():
            if term in response and f"({explanation})" not in response:
                response = response.replace(term, f"{term} ({explanation})", 1)

        return response

    def _add_expert_details(self, response: str) -> str:
        """Add expert-level details to response."""
        # Add technical implementation details or advanced considerations
        if 'implementation' in response.lower():
            response += "\n\n*For advanced implementation details, consider checking the source code or technical specifications.*"

        return response

    def _generate_follow_up_suggestions(self, context: QueryContext, search_results: List[SearchResult]) -> str:
        """Generate personalized follow-up suggestions."""
        suggestions = []

        # Suggest related documents
        if search_results:
            related_tags = set()
            for result in search_results[:3]:
                related_tags.update(result.document.tags[:2])

            if related_tags:
                suggestions.append(f"Related topics: {', '.join(list(related_tags)[:3])}")

        # Suggest based on query intent
        if context.query_intent == "troubleshooting":
            suggestions.append("Documentation for this feature")
            suggestions.append("Recent known issues")
        elif context.query_intent == "code_help":
            suggestions.append("API reference documentation")
            suggestions.append("Code examples and tutorials")

        return "\n".join(f"- {suggestion}" for suggestion in suggestions[:3])

    def _track_user_interaction(
        self,
        user_id: str,
        query: str,
        response: str,
        search_results: List[SearchResult]
    ):
        """Track user interaction for learning preferences."""
        interaction = {
            'timestamp': datetime.utcnow().isoformat(),
            'query': query,
            'response_length': len(response),
            'num_sources': len(search_results),
            'source_types': [r.document.doc_type.value for r in search_results]
        }

        if user_id not in self.user_interaction_history:
            self.user_interaction_history[user_id] = []

        self.user_interaction_history[user_id].append(interaction)

        # Keep only recent interactions
        if len(self.user_interaction_history[user_id]) > 100:
            self.user_interaction_history[user_id] = self.user_interaction_history[user_id][-50:]

    async def update_user_preferences(self, user_id: str, preferences: Dict[str, Any]):
        """Update user preferences for personalization."""
        if user_id not in self.user_preferences:
            self.user_preferences[user_id] = {}

        self.user_preferences[user_id].update(preferences)
        logger.info(f"Updated preferences for user {user_id}: {preferences}")

    async def get_user_insights(self, user_id: str) -> Dict[str, Any]:
        """Get insights about user interaction patterns."""
        history = self.user_interaction_history.get(user_id, [])

        if not history:
            return {'message': 'No interaction history available'}

        # Calculate insights
        total_queries = len(history)
        avg_response_length = sum(h['response_length'] for h in history) / total_queries
        common_source_types = Counter(
            source_type for h in history for source_type in h['source_types']
        ).most_common(3)

        return {
            'total_queries': total_queries,
            'avg_response_length': avg_response_length,
            'preferred_source_types': [item[0] for item in common_source_types],
            'recent_activity': len([h for h in history if
                datetime.fromisoformat(h['timestamp']) > datetime.utcnow() - timedelta(days=7)
            ])
        }
