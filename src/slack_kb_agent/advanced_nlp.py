"""Advanced NLP Query Understanding System.

This module provides sophisticated natural language processing capabilities for 
understanding user queries beyond simple pattern matching.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import torch
    from sentence_transformers import SentenceTransformer
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch and sentence-transformers not available, falling back to rule-based NLP")


class QueryComplexity(Enum):
    """Query complexity levels for routing decisions."""
    SIMPLE = "simple"        # Direct lookups, basic questions
    MODERATE = "moderate"    # Multi-step reasoning, context needed
    COMPLEX = "complex"      # Requires domain expertise, escalation


class QueryIntent(Enum):
    """Enhanced query intent classification."""
    # Informational
    DEFINITION = "definition"           # What is X? Define Y
    EXPLANATION = "explanation"         # How does X work? Explain Y
    COMPARISON = "comparison"           # X vs Y, differences between

    # Procedural
    HOWTO = "howto"                    # How to do X, step by step
    TROUBLESHOOTING = "troubleshooting" # Fix X, X is broken, error Y
    CONFIGURATION = "configuration"     # How to configure X, setup Y

    # Factual
    STATUS = "status"                  # What's the status of X?
    HISTORY = "history"                # When was X changed? Who did Y?
    LOCATION = "location"              # Where is X? Find Y

    # Meta
    CONVERSATIONAL = "conversational"   # Hello, thanks, etc.
    ESCALATION = "escalation"          # I need help, speak to human


@dataclass
class QueryContext:
    """Context information extracted from a query."""
    entities: List[str]          # Named entities (projects, components, etc.)
    technical_terms: List[str]   # Technical terminology
    urgency_level: int           # 1-5, where 5 is most urgent
    requires_code: bool          # Query likely needs code examples
    requires_diagram: bool       # Query would benefit from visual aid
    domain_areas: List[str]      # Technical domains (auth, deployment, etc.)


@dataclass
class EnhancedQuery:
    """Enhanced query with NLP processing results."""
    original_query: str
    intent: QueryIntent
    complexity: QueryComplexity
    context: QueryContext
    confidence: float            # 0-1 confidence in classification
    expanded_query: str          # Query with synonyms and expansions
    key_concepts: List[str]      # Core concepts extracted


class EntityExtractor:
    """Extract named entities and technical terms from queries."""

    def __init__(self):
        # Technical domain patterns
        self.domain_patterns = {
            'authentication': r'\b(auth|login|password|token|oauth|jwt|sso)\b',
            'deployment': r'\b(deploy|release|build|ci|cd|pipeline|docker)\b',
            'database': r'\b(db|database|sql|postgres|redis|query|table)\b',
            'api': r'\b(api|endpoint|rest|graphql|webhook|request|response)\b',
            'monitoring': r'\b(log|metric|alert|monitor|dashboard|grafana|prometheus)\b',
            'security': r'\b(security|vulnerability|encrypt|ssl|tls|cert)\b',
            'performance': r'\b(slow|fast|performance|optimize|cache|memory|cpu)\b',
            'networking': r'\b(network|ip|port|dns|load|balancer|proxy)\b'
        }

        # Common technical entities
        self.technical_patterns = [
            r'\b[A-Z][a-z]*(?:[A-Z][a-z]*)+\b',  # CamelCase
            r'\b[a-z]+_[a-z_]+\b',               # snake_case
            r'\b[a-z-]+\.[a-z-]+\b',            # kebab-case or domain
            r'\b\d+\.\d+\.\d+\b',               # Version numbers
            r'\bv\d+(?:\.\d+)*\b',              # Version tags
            r'\b[A-Z_]{3,}\b',                  # Constants/env vars
        ]

        # Urgency indicators
        self.urgency_patterns = {
            5: [r'\b(urgent|critical|emergency|down|broken|failing)\b'],
            4: [r'\b(asap|quickly|soon|important|issue|problem)\b'],
            3: [r'\b(help|stuck|confused|error)\b'],
            2: [r'\b(question|wondering|curious)\b'],
            1: [r'\b(thanks|hello|hi|general)\b']
        }

    def extract_entities(self, query: str) -> List[str]:
        """Extract named entities from query."""
        entities = []
        query_lower = query.lower()

        # Extract technical patterns
        for pattern in self.technical_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            entities.extend(matches)

        # Remove duplicates while preserving order
        return list(dict.fromkeys(entities))

    def extract_technical_terms(self, query: str) -> List[str]:
        """Extract technical terminology."""
        terms = []
        query_lower = query.lower()

        for domain, pattern in self.domain_patterns.items():
            if re.search(pattern, query_lower):
                terms.append(domain)

        return terms

    def assess_urgency(self, query: str) -> int:
        """Assess query urgency level (1-5)."""
        query_lower = query.lower()

        for urgency_level, patterns in self.urgency_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    return urgency_level

        return 2  # Default moderate urgency

    def extract_context(self, query: str) -> QueryContext:
        """Extract comprehensive context from query."""
        return QueryContext(
            entities=self.extract_entities(query),
            technical_terms=self.extract_technical_terms(query),
            urgency_level=self.assess_urgency(query),
            requires_code=self._requires_code(query),
            requires_diagram=self._requires_diagram(query),
            domain_areas=self.extract_technical_terms(query)
        )

    def _requires_code(self, query: str) -> bool:
        """Check if query likely needs code examples."""
        code_indicators = [
            r'\b(code|example|snippet|implementation|syntax)\b',
            r'\b(how to.*code|show me.*code)\b',
            r'\b(function|method|class|variable)\b'
        ]

        query_lower = query.lower()
        return any(re.search(pattern, query_lower) for pattern in code_indicators)

    def _requires_diagram(self, query: str) -> bool:
        """Check if query would benefit from visual aids."""
        visual_indicators = [
            r'\b(architecture|diagram|flow|structure)\b',
            r'\b(how.*work|explain.*process)\b',
            r'\b(relationship|connection|interaction)\b'
        ]

        query_lower = query.lower()
        return any(re.search(pattern, query_lower) for pattern in visual_indicators)


class SemanticIntentClassifier:
    """ML-based intent classification using sentence transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.intent_templates = None

        if TORCH_AVAILABLE:
            try:
                self._initialize_model()
            except Exception as e:
                logger.warning(f"Failed to initialize semantic classifier: {e}")
                self.model = None

    def _initialize_model(self):
        """Initialize the sentence transformer model."""
        self.model = SentenceTransformer(self.model_name)

        # Pre-defined intent templates for semantic matching
        self.intent_templates = {
            QueryIntent.DEFINITION: [
                "What is X?", "Define X", "What does X mean?", "Explain the concept of X"
            ],
            QueryIntent.EXPLANATION: [
                "How does X work?", "Explain how X functions", "What is the process of X?",
                "How is X implemented?"
            ],
            QueryIntent.HOWTO: [
                "How to do X?", "Steps to accomplish X", "Guide for X", "Tutorial for X"
            ],
            QueryIntent.TROUBLESHOOTING: [
                "X is broken", "Fix X problem", "X error", "X not working", "Debug X"
            ],
            QueryIntent.STATUS: [
                "Status of X", "What is the current state of X?", "Is X ready?",
                "Progress on X"
            ],
            QueryIntent.LOCATION: [
                "Where is X?", "Find X", "Location of X", "Where can I find X?"
            ],
            QueryIntent.COMPARISON: [
                "X vs Y", "Difference between X and Y", "Compare X and Y",
                "Which is better X or Y?"
            ],
            QueryIntent.CONFIGURATION: [
                "How to configure X?", "Setup X", "Install X", "X configuration"
            ],
            QueryIntent.HISTORY: [
                "When was X changed?", "History of X", "Who modified X?", "X changelog"
            ]
        }

        # Pre-compute embeddings for all templates
        self.template_embeddings = {}
        for intent, templates in self.intent_templates.items():
            embeddings = self.model.encode(templates)
            self.template_embeddings[intent] = embeddings

    def classify_intent(self, query: str) -> Tuple[QueryIntent, float]:
        """Classify query intent using semantic similarity."""
        if not self.model or not self.template_embeddings:
            return self._fallback_classification(query)

        try:
            # Encode the query
            query_embedding = self.model.encode([query])

            best_intent = QueryIntent.CONVERSATIONAL
            best_score = 0.0

            # Compare with all intent templates
            for intent, template_embeddings in self.template_embeddings.items():
                # Calculate similarity with all templates for this intent
                similarities = torch.cosine_similarity(
                    torch.tensor(query_embedding),
                    torch.tensor(template_embeddings)
                )

                # Take the maximum similarity for this intent
                max_similarity = float(similarities.max())

                if max_similarity > best_score:
                    best_score = max_similarity
                    best_intent = intent

            return best_intent, best_score

        except Exception as e:
            logger.warning(f"Semantic classification failed: {e}")
            return self._fallback_classification(query)

    def _fallback_classification(self, query: str) -> Tuple[QueryIntent, float]:
        """Rule-based fallback classification."""
        query_lower = query.lower().strip()

        # Definition patterns
        if re.search(r'\b(what is|define|meaning of|explain)\b.*\?', query_lower):
            return QueryIntent.DEFINITION, 0.8

        # How-to patterns
        if re.search(r'\b(how to|how do i|steps to|guide|tutorial)\b', query_lower):
            return QueryIntent.HOWTO, 0.8

        # Troubleshooting patterns
        if re.search(r'\b(error|broken|not working|fix|debug|problem|issue)\b', query_lower):
            return QueryIntent.TROUBLESHOOTING, 0.8

        # Status patterns
        if re.search(r'\b(status|state|ready|progress|current)\b', query_lower):
            return QueryIntent.STATUS, 0.7

        # Location patterns
        if re.search(r'\b(where|find|location|locate)\b', query_lower):
            return QueryIntent.LOCATION, 0.7

        # Comparison patterns
        if re.search(r'\bvs\b|versus|compared to|difference between', query_lower):
            return QueryIntent.COMPARISON, 0.8

        return QueryIntent.CONVERSATIONAL, 0.5


class QueryExpander:
    """Expand queries with synonyms and related terms."""

    def __init__(self):
        # Technical synonyms mapping
        self.synonyms = {
            'api': ['endpoint', 'service', 'interface'],
            'database': ['db', 'data store', 'storage'],
            'deploy': ['release', 'publish', 'ship'],
            'auth': ['authentication', 'login', 'access'],
            'config': ['configuration', 'settings', 'setup'],
            'error': ['bug', 'issue', 'problem', 'exception'],
            'fix': ['resolve', 'solve', 'repair'],
            'fast': ['quick', 'rapid', 'speedy'],
            'slow': ['sluggish', 'delayed', 'laggy']
        }

    def expand_query(self, query: str, max_expansions: int = 3) -> str:
        """Expand query with relevant synonyms."""
        words = query.lower().split()
        expanded_terms = []

        for word in words:
            # Clean word (remove punctuation)
            clean_word = re.sub(r'[^\w]', '', word)

            if clean_word in self.synonyms:
                # Add original word and synonyms (limited)
                synonyms = self.synonyms[clean_word][:max_expansions]
                expanded_terms.append(f"({clean_word}|{' '.join(synonyms)})")
            else:
                expanded_terms.append(word)

        return ' '.join(expanded_terms)


class AdvancedQueryProcessor:
    """Advanced NLP-powered query processing system."""

    def __init__(self):
        self.entity_extractor = EntityExtractor()
        self.intent_classifier = SemanticIntentClassifier()
        self.query_expander = QueryExpander()

        # Complexity assessment patterns
        self.complexity_indicators = {
            QueryComplexity.SIMPLE: [
                r'^\w+\s*\?$',  # Single word questions
                r'^(hi|hello|thanks|thank you)$'  # Greetings
            ],
            QueryComplexity.COMPLEX: [
                r'\b(integrate|architecture|design pattern|best practice)\b',
                r'\b(multiple|several|various|different)\b.*\b(options|approaches)\b',
                r'\b(enterprise|production|scale|performance)\b'
            ]
        }

    def process_query(self, query: str, user_context: Optional[Dict] = None) -> EnhancedQuery:
        """Process query with advanced NLP techniques."""
        # Extract context information
        context = self.entity_extractor.extract_context(query)

        # Classify intent
        intent, confidence = self.intent_classifier.classify_intent(query)

        # Assess complexity
        complexity = self._assess_complexity(query, context)

        # Expand query
        expanded_query = self.query_expander.expand_query(query)

        # Extract key concepts
        key_concepts = self._extract_key_concepts(query, context)

        return EnhancedQuery(
            original_query=query,
            intent=intent,
            complexity=complexity,
            context=context,
            confidence=confidence,
            expanded_query=expanded_query,
            key_concepts=key_concepts
        )

    def _assess_complexity(self, query: str, context: QueryContext) -> QueryComplexity:
        """Assess query complexity based on multiple factors."""
        query_lower = query.lower()

        # Check for explicit complexity indicators
        for complexity, patterns in self.complexity_indicators.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    return complexity

        # Assess based on context
        complexity_score = 0

        # Length-based scoring
        if len(query.split()) > 15:
            complexity_score += 2
        elif len(query.split()) > 8:
            complexity_score += 1

        # Technical terms increase complexity
        complexity_score += len(context.technical_terms)

        # Multiple entities increase complexity
        if len(context.entities) > 3:
            complexity_score += 2
        elif len(context.entities) > 1:
            complexity_score += 1

        # Urgency can indicate complexity
        if context.urgency_level >= 4:
            complexity_score += 1

        # Map score to complexity level
        if complexity_score >= 4:
            return QueryComplexity.COMPLEX
        elif complexity_score >= 2:
            return QueryComplexity.MODERATE
        else:
            return QueryComplexity.SIMPLE

    def _extract_key_concepts(self, query: str, context: QueryContext) -> List[str]:
        """Extract key concepts from the query."""
        concepts = []

        # Add entities as key concepts
        concepts.extend(context.entities)

        # Add domain areas as concepts
        concepts.extend(context.domain_areas)

        # Extract important nouns (simple approach)
        words = re.findall(r'\b[a-zA-Z]{3,}\b', query)
        important_words = [w for w in words if w.lower() not in {
            'the', 'and', 'or', 'but', 'can', 'how', 'what', 'where', 'when', 'why',
            'this', 'that', 'these', 'those', 'with', 'for', 'from'
        }]

        concepts.extend(important_words[:5])  # Limit to top 5

        # Remove duplicates and return
        return list(dict.fromkeys(concepts))

    def suggest_follow_up_questions(self, enhanced_query: EnhancedQuery) -> List[str]:
        """Suggest relevant follow-up questions based on the query."""
        suggestions = []

        if enhanced_query.intent == QueryIntent.DEFINITION:
            suggestions = [
                f"How to implement {enhanced_query.key_concepts[0]}?" if enhanced_query.key_concepts else "How to implement this?",
                f"Best practices for {enhanced_query.key_concepts[0]}" if enhanced_query.key_concepts else "What are the best practices?",
                "Are there any examples available?"
            ]
        elif enhanced_query.intent == QueryIntent.HOWTO:
            suggestions = [
                "What are common issues with this approach?",
                "Are there alternative methods?",
                "What prerequisites are needed?"
            ]
        elif enhanced_query.intent == QueryIntent.TROUBLESHOOTING:
            suggestions = [
                "How to prevent this issue in the future?",
                "What are the root causes?",
                "Are there monitoring alerts for this?"
            ]

        return suggestions[:3]  # Limit to 3 suggestions
