"""Smart content curation system for autonomous knowledge base optimization.

This module provides intelligent content management capabilities including:
- Automatic quality assessment of documents
- Knowledge gap detection and analysis
- Content freshness monitoring
- Intelligent content recommendations
- Automated content tagging and categorization
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from .models import Document

logger = logging.getLogger(__name__)


class ContentQuality(Enum):
    """Content quality levels."""
    EXCELLENT = "excellent"    # Highly valuable, comprehensive, up-to-date
    GOOD = "good"             # Useful, mostly accurate, some value
    FAIR = "fair"             # Basic information, limited value
    POOR = "poor"             # Outdated, incomplete, or low quality
    DUPLICATE = "duplicate"    # Redundant with existing content


class ContentType(Enum):
    """Types of content identified."""
    TUTORIAL = "tutorial"          # Step-by-step guides
    REFERENCE = "reference"        # API docs, specifications
    TROUBLESHOOTING = "troubleshooting"  # Error resolution guides
    FAQ = "faq"                   # Frequently asked questions
    ANNOUNCEMENT = "announcement"  # News, updates, releases
    DISCUSSION = "discussion"      # Conversations, Q&A
    CODE_EXAMPLE = "code_example" # Code snippets, samples


class KnowledgeGap(Enum):
    """Types of knowledge gaps."""
    MISSING_TOPIC = "missing_topic"      # Completely missing subject
    OUTDATED_INFO = "outdated_info"      # Information needs updating
    INCOMPLETE_COVERAGE = "incomplete_coverage"  # Partial information
    BROKEN_REFERENCES = "broken_references"     # Dead links, missing docs
    CONFLICTING_INFO = "conflicting_info"       # Contradictory content


@dataclass
class ContentMetrics:
    """Metrics for content quality assessment."""
    readability_score: float       # 0-1, higher is better
    completeness_score: float      # 0-1, how comprehensive
    accuracy_confidence: float     # 0-1, confidence in accuracy
    freshness_score: float         # 0-1, how recent/current
    engagement_score: float        # 0-1, user interaction level
    technical_depth_score: float   # 0-1, level of technical detail
    structure_quality_score: float # 0-1, organization and formatting

    def overall_quality_score(self) -> float:
        """Calculate overall quality score."""
        weights = {
            'readability': 0.15,
            'completeness': 0.20,
            'accuracy': 0.25,
            'freshness': 0.15,
            'engagement': 0.10,
            'technical_depth': 0.10,
            'structure': 0.05
        }

        return (
            self.readability_score * weights['readability'] +
            self.completeness_score * weights['completeness'] +
            self.accuracy_confidence * weights['accuracy'] +
            self.freshness_score * weights['freshness'] +
            self.engagement_score * weights['engagement'] +
            self.technical_depth_score * weights['technical_depth'] +
            self.structure_quality_score * weights['structure']
        )


@dataclass
class CuratedContent:
    """Content with curation metadata."""
    document: Document
    content_type: ContentType
    quality: ContentQuality
    metrics: ContentMetrics
    tags: Set[str]
    last_curated: float
    curation_confidence: float     # 0-1, confidence in curation
    suggested_improvements: List[str]
    related_content: List[str]     # IDs of related documents


@dataclass
class KnowledgeGapAnalysis:
    """Analysis of knowledge gaps."""
    gap_type: KnowledgeGap
    topic: str
    description: str
    severity: float               # 0-1, how critical the gap is
    evidence: List[str]           # Evidence for the gap
    suggested_content: List[str]  # Suggestions to fill the gap
    related_queries: List[str]    # Queries that revealed the gap


class ContentQualityAssessor:
    """Assess content quality using multiple heuristics."""

    def __init__(self):
        # Quality indicators
        self.quality_indicators = {
            'positive': [
                r'\b(step-by-step|tutorial|guide|example|detailed)\b',
                r'\b(updated|current|latest|new|recent)\b',
                r'\b(complete|comprehensive|thorough)\b',
                r'```.*```',  # Code blocks
                r'\n\s*[-*]\s+',  # Lists
                r'\n#{1,6}\s+',  # Headers
            ],
            'negative': [
                r'\b(outdated|old|deprecated|legacy)\b',
                r'\b(TODO|FIXME|hack|workaround)\b',
                r'\b(unclear|confusing|unknown)\b',
                r'\?\?\?',  # Question marks indicating uncertainty
                r'\b(broken|not working|doesn\'t work)\b'
            ]
        }

        # Technical depth indicators
        self.technical_indicators = [
            r'\b(API|SDK|framework|library|module)\b',
            r'\b(function|method|class|variable|parameter)\b',
            r'\b(database|query|schema|table|index)\b',
            r'\b(server|client|endpoint|request|response)\b',
            r'\b(authentication|authorization|security)\b',
            r'[a-zA-Z_][a-zA-Z0-9_]*\([^)]*\)',  # Function calls
            r'[A-Z][a-zA-Z0-9]*[A-Z][a-zA-Z0-9]*',  # CamelCase
        ]

        # Stop words for readability analysis
        self.stop_words = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'you', 'your', 'this', 'can', 'have'
        }

    def assess_content_quality(self, document: Document) -> ContentMetrics:
        """Assess overall quality of a document."""
        content = document.content

        return ContentMetrics(
            readability_score=self._assess_readability(content),
            completeness_score=self._assess_completeness(content),
            accuracy_confidence=self._assess_accuracy_confidence(content),
            freshness_score=self._assess_freshness(document),
            engagement_score=self._assess_engagement(document),
            technical_depth_score=self._assess_technical_depth(content),
            structure_quality_score=self._assess_structure_quality(content)
        )

    def _assess_readability(self, content: str) -> float:
        """Assess content readability (simplified Flesch score approximation)."""
        sentences = len(re.findall(r'[.!?]+', content))
        words = len(re.findall(r'\w+', content))

        if sentences == 0 or words == 0:
            return 0.0

        # Average sentence length
        avg_sentence_length = words / sentences

        # Complex word ratio (words > 6 characters)
        complex_words = len(re.findall(r'\b\w{7,}\b', content))
        complex_ratio = complex_words / words if words > 0 else 0

        # Simplified readability score (inverse of complexity)
        if avg_sentence_length > 25:  # Very long sentences
            sentence_penalty = 0.3
        elif avg_sentence_length > 15:  # Moderately long sentences
            sentence_penalty = 0.1
        else:
            sentence_penalty = 0.0

        complexity_penalty = complex_ratio * 0.5

        readability = max(0.0, 1.0 - sentence_penalty - complexity_penalty)
        return min(1.0, readability)

    def _assess_completeness(self, content: str) -> float:
        """Assess how complete/comprehensive the content is."""
        score = 0.0

        # Length-based completeness
        length = len(content)
        if length > 2000:  # Comprehensive
            score += 0.4
        elif length > 500:  # Decent coverage
            score += 0.2
        elif length < 100:  # Too brief
            score -= 0.2

        # Structural completeness indicators
        structure_indicators = [
            (r'\n#{1,6}\s+', 0.1),      # Headers
            (r'\n\s*[-*]\s+', 0.1),     # Lists
            (r'```.*?```', 0.1),        # Code blocks
            (r'\b(example|example[s]?)\b', 0.1),  # Examples
            (r'\b(step\s*\d+|first|second|third|then|next|finally)\b', 0.1),  # Sequential steps
            (r'\b(note|warning|important|tip)\b', 0.1),  # Callouts
        ]

        for pattern, weight in structure_indicators:
            if re.search(pattern, content, re.IGNORECASE | re.DOTALL):
                score += weight

        # Topic coverage (check for multiple aspects)
        topic_coverage = [
            r'\b(what|definition|overview)\b',     # What it is
            r'\b(why|benefits?|advantages?)\b',    # Why use it
            r'\b(how|steps?|process|procedure)\b', # How to do it
            r'\b(when|use\s+cases?)\b',           # When to use
            r'\b(example|demo|sample)\b',         # Examples
        ]

        coverage_count = sum(1 for pattern in topic_coverage
                           if re.search(pattern, content, re.IGNORECASE))
        score += (coverage_count / len(topic_coverage)) * 0.3

        return min(1.0, max(0.0, score))

    def _assess_accuracy_confidence(self, content: str) -> float:
        """Assess confidence in content accuracy based on language used."""
        confidence = 0.7  # Start with neutral confidence

        # Confidence indicators (positive)
        confidence_patterns = [
            r'\b(verified|tested|confirmed|documented)\b',
            r'\b(official|standard|recommended)\b',
            r'\b(version\s+\d+\.\d+|as\s+of\s+\d{4})\b',  # Version/date specificity
        ]

        # Uncertainty indicators (negative)
        uncertainty_patterns = [
            r'\b(might|maybe|possibly|probably|seems?)\b',
            r'\b(unsure|uncertain|unclear|unknown)\b',
            r'\?\?\?|\bTODO\b|\bFIXME\b',
        ]

        for pattern in confidence_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                confidence += 0.1

        for pattern in uncertainty_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                confidence -= 0.1

        return min(1.0, max(0.0, confidence))

    def _assess_freshness(self, document: Document) -> float:
        """Assess how fresh/current the content is."""
        # Check if document has timestamp
        if hasattr(document, 'created_at') and document.created_at:
            try:
                created_time = datetime.fromisoformat(document.created_at)
                age_days = (datetime.now() - created_time).days

                # Fresh content scoring
                if age_days <= 7:
                    return 1.0
                elif age_days <= 30:
                    return 0.8
                elif age_days <= 90:
                    return 0.6
                elif age_days <= 365:
                    return 0.4
                else:
                    return 0.2
            except:
                pass

        # Content-based freshness indicators
        content = document.content.lower()

        # Recent indicators
        current_year = datetime.now().year
        recent_patterns = [
            rf'\b{current_year}\b',
            rf'\b{current_year-1}\b',  # Last year is still reasonably fresh
            r'\b(new|recent|latest|updated|current)\b',
            r'\b(version\s+[2-9]\.\d+)\b',  # Higher version numbers
        ]

        # Outdated indicators
        outdated_patterns = [
            r'\b(old|legacy|deprecated|obsolete)\b',
            r'\b(no longer|not supported|discontinued)\b',
            rf'\b{current_year-3}\b',  # 3+ years old mentions
            r'\b(version\s+[01]\.\d+)\b',  # Very old versions
        ]

        freshness_score = 0.5  # Neutral baseline

        for pattern in recent_patterns:
            if re.search(pattern, content):
                freshness_score += 0.1

        for pattern in outdated_patterns:
            if re.search(pattern, content):
                freshness_score -= 0.2

        return min(1.0, max(0.0, freshness_score))

    def _assess_engagement(self, document: Document) -> float:
        """Assess user engagement potential based on content characteristics."""
        content = document.content

        # Engagement indicators
        engagement_score = 0.0

        # Interactive elements
        interactive_patterns = [
            (r'\b(try\s+it|follow\s+along|hands-on)\b', 0.2),
            (r'\b(exercise|practice|assignment)\b', 0.1),
            (r'\b(quiz|test|check)\b', 0.1),
        ]

        # Visual elements (implied)
        visual_patterns = [
            (r'\b(screenshot|image|diagram|chart)\b', 0.1),
            (r'\b(see\s+figure|shown\s+below)\b', 0.1),
        ]

        # Conversational tone
        conversational_patterns = [
            (r'\b(you|your|we|let\'s)\b', 0.05),
            (r'\b(question|ask|wonder)\b', 0.05),
        ]

        all_patterns = interactive_patterns + visual_patterns + conversational_patterns

        for pattern, weight in all_patterns:
            matches = len(re.findall(pattern, content, re.IGNORECASE))
            engagement_score += min(matches * weight, weight)  # Cap contribution per pattern

        return min(1.0, max(0.0, engagement_score))

    def _assess_technical_depth(self, content: str) -> float:
        """Assess the technical depth and sophistication of content."""
        technical_score = 0.0

        # Count technical indicators
        for pattern in self.technical_indicators:
            matches = len(re.findall(pattern, content, re.IGNORECASE))
            technical_score += min(matches * 0.05, 0.2)  # Each type contributes max 0.2

        # Code blocks indicate high technical depth
        code_blocks = len(re.findall(r'```.*?```', content, re.DOTALL))
        technical_score += min(code_blocks * 0.1, 0.3)

        # Technical complexity indicators
        complexity_patterns = [
            r'\b(algorithm|complexity|optimization|performance)\b',
            r'\b(architecture|design\s+pattern|scalability)\b',
            r'\b(concurrent|parallel|asynchronous|threading)\b',
        ]

        for pattern in complexity_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                technical_score += 0.1

        return min(1.0, max(0.0, technical_score))

    def _assess_structure_quality(self, content: str) -> float:
        """Assess the structural organization and formatting quality."""
        structure_score = 0.0

        # Headers (good structure)
        headers = len(re.findall(r'\n#{1,6}\s+', content))
        if headers > 0:
            structure_score += min(headers * 0.1, 0.3)

        # Lists (organized information)
        lists = len(re.findall(r'\n\s*[-*]\s+', content))
        if lists > 0:
            structure_score += min(lists * 0.02, 0.2)

        # Paragraphs (not wall of text)
        paragraphs = len(re.findall(r'\n\s*\n', content))
        if paragraphs > 0:
            structure_score += min(paragraphs * 0.05, 0.2)

        # Code formatting
        if re.search(r'```.*?```', content, re.DOTALL):
            structure_score += 0.1

        # Consistent formatting indicators
        if re.search(r'\n\s*\d+\.\s+', content):  # Numbered lists
            structure_score += 0.1

        return min(1.0, max(0.0, structure_score))


class ContentTypeClassifier:
    """Classify content into different types."""

    def __init__(self):
        self.type_patterns = {
            ContentType.TUTORIAL: [
                r'\b(tutorial|guide|walkthrough|step-by-step)\b',
                r'\b(how\s+to|getting\s+started)\b',
                r'\b(step\s+\d+|first|second|then|next|finally)\b',
            ],
            ContentType.REFERENCE: [
                r'\b(API|reference|documentation|specification)\b',
                r'\b(function|method|parameter|return)\b',
                r'\b(class|interface|type|enum)\b',
            ],
            ContentType.TROUBLESHOOTING: [
                r'\b(troubleshoot|debug|fix|solve|error|issue)\b',
                r'\b(problem|not\s+working|broken)\b',
                r'\b(solution|resolution|workaround)\b',
            ],
            ContentType.FAQ: [
                r'\b(FAQ|frequently\s+asked|common\s+questions)\b',
                r'\b(Q:|A:|Question:|Answer:)\b',
                r'^\s*Q\d*[.:]\s+',  # Q1: or Q:
            ],
            ContentType.ANNOUNCEMENT: [
                r'\b(announce|release|update|news)\b',
                r'\b(version|v\d+\.\d+|changelog)\b',
                r'\b(new\s+feature|improvement|fix)\b',
            ],
            ContentType.DISCUSSION: [
                r'\b(discussion|conversation|thread)\b',
                r'\b(comment|reply|response)\b',
                r'\b(what\s+do\s+you\s+think|opinions?)\b',
            ],
            ContentType.CODE_EXAMPLE: [
                r'```.*```',  # Code blocks
                r'\b(example|sample|snippet|demo)\b',
                r'\b(code|script|program|implementation)\b',
            ]
        }

    def classify_content_type(self, document: Document) -> Tuple[ContentType, float]:
        """Classify content type with confidence score."""
        content = document.content.lower()
        scores = {}

        for content_type, patterns in self.type_patterns.items():
            score = 0.0
            for pattern in patterns:
                matches = len(re.findall(pattern, content))
                score += matches

            # Normalize by content length
            normalized_score = score / max(len(content.split()), 1)
            scores[content_type] = normalized_score

        if not scores or max(scores.values()) == 0:
            return ContentType.DISCUSSION, 0.0  # Default type

        best_type = max(scores, key=scores.get)
        confidence = min(scores[best_type] * 10, 1.0)  # Scale and cap at 1.0

        return best_type, confidence


class KnowledgeGapDetector:
    """Detect gaps in knowledge coverage."""

    def __init__(self):
        self.topic_patterns = {
            'authentication': r'\b(auth|login|password|jwt|oauth|sso)\b',
            'deployment': r'\b(deploy|release|build|ci|cd|docker|kubernetes)\b',
            'database': r'\b(database|sql|postgres|mysql|redis|mongo)\b',
            'api': r'\b(api|endpoint|rest|graphql|service|microservice)\b',
            'frontend': r'\b(react|vue|angular|html|css|javascript|ui|ux)\b',
            'backend': r'\b(server|node|python|java|spring|framework)\b',
            'monitoring': r'\b(monitor|log|metric|alert|observability)\b',
            'security': r'\b(security|vulnerability|encrypt|https|ssl|tls)\b',
            'testing': r'\b(test|testing|unit|integration|e2e|qa)\b',
            'documentation': r'\b(docs|documentation|readme|guide|manual)\b'
        }

        # Track query patterns that don't find good results
        self.unresolved_queries = defaultdict(list)

    def analyze_knowledge_gaps(self, documents: List[Document],
                             query_analytics: Optional[Dict] = None) -> List[KnowledgeGapAnalysis]:
        """Analyze knowledge gaps in the document collection."""
        gaps = []

        # Analyze topic coverage gaps
        topic_coverage = self._analyze_topic_coverage(documents)
        gaps.extend(self._identify_coverage_gaps(topic_coverage))

        # Analyze freshness gaps
        gaps.extend(self._identify_freshness_gaps(documents))

        # Analyze query-based gaps (if analytics provided)
        if query_analytics:
            gaps.extend(self._identify_query_based_gaps(query_analytics))

        # Analyze content conflicts
        gaps.extend(self._identify_content_conflicts(documents))

        return sorted(gaps, key=lambda x: x.severity, reverse=True)

    def _analyze_topic_coverage(self, documents: List[Document]) -> Dict[str, int]:
        """Analyze how well each topic is covered."""
        coverage = defaultdict(int)

        for document in documents:
            content = document.content.lower()
            for topic, pattern in self.topic_patterns.items():
                matches = len(re.findall(pattern, content))
                if matches > 0:
                    coverage[topic] += 1

        return coverage

    def _identify_coverage_gaps(self, coverage: Dict[str, int]) -> List[KnowledgeGapAnalysis]:
        """Identify topics with insufficient coverage."""
        gaps = []

        total_topics = len(self.topic_patterns)
        avg_coverage = sum(coverage.values()) / total_topics if total_topics > 0 else 0

        for topic, count in coverage.items():
            if count == 0:
                # Complete gap
                gaps.append(KnowledgeGapAnalysis(
                    gap_type=KnowledgeGap.MISSING_TOPIC,
                    topic=topic,
                    description=f"No content found for topic: {topic}",
                    severity=0.9,
                    evidence=[f"No documents match {topic} patterns"],
                    suggested_content=[
                        f"Create {topic} overview documentation",
                        f"Add {topic} tutorial or guide",
                        f"Include {topic} best practices"
                    ],
                    related_queries=[]
                ))
            elif count < avg_coverage * 0.5:
                # Insufficient coverage
                gaps.append(KnowledgeGapAnalysis(
                    gap_type=KnowledgeGap.INCOMPLETE_COVERAGE,
                    topic=topic,
                    description=f"Limited content for topic: {topic} ({count} documents)",
                    severity=0.6,
                    evidence=[f"Only {count} documents cover {topic}, below average of {avg_coverage:.1f}"],
                    suggested_content=[
                        f"Expand {topic} documentation",
                        f"Add more {topic} examples",
                        f"Create {topic} FAQ"
                    ],
                    related_queries=[]
                ))

        return gaps

    def _identify_freshness_gaps(self, documents: List[Document]) -> List[KnowledgeGapAnalysis]:
        """Identify outdated content that needs updating."""
        gaps = []
        current_time = time.time()

        for document in documents:
            # Check for age-based staleness (if timestamp available)
            if hasattr(document, 'created_at') and document.created_at:
                try:
                    created_time = datetime.fromisoformat(document.created_at)
                    age_days = (datetime.now() - created_time).days

                    if age_days > 365:  # Over a year old
                        gaps.append(KnowledgeGapAnalysis(
                            gap_type=KnowledgeGap.OUTDATED_INFO,
                            topic=f"Outdated document: {document.source}",
                            description=f"Document is {age_days} days old and may need updating",
                            severity=min(age_days / 730, 0.8),  # Severity increases with age
                            evidence=[f"Created {age_days} days ago"],
                            suggested_content=[
                                "Review and update content for accuracy",
                                "Verify all links and references",
                                "Update examples and code samples"
                            ],
                            related_queries=[]
                        ))
                except:
                    pass

            # Content-based staleness detection
            content = document.content.lower()
            staleness_indicators = [
                r'\b(deprecated|obsolete|no\s+longer|discontinued)\b',
                r'\b(old\s+version|legacy|outdated)\b',
                r'\b20(0[0-9]|1[0-8])\b',  # Years 2000-2018
            ]

            for indicator in staleness_indicators:
                if re.search(indicator, content):
                    gaps.append(KnowledgeGapAnalysis(
                        gap_type=KnowledgeGap.OUTDATED_INFO,
                        topic=f"Potentially outdated: {document.source}",
                        description="Content contains indicators of being outdated",
                        severity=0.5,
                        evidence=[f"Contains pattern: {indicator}"],
                        suggested_content=[
                            "Review content for currency",
                            "Update deprecated references",
                            "Add current alternatives"
                        ],
                        related_queries=[]
                    ))
                    break  # Don't add multiple gaps for same document

        return gaps

    def _identify_query_based_gaps(self, query_analytics: Dict) -> List[KnowledgeGapAnalysis]:
        """Identify gaps based on unresolved queries."""
        gaps = []

        # Look for common query patterns that don't get good results
        unresolved_queries = query_analytics.get('low_satisfaction_queries', [])

        if unresolved_queries:
            # Group similar queries
            query_groups = self._group_similar_queries(unresolved_queries)

            for group_topic, queries in query_groups.items():
                if len(queries) >= 3:  # Multiple similar failed queries
                    gaps.append(KnowledgeGapAnalysis(
                        gap_type=KnowledgeGap.MISSING_TOPIC,
                        topic=group_topic,
                        description=f"Multiple unresolved queries about {group_topic}",
                        severity=min(len(queries) / 10, 0.9),
                        evidence=[f"{len(queries)} similar unresolved queries"],
                        suggested_content=[
                            f"Create comprehensive guide for {group_topic}",
                            f"Add FAQ section for {group_topic}",
                            f"Include troubleshooting for {group_topic}"
                        ],
                        related_queries=queries[:5]  # Include sample queries
                    ))

        return gaps

    def _group_similar_queries(self, queries: List[str]) -> Dict[str, List[str]]:
        """Group similar queries by topic."""
        groups = defaultdict(list)

        for query in queries:
            query_lower = query.lower()

            # Try to match to known topics
            matched_topic = None
            for topic, pattern in self.topic_patterns.items():
                if re.search(pattern, query_lower):
                    matched_topic = topic
                    break

            if matched_topic:
                groups[matched_topic].append(query)
            else:
                # Extract key terms for grouping
                key_terms = re.findall(r'\b\w{4,}\b', query_lower)
                if key_terms:
                    primary_term = key_terms[0]  # Use first significant term
                    groups[primary_term].append(query)

        return groups

    def _identify_content_conflicts(self, documents: List[Document]) -> List[KnowledgeGapAnalysis]:
        """Identify conflicting information across documents."""
        gaps = []

        # Simple conflict detection - look for documents on same topic with different conclusions
        topic_docs = defaultdict(list)

        for document in documents:
            content = document.content.lower()
            for topic, pattern in self.topic_patterns.items():
                if re.search(pattern, content):
                    topic_docs[topic].append(document)

        # Look for conflicting statements in same-topic documents
        for topic, docs in topic_docs.items():
            if len(docs) >= 2:
                conflict_indicators = self._detect_conflicts_in_documents(docs)
                if conflict_indicators:
                    gaps.append(KnowledgeGapAnalysis(
                        gap_type=KnowledgeGap.CONFLICTING_INFO,
                        topic=topic,
                        description=f"Potential conflicting information about {topic}",
                        severity=0.7,
                        evidence=conflict_indicators,
                        suggested_content=[
                            f"Review {topic} documentation for consistency",
                            f"Consolidate {topic} information",
                            f"Create authoritative {topic} reference"
                        ],
                        related_queries=[]
                    ))

        return gaps

    def _detect_conflicts_in_documents(self, documents: List[Document]) -> List[str]:
        """Detect potential conflicts between documents."""
        conflicts = []

        # Look for contradictory statements
        contradiction_patterns = [
            (r'\b(recommended|should|must)\b', r'\b(not\s+recommended|shouldn\'t|mustn\'t)\b'),
            (r'\b(secure|safe)\b', r'\b(insecure|unsafe|vulnerable)\b'),
            (r'\b(fast|quick|efficient)\b', r'\b(slow|sluggish|inefficient)\b'),
            (r'\b(deprecated)\b', r'\b(recommended|current|latest)\b'),
        ]

        for i, doc1 in enumerate(documents):
            for doc2 in documents[i+1:]:
                for positive_pattern, negative_pattern in contradiction_patterns:
                    if (re.search(positive_pattern, doc1.content.lower()) and
                        re.search(negative_pattern, doc2.content.lower())):
                        conflicts.append(
                            f"Potential conflict between {doc1.source} and {doc2.source}"
                        )
                        break

        return conflicts


class ContentCurationSystem:
    """Main content curation system orchestrating all components."""

    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = Path(storage_path) if storage_path else Path("data/content_curation.json")

        self.quality_assessor = ContentQualityAssessor()
        self.type_classifier = ContentTypeClassifier()
        self.gap_detector = KnowledgeGapDetector()

        self.curated_content: Dict[str, CuratedContent] = {}
        self.knowledge_gaps: List[KnowledgeGapAnalysis] = []

        # Load existing curation data
        self._load_curation_data()

    def curate_document(self, document: Document) -> CuratedContent:
        """Curate a single document."""
        doc_id = self._generate_document_id(document)

        # Assess quality
        metrics = self.quality_assessor.assess_content_quality(document)

        # Classify type
        content_type, type_confidence = self.type_classifier.classify_content_type(document)

        # Determine overall quality level
        quality_score = metrics.overall_quality_score()
        if quality_score >= 0.8:
            quality = ContentQuality.EXCELLENT
        elif quality_score >= 0.6:
            quality = ContentQuality.GOOD
        elif quality_score >= 0.4:
            quality = ContentQuality.FAIR
        else:
            quality = ContentQuality.POOR

        # Generate tags
        tags = self._generate_content_tags(document)

        # Generate improvement suggestions
        suggestions = self._generate_improvement_suggestions(metrics, document)

        # Find related content
        related_content = self._find_related_content(document)

        curated = CuratedContent(
            document=document,
            content_type=content_type,
            quality=quality,
            metrics=metrics,
            tags=tags,
            last_curated=time.time(),
            curation_confidence=type_confidence,
            suggested_improvements=suggestions,
            related_content=related_content
        )

        self.curated_content[doc_id] = curated
        return curated

    def curate_document_collection(self, documents: List[Document]) -> Dict[str, CuratedContent]:
        """Curate an entire document collection."""
        curated_docs = {}

        for document in documents:
            curated = self.curate_document(document)
            doc_id = self._generate_document_id(document)
            curated_docs[doc_id] = curated

        # Detect knowledge gaps across the collection
        self.knowledge_gaps = self.gap_detector.analyze_knowledge_gaps(documents)

        # Persist curation data
        self._save_curation_data()

        return curated_docs

    def get_high_quality_content(self, limit: int = 10) -> List[CuratedContent]:
        """Get the highest quality content."""
        return sorted(
            self.curated_content.values(),
            key=lambda x: x.metrics.overall_quality_score(),
            reverse=True
        )[:limit]

    def get_content_needing_improvement(self, limit: int = 10) -> List[CuratedContent]:
        """Get content that needs improvement."""
        poor_content = [
            content for content in self.curated_content.values()
            if content.quality in [ContentQuality.POOR, ContentQuality.FAIR]
        ]

        return sorted(
            poor_content,
            key=lambda x: len(x.suggested_improvements),
            reverse=True
        )[:limit]

    def get_knowledge_gaps(self, min_severity: float = 0.5) -> List[KnowledgeGapAnalysis]:
        """Get knowledge gaps above minimum severity."""
        return [gap for gap in self.knowledge_gaps if gap.severity >= min_severity]

    def get_content_by_type(self, content_type: ContentType) -> List[CuratedContent]:
        """Get all content of a specific type."""
        return [
            content for content in self.curated_content.values()
            if content.content_type == content_type
        ]

    def suggest_content_priorities(self) -> List[str]:
        """Suggest content creation priorities based on gaps and quality."""
        priorities = []

        # High-severity gaps get top priority
        high_priority_gaps = [gap for gap in self.knowledge_gaps if gap.severity >= 0.8]
        for gap in high_priority_gaps[:5]:
            priorities.append(f"CRITICAL: {gap.description}")

        # Popular topics with poor content
        poor_content_topics = Counter()
        for content in self.curated_content.values():
            if content.quality == ContentQuality.POOR:
                for tag in content.tags:
                    poor_content_topics[tag] += 1

        for topic, count in poor_content_topics.most_common(3):
            priorities.append(f"IMPROVE: {topic} content quality ({count} poor documents)")

        # Missing content types
        type_counts = Counter(content.content_type for content in self.curated_content.values())
        total_content = len(self.curated_content)

        for content_type in ContentType:
            percentage = (type_counts.get(content_type, 0) / total_content) * 100 if total_content > 0 else 0
            if percentage < 10:  # Less than 10% of content
                priorities.append(f"ADD: More {content_type.value} content (only {percentage:.1f}% of collection)")

        return priorities[:10]  # Return top 10 priorities

    def _generate_document_id(self, document: Document) -> str:
        """Generate unique ID for document."""
        content_hash = hashlib.md5(document.content.encode()).hexdigest()
        source_hash = hashlib.md5(document.source.encode()).hexdigest()
        return f"{source_hash[:8]}_{content_hash[:8]}"

    def _generate_content_tags(self, document: Document) -> Set[str]:
        """Generate tags for content."""
        tags = set()
        content = document.content.lower()

        # Technology tags
        tech_patterns = {
            'python': r'\bpython\b',
            'javascript': r'\b(javascript|js|node)\b',
            'docker': r'\bdocker\b',
            'kubernetes': r'\b(kubernetes|k8s)\b',
            'api': r'\bapi\b',
            'database': r'\b(database|sql|db)\b',
            'authentication': r'\b(auth|login|jwt|oauth)\b',
            'deployment': r'\b(deploy|deployment|ci|cd)\b',
            'monitoring': r'\b(monitor|metric|log|alert)\b',
            'security': r'\b(security|secure|vulnerability)\b',
        }

        for tag, pattern in tech_patterns.items():
            if re.search(pattern, content):
                tags.add(tag)

        # Difficulty level tags
        if re.search(r'\b(beginner|basic|intro|getting\s+started)\b', content):
            tags.add('beginner')
        elif re.search(r'\b(advanced|expert|complex|sophisticated)\b', content):
            tags.add('advanced')
        else:
            tags.add('intermediate')

        # Length tags
        word_count = len(content.split())
        if word_count < 200:
            tags.add('short')
        elif word_count > 1000:
            tags.add('comprehensive')

        return tags

    def _generate_improvement_suggestions(self, metrics: ContentMetrics, document: Document) -> List[str]:
        """Generate specific improvement suggestions."""
        suggestions = []

        if metrics.readability_score < 0.6:
            suggestions.append("Improve readability by using shorter sentences and simpler vocabulary")

        if metrics.completeness_score < 0.5:
            suggestions.append("Add more comprehensive coverage with examples and detailed explanations")

        if metrics.freshness_score < 0.5:
            suggestions.append("Update content with current information and remove outdated references")

        if metrics.structure_quality_score < 0.6:
            suggestions.append("Improve structure with headers, lists, and better formatting")

        if metrics.engagement_score < 0.3:
            suggestions.append("Make content more engaging with interactive elements and conversational tone")

        if metrics.technical_depth_score < 0.4 and 'tutorial' in document.content.lower():
            suggestions.append("Add code examples and technical implementation details")

        if not re.search(r'```.*```', document.content, re.DOTALL) and 'code' in document.content.lower():
            suggestions.append("Include formatted code examples for better clarity")

        return suggestions

    def _find_related_content(self, document: Document) -> List[str]:
        """Find related content based on topic similarity."""
        related = []

        # Simple keyword-based similarity
        doc_words = set(re.findall(r'\w+', document.content.lower()))

        for doc_id, curated in self.curated_content.items():
            other_words = set(re.findall(r'\w+', curated.document.content.lower()))

            # Calculate Jaccard similarity
            intersection = len(doc_words & other_words)
            union = len(doc_words | other_words)

            if union > 0 and intersection / union > 0.3:  # 30% similarity threshold
                related.append(doc_id)

        return related[:5]  # Return top 5 related documents

    def _load_curation_data(self):
        """Load existing curation data."""
        if self.storage_path.exists():
            try:
                import json
                with open(self.storage_path) as f:
                    data = json.load(f)

                # Restore curated content (simplified - would need full deserialization)
                logger.info(f"Loaded curation data from {self.storage_path}")

            except Exception as e:
                logger.warning(f"Failed to load curation data: {e}")

    def _save_curation_data(self):
        """Save curation data to storage."""
        try:
            # Create directory if it doesn't exist
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)

            # Simplified serialization (would need full implementation)
            data = {
                'curated_count': len(self.curated_content),
                'knowledge_gaps_count': len(self.knowledge_gaps),
                'last_updated': time.time()
            }

            import json
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)

            logger.info(f"Saved curation data to {self.storage_path}")

        except Exception as e:
            logger.error(f"Failed to save curation data: {e}")


# Convenience functions
def assess_document_quality(document: Document) -> ContentMetrics:
    """Assess quality of a single document."""
    assessor = ContentQualityAssessor()
    return assessor.assess_content_quality(document)


def detect_knowledge_gaps(documents: List[Document]) -> List[KnowledgeGapAnalysis]:
    """Detect knowledge gaps in document collection."""
    detector = KnowledgeGapDetector()
    return detector.analyze_knowledge_gaps(documents)


def create_curation_system(storage_path: Optional[str] = None) -> ContentCurationSystem:
    """Create and return a content curation system."""
    return ContentCurationSystem(storage_path)
