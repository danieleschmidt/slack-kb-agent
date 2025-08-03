"""Tests for advanced algorithms and intelligent query processing."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.slack_kb_agent.advanced_algorithms import (
    IntelligentQueryRouter,
    KnowledgeGapAnalyzer,
    ContentQualityOptimizer,
    PerformanceOptimizer,
    QueryComplexity,
    KnowledgeGap
)
from src.slack_kb_agent.models import Document, DocumentType, SourceType


class TestIntelligentQueryRouter:
    """Test intelligent query routing and complexity classification."""
    
    @pytest.fixture
    def router(self):
        """Create router instance for testing."""
        return IntelligentQueryRouter()
    
    def test_classify_simple_query(self, router):
        """Test classification of simple queries."""
        simple_queries = [
            "what is Python",
            "define API",
            "show me docs",
            "list files"
        ]
        
        for query in simple_queries:
            complexity = router.classify_query_complexity(query)
            assert complexity in [QueryComplexity.SIMPLE, QueryComplexity.MODERATE]
    
    def test_classify_complex_query(self, router):
        """Test classification of complex queries."""
        complex_queries = [
            "how does the authentication system work with microservices",
            "compare the performance implications of different caching strategies",
            "explain the relationship between database schema and API design"
        ]
        
        for query in complex_queries:
            complexity = router.classify_query_complexity(query)
            assert complexity in [QueryComplexity.COMPLEX, QueryComplexity.EXPERT]
    
    def test_classify_expert_query(self, router):
        """Test classification of expert-level queries."""
        expert_queries = [
            "how to implement distributed consensus algorithm with Byzantine fault tolerance",
            "architecture design patterns for scalable microservices with event sourcing",
            "performance optimization strategies for high-throughput database systems"
        ]
        
        for query in expert_queries:
            complexity = router.classify_query_complexity(query)
            assert complexity == QueryComplexity.EXPERT
    
    def test_route_query_basic(self, router):
        """Test basic query routing functionality."""
        query = "how to deploy the application"
        user_context = {"expertise_level": "intermediate"}
        
        routing = router.route_query(query, user_context)
        
        assert "complexity" in routing
        assert "suggested_search_strategy" in routing
        assert "context_requirements" in routing
        assert "response_style" in routing
        assert "escalation_needed" in routing
    
    def test_route_query_escalation(self, router):
        """Test query escalation logic."""
        expert_query = "implement custom distributed lock manager"
        low_confidence_context = {
            "expertise_level": "expert",
            "topic_confidence": 0.3
        }
        
        routing = router.route_query(expert_query, low_confidence_context)
        
        assert routing["escalation_needed"] is True
    
    def test_search_strategy_mapping(self, router):
        """Test search strategy recommendations."""
        test_cases = [
            (QueryComplexity.SIMPLE, "keyword_exact"),
            (QueryComplexity.MODERATE, "hybrid_weighted"),
            (QueryComplexity.COMPLEX, "semantic_deep"),
            (QueryComplexity.EXPERT, "multi_step_reasoning")
        ]
        
        for complexity, expected_strategy in test_cases:
            strategy = router._get_search_strategy(complexity)
            assert strategy == expected_strategy
    
    def test_response_style_adaptation(self, router):
        """Test response style adaptation."""
        # Expert user with expert query
        style = router._get_response_style(QueryComplexity.EXPERT, "expert")
        assert style == "technical_detailed"
        
        # Beginner user with any complex query
        style = router._get_response_style(QueryComplexity.COMPLEX, "beginner")
        assert style == "explanatory_with_examples"
        
        # Simple query
        style = router._get_response_style(QueryComplexity.SIMPLE, "intermediate")
        assert style == "concise"


class TestKnowledgeGapAnalyzer:
    """Test knowledge gap analysis and identification."""
    
    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance for testing."""
        return KnowledgeGapAnalyzer(min_frequency_threshold=2)
    
    def test_record_unanswered_query(self, analyzer):
        """Test recording unanswered queries."""
        query = "how to configure distributed tracing"
        analyzer.record_unanswered_query(query)
        
        assert len(analyzer.unanswered_queries) == 1
        assert analyzer.unanswered_queries[0]["query"] == query
        assert analyzer.unanswered_queries[0]["topic"] == "general"
    
    def test_record_low_confidence_response(self, analyzer):
        """Test recording low confidence responses."""
        query = "database migration best practices"
        analyzer.record_low_confidence_response(query, 0.3)
        
        assert len(analyzer.low_confidence_responses) == 1
        assert analyzer.low_confidence_responses[0]["confidence"] == 0.3
        assert analyzer.low_confidence_responses[0]["topic"] == "database"
    
    def test_identify_knowledge_gaps(self, analyzer):
        """Test knowledge gap identification."""
        # Record multiple queries about deployment
        deployment_queries = [
            "how to deploy to kubernetes",
            "deployment pipeline setup",
            "docker deployment issues"
        ]
        
        for query in deployment_queries:
            analyzer.record_unanswered_query(query)
        
        gaps = analyzer.identify_knowledge_gaps(days_window=1)
        
        assert len(gaps) > 0
        deployment_gap = next((gap for gap in gaps if gap.topic == "deployment"), None)
        assert deployment_gap is not None
        assert deployment_gap.frequency >= 3
        assert len(deployment_gap.suggested_sources) > 0
    
    def test_topic_extraction(self, analyzer):
        """Test topic extraction from queries."""
        test_cases = [
            ("deploy with docker", "deployment"),
            ("sql query optimization", "database"),
            ("rest api endpoints", "api"),
            ("authentication error", "authentication"),
            ("pytest fixtures", "testing"),
            ("general question", "general")
        ]
        
        for query, expected_topic in test_cases:
            topic = analyzer._extract_topic(query)
            assert topic == expected_topic
    
    def test_priority_scoring(self, analyzer):
        """Test priority score calculation."""
        # Recent, frequent topic should have high priority
        recent_time = datetime.utcnow() - timedelta(hours=1)
        score = analyzer._calculate_priority_score("authentication", 5.0, recent_time)
        
        assert score > 0.8  # Should be high priority
        
        # Old, infrequent topic should have low priority
        old_time = datetime.utcnow() - timedelta(days=20)
        score = analyzer._calculate_priority_score("general", 1.0, old_time)
        
        assert score < 0.5  # Should be low priority


class TestContentQualityOptimizer:
    """Test content quality optimization and scoring."""
    
    @pytest.fixture
    def optimizer(self):
        """Create optimizer instance for testing."""
        return ContentQualityOptimizer()
    
    @pytest.fixture
    def sample_document(self):
        """Create sample document for testing."""
        return Document(
            content="Python is a programming language with clear syntax",
            source="python_docs",
            doc_type=DocumentType.API_DOCUMENTATION,
            source_type=SourceType.GITHUB,
            priority=4,
            tags=["python", "programming"],
            created_at=datetime.utcnow() - timedelta(days=5)
        )
    
    def test_calculate_document_relevance_score(self, optimizer, sample_document):
        """Test comprehensive relevance scoring."""
        query = "python programming syntax"
        user_context = {"interests": ["python"]}
        
        score = optimizer.calculate_document_relevance_score(
            sample_document, query, user_context
        )
        
        assert 0.0 <= score <= 1.0
        assert score > 0.5  # Should be relevant due to matching keywords and tags
    
    def test_text_similarity_calculation(self, optimizer):
        """Test text similarity calculations."""
        content = "Python programming language with object-oriented features"
        query = "python programming"
        
        similarity = optimizer._calculate_text_similarity(content, query)
        
        assert 0.0 <= similarity <= 1.0
        assert similarity > 0.1  # Should have some similarity
    
    def test_freshness_scoring(self, optimizer, sample_document):
        """Test freshness score calculation."""
        # Very new document
        new_doc = Document(
            content="test",
            source="test",
            created_at=datetime.utcnow() - timedelta(days=1)
        )
        new_score = optimizer._calculate_freshness_score(new_doc)
        
        # Old document
        old_doc = Document(
            content="test",
            source="test",
            created_at=datetime.utcnow() - timedelta(days=400)
        )
        old_score = optimizer._calculate_freshness_score(old_doc)
        
        assert new_score > old_score
        assert new_score >= 0.8  # New documents should have high freshness
        assert old_score <= 0.3   # Old documents should have low freshness
    
    def test_authority_scoring(self, optimizer):
        """Test authority score calculation."""
        # High authority document
        high_auth_doc = Document(
            content="test",
            source="test",
            doc_type=DocumentType.API_DOCUMENTATION,
            source_type=SourceType.GITHUB
        )
        
        # Low authority document
        low_auth_doc = Document(
            content="test",
            source="test",
            doc_type=DocumentType.SLACK_MESSAGE,
            source_type=SourceType.MANUAL_ENTRY
        )
        
        high_score = optimizer._calculate_authority_score(high_auth_doc)
        low_score = optimizer._calculate_authority_score(low_auth_doc)
        
        assert high_score > low_score
    
    def test_user_feedback_integration(self, optimizer):
        """Test user feedback recording and scoring updates."""
        source = "test_document"
        
        # Record positive feedback
        optimizer.record_user_feedback(source, "test query", True)
        optimizer.record_user_feedback(source, "another query", True)
        optimizer.record_user_feedback(source, "third query", False)
        
        # Check that score was updated
        assert source in optimizer.document_scores
        score = optimizer.document_scores[source]
        assert 0.0 <= score <= 1.0
        # Should be positive since 2/3 feedback was positive
        assert score > 0.5


class TestPerformanceOptimizer:
    """Test performance optimization and analysis."""
    
    @pytest.fixture
    def optimizer(self):
        """Create optimizer instance for testing."""
        return PerformanceOptimizer()
    
    def test_record_query_performance(self, optimizer):
        """Test recording query performance metrics."""
        optimizer.record_query_performance(
            query="test query",
            response_time=1.5,
            cache_hit=True,
            resource_usage={"memory_mb": 128, "cpu_percent": 15}
        )
        
        assert len(optimizer.query_response_times) == 1
        assert len(optimizer.cache_hit_rates) == 1
        assert len(optimizer.resource_usage) == 1
        
        # Check data integrity
        record = optimizer.query_response_times[0]
        assert record["query"] == "test query"
        assert record["response_time"] == 1.5
        assert record["cache_hit"] is True
    
    def test_analyze_performance_patterns(self, optimizer):
        """Test performance pattern analysis."""
        # Record some performance data
        for i in range(10):
            optimizer.record_query_performance(
                query=f"query_{i}",
                response_time=1.0 + (i * 0.1),  # Gradually increasing response time
                cache_hit=(i % 3 == 0),  # 33% cache hit rate
                resource_usage={"memory_mb": 100 + i}
            )
        
        analysis = optimizer.analyze_performance_patterns()
        
        assert "avg_response_time" in analysis
        assert "p95_response_time" in analysis
        assert "cache_hit_rate" in analysis
        assert "slow_query_count" in analysis
        assert "recommendations" in analysis
        assert "performance_score" in analysis
        
        # Check that analysis makes sense
        assert analysis["avg_response_time"] > 1.0
        assert 0.0 <= analysis["cache_hit_rate"] <= 1.0
        assert 0.0 <= analysis["performance_score"] <= 1.0
    
    def test_performance_score_calculation(self, optimizer):
        """Test performance score calculation."""
        # Good performance scenario
        good_score = optimizer._calculate_performance_score(
            avg_response_time=0.5,  # Fast
            cache_hit_rate=0.8,     # High hit rate
            slow_query_ratio=0.05   # Few slow queries
        )
        
        # Poor performance scenario
        poor_score = optimizer._calculate_performance_score(
            avg_response_time=3.0,  # Slow
            cache_hit_rate=0.2,     # Low hit rate
            slow_query_ratio=0.3    # Many slow queries
        )
        
        assert 0.0 <= good_score <= 1.0
        assert 0.0 <= poor_score <= 1.0
        assert good_score > poor_score
    
    def test_insufficient_data_handling(self, optimizer):
        """Test handling of insufficient data scenarios."""
        # Should handle empty data gracefully
        analysis = optimizer.analyze_performance_patterns()
        
        assert analysis["status"] == "insufficient_data"
    
    @pytest.mark.parametrize("response_time,cache_hit_rate,expected_recommendations", [
        (3.0, 0.2, 2),  # Slow + low cache hit = 2 recommendations
        (1.0, 0.8, 0),  # Good performance = no recommendations
        (2.5, 0.5, 1),  # Moderate issues = some recommendations
    ])
    def test_recommendation_generation(self, optimizer, response_time, cache_hit_rate, expected_recommendations):
        """Test recommendation generation based on performance metrics."""
        # Record performance data that matches test parameters
        for i in range(10):
            optimizer.record_query_performance(
                query=f"query_{i}",
                response_time=response_time,
                cache_hit=i < (cache_hit_rate * 10),  # Simulate cache hit rate
                resource_usage={"memory_mb": 100}
            )
        
        analysis = optimizer.analyze_performance_patterns()
        
        # Check that recommendations match expectations
        if expected_recommendations == 0:
            assert len(analysis["recommendations"]) == 0
        else:
            assert len(analysis["recommendations"]) >= expected_recommendations


class TestIntegration:
    """Integration tests for advanced algorithms working together."""
    
    def test_query_flow_integration(self):
        """Test complete query processing flow with all components."""
        router = IntelligentQueryRouter()
        gap_analyzer = KnowledgeGapAnalyzer()
        quality_optimizer = ContentQualityOptimizer()
        performance_optimizer = PerformanceOptimizer()
        
        # Simulate a complex query flow
        query = "how to implement microservices architecture with event sourcing"
        user_context = {"expertise_level": "intermediate"}
        
        # Route the query
        routing = router.route_query(query, user_context)
        assert routing["complexity"] == "expert"
        
        # Simulate no good answer found
        gap_analyzer.record_unanswered_query(query)
        
        # Record performance
        start_time = datetime.utcnow()
        performance_optimizer.record_query_performance(
            query=query,
            response_time=2.5,
            cache_hit=False,
            resource_usage={"memory_mb": 256}
        )
        
        # Verify integration works
        gaps = gap_analyzer.identify_knowledge_gaps()
        assert len(gaps) > 0
        
        performance_analysis = performance_optimizer.analyze_performance_patterns()
        assert "avg_response_time" in performance_analysis
    
    def test_learning_and_adaptation(self):
        """Test that components learn and adapt over time."""
        router = IntelligentQueryRouter()
        
        # Record multiple similar queries
        queries = [
            "deploy application",
            "deployment process",
            "how to deploy",
            "deployment pipeline"
        ]
        
        user_context = {"expertise_level": "intermediate"}
        
        for query in queries:
            routing = router.route_query(query, user_context)
            # Verify routing is consistent
            assert "deployment" in router._extract_main_topic(query)
        
        # Check that pattern tracking works
        assert len(router.query_patterns) > 0