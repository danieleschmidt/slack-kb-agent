"""Test feedback learning system functionality."""

import pytest
import tempfile
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

from slack_kb_agent.feedback_learning import (
    FeedbackLearningSystem,
    FeedbackCollector,
    PatternLearner,
    AdaptiveResponseRanker,
    FeedbackEvent,
    FeedbackType,
    LearningDomain,
    LearningMetrics
)


class TestFeedbackEvent:
    """Test feedback event data structure."""
    
    def test_feedback_event_creation(self):
        """Test creating feedback events."""
        event = FeedbackEvent(
            event_id="test123",
            query="How to deploy?",
            response="Use docker deploy",
            feedback_type=FeedbackType.POSITIVE,
            user_id="user123",
            timestamp=time.time(),
            context={"channel": "dev-team"}
        )
        
        assert event.event_id == "test123"
        assert event.feedback_type == FeedbackType.POSITIVE
        assert event.user_id == "user123"
        assert event.documents_used == []  # Default empty list
        
    def test_feedback_event_with_rating(self):
        """Test feedback event with rating."""
        event = FeedbackEvent(
            event_id="test123",
            query="How to deploy?",
            response="Use docker deploy",
            feedback_type=FeedbackType.RATING,
            user_id="user123",
            timestamp=time.time(),
            context={},
            rating=5,
            documents_used=["doc1", "doc2"]
        )
        
        assert event.rating == 5
        assert len(event.documents_used) == 2


class TestPatternLearner:
    """Test pattern learning functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.learner = PatternLearner(max_patterns=10)
    
    def test_tokenize_query(self):
        """Test query tokenization."""
        tokens = self.learner._tokenize_query("How to deploy the authentication service?")
        
        # Should extract meaningful words, filter stop words
        assert "deploy" in tokens
        assert "authentication" in tokens
        assert "service" in tokens
        assert "how" not in tokens  # Stop word
        assert "the" not in tokens  # Stop word
        
    def test_generate_signature(self):
        """Test signature generation."""
        tokens1 = ["deploy", "auth", "service"]
        tokens2 = ["service", "deploy", "auth"]  # Same tokens, different order
        
        sig1 = self.learner._generate_signature(tokens1)
        sig2 = self.learner._generate_signature(tokens2)
        
        # Should generate same signature for same tokens regardless of order
        assert sig1 == sig2
        assert len(sig1) == 8  # MD5 hash truncated to 8 chars
    
    def test_learn_from_positive_feedback(self):
        """Test learning from positive feedback."""
        event = FeedbackEvent(
            event_id="test1",
            query="How to deploy applications?",
            response="Use docker",
            feedback_type=FeedbackType.POSITIVE,
            user_id="user123",
            timestamp=time.time(),
            context={}
        )
        
        self.learner.learn_from_feedback(event)
        
        # Should have learned pattern
        assert len(self.learner.query_patterns) == 1
        
        # Pattern should have positive weight
        pattern = list(self.learner.query_patterns.values())[0]
        assert pattern['positive_weight'] == 1.0
        assert pattern['total_interactions'] == 1.0
    
    def test_learn_from_negative_feedback(self):
        """Test learning from negative feedback."""
        event = FeedbackEvent(
            event_id="test1",
            query="How to deploy applications?",
            response="Use docker",
            feedback_type=FeedbackType.NEGATIVE,
            user_id="user123",
            timestamp=time.time(),
            context={}
        )
        
        self.learner.learn_from_feedback(event)
        
        # Pattern should have negative weight
        pattern = list(self.learner.query_patterns.values())[0]
        assert pattern['negative_weight'] == 1.0
        assert pattern['total_interactions'] == 1.0
    
    def test_learn_from_escalation_feedback(self):
        """Test learning from escalation feedback."""
        event = FeedbackEvent(
            event_id="test1",
            query="Complex enterprise integration question?",
            response="I don't know",
            feedback_type=FeedbackType.ESCALATION,
            user_id="user123",
            timestamp=time.time(),
            context={}
        )
        
        self.learner.learn_from_feedback(event)
        
        # Should add to escalation patterns
        assert len(self.learner.escalation_patterns) == 1
        
        # Pattern should have escalation weight
        pattern = list(self.learner.query_patterns.values())[0]
        assert pattern['escalation_weight'] == 1.0
    
    def test_learn_from_rating_feedback(self):
        """Test learning from rating feedback."""
        event = FeedbackEvent(
            event_id="test1",
            query="How to configure OAuth?",
            response="Check the docs",
            feedback_type=FeedbackType.RATING,
            user_id="user123",
            timestamp=time.time(),
            context={},
            rating=5
        )
        
        self.learner.learn_from_feedback(event)
        
        # Pattern should have rating data
        pattern = list(self.learner.query_patterns.values())[0]
        assert pattern['rating_sum'] == 5.0
        assert pattern['rating_count'] == 1.0
        
        # High rating should add to high-value patterns
        assert len(self.learner.high_value_patterns) == 1
    
    def test_predict_query_success(self):
        """Test query success prediction."""
        # Add positive feedback for a query
        positive_event = FeedbackEvent(
            event_id="test1",
            query="How to deploy with Docker?",
            response="Use docker build",
            feedback_type=FeedbackType.POSITIVE,
            user_id="user123",
            timestamp=time.time(),
            context={}
        )
        self.learner.learn_from_feedback(positive_event)
        
        # Test prediction for similar query
        success_rate = self.learner.predict_query_success("How to deploy with Docker?")
        assert success_rate > 0.5  # Should predict high success
        
        # Test prediction for unknown query
        unknown_success = self.learner.predict_query_success("Completely different query")
        assert unknown_success == 0.5  # Should return neutral
    
    def test_escalation_prediction(self):
        """Test escalation decision prediction."""
        # Add escalation feedback
        escalation_event = FeedbackEvent(
            event_id="test1",
            query="Complex enterprise architecture question",
            response="I can't help",
            feedback_type=FeedbackType.ESCALATION,
            user_id="user123",
            timestamp=time.time(),
            context={}
        )
        self.learner.learn_from_feedback(escalation_event)
        
        # Should recommend escalation for similar query
        should_escalate = self.learner.should_escalate("Complex enterprise architecture question")
        assert should_escalate
        
        # Should not recommend escalation for different query
        should_not_escalate = self.learner.should_escalate("Simple deployment question")
        assert not should_not_escalate
    
    def test_pattern_pruning(self):
        """Test pattern limit enforcement."""
        learner = PatternLearner(max_patterns=3)
        
        # Add more patterns than limit
        for i in range(5):
            event = FeedbackEvent(
                event_id=f"test{i}",
                query=f"Query number {i}",
                response="Response",
                feedback_type=FeedbackType.POSITIVE,
                user_id="user123",
                timestamp=time.time(),
                context={}
            )
            learner.learn_from_feedback(event)
        
        # Should not exceed max patterns
        assert len(learner.query_patterns) <= 3


class TestAdaptiveResponseRanker:
    """Test adaptive response ranking."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.ranker = AdaptiveResponseRanker()
    
    def test_learn_from_positive_feedback(self):
        """Test learning from positive document feedback."""
        event = FeedbackEvent(
            event_id="test1",
            query="How to deploy?",
            response="Use docker",
            feedback_type=FeedbackType.POSITIVE,
            user_id="user123",
            timestamp=time.time(),
            context={},
            documents_used=["doc1", "doc2"]
        )
        
        self.ranker.learn_from_feedback(event)
        
        # Documents should have positive scores
        assert self.ranker.document_scores["doc1"] > 0
        assert self.ranker.document_scores["doc2"] > 0
        
    def test_learn_from_negative_feedback(self):
        """Test learning from negative document feedback."""
        event = FeedbackEvent(
            event_id="test1",
            query="How to deploy?",
            response="Use docker",
            feedback_type=FeedbackType.NEGATIVE,
            user_id="user123",
            timestamp=time.time(),
            context={},
            documents_used=["doc1"]
        )
        
        self.ranker.learn_from_feedback(event)
        
        # Document should have negative score
        assert self.ranker.document_scores["doc1"] < 0
    
    def test_document_ranking(self):
        """Test document ranking based on learned feedback."""
        # Add positive feedback for doc1
        positive_event = FeedbackEvent(
            event_id="test1",
            query="deployment process",
            response="Great response",
            feedback_type=FeedbackType.POSITIVE,
            user_id="user123",
            timestamp=time.time(),
            context={},
            documents_used=["doc1"]
        )
        self.ranker.learn_from_feedback(positive_event)
        
        # Add negative feedback for doc2
        negative_event = FeedbackEvent(
            event_id="test2",
            query="deployment process",
            response="Bad response",
            feedback_type=FeedbackType.NEGATIVE,
            user_id="user123",
            timestamp=time.time(),
            context={},
            documents_used=["doc2"]
        )
        self.ranker.learn_from_feedback(negative_event)
        
        # Rank documents for similar query
        ranked = self.ranker.rank_documents(["doc1", "doc2", "doc3"], "deployment process")
        
        # doc1 should rank higher than doc2
        doc_scores = {doc_id: score for doc_id, score in ranked}
        assert doc_scores["doc1"] > doc_scores["doc2"]
    
    def test_query_specific_associations(self):
        """Test query-specific document associations."""
        # Learn association between query and document
        event = FeedbackEvent(
            event_id="test1",
            query="database connection",
            response="Good response",
            feedback_type=FeedbackType.RATING,
            user_id="user123",
            timestamp=time.time(),
            context={},
            rating=5,
            documents_used=["database_doc"]
        )
        
        self.ranker.learn_from_feedback(event)
        
        # Query signature should have association
        query_sig = self.ranker._get_query_signature("database connection")
        assert query_sig in self.ranker.query_document_associations
        assert "database_doc" in self.ranker.query_document_associations[query_sig]


class TestFeedbackCollector:
    """Test feedback collection and storage."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.json')
        self.temp_file.close()
        self.collector = FeedbackCollector(self.temp_file.name)
    
    def teardown_method(self):
        """Cleanup test fixtures."""
        Path(self.temp_file.name).unlink(missing_ok=True)
    
    def test_collect_feedback(self):
        """Test feedback collection."""
        event = FeedbackEvent(
            event_id="test1",
            query="Test query",
            response="Test response",
            feedback_type=FeedbackType.POSITIVE,
            user_id="user123",
            timestamp=time.time(),
            context={}
        )
        
        self.collector.collect_feedback(event)
        
        # Should be in buffer
        assert len(self.collector.feedback_buffer) == 1
        assert self.collector.feedback_buffer[0].event_id == "test1"
        
        # Should be in user history
        assert len(self.collector.user_feedback_history["user123"]) == 1
    
    def test_user_satisfaction_calculation(self):
        """Test user satisfaction calculation."""
        # Add positive feedback
        positive_event = FeedbackEvent(
            event_id="test1",
            query="Test query",
            response="Great response",
            feedback_type=FeedbackType.POSITIVE,
            user_id="user123",
            timestamp=time.time(),
            context={}
        )
        self.collector.collect_feedback(positive_event)
        
        # Add negative feedback
        negative_event = FeedbackEvent(
            event_id="test2",
            query="Test query 2",
            response="Bad response",
            feedback_type=FeedbackType.NEGATIVE,
            user_id="user123",
            timestamp=time.time(),
            context={}
        )
        self.collector.collect_feedback(negative_event)
        
        # Satisfaction should be 50% (1 positive, 1 negative)
        satisfaction = self.collector.calculate_user_satisfaction("user123")
        assert satisfaction == 0.5
    
    def test_rating_based_satisfaction(self):
        """Test satisfaction calculation with ratings."""
        # Add rating feedback
        rating_event = FeedbackEvent(
            event_id="test1",
            query="Test query",
            response="Good response",
            feedback_type=FeedbackType.RATING,
            user_id="user123",
            timestamp=time.time(),
            context={},
            rating=4
        )
        self.collector.collect_feedback(rating_event)
        
        # Satisfaction should be 4/5 = 0.8
        satisfaction = self.collector.calculate_user_satisfaction("user123")
        assert satisfaction == 0.8
    
    def test_feedback_persistence(self):
        """Test feedback data persistence."""
        event = FeedbackEvent(
            event_id="test1",
            query="Test query",
            response="Test response",
            feedback_type=FeedbackType.POSITIVE,
            user_id="user123",
            timestamp=time.time(),
            context={}
        )
        
        self.collector.collect_feedback(event)
        self.collector._persist_feedback_data()
        
        # Create new collector with same file
        new_collector = FeedbackCollector(self.temp_file.name)
        
        # Should load existing data
        assert len(new_collector.feedback_buffer) == 1
        assert new_collector.feedback_buffer[0].event_id == "test1"


class TestFeedbackLearningSystem:
    """Test the integrated feedback learning system."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.json')
        self.temp_file.close()
        self.system = FeedbackLearningSystem(self.temp_file.name)
    
    def teardown_method(self):
        """Cleanup test fixtures."""
        Path(self.temp_file.name).unlink(missing_ok=True)
    
    def test_record_feedback(self):
        """Test recording feedback through the main system."""
        event_id = self.system.record_feedback(
            query="How to deploy?",
            response="Use docker deploy",
            feedback_type=FeedbackType.POSITIVE,
            user_id="user123",
            documents_used=["doc1", "doc2"]
        )
        
        assert event_id is not None
        assert len(event_id) > 0
        
        # Should update all learning components
        assert len(self.system.feedback_collector.feedback_buffer) == 1
        assert len(self.system.pattern_learner.query_patterns) == 1
        assert self.system.response_ranker.document_scores["doc1"] > 0
    
    def test_escalation_decision(self):
        """Test escalation decision based on learning."""
        # Record escalation feedback
        self.system.record_feedback(
            query="Complex enterprise architecture",
            response="Cannot help",
            feedback_type=FeedbackType.ESCALATION,
            user_id="user123"
        )
        
        # Should recommend escalation for similar queries
        should_escalate = self.system.should_escalate_query("Complex enterprise architecture")
        assert should_escalate
    
    def test_success_prediction(self):
        """Test query success prediction."""
        # Record positive feedback
        self.system.record_feedback(
            query="Docker deployment guide",
            response="Here's how to deploy",
            feedback_type=FeedbackType.POSITIVE,
            user_id="user123"
        )
        
        # Should predict high success for similar query
        success_rate = self.system.predict_query_success_rate("Docker deployment guide")
        assert success_rate > 0.5
    
    def test_document_ranking(self):
        """Test document ranking integration."""
        # Record positive feedback for specific documents
        self.system.record_feedback(
            query="deployment process",
            response="Helpful response",
            feedback_type=FeedbackType.POSITIVE,
            user_id="user123",
            documents_used=["good_doc", "ok_doc"]
        )
        
        # Record negative feedback for another document
        self.system.record_feedback(
            query="deployment process",
            response="Unhelpful response",
            feedback_type=FeedbackType.NEGATIVE,
            user_id="user123",
            documents_used=["bad_doc"]
        )
        
        # Rank documents for similar query
        ranked = self.system.rank_response_documents(
            ["good_doc", "bad_doc", "ok_doc"], 
            "deployment process"
        )
        
        # Good doc should rank higher than bad doc
        doc_scores = {doc_id: score for doc_id, score in ranked}
        assert doc_scores["good_doc"] > doc_scores["bad_doc"]
    
    def test_user_satisfaction_tracking(self):
        """Test user satisfaction tracking."""
        # Record mixed feedback for user
        self.system.record_feedback(
            query="Query 1",
            response="Good response",
            feedback_type=FeedbackType.POSITIVE,
            user_id="user123"
        )
        
        self.system.record_feedback(
            query="Query 2",
            response="Bad response",
            feedback_type=FeedbackType.NEGATIVE,
            user_id="user123"
        )
        
        # Satisfaction should be 50%
        satisfaction = self.system.get_user_satisfaction_score("user123")
        assert satisfaction == 0.5
    
    def test_learning_metrics(self):
        """Test learning metrics calculation."""
        # Record various feedback types
        self.system.record_feedback(
            query="Query 1",
            response="Response 1",
            feedback_type=FeedbackType.POSITIVE,
            user_id="user123"
        )
        
        self.system.record_feedback(
            query="Query 2",
            response="Response 2",
            feedback_type=FeedbackType.NEGATIVE,
            user_id="user123"
        )
        
        self.system.record_feedback(
            query="Query 3",
            response="Response 3",
            feedback_type=FeedbackType.RATING,
            user_id="user123",
            rating=4
        )
        
        metrics = self.system.get_learning_metrics()
        
        assert metrics.total_feedback_events == 3
        assert 0 <= metrics.positive_feedback_rate <= 1
        assert metrics.average_rating > 0
        assert metrics.learned_patterns > 0


if __name__ == "__main__":
    pytest.main([__file__])