#!/usr/bin/env python3
"""
Test suite for enhanced query processing with LLM integration.

Tests intelligent query understanding, intent classification, and query expansion.
"""

import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from slack_kb_agent.query_processor import (
    QueryProcessor,
    EnhancedQueryProcessor,
    QueryIntent,
    QueryExpansion,
    QueryContext,
    QueryResult
)
from slack_kb_agent.models import Document
from slack_kb_agent.llm import LLMResponse


class TestQueryIntent(unittest.TestCase):
    """Test query intent classification."""
    
    def test_intent_classification(self):
        """Test basic intent classification."""
        intents = [
            ("How do I deploy the app?", QueryIntent.QUESTION),
            ("Deploy the application now", QueryIntent.COMMAND),
            ("The app is not starting", QueryIntent.TROUBLESHOOTING),
            ("What is OAuth?", QueryIntent.DEFINITION),
            ("Show me the API docs", QueryIntent.SEARCH),
            ("Thanks for helping", QueryIntent.CONVERSATIONAL)
        ]
        
        for query, expected_intent in intents:
            with self.subTest(query=query):
                intent = QueryIntent.classify(query)
                self.assertEqual(intent, expected_intent)
    
    def test_intent_confidence(self):
        """Test intent classification with confidence scores."""
        result = QueryIntent.classify_with_confidence("How do I deploy?")
        
        self.assertIsInstance(result, dict)
        self.assertIn("intent", result)
        self.assertIn("confidence", result)
        self.assertIsInstance(result["confidence"], float)
        self.assertGreaterEqual(result["confidence"], 0.0)
        self.assertLessEqual(result["confidence"], 1.0)


class TestQueryExpansion(unittest.TestCase):
    """Test query expansion functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.expander = QueryExpansion()
    
    def test_synonym_expansion(self):
        """Test expansion with synonyms."""
        query = "deploy app"
        expanded = self.expander.expand_synonyms(query)
        
        # Should include deployment-related terms
        self.assertIn("deploy", expanded)
        expected_synonyms = ["deployment", "release", "publish"]
        self.assertTrue(any(syn in expanded for syn in expected_synonyms))
    
    def test_technical_term_expansion(self):
        """Test expansion of technical terms."""
        expansions = [
            ("CI/CD", ["continuous integration", "continuous deployment", "pipeline"]),
            ("API", ["application programming interface", "endpoint", "service"]),
            ("OAuth", ["authentication", "authorization", "login"]),
            ("Docker", ["container", "containerization", "deployment"])
        ]
        
        for term, expected_terms in expansions:
            with self.subTest(term=term):
                expanded = self.expander.expand_technical_terms(term)
                self.assertTrue(any(exp in expanded for exp in expected_terms))
    
    @patch('slack_kb_agent.query_processor.get_response_generator')
    def test_llm_query_expansion(self, mock_get_generator):
        """Test LLM-powered query expansion."""
        # Mock LLM response
        mock_generator = MagicMock()
        mock_generator.is_available.return_value = True
        mock_generator.generate_response.return_value = LLMResponse(
            content="Related terms: deployment, release, continuous integration, production",
            success=True
        )
        mock_get_generator.return_value = mock_generator
        
        expanded = self.expander.expand_with_llm("deploy app")
        
        self.assertIn("deployment", expanded)
        self.assertIn("release", expanded)
        self.assertIn("continuous integration", expanded)
    
    def test_fallback_expansion(self):
        """Test fallback when LLM unavailable."""
        expanded = self.expander.expand_with_llm("deploy app", fallback_to_synonyms=True)
        
        # Should still return useful expansions
        self.assertIsInstance(expanded, list)
        self.assertGreater(len(expanded), 0)


class TestQueryContext(unittest.TestCase):
    """Test query context management."""
    
    def setUp(self):
        """Set up test environment."""
        self.context = QueryContext(user_id="U123456")
    
    def test_context_tracking(self):
        """Test conversation context tracking."""
        # First query
        self.context.add_query("How do I deploy?", ["doc1", "doc2"])
        
        # Follow-up query
        result = self.context.get_context_for_query("What about staging?")
        
        self.assertEqual(len(self.context.history), 1)
        self.assertIn("deploy", result["previous_topics"])
    
    def test_context_relevance(self):
        """Test context relevance scoring."""
        self.context.add_query("How do I deploy the application?", ["deploy.md"])
        
        # Related follow-up
        relevance = self.context.calculate_relevance("What about staging deployment?")
        self.assertGreater(relevance, 0.2)
        
        # Unrelated follow-up  
        relevance = self.context.calculate_relevance("What is the weather?")
        self.assertLess(relevance, 0.3)
    
    def test_context_expiry(self):
        """Test context expiry and cleanup."""
        import time
        
        # Add old context
        old_time = time.time() - 3700  # Over an hour ago
        self.context.add_query("Old query", ["doc1"], timestamp=old_time)
        
        # Add recent context
        self.context.add_query("Recent query", ["doc2"])
        
        # Cleanup should remove old context
        self.context.cleanup_expired()
        
        self.assertEqual(len(self.context.history), 1)
        self.assertEqual(self.context.history[0]["query"], "Recent query")


class TestEnhancedQueryProcessor(unittest.TestCase):
    """Test enhanced query processor integration."""
    
    def setUp(self):
        """Set up test environment."""
        self.knowledge_base = MagicMock()
        self.knowledge_base.search.return_value = [
            Document(content="Deploy with npm run deploy", source="docs"),
            Document(content="Use Docker for containerization", source="deployment.md")
        ]
        
        self.processor = EnhancedQueryProcessor(self.knowledge_base)
    
    def test_enhanced_processing_pipeline(self):
        """Test the complete enhanced processing pipeline."""
        query = "how to deploy"
        user_id = "U123456"
        
        result = self.processor.process_query(query, user_id=user_id)
        
        self.assertIsInstance(result, QueryResult)
        self.assertEqual(result.original_query, query)
        self.assertIsNotNone(result.intent)
        self.assertIsInstance(result.expanded_terms, list)
        self.assertIsInstance(result.documents, list)
    
    def test_intent_based_processing(self):
        """Test processing behavior based on intent."""
        # Question intent
        result = self.processor.process_query("How do I deploy?", user_id="U123456")
        self.assertEqual(result.intent, QueryIntent.QUESTION)
        
        # Command intent
        result = self.processor.process_query("Deploy the app", user_id="U123456")
        self.assertEqual(result.intent, QueryIntent.COMMAND)
    
    @patch('slack_kb_agent.query_processor.get_response_generator')
    def test_query_refinement_suggestions(self, mock_get_generator):
        """Test query refinement when no results found."""
        # Mock empty search results
        self.knowledge_base.search.return_value = []
        
        # Mock LLM response with suggestions
        mock_generator = MagicMock()
        mock_generator.is_available.return_value = True
        mock_generator.generate_response.return_value = LLMResponse(
            content="Try searching for: deployment guide, release process, CI/CD pipeline",
            success=True
        )
        mock_get_generator.return_value = mock_generator
        
        result = self.processor.process_query("xyz deployment", user_id="U123456")
        
        self.assertIsNotNone(result.suggestions)
        self.assertIn("deployment guide", result.suggestions)
    
    def test_follow_up_context(self):
        """Test follow-up query processing with context."""
        user_id = "U123456"
        
        # First query
        result1 = self.processor.process_query("How do I deploy?", user_id=user_id)
        
        # Follow-up query
        result2 = self.processor.process_query("What about staging?", user_id=user_id)
        
        # Should have context from previous query
        self.assertIn("deployment", " ".join(result2.expanded_terms))
    
    def test_semantic_similarity_boost(self):
        """Test semantic similarity boosting in search."""
        # Mock vector search with similarity scores
        self.knowledge_base.search_semantic.return_value = [
            (Document(content="Deploy to production", source="prod.md"), 0.9),
            (Document(content="Deploy to staging", source="staging.md"), 0.8)
        ]
        
        result = self.processor.process_query("production deployment", user_id="U123456")
        
        # Should prefer higher similarity documents
        self.assertGreater(len(result.documents), 0)
        # First document should be the one with higher similarity
        self.assertIn("production", result.documents[0].content)
    
    def test_error_handling(self):
        """Test error handling in enhanced processing."""
        # Mock intent classification error by causing exception in processing
        processor_with_error = EnhancedQueryProcessor(self.knowledge_base)
        
        # Cause an error in the processing pipeline
        original_classify = processor_with_error.query_expansion.expand_synonyms
        def error_classify(*args, **kwargs):
            raise Exception("Processing failed")
        processor_with_error.query_expansion.expand_synonyms = error_classify
        
        result = processor_with_error.process_query("test query", user_id="U123456")
        
        # Should handle errors gracefully with fallback
        self.assertIsInstance(result, QueryResult)
        self.assertIsNotNone(result.error_message)
    
    def test_performance_metrics(self):
        """Test performance tracking in enhanced processing."""
        result = self.processor.process_query("deploy app", user_id="U123456")
        
        self.assertIsNotNone(result.processing_time)
        self.assertGreater(result.processing_time, 0)
        self.assertIsInstance(result.metrics, dict)


class TestIntegration(unittest.TestCase):
    """Integration tests for enhanced query processing."""
    
    def test_backward_compatibility(self):
        """Test that enhanced processor maintains backward compatibility."""
        from slack_kb_agent.query_processor import QueryProcessor
        
        # Original processor should still work
        kb = MagicMock()
        kb.search.return_value = [Document(content="test", source="test")]
        
        processor = QueryProcessor(kb)
        result = processor.process_query("test query")
        
        # Should return simple document list (backward compatible)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
    
    def test_enhanced_vs_basic(self):
        """Test differences between enhanced and basic processing."""
        kb = MagicMock()
        kb.search.return_value = [Document(content="test", source="test")]
        
        basic_processor = QueryProcessor(kb)
        enhanced_processor = EnhancedQueryProcessor(kb)
        
        basic_result = basic_processor.process_query("test query")
        enhanced_result = enhanced_processor.process_query("test query", user_id="U123456")
        
        # Basic returns list, enhanced returns QueryResult
        self.assertIsInstance(basic_result, list)
        self.assertIsInstance(enhanced_result, QueryResult)
        
        # Enhanced has additional metadata
        self.assertIsNotNone(enhanced_result.intent)
        self.assertIsNotNone(enhanced_result.expanded_terms)


if __name__ == "__main__":
    unittest.main()