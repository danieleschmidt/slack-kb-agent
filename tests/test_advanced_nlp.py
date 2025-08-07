"""Test advanced NLP query processing capabilities."""

import pytest
from unittest.mock import patch, MagicMock

from slack_kb_agent.advanced_nlp import (
    AdvancedQueryProcessor, 
    EntityExtractor,
    SemanticIntentClassifier,
    QueryExpander,
    QueryIntent,
    QueryComplexity,
    EnhancedQuery
)


class TestEntityExtractor:
    """Test entity extraction functionality."""
    
    def test_extract_entities(self):
        """Test extraction of named entities."""
        extractor = EntityExtractor()
        
        # Test CamelCase detection
        query = "How do I configure AuthService and DatabaseConnection?"
        entities = extractor.extract_entities(query)
        assert "AuthService" in entities
        assert "DatabaseConnection" in entities
        
        # Test snake_case detection
        query = "Where is user_management and api_key_validation?"
        entities = extractor.extract_entities(query)
        assert "user_management" in entities
        assert "api_key_validation" in entities
        
        # Test version numbers
        query = "Upgrade to version 2.1.0 and use v1.5"
        entities = extractor.extract_entities(query)
        assert "2.1.0" in entities
        assert "v1.5" in entities
    
    def test_extract_technical_terms(self):
        """Test technical terminology extraction."""
        extractor = EntityExtractor()
        
        query = "How to configure OAuth authentication and setup database connections?"
        terms = extractor.extract_technical_terms(query)
        
        assert "authentication" in terms
        assert "database" in terms
        assert len(terms) >= 2
    
    def test_assess_urgency(self):
        """Test urgency level assessment."""
        extractor = EntityExtractor()
        
        # Critical urgency
        assert extractor.assess_urgency("System is down and critical!") == 5
        assert extractor.assess_urgency("Emergency: production is broken") == 5
        
        # High urgency
        assert extractor.assess_urgency("Need help ASAP with this issue") == 4
        assert extractor.assess_urgency("Important problem needs solving") == 4
        
        # Medium urgency
        assert extractor.assess_urgency("I'm stuck on this error") == 3
        assert extractor.assess_urgency("Need help with configuration") == 3
        
        # Low urgency
        assert extractor.assess_urgency("Just wondering about deployment") == 2
        assert extractor.assess_urgency("Curious about the API") == 2
        
        # Very low urgency
        assert extractor.assess_urgency("Thanks for the help!") == 1
        assert extractor.assess_urgency("Hello, general question") == 1
    
    def test_requires_code_detection(self):
        """Test detection of queries requiring code examples."""
        extractor = EntityExtractor()
        
        assert extractor._requires_code("Show me code example for authentication")
        assert extractor._requires_code("How to implement this function?")
        assert extractor._requires_code("What's the syntax for this method?")
        assert not extractor._requires_code("What is authentication?")
        assert not extractor._requires_code("Explain the deployment process")
    
    def test_requires_diagram_detection(self):
        """Test detection of queries requiring visual aids."""
        extractor = EntityExtractor()
        
        assert extractor._requires_diagram("Show me the system architecture")
        assert extractor._requires_diagram("How does the authentication flow work?")
        assert extractor._requires_diagram("Explain the relationship between services")
        assert not extractor._requires_diagram("What is the API key?")
        assert not extractor._requires_diagram("How to deploy the application?")


class TestSemanticIntentClassifier:
    """Test semantic intent classification."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.classifier = SemanticIntentClassifier()
    
    @patch('slack_kb_agent.advanced_nlp.TORCH_AVAILABLE', False)
    def test_fallback_classification(self):
        """Test rule-based fallback classification."""
        classifier = SemanticIntentClassifier()
        
        # Definition queries
        intent, confidence = classifier._fallback_classification("What is Docker?")
        assert intent == QueryIntent.DEFINITION
        assert confidence == 0.8
        
        # How-to queries
        intent, confidence = classifier._fallback_classification("How to deploy the app?")
        assert intent == QueryIntent.HOWTO
        assert confidence == 0.8
        
        # Troubleshooting queries
        intent, confidence = classifier._fallback_classification("API is broken")
        assert intent == QueryIntent.TROUBLESHOOTING
        assert confidence == 0.8
        
        # Status queries
        intent, confidence = classifier._fallback_classification("What's the status?")
        assert intent == QueryIntent.STATUS
        assert confidence == 0.7
        
        # Location queries
        intent, confidence = classifier._fallback_classification("Where is the config?")
        assert intent == QueryIntent.LOCATION
        assert confidence == 0.7
        
        # Comparison queries
        intent, confidence = classifier._fallback_classification("Docker vs Kubernetes")
        assert intent == QueryIntent.COMPARISON
        assert confidence == 0.8
    
    def test_intent_templates(self):
        """Test intent template initialization."""
        classifier = SemanticIntentClassifier()
        
        if classifier.intent_templates:
            assert QueryIntent.DEFINITION in classifier.intent_templates
            assert QueryIntent.HOWTO in classifier.intent_templates
            assert QueryIntent.TROUBLESHOOTING in classifier.intent_templates
            
            # Check template content
            definition_templates = classifier.intent_templates[QueryIntent.DEFINITION]
            assert "What is X?" in definition_templates
            assert "Define X" in definition_templates


class TestQueryExpander:
    """Test query expansion functionality."""
    
    def test_expand_query(self):
        """Test query expansion with synonyms."""
        expander = QueryExpander()
        
        # Test API expansion
        expanded = expander.expand_query("api documentation")
        assert "api" in expanded.lower()
        assert any(synonym in expanded.lower() for synonym in ["endpoint", "service", "interface"])
        
        # Test database expansion
        expanded = expander.expand_query("database connection")
        assert "database" in expanded.lower() or "db" in expanded.lower()
        
        # Test error expansion
        expanded = expander.expand_query("fix the error")
        assert "fix" in expanded.lower() or any(synonym in expanded.lower() for synonym in ["resolve", "solve", "repair"])
    
    def test_max_expansions_limit(self):
        """Test expansion limit functionality."""
        expander = QueryExpander()
        
        # With max_expansions=1, should only get 1 synonym per word
        expanded = expander.expand_query("api error", max_expansions=1)
        
        # Count expansions (this is approximate due to regex format)
        expansion_count = expanded.count("|")
        assert expansion_count <= 2  # One per expanded word


class TestAdvancedQueryProcessor:
    """Test the main advanced query processor."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.processor = AdvancedQueryProcessor()
    
    def test_simple_query_complexity(self):
        """Test simple query complexity assessment."""
        enhanced = self.processor.process_query("Hi there")
        assert enhanced.complexity == QueryComplexity.SIMPLE
        
        enhanced = self.processor.process_query("What?")
        assert enhanced.complexity == QueryComplexity.SIMPLE
    
    def test_complex_query_complexity(self):
        """Test complex query complexity assessment."""
        query = "How to design a scalable enterprise architecture with multiple microservices and different deployment strategies?"
        enhanced = self.processor.process_query(query)
        assert enhanced.complexity == QueryComplexity.COMPLEX
        
        query = "Best practices for integrating multiple authentication systems in production"
        enhanced = self.processor.process_query(query)
        assert enhanced.complexity == QueryComplexity.COMPLEX
    
    def test_moderate_query_complexity(self):
        """Test moderate query complexity assessment."""
        query = "How to configure OAuth with two factor authentication?"
        enhanced = self.processor.process_query(query)
        assert enhanced.complexity == QueryComplexity.MODERATE
    
    def test_key_concepts_extraction(self):
        """Test key concept extraction."""
        query = "How to configure DatabaseConnection and AuthService for production deployment?"
        enhanced = self.processor.process_query(query)
        
        # Should extract entities and important terms
        assert len(enhanced.key_concepts) > 0
        assert any("DatabaseConnection" in concept or "AuthService" in concept for concept in enhanced.key_concepts)
    
    def test_query_expansion(self):
        """Test that queries are expanded properly."""
        query = "API error"
        enhanced = self.processor.process_query(query)
        
        # Expanded query should be different from original
        assert enhanced.expanded_query != enhanced.original_query
        assert enhanced.original_query == query
    
    def test_intent_classification_integration(self):
        """Test intent classification integration."""
        # Definition query
        enhanced = self.processor.process_query("What is Docker?")
        assert enhanced.intent == QueryIntent.DEFINITION
        
        # How-to query
        enhanced = self.processor.process_query("How to deploy applications?")
        assert enhanced.intent == QueryIntent.HOWTO
        
        # Troubleshooting query
        enhanced = self.processor.process_query("Service is not working")
        assert enhanced.intent == QueryIntent.TROUBLESHOOTING
    
    def test_context_extraction(self):
        """Test context information extraction."""
        query = "Critical: AuthService authentication is broken in production!"
        enhanced = self.processor.process_query(query)
        
        # Should detect high urgency
        assert enhanced.context.urgency_level >= 4
        
        # Should extract technical terms
        assert "authentication" in enhanced.context.technical_terms
        
        # Should extract entities
        assert any("AuthService" in entity for entity in enhanced.context.entities)
    
    def test_follow_up_suggestions(self):
        """Test follow-up question suggestions."""
        # Definition query
        enhanced = self.processor.process_query("What is Docker?")
        suggestions = self.processor.suggest_follow_up_questions(enhanced)
        
        assert len(suggestions) <= 3
        assert any("implement" in suggestion.lower() for suggestion in suggestions)
        
        # How-to query
        enhanced = self.processor.process_query("How to deploy?")
        suggestions = self.processor.suggest_follow_up_questions(enhanced)
        
        assert len(suggestions) <= 3
        assert any("issue" in suggestion.lower() or "alternative" in suggestion.lower() for suggestion in suggestions)
    
    def test_enhanced_query_structure(self):
        """Test EnhancedQuery structure completeness."""
        query = "How to fix authentication errors in production?"
        enhanced = self.processor.process_query(query)
        
        # Check all required fields are present
        assert enhanced.original_query == query
        assert enhanced.intent is not None
        assert enhanced.complexity is not None
        assert enhanced.context is not None
        assert 0.0 <= enhanced.confidence <= 1.0
        assert enhanced.expanded_query is not None
        assert isinstance(enhanced.key_concepts, list)
    
    def test_user_context_integration(self):
        """Test user context parameter handling."""
        user_context = {"user_id": "test_user", "previous_queries": ["What is Docker?"]}
        
        # Should not crash with user context
        enhanced = self.processor.process_query("How to deploy Docker?", user_context)
        assert enhanced is not None
        assert enhanced.original_query == "How to deploy Docker?"


class TestIntegrationWithEnhancedQueryProcessor:
    """Test integration with existing EnhancedQueryProcessor."""
    
    def test_advanced_intent_mapping(self):
        """Test advanced intent mapping to legacy intents."""
        from slack_kb_agent.query_processor import EnhancedQueryProcessor
        from slack_kb_agent.knowledge_base import KnowledgeBase
        
        kb = KnowledgeBase()
        processor = EnhancedQueryProcessor(kb)
        
        # Test mapping function
        from slack_kb_agent.advanced_nlp import QueryIntent as AdvancedQueryIntent
        from slack_kb_agent.query_processor import QueryIntent
        
        assert processor._map_advanced_intent(AdvancedQueryIntent.DEFINITION) == QueryIntent.DEFINITION
        assert processor._map_advanced_intent(AdvancedQueryIntent.TROUBLESHOOTING) == QueryIntent.TROUBLESHOOTING
        assert processor._map_advanced_intent(AdvancedQueryIntent.HOWTO) == QueryIntent.QUESTION
        assert processor._map_advanced_intent(AdvancedQueryIntent.CONVERSATIONAL) == QueryIntent.CONVERSATIONAL


if __name__ == "__main__":
    pytest.main([__file__])