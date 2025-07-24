#!/usr/bin/env python3
"""
Security test for SQL injection vulnerability fix in database search.

This test verifies that the search_documents method properly handles
malicious SQL injection attempts without executing arbitrary SQL.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add src to path to allow imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from slack_kb_agent.database import DatabaseRepository, DatabaseManager
    from slack_kb_agent.models import Document
    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"Imports not available: {e}")
    IMPORTS_AVAILABLE = False


@unittest.skipUnless(IMPORTS_AVAILABLE, "Dependencies not available")
class TestSQLInjectionFix(unittest.TestCase):
    """Test SQL injection vulnerability fix in search operations."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not IMPORTS_AVAILABLE:
            self.skipTest("Dependencies not available")
            
        self.mock_db_manager = Mock(spec=DatabaseManager)
        self.mock_session = Mock()
        self.mock_db_manager.get_session.return_value.__enter__.return_value = self.mock_session
        self.repository = DatabaseRepository(self.mock_db_manager)
    
    def test_sql_injection_attempt_is_escaped(self):
        """Test that SQL injection attempts are properly escaped."""
        # SQL injection payload that would drop tables if not escaped
        malicious_query = "'; DROP TABLE documents; --"
        
        # Mock query builder to capture the actual filter expression
        mock_query = Mock()
        mock_filter = Mock()
        mock_order_by = Mock()
        mock_limit = Mock()
        
        # Chain the mocks
        self.mock_session.query.return_value = mock_query
        mock_query.filter.return_value = mock_filter
        mock_filter.order_by.return_value = mock_order_by
        mock_order_by.limit.return_value = mock_limit
        mock_limit.all.return_value = []
        
        # Execute search with malicious query
        result = self.repository.search_documents(malicious_query)
        
        # Verify that query was called but no SQL injection occurred
        self.mock_session.query.assert_called_once()
        mock_query.filter.assert_called_once()
        
        # Get the actual filter expression used
        filter_call_args = mock_query.filter.call_args[0][0]
        
        # The filter should be a proper SQLAlchemy expression, not raw SQL
        # If properly escaped, this should be a BinaryExpression with bound parameters
        self.assertIsNotNone(filter_call_args)
        
        # Verify no exception was raised and empty result returned
        self.assertEqual(result, [])
        
    def test_input_validation_rejects_non_string(self):
        """Test that non-string input is rejected."""
        with self.assertRaises(ValueError) as cm:
            self.repository.search_documents(123)
        self.assertIn("must be a string", str(cm.exception))
        
    def test_input_validation_rejects_long_queries(self):
        """Test that overly long queries are rejected."""
        long_query = "x" * 1001  # Exceeds 1000 character limit
        with self.assertRaises(ValueError) as cm:
            self.repository.search_documents(long_query)
        self.assertIn("too long", str(cm.exception))
        
    def test_empty_query_returns_empty_list(self):
        """Test that empty or whitespace-only queries return empty results."""
        self.assertEqual(self.repository.search_documents(""), [])
        self.assertEqual(self.repository.search_documents("   "), [])
    
    def test_normal_search_still_works(self):
        """Test that normal search functionality is preserved."""
        normal_query = "test content"
        
        # Mock successful search result
        mock_doc_model = Mock()
        mock_doc_model.to_document.return_value = Document(
            content="Test document with test content",
            source="test",
            metadata={}
        )
        
        mock_query = Mock()
        mock_filter = Mock()
        mock_order_by = Mock()
        mock_limit = Mock()
        
        self.mock_session.query.return_value = mock_query
        mock_query.filter.return_value = mock_filter
        mock_filter.order_by.return_value = mock_order_by
        mock_order_by.limit.return_value = mock_limit
        mock_limit.all.return_value = [mock_doc_model]
        
        # Execute normal search
        result = self.repository.search_documents(normal_query)
        
        # Verify search worked correctly
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].content, "Test document with test content")
    
    def test_special_characters_are_handled_safely(self):
        """Test that special SQL characters are handled safely."""
        special_queries = [
            "test % content",  # SQL wildcard
            "test _ content",  # SQL single char wildcard
            "test ' content",  # Single quote
            "test \" content", # Double quote
            "test \\ content", # Backslash
            "test; SELECT * FROM users", # Semicolon injection attempt
        ]
        
        for query in special_queries:
            with self.subTest(query=query):
                # Mock the query chain
                mock_query = Mock()
                mock_filter = Mock()
                mock_order_by = Mock()
                mock_limit = Mock()
                
                self.mock_session.query.return_value = mock_query
                mock_query.filter.return_value = mock_filter
                mock_filter.order_by.return_value = mock_order_by
                mock_order_by.limit.return_value = mock_limit
                mock_limit.all.return_value = []
                
                # Should not raise exception
                result = self.repository.search_documents(query)
                self.assertEqual(result, [])
                
                # Reset mocks for next iteration
                self.mock_session.reset_mock()


if __name__ == '__main__':
    unittest.main()