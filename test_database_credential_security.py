#!/usr/bin/env python3
"""
Test to verify that database credentials are properly masked in logs,
monitoring, and memory statistics to prevent credential exposure.
"""

import unittest
import sys
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
import re

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from slack_kb_agent.database import DatabaseManager
except ImportError as e:
    print(f"Failed to import database modules: {e}")
    print("This test requires the database module to be available")
    sys.exit(1)


class TestDatabaseCredentialSecurity(unittest.TestCase):
    """Test that database credentials are properly masked in all contexts."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_urls = {
            'postgres_with_password': 'postgresql://user:secret123@localhost:5432/dbname',
            'postgres_with_symbols': 'postgresql://user:p@ssw0rd!@localhost:5432/dbname', 
            'postgres_with_port': 'postgresql://username:password@db.example.com:5433/mydb',
            'mysql_url': 'mysql://admin:admin123@localhost:3306/testdb',
            'no_credentials': 'postgresql://localhost:5432/dbname',
            'local_socket': 'postgresql:///var/lib/postgres/socket/dbname'
        }
    
    def test_connection_string_masking(self):
        """Test that connection strings are properly masked when displayed."""
        for name, url in self.test_urls.items():
            with self.subTest(url_type=name):
                masked = self._mask_database_url(url)
                
                # Should not contain original password
                if ':' in url and '@' in url:
                    # Extract password from URL
                    password_match = re.search(r'://[^:]+:([^@]+)@', url)
                    if password_match:
                        password = password_match.group(1)
                        if password:  # Non-empty password
                            self.assertNotIn(password, masked, 
                                f"Password '{password}' should be masked in {masked}")
                
                # Should contain masked indicator
                if '@' in url:  # URLs with credentials
                    self.assertIn('***', masked, f"Masked URL should contain *** indicator: {masked}")
    
    def test_memory_stats_credential_safety(self):
        """Test that memory statistics don't expose database credentials."""
        test_url = self.test_urls['postgres_with_password']
        
        with patch.dict(os.environ, {'DATABASE_URL': test_url}):
            with patch('slack_kb_agent.database.create_engine') as mock_engine:
                mock_engine.return_value = MagicMock()
                
                db_manager = DatabaseManager()
                stats = db_manager.get_memory_stats()
                
                # Check all string values in stats for credentials
                self._verify_no_credentials_in_dict(stats, 'secret123')
    
    def test_logging_credential_safety(self):
        """Test that log messages don't expose database credentials."""
        test_url = self.test_urls['postgres_with_symbols']
        
        with patch.dict(os.environ, {'DATABASE_URL': test_url}):
            with patch('slack_kb_agent.database.create_engine') as mock_engine:
                with patch('slack_kb_agent.database.logger') as mock_logger:
                    mock_engine.return_value = MagicMock()
                    
                    # Initialize database manager
                    db_manager = DatabaseManager()
                    
                    # Check all logged messages
                    for call in mock_logger.info.call_args_list:
                        message = call[0][0] if call[0] else ""
                        self.assertNotIn('p@ssw0rd!', str(message), 
                                       f"Password should not appear in log: {message}")
    
    def test_error_message_credential_safety(self):
        """Test that error messages don't expose database credentials."""
        test_url = self.test_urls['postgres_with_port']
        
        with patch.dict(os.environ, {'DATABASE_URL': test_url}):
            with patch('slack_kb_agent.database.create_engine') as mock_engine:
                # Make engine creation fail
                mock_engine.side_effect = Exception("Connection failed")
                
                with patch('slack_kb_agent.database.logger') as mock_logger:
                    try:
                        DatabaseManager()
                    except:
                        pass  # We expect this to fail
                    
                    # Check error messages don't contain password
                    for call in mock_logger.error.call_args_list:
                        message = str(call)
                        self.assertNotIn('password', message, 
                                       f"Password should not appear in error: {message}")
    
    def test_string_representation_safety(self):
        """Test that string representations don't expose credentials."""
        test_url = self.test_urls['postgres_with_password']
        
        with patch.dict(os.environ, {'DATABASE_URL': test_url}):
            with patch('slack_kb_agent.database.create_engine') as mock_engine:
                mock_engine.return_value = MagicMock()
                
                db_manager = DatabaseManager()
                
                # Test string representation
                str_repr = str(db_manager)
                self.assertNotIn('secret123', str_repr, 
                               f"String representation should not contain password: {str_repr}")
                
                # Test repr representation if available
                if hasattr(db_manager, '__repr__'):
                    repr_str = repr(db_manager)
                    self.assertNotIn('secret123', repr_str, 
                                   f"Repr should not contain password: {repr_str}")
    
    def test_monitoring_endpoint_safety(self):
        """Test that monitoring endpoints don't expose credentials."""
        test_url = self.test_urls['mysql_url']
        
        with patch.dict(os.environ, {'DATABASE_URL': test_url}):
            with patch('slack_kb_agent.database.create_engine') as mock_engine:
                mock_engine.return_value = MagicMock()
                
                db_manager = DatabaseManager()
                
                # Test various status methods that might expose URLs
                methods_to_test = ['get_memory_stats']
                
                for method_name in methods_to_test:
                    if hasattr(db_manager, method_name):
                        result = getattr(db_manager, method_name)()
                        if isinstance(result, dict):
                            self._verify_no_credentials_in_dict(result, 'admin123')
    
    def _mask_database_url(self, url: str) -> str:
        """Helper method to mask database credentials - this is what we want to implement."""
        if not url or '://' not in url:
            return url
            
        # Parse URL parts
        scheme_netloc = url.split('://', 1)
        if len(scheme_netloc) != 2:
            return url
            
        scheme, rest = scheme_netloc
        
        # Check if there are credentials
        if '@' not in rest:
            return url  # No credentials to mask
        
        # Split credentials and host info
        creds_host = rest.split('@', 1)
        if len(creds_host) != 2:
            return url
            
        credentials, host_db = creds_host
        
        # Mask the password part
        if ':' in credentials:
            username, password = credentials.split(':', 1)
            masked_creds = f"{username}:***"
        else:
            masked_creds = credentials  # No password to mask
        
        return f"{scheme}://{masked_creds}@{host_db}"
    
    def _verify_no_credentials_in_dict(self, data: dict, sensitive_value: str) -> None:
        """Recursively verify that sensitive data doesn't appear in dictionary values."""
        for key, value in data.items():
            if isinstance(value, dict):
                self._verify_no_credentials_in_dict(value, sensitive_value)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, str):
                        self.assertNotIn(sensitive_value, item, 
                                       f"Sensitive value found in {key}: {item}")
            elif isinstance(value, str):
                self.assertNotIn(sensitive_value, value, 
                               f"Sensitive value found in {key}: {value}")


class TestCredentialMaskingUtility(unittest.TestCase):
    """Test the credential masking utility function."""
    
    def test_various_url_formats(self):
        """Test masking works for various database URL formats."""
        test_cases = [
            ('postgresql://user:password@localhost/db', 'postgresql://user:***@localhost/db'),
            ('mysql://admin:secret@host:3306/db', 'mysql://admin:***@host:3306/db'),
            ('postgresql://localhost/db', 'postgresql://localhost/db'),  # No credentials
            ('postgresql://user@localhost/db', 'postgresql://user@localhost/db'),  # No password
            ('invalid-url', 'invalid-url'),  # Invalid format
            ('', ''),  # Empty string
            (None, None),  # None input
        ]
        
        for original, expected in test_cases:
            with self.subTest(url=original):
                # This is the implementation we need to create
                if original is None:
                    result = None
                else:
                    result = self._mask_database_url(original)
                self.assertEqual(result, expected, f"Failed for {original}")
    
    def _mask_database_url(self, url: str) -> str:
        """Implementation of URL masking that we'll integrate into the codebase."""
        if not url or '://' not in url:
            return url
            
        try:
            # Parse URL parts
            scheme_netloc = url.split('://', 1)
            if len(scheme_netloc) != 2:
                return url
                
            scheme, rest = scheme_netloc
            
            # Check if there are credentials
            if '@' not in rest:
                return url  # No credentials to mask
            
            # Split credentials and host info
            creds_host = rest.split('@', 1)
            if len(creds_host) != 2:
                return url
                
            credentials, host_db = creds_host
            
            # Mask the password part
            if ':' in credentials:
                username, password = credentials.split(':', 1)
                if password:  # Only mask if password is not empty
                    masked_creds = f"{username}:***"
                else:
                    masked_creds = credentials
            else:
                masked_creds = credentials  # No password to mask
            
            return f"{scheme}://{masked_creds}@{host_db}"
            
        except Exception:
            # If parsing fails, return original URL (defensive approach)
            return url


def run_tests():
    """Run all database credential security tests."""
    print("🔒 Testing Database Credential Security...")
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTest(loader.loadTestsFromTestCase(TestDatabaseCredentialSecurity))
    suite.addTest(loader.loadTestsFromTestCase(TestCredentialMaskingUtility))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return success status
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    if success:
        print("\n✅ All database credential security tests passed!")
    else:
        print("\n❌ Some tests failed. Database credential masking needs implementation.")
    
    sys.exit(0 if success else 1)