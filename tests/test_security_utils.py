#!/usr/bin/env python3
"""
Comprehensive tests for security_utils module.

This test suite verifies that credential masking and sensitive data protection
functions work correctly to prevent accidental exposure of secrets in logs,
monitoring, and error messages.
"""

import unittest
from unittest.mock import Mock
import sys
import os

# Add src to path to allow imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from slack_kb_agent.security_utils import (
    mask_database_url,
    mask_connection_string,
    mask_sensitive_dict,
    get_safe_repr,
    quick_mask_credentials
)


class TestMaskDatabaseUrl(unittest.TestCase):
    """Test database URL credential masking functionality."""
    
    def test_postgresql_url_masking(self):
        """Test PostgreSQL URL credential masking."""
        url = "postgresql://user:password@localhost:5432/database"
        masked = mask_database_url(url)
        self.assertEqual(masked, "postgresql://user:***@localhost:5432/database")
    
    def test_mysql_url_masking(self):
        """Test MySQL URL credential masking."""
        url = "mysql://admin:secret123@db.example.com:3306/mydb"
        masked = mask_database_url(url)
        self.assertEqual(masked, "mysql://admin:***@db.example.com:3306/mydb")
    
    def test_complex_password_masking(self):
        """Test masking of complex passwords with special characters."""
        url = "postgresql://user:p@ssw0rd!@#$%@localhost/db"
        masked = mask_database_url(url)
        self.assertEqual(masked, "postgresql://user:***@localhost/db")
    
    def test_url_without_credentials(self):
        """Test URL without credentials remains unchanged."""
        url = "postgresql://localhost:5432/database"
        masked = mask_database_url(url)
        self.assertEqual(masked, url)
    
    def test_url_with_username_only(self):
        """Test URL with username but no password."""
        url = "postgresql://user@localhost:5432/database"
        masked = mask_database_url(url)
        self.assertEqual(masked, url)
    
    def test_url_with_empty_password(self):
        """Test URL with empty password."""
        url = "postgresql://user:@localhost:5432/database"
        masked = mask_database_url(url)
        self.assertEqual(masked, url)  # No password to mask
    
    def test_none_input(self):
        """Test None input handling."""
        self.assertEqual(mask_database_url(None), "None")
    
    def test_empty_string_input(self):
        """Test empty string input."""
        self.assertEqual(mask_database_url(""), "")
    
    def test_non_string_input(self):
        """Test non-string input handling."""
        self.assertEqual(mask_database_url(123), "123")
        self.assertEqual(mask_database_url([]), "[]")
    
    def test_malformed_url_handling(self):
        """Test handling of malformed URLs."""
        malformed_urls = [
            "not_a_url",
            "http://",
            "postgresql://user@",
            "postgresql://user:pass@@localhost",
            "postgresql://user:pass@"
        ]
        
        for url in malformed_urls:
            with self.subTest(url=url):
                result = mask_database_url(url)
                # Should either return original or safe error message
                self.assertIsInstance(result, str)
                self.assertNotIn("password", result.lower())
    
    def test_url_parsing_exception_handling(self):
        """Test graceful handling of URL parsing exceptions."""
        # Create a malformed URL that might cause parsing errors
        malformed_url = "postgresql://user:pass@host:port:extra:stuff"
        result = mask_database_url(malformed_url)
        
        # Should mask the password even in malformed URLs
        self.assertEqual(result, "postgresql://user:***@host:port:extra:stuff")


class TestMaskConnectionString(unittest.TestCase):
    """Test connection string masking functionality."""
    
    def test_url_format_connection_string(self):
        """Test URL format connection strings."""
        conn_str = "postgresql://user:password@localhost:5432/db"
        masked = mask_connection_string(conn_str)
        self.assertEqual(masked, "postgresql://user:***@localhost:5432/db")
    
    def test_key_value_format_connection_string(self):
        """Test key-value format connection strings."""
        conn_str = "host=localhost port=5432 database=mydb password=secret user=admin"
        masked = mask_connection_string(conn_str)
        self.assertIn("password=***", masked)
        self.assertIn("host=localhost", masked)
        self.assertIn("user=admin", masked)
        self.assertNotIn("secret", masked)
    
    def test_mixed_sensitive_keys(self):
        """Test masking of various sensitive key names."""
        sensitive_keys = ["password", "pwd", "pass", "secret", "key"]
        
        for key in sensitive_keys:
            with self.subTest(key=key):
                conn_str = f"host=localhost {key}=sensitive_value"
                masked = mask_connection_string(conn_str)
                self.assertIn(f"{key}=***", masked)
                self.assertNotIn("sensitive_value", masked)
    
    def test_non_connection_string_input(self):
        """Test handling of non-connection string input."""
        non_conn_str = "just a regular string"
        masked = mask_connection_string(non_conn_str)
        self.assertEqual(masked, "<masked_connection_string>")
    
    def test_none_and_empty_input(self):
        """Test None and empty input handling."""
        self.assertEqual(mask_connection_string(None), "None")
        self.assertEqual(mask_connection_string(""), "")
    
    def test_exception_handling(self):
        """Test graceful exception handling."""
        # This would cause an exception in string processing
        malformed_str = "key=value=extra=stuff"
        result = mask_connection_string(malformed_str)
        self.assertIsInstance(result, str)


class TestMaskSensitiveDict(unittest.TestCase):
    """Test dictionary masking functionality."""
    
    def test_basic_sensitive_key_masking(self):
        """Test masking of basic sensitive keys."""
        data = {
            "username": "admin",
            "password": "secret123",
            "host": "localhost"
        }
        
        masked = mask_sensitive_dict(data)
        
        self.assertEqual(masked["username"], "admin")
        self.assertEqual(masked["password"], "***")
        self.assertEqual(masked["host"], "localhost")
    
    def test_nested_dictionary_masking(self):
        """Test recursive masking of nested dictionaries."""
        data = {
            "database": {
                "host": "localhost",
                "password": "secret"
            },
            "api": {
                "token": "abc123",
                "endpoint": "https://api.example.com"
            }
        }
        
        masked = mask_sensitive_dict(data)
        
        self.assertEqual(masked["database"]["host"], "localhost")
        self.assertEqual(masked["database"]["password"], "***")
        self.assertEqual(masked["api"]["token"], "***")
        self.assertEqual(masked["api"]["endpoint"], "https://api.example.com")
    
    def test_url_in_sensitive_value(self):
        """Test URL masking within sensitive values."""
        data = {
            "database_url": "postgresql://user:password@localhost/db"
        }
        
        masked = mask_sensitive_dict(data)
        
        self.assertEqual(masked["database_url"], "postgresql://user:***@localhost/db")
    
    def test_custom_sensitive_keys(self):
        """Test custom sensitive keys parameter."""
        data = {
            "custom_secret": "sensitive",
            "password": "also_sensitive",
            "normal_key": "not_sensitive"
        }
        
        custom_keys = {"custom_secret"}
        masked = mask_sensitive_dict(data, custom_keys)
        
        self.assertEqual(masked["custom_secret"], "***")
        self.assertEqual(masked["password"], "also_sensitive")  # Not in custom keys
        self.assertEqual(masked["normal_key"], "not_sensitive")
    
    def test_string_value_connection_detection(self):
        """Test detection of connection strings in string values."""
        data = {
            "config": "host=localhost password=secret user=admin",
            "normal_string": "just text"
        }
        
        masked = mask_sensitive_dict(data)
        
        self.assertIn("password=***", masked["config"])
        self.assertEqual(masked["normal_string"], "just text")
    
    def test_non_dict_input(self):
        """Test handling of non-dictionary input."""
        non_dict_inputs = ["string", 123, None, [1, 2, 3]]
        
        for input_val in non_dict_inputs:
            with self.subTest(input_val=input_val):
                result = mask_sensitive_dict(input_val)
                self.assertEqual(result, input_val)
    
    def test_empty_string_values(self):
        """Test handling of empty string values."""
        data = {
            "password": "",
            "secret": None,
            "token": "valid_token"
        }
        
        masked = mask_sensitive_dict(data)
        
        self.assertEqual(masked["password"], "***")
        self.assertEqual(masked["secret"], "***")
        self.assertEqual(masked["token"], "***")


class TestGetSafeRepr(unittest.TestCase):
    """Test safe object representation functionality."""
    
    def setUp(self):
        """Set up test objects."""
        class TestObject:
            def __init__(self):
                self.database_url = "postgresql://user:password@localhost/db"
                self.api_key = "secret_key_123"
                self.normal_attr = "normal_value"
                self.long_attr = "x" * 100
        
        self.test_obj = TestObject()
    
    def test_sensitive_attribute_masking(self):
        """Test masking of sensitive attributes."""
        repr_str = get_safe_repr(self.test_obj)
        
        self.assertIn("TestObject", repr_str)
        self.assertIn("api_key=***", repr_str)
        self.assertIn("database_url=postgresql://user:***@localhost/db", repr_str)
        self.assertNotIn("password", repr_str)
        self.assertNotIn("secret_key_123", repr_str)
    
    def test_normal_attribute_preservation(self):
        """Test that normal attributes are preserved."""
        repr_str = get_safe_repr(self.test_obj)
        
        self.assertIn("normal_attr='normal_value'", repr_str)
    
    def test_long_attribute_truncation(self):
        """Test truncation of long attribute values."""
        repr_str = get_safe_repr(self.test_obj)
        
        self.assertIn("long_attr=", repr_str)
        self.assertIn("...", repr_str)
        # Should not contain the full 100 character string
        self.assertLess(len(repr_str), 200)
    
    def test_custom_mask_attributes(self):
        """Test custom mask attributes parameter."""
        custom_attrs = {"normal_attr"}
        repr_str = get_safe_repr(self.test_obj, custom_attrs)
        
        self.assertIn("normal_attr=***", repr_str)
        # Database URL should not be masked with custom attrs
        self.assertIn("database_url=", repr_str)
        self.assertNotIn("database_url=***", repr_str)
    
    def test_attribute_limits(self):
        """Test that representation is limited to prevent excessive output."""
        class ObjectWithManyAttrs:
            def __init__(self):
                for i in range(20):
                    setattr(self, f"attr_{i}", f"value_{i}")
        
        obj = ObjectWithManyAttrs()
        repr_str = get_safe_repr(obj)
        
        # Should limit to 5 attributes and add ...
        attr_count = repr_str.count("attr_")
        self.assertLessEqual(attr_count, 5)
        if attr_count == 5:
            self.assertIn("...", repr_str)
    
    def test_exception_handling(self):
        """Test graceful handling of objects that raise exceptions."""
        class ProblematicObject:
            @property
            def bad_property(self):
                raise ValueError("Cannot access this property")
        
        obj = ProblematicObject()
        repr_str = get_safe_repr(obj)
        
        # Should not crash and should return something reasonable
        self.assertIsInstance(repr_str, str)
        self.assertIn("ProblematicObject", repr_str)


class TestQuickMaskCredentials(unittest.TestCase):
    """Test quick regex-based credential masking."""
    
    def test_url_password_masking(self):
        """Test quick masking of passwords in URLs."""
        text = "Connection: postgresql://user:password@localhost/db"
        masked = quick_mask_credentials(text)
        
        self.assertIn("postgresql://user:***@localhost/db", masked)
        self.assertNotIn("password", masked)
    
    def test_key_value_password_masking(self):
        """Test quick masking of key-value password pairs."""
        text = "Config: host=localhost password=secret user=admin"
        masked = quick_mask_credentials(text)
        
        self.assertIn("password=***", masked)
        self.assertNotIn("secret", masked)
        self.assertIn("host=localhost", masked)
    
    def test_multiple_patterns_in_text(self):
        """Test masking multiple credential patterns in the same text."""
        text = "Database: postgresql://user:mypass@db password=secret key=value"
        masked = quick_mask_credentials(text)
        
        self.assertIn("postgresql://user:***@db", masked)
        self.assertIn("password=***", masked)
        self.assertNotIn("mypass", masked)
        self.assertNotIn("secret", masked)
        self.assertIn("key=value", masked)  # key=value should not be masked
    
    def test_case_insensitive_matching(self):
        """Test case-insensitive matching of password keys."""
        text = "PASSWORD=secret Pwd=another_secret"
        masked = quick_mask_credentials(text)
        
        self.assertIn("PASSWORD=***", masked)
        self.assertIn("Pwd=***", masked)
        self.assertNotIn("secret", masked)
        self.assertNotIn("another_secret", masked)
    
    def test_empty_and_none_input(self):
        """Test handling of empty and None input."""
        self.assertEqual(quick_mask_credentials(""), "")
        self.assertEqual(quick_mask_credentials(None), None)
    
    def test_text_without_credentials(self):
        """Test that text without credentials remains unchanged."""
        text = "This is just normal text with no credentials"
        masked = quick_mask_credentials(text)
        self.assertEqual(masked, text)
    
    def test_performance_compared_to_full_masking(self):
        """Test that quick masking is indeed faster (basic check)."""
        import time
        
        text = "postgresql://user:password@localhost/db " * 100
        
        # Quick masking
        start = time.time()
        for _ in range(1000):
            quick_mask_credentials(text)
        quick_time = time.time() - start
        
        # Should complete reasonably quickly
        self.assertLess(quick_time, 1.0)  # Should complete in under 1 second


class TestSecurityUtilsIntegration(unittest.TestCase):
    """Integration tests for security utilities."""
    
    def test_comprehensive_credential_protection(self):
        """Test comprehensive protection across all utility functions."""
        # Simulate a real-world scenario with multiple credential types
        config_data = {
            "database": {
                "url": "postgresql://admin:supersecret@prod-db:5432/app_db",
                "backup_url": "mysql://backup:backup123@backup-db/backup_db"
            },
            "api": {
                "key": "api_key_abc123xyz",
                "secret": "api_secret_def456"
            },
            "conn_config": "host=localhost password=dbpassword user=dbuser"
        }
        
        # Test dictionary masking
        masked_dict = mask_sensitive_dict(config_data)
        
        # Verify all credentials are masked
        self.assertNotIn("supersecret", str(masked_dict))
        self.assertNotIn("backup123", str(masked_dict))
        self.assertNotIn("api_key_abc123xyz", str(masked_dict))
        self.assertNotIn("api_secret_def456", str(masked_dict))
        self.assertNotIn("dbpassword", str(masked_dict))
        
        # Verify structure is preserved
        self.assertEqual(masked_dict["database"]["url"], "postgresql://admin:***@prod-db:5432/app_db")
        # conn_config contains "password=" so gets processed by mask_connection_string
        self.assertEqual(masked_dict["conn_config"], "host=localhost password=*** user=dbuser")
    
    def test_logging_safety(self):
        """Test that masked data is safe for logging."""
        sensitive_data = {
            "database_url": "postgresql://user:password@localhost/db",
            "api_key": "secret123",
            "normal_data": "safe_to_log"
        }
        
        masked = mask_sensitive_dict(sensitive_data)
        log_message = f"Configuration: {masked}"
        
        # Verify no credentials in log message
        self.assertNotIn("password", log_message)
        self.assertNotIn("secret123", log_message)
        
        # Verify safe data is preserved
        self.assertIn("safe_to_log", log_message)
    
    def test_error_message_safety(self):
        """Test that error messages containing URLs are safe."""
        error_message = "Database connection failed: postgresql://user:password@localhost/db"
        safe_message = quick_mask_credentials(error_message)
        
        self.assertNotIn("password", safe_message)
        self.assertIn("postgresql://user:***@localhost/db", safe_message)


if __name__ == '__main__':
    unittest.main()