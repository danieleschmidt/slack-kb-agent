"""Tests for database security configurations."""

import pytest
import os
from unittest.mock import patch

from slack_kb_agent.database import DatabaseManager


class TestDatabaseSecurity:
    """Test database security configurations."""
    
    def test_database_url_required(self):
        """Test that DatabaseManager requires DATABASE_URL environment variable."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="DATABASE_URL environment variable is required"):
                DatabaseManager()
    
    def test_database_url_from_environment(self):
        """Test that DatabaseManager correctly reads DATABASE_URL from environment."""
        test_url = "postgresql://testuser:testpass@testhost:5432/testdb"
        
        with patch.dict(os.environ, {'DATABASE_URL': test_url}):
            manager = DatabaseManager()
            assert manager.database_url == test_url
    
    def test_database_url_parameter_override(self):
        """Test that explicit database_url parameter overrides environment."""
        env_url = "postgresql://envuser:envpass@envhost:5432/envdb"
        param_url = "postgresql://paramuser:parampass@paramhost:5432/paramdb"
        
        with patch.dict(os.environ, {'DATABASE_URL': env_url}):
            manager = DatabaseManager(database_url=param_url)
            assert manager.database_url == param_url
    
    def test_no_hardcoded_credentials_in_defaults(self):
        """Test that no hard-coded credentials exist in default configuration."""
        # This test ensures we don't accidentally introduce hard-coded credentials again
        with patch.dict(os.environ, {}, clear=True):
            try:
                DatabaseManager()
                # Should raise ValueError, not create with default credentials
                pytest.fail("DatabaseManager should require DATABASE_URL")
            except ValueError as e:
                assert "DATABASE_URL environment variable is required" in str(e)
    
    def test_database_url_validation_message(self):
        """Test that error message provides helpful guidance."""
        with patch.dict(os.environ, {}, clear=True):
            try:
                DatabaseManager()
            except ValueError as e:
                error_message = str(e)
                assert "DATABASE_URL environment variable is required" in error_message
                assert "Example: postgresql://user:password@localhost:5432/dbname" in error_message