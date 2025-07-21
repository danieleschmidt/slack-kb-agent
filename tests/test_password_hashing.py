"""Tests for secure password hashing functionality."""

import pytest
import os
import time
from unittest.mock import patch, MagicMock

from slack_kb_agent.auth import (
    AuthConfig,
    BasicAuthenticator,
    PasswordHasher
)


class TestPasswordHasher:
    """Test password hashing functionality."""
    
    def test_hash_password_returns_different_hash_each_time(self):
        """Test that hashing the same password produces different hashes (due to salt)."""
        hasher = PasswordHasher()
        password = "test_password_123"
        
        hash1 = hasher.hash_password(password)
        hash2 = hasher.hash_password(password)
        
        # Should be different due to different salts
        assert hash1 != hash2
        assert len(hash1) > 50  # bcrypt hashes are typically 60 characters
        assert len(hash2) > 50
    
    def test_verify_password_correct_password(self):
        """Test that correct password verifies successfully."""
        hasher = PasswordHasher()
        password = "test_password_123"
        
        hashed = hasher.hash_password(password)
        
        # Correct password should verify
        assert hasher.verify_password(password, hashed)
    
    def test_verify_password_incorrect_password(self):
        """Test that incorrect password fails verification."""
        hasher = PasswordHasher()
        password = "test_password_123"
        wrong_password = "wrong_password"
        
        hashed = hasher.hash_password(password)
        
        # Wrong password should not verify
        assert not hasher.verify_password(wrong_password, hashed)
    
    def test_verify_password_empty_password(self):
        """Test that empty password is handled correctly."""
        hasher = PasswordHasher()
        password = ""
        
        hashed = hasher.hash_password(password)
        
        # Empty password should verify against its own hash
        assert hasher.verify_password(password, hashed)
        
        # Non-empty password should not verify against empty password hash
        assert not hasher.verify_password("nonempty", hashed)
    
    def test_verify_password_timing_safety(self):
        """Test that password verification has consistent timing (prevents timing attacks)."""
        hasher = PasswordHasher()
        password = "test_password_123"
        hashed = hasher.hash_password(password)
        
        # Measure timing for correct password
        start_time = time.time()
        hasher.verify_password(password, hashed)
        correct_time = time.time() - start_time
        
        # Measure timing for incorrect password
        start_time = time.time()
        hasher.verify_password("wrong_password", hashed)
        incorrect_time = time.time() - start_time
        
        # Times should be similar (within reasonable tolerance)
        # This is a basic test - bcrypt naturally provides timing safety
        time_diff = abs(correct_time - incorrect_time)
        assert time_diff < 0.1  # Allow 100ms difference
    
    def test_hash_special_characters(self):
        """Test hashing passwords with special characters."""
        hasher = PasswordHasher()
        
        special_passwords = [
            "password!@#$%^&*()",
            "пароль",  # Non-ASCII characters
            "pass word with spaces",
            "🔒🔑💻",  # Unicode symbols
            "\n\t\r",  # Whitespace characters
        ]
        
        for password in special_passwords:
            hashed = hasher.hash_password(password)
            assert hasher.verify_password(password, hashed)
            assert len(hashed) > 50
    
    def test_hash_long_password(self):
        """Test hashing very long passwords."""
        hasher = PasswordHasher()
        
        # bcrypt truncates at 72 bytes, but should still work
        long_password = "a" * 1000
        hashed = hasher.hash_password(long_password)
        
        assert hasher.verify_password(long_password, hashed)
        
        # Slightly different long password should not verify
        different_long = "a" * 999 + "b"
        assert not hasher.verify_password(different_long, hashed)
    
    def test_verify_invalid_hash_format(self):
        """Test verification with invalid hash format."""
        hasher = PasswordHasher()
        
        invalid_hashes = [
            "",
            "not_a_hash",
            "too_short",
            "invalid$2b$format",
        ]
        
        for invalid_hash in invalid_hashes:
            # Should return False rather than raising exception
            assert not hasher.verify_password("password", invalid_hash)
    
    def test_cost_parameter_affects_performance(self):
        """Test that higher cost parameters take longer (basic performance test)."""
        low_cost_hasher = PasswordHasher(cost=4)  # Very low for testing
        high_cost_hasher = PasswordHasher(cost=6)  # Still low for testing
        
        password = "test_password"
        
        # Measure time for low cost
        start_time = time.time()
        low_cost_hasher.hash_password(password)
        low_cost_time = time.time() - start_time
        
        # Measure time for higher cost
        start_time = time.time()
        high_cost_hasher.hash_password(password)
        high_cost_time = time.time() - start_time
        
        # Higher cost should take longer (though this might be flaky in fast environments)
        # This is more of a sanity check than a strict requirement
        assert high_cost_time >= low_cost_time * 0.5  # Allow some variance


class TestBasicAuthenticatorWithHashing:
    """Test BasicAuthenticator with password hashing integration."""
    
    def test_authenticator_uses_hashed_passwords(self):
        """Test that BasicAuthenticator stores and verifies hashed passwords."""
        config = AuthConfig(
            enabled=True,
            api_keys=set(),
            basic_users={"testuser": "plaintext_password"},
            basic_password=None
        )
        
        authenticator = BasicAuthenticator(config)
        
        # Password should be hashed during initialization
        stored_hash = authenticator.users["testuser"]
        assert stored_hash != "plaintext_password"
        assert len(stored_hash) > 50  # bcrypt hash length
        
        # Authentication should work with original password
        assert authenticator.verify_basic_auth("testuser", "plaintext_password")
        assert not authenticator.verify_basic_auth("testuser", "wrong_password")
    
    def test_authenticator_handles_empty_users(self):
        """Test BasicAuthenticator with no users configured."""
        config = AuthConfig(
            enabled=True,
            api_keys=set(),
            basic_users={},
            basic_password=None
        )
        
        authenticator = BasicAuthenticator(config)
        
        # Should handle empty users gracefully
        assert not authenticator.verify_basic_auth("anyuser", "anypassword")
    
    def test_authenticator_with_basic_password(self):
        """Test BasicAuthenticator with default basic password."""
        config = AuthConfig(
            enabled=True,
            api_keys=set(),
            basic_users={},
            basic_password="default_password"
        )
        
        authenticator = BasicAuthenticator(config)
        
        # Should hash the default password
        assert hasattr(authenticator, 'default_password_hash')
        assert authenticator.default_password_hash != "default_password"
        assert len(authenticator.default_password_hash) > 50
        
        # Should authenticate with default password
        assert authenticator.verify_basic_auth("admin", "default_password")
        assert not authenticator.verify_basic_auth("admin", "wrong_password")
    
    def test_authenticator_migration_from_plaintext(self):
        """Test that existing plaintext passwords are migrated to hashed."""
        # This test ensures backward compatibility during the migration
        config = AuthConfig(
            enabled=True,
            api_keys=set(),
            basic_users={
                "user1": "password123",
                "user2": "another_password"
            },
            basic_password="default_pass"
        )
        
        authenticator = BasicAuthenticator(config)
        
        # All passwords should be hashed
        for username, stored_hash in authenticator.users.items():
            assert stored_hash != config.basic_users[username]
            assert len(stored_hash) > 50
        
        # Original passwords should still work
        assert authenticator.verify_basic_auth("user1", "password123")
        assert authenticator.verify_basic_auth("user2", "another_password")
        assert authenticator.verify_basic_auth("admin", "default_pass")
        
        # Wrong passwords should not work
        assert not authenticator.verify_basic_auth("user1", "wrong")
        assert not authenticator.verify_basic_auth("user2", "wrong")
    
    def test_authenticator_with_pre_hashed_passwords(self):
        """Test that pre-hashed passwords are not double-hashed."""
        hasher = PasswordHasher()
        pre_hashed = hasher.hash_password("original_password")
        
        config = AuthConfig(
            enabled=True,
            api_keys=set(),
            basic_users={"testuser": pre_hashed},
            basic_password=None
        )
        
        authenticator = BasicAuthenticator(config)
        
        # Should detect and preserve pre-hashed password
        assert authenticator.users["testuser"] == pre_hashed
        
        # Should still work with original password
        assert authenticator.verify_basic_auth("testuser", "original_password")


class TestPasswordHashingSecurityProperties:
    """Test security properties of password hashing implementation."""
    
    def test_salt_randomness(self):
        """Test that salts are properly random."""
        hasher = PasswordHasher()
        password = "test_password"
        
        hashes = [hasher.hash_password(password) for _ in range(10)]
        
        # All hashes should be different (due to random salts)
        assert len(set(hashes)) == 10
        
        # All should verify with the original password
        for hash_value in hashes:
            assert hasher.verify_password(password, hash_value)
    
    def test_no_password_leakage_in_exceptions(self):
        """Test that password values don't leak in exception messages."""
        hasher = PasswordHasher()
        secret_password = "super_secret_password_123"
        
        # Test various error conditions
        try:
            hasher.verify_password(secret_password, "invalid_hash_format")
        except Exception as e:
            error_message = str(e)
            assert secret_password not in error_message
        
        # Should not raise exception anyway - just return False
        result = hasher.verify_password(secret_password, "invalid")
        assert result is False
    
    def test_timing_attack_resistance(self):
        """Test basic timing attack resistance properties."""
        hasher = PasswordHasher()
        password = "test_password"
        hashed = hasher.hash_password(password)
        
        # Test that verification time is similar for different wrong passwords
        wrong_passwords = [
            "x",  # Short wrong password
            "completely_different_password_that_is_much_longer",  # Long wrong password
            password[:-1],  # Almost correct password
        ]
        
        times = []
        for wrong_pwd in wrong_passwords:
            start = time.time()
            result = hasher.verify_password(wrong_pwd, hashed)
            end = time.time()
            assert not result
            times.append(end - start)
        
        # Times should be relatively similar (bcrypt provides natural timing safety)
        if len(times) > 1:
            max_time = max(times)
            min_time = min(times)
            # Allow significant variance since we're not doing micro-benchmarks
            assert max_time / min_time < 10  # No more than 10x difference