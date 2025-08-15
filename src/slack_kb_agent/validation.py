#!/usr/bin/env python3
"""
Input validation and sanitization module for Slack KB Agent.

Provides comprehensive input validation, sanitization, and security filtering
for user queries, bot commands, and API inputs.
"""

import html
import logging
import os
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, NamedTuple, Optional

from .constants import ValidationDefaults

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ValidationResult(NamedTuple):
    """Result of input validation."""
    is_valid: bool
    sanitized_input: Optional[str] = None
    error_message: Optional[str] = None
    warning_message: Optional[str] = None


@dataclass
class ValidationConfig:
    """Configuration for input validation."""
    enabled: bool = True
    max_query_length: int = ValidationDefaults.MAX_QUERY_LENGTH
    max_user_id_length: int = ValidationDefaults.MAX_USER_ID_LENGTH
    max_channel_id_length: int = ValidationDefaults.MAX_CHANNEL_ID_LENGTH
    strip_html: bool = True
    block_sql_injection: bool = True
    block_command_injection: bool = True
    block_xss: bool = True
    allow_unicode: bool = True
    normalize_whitespace: bool = True

    @classmethod
    def from_env(cls) -> 'ValidationConfig':
        """Create ValidationConfig from environment variables."""
        return cls(
            enabled=os.getenv("VALIDATION_ENABLED", "true").lower() == "true",
            max_query_length=int(os.getenv("VALIDATION_MAX_QUERY_LENGTH", "1000")),
            strip_html=os.getenv("VALIDATION_STRIP_HTML", "true").lower() == "true",
            block_sql_injection=os.getenv("VALIDATION_BLOCK_SQL", "true").lower() == "true",
            block_command_injection=os.getenv("VALIDATION_BLOCK_COMMAND", "true").lower() == "true",
            block_xss=os.getenv("VALIDATION_BLOCK_XSS", "true").lower() == "true",
            allow_unicode=os.getenv("VALIDATION_ALLOW_UNICODE", "true").lower() == "true",
            normalize_whitespace=os.getenv("VALIDATION_NORMALIZE_WHITESPACE", "true").lower() == "true"
        )


class InputValidator:
    """Comprehensive input validator and sanitizer."""

    # SQL injection patterns
    SQL_INJECTION_PATTERNS = [
        r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION)\b)",
        r"(--|#|/\*|\*/)",
        r"(\bOR\b.*=.*=|\bAND\b.*=.*=)",
        r"(;.*DROP|;.*DELETE|;.*INSERT)",
        r"(\x00|\x1a)"  # Null bytes and substitute characters
    ]

    # Command injection patterns (more targeted)
    COMMAND_INJECTION_PATTERNS = [
        r"(&&|\|\|?)",       # Command chaining (including single pipe)
        r";\s*(rm|curl|nc|wget)", # Dangerous commands after semicolon
        r"\$\([^)]*\)",      # Command substitution
        r"`[^`]*`",          # Backtick command execution
        r"\\x[0-9a-fA-F]{2}", # Hex-encoded characters
    ]

    # XSS patterns
    XSS_PATTERNS = [
        r"<script[^>]*>.*?</script>",
        r"javascript:",
        r"on\w+\s*=",  # Event handlers like onclick=
        r"<iframe[^>]*>",
        r"<object[^>]*>",
        r"<embed[^>]*>",
        r"<link[^>]*>",
        r"<meta[^>]*>",
    ]

    # Slack ID patterns (more permissive for real Slack IDs)
    SLACK_USER_ID_PATTERN = r"^[UW][A-Z0-9]{6,}$"
    SLACK_CHANNEL_ID_PATTERN = r"^[CDG][A-Z0-9]{6,}$"

    def __init__(self, config: Optional[ValidationConfig] = None):
        self.config = config or ValidationConfig()

        # Compile regex patterns for performance
        self.sql_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.SQL_INJECTION_PATTERNS]
        self.command_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.COMMAND_INJECTION_PATTERNS]
        self.xss_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.XSS_PATTERNS]
        self.user_id_pattern = re.compile(self.SLACK_USER_ID_PATTERN)
        self.channel_id_pattern = re.compile(self.SLACK_CHANNEL_ID_PATTERN)

    def validate_query(self, query: str) -> ValidationResult:
        """Validate and sanitize a user query."""
        if not self.config.enabled:
            return ValidationResult(is_valid=True, sanitized_input=query)

        # Check for empty query
        if not query or not query.strip():
            return ValidationResult(
                is_valid=False,
                error_message="Query cannot be empty"
            )

        # Check length
        if len(query) > self.config.max_query_length:
            return ValidationResult(
                is_valid=False,
                error_message=f"Query too long (max {self.config.max_query_length} characters)"
            )

        # Check for dangerous patterns in ORIGINAL query before sanitization
        validation_errors = []

        if self.config.block_sql_injection and self._contains_sql_injection(query):
            validation_errors.append("Potential SQL injection detected")

        if self.config.block_command_injection and self._contains_command_injection(query):
            validation_errors.append("Potential command injection detected")

        if self.config.block_xss and self._contains_xss(query):
            validation_errors.append("Potential XSS detected")

        # Sanitize the query (for safe processing if validation passes)
        sanitized = self._sanitize_text(query)

        # Check for control characters
        if self._contains_dangerous_characters(sanitized):
            validation_errors.append("Invalid characters detected")

        if validation_errors:
            return ValidationResult(
                is_valid=False,
                error_message="; ".join(validation_errors)
            )

        return ValidationResult(
            is_valid=True,
            sanitized_input=sanitized,
            warning_message="Query was sanitized" if sanitized != query else None
        )

    def validate_user_id(self, user_id: str) -> ValidationResult:
        """Validate Slack user ID format."""
        if not user_id:
            return ValidationResult(
                is_valid=False,
                error_message="User ID cannot be empty"
            )

        if len(user_id) > self.config.max_user_id_length:
            return ValidationResult(
                is_valid=False,
                error_message="User ID too long"
            )

        if not self.user_id_pattern.match(user_id):
            return ValidationResult(
                is_valid=False,
                error_message="Invalid user ID format"
            )

        return ValidationResult(is_valid=True, sanitized_input=user_id)

    def validate_channel_id(self, channel_id: str) -> ValidationResult:
        """Validate Slack channel ID format."""
        if not channel_id:
            return ValidationResult(
                is_valid=False,
                error_message="Channel ID cannot be empty"
            )

        if len(channel_id) > self.config.max_channel_id_length:
            return ValidationResult(
                is_valid=False,
                error_message="Channel ID too long"
            )

        if not self.channel_id_pattern.match(channel_id):
            return ValidationResult(
                is_valid=False,
                error_message="Invalid channel ID format"
            )

        return ValidationResult(is_valid=True, sanitized_input=channel_id)

    def _sanitize_text(self, text: str) -> str:
        """Sanitize text input."""
        result = text

        # Normalize Unicode if enabled
        if self.config.allow_unicode:
            result = unicodedata.normalize('NFKC', result)
        else:
            # Remove non-ASCII characters
            result = result.encode('ascii', 'ignore').decode('ascii')

        # Remove dangerous patterns first, then HTML escape
        if self.config.block_xss:
            for pattern in self.xss_patterns:
                result = pattern.sub('', result)

        if self.config.block_sql_injection:
            for pattern in self.sql_patterns:
                result = pattern.sub(' ', result)  # Replace with space to preserve readability

        if self.config.block_command_injection:
            for pattern in self.command_patterns:
                result = pattern.sub(' ', result)  # Replace with space to preserve readability

        # HTML escape after pattern removal
        if self.config.strip_html:
            result = html.escape(result)

        # Normalize whitespace
        if self.config.normalize_whitespace:
            result = ' '.join(result.split())

        return result

    def _contains_sql_injection(self, text: str) -> bool:
        """Check for SQL injection patterns."""
        return any(pattern.search(text) for pattern in self.sql_patterns)

    def _contains_command_injection(self, text: str) -> bool:
        """Check for command injection patterns."""
        found = any(pattern.search(text) for pattern in self.command_patterns)
        if found:
            logger.debug(f"Command injection detected in: {text}")
        return found

    def _contains_xss(self, text: str) -> bool:
        """Check for XSS patterns."""
        return any(pattern.search(text) for pattern in self.xss_patterns)

    def _contains_dangerous_characters(self, text: str) -> bool:
        """Check for dangerous control characters."""
        # Check for null bytes and other control characters
        dangerous_chars = ['\x00', '\x01', '\x02', '\x03', '\x04', '\x05', '\x06', '\x07',
                          '\x08', '\x0b', '\x0c', '\x0e', '\x0f', '\x10', '\x11', '\x12',
                          '\x13', '\x14', '\x15', '\x16', '\x17', '\x18', '\x19', '\x1a',
                          '\x1b', '\x1c', '\x1d', '\x1e', '\x1f', '\x7f']

        return any(char in text for char in dangerous_chars)


# Global validator instance
_global_validator: Optional[InputValidator] = None


def get_validator() -> InputValidator:
    """Get global input validator instance."""
    global _global_validator

    if _global_validator is None:
        config = ValidationConfig.from_env()
        _global_validator = InputValidator(config)

    return _global_validator


def sanitize_query(query: str) -> str:
    """Sanitize a query string (convenience function)."""
    validator = get_validator()
    result = validator.validate_query(query)

    if result.is_valid:
        return result.sanitized_input or query
    else:
        # Log the validation failure but return a safe empty string
        logger.warning(f"Query validation failed: {result.error_message}")
        return ""


def validate_slack_input(event: Dict[str, Any]) -> ValidationResult:
    """Validate Slack event input."""
    validator = get_validator()

    # Check required fields
    if "text" not in event:
        return ValidationResult(
            is_valid=False,
            error_message="Missing required field: text"
        )

    if "user" not in event:
        return ValidationResult(
            is_valid=False,
            error_message="Missing required field: user"
        )

    # Validate text content
    text_result = validator.validate_query(event["text"])
    if not text_result.is_valid:
        return text_result

    # Validate user ID
    user_result = validator.validate_user_id(event["user"])
    if not user_result.is_valid:
        return ValidationResult(
            is_valid=False,
            error_message=f"Invalid user ID: {user_result.error_message}"
        )

    # Validate channel ID if present
    if "channel" in event:
        channel_result = validator.validate_channel_id(event["channel"])
        if not channel_result.is_valid:
            return ValidationResult(
                is_valid=False,
                error_message=f"Invalid channel ID: {channel_result.error_message}"
            )

    return ValidationResult(
        is_valid=True,
        sanitized_input=text_result.sanitized_input,
        warning_message=text_result.warning_message
    )


def require_valid_input(func):
    """Decorator to validate function inputs."""
    def wrapper(*args, **kwargs):
        # This decorator can be extended for specific validation requirements
        return func(*args, **kwargs)
    return wrapper
