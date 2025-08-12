"""Robust Validation Engine with Comprehensive Error Handling and Security.

This module provides enterprise-grade validation, error handling, and security
measures for the Slack KB Agent research and production systems.
"""

import asyncio
import json
import time
import logging
import hashlib
import re
import sys
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any, Callable, Union
from datetime import datetime, timedelta
from pathlib import Path
import traceback
import threading
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Validation levels for different security contexts."""
    BASIC = "basic"
    STANDARD = "standard"
    STRICT = "strict"
    PARANOID = "paranoid"


class SecurityThreatLevel(Enum):
    """Security threat assessment levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ValidationResult:
    """Result of validation operation."""
    is_valid: bool
    threat_level: SecurityThreatLevel
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    sanitized_data: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ErrorContext:
    """Context information for error handling."""
    component: str
    operation: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    stack_trace: Optional[str] = None


class RobustValidator:
    """Comprehensive validation engine with security and error handling."""
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STANDARD):
        self.validation_level = validation_level
        self.threat_patterns = self._load_threat_patterns()
        self.sanitization_rules = self._load_sanitization_rules()
        self.validation_cache = {}
        self.error_history = deque(maxlen=1000)
        self._lock = threading.Lock()
        
    def validate_query_input(self, query: str, context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """Validate user query input with comprehensive security checks."""
        try:
            if not isinstance(query, str):
                return ValidationResult(
                    is_valid=False,
                    threat_level=SecurityThreatLevel.MEDIUM,
                    errors=["Query must be a string"],
                    sanitized_data=str(query) if query is not None else ""
                )
            
            # Basic length validation
            if len(query) > 10000:
                return ValidationResult(
                    is_valid=False,
                    threat_level=SecurityThreatLevel.HIGH,
                    errors=["Query exceeds maximum length (10,000 characters)"],
                    sanitized_data=query[:10000]
                )
            
            # Check for malicious patterns
            threat_results = self._detect_threats(query)
            
            # SQL injection detection
            sql_threats = self._detect_sql_injection(query)
            threat_results.extend(sql_threats)
            
            # XSS detection
            xss_threats = self._detect_xss_attempts(query)
            threat_results.extend(xss_threats)
            
            # Command injection detection
            cmd_threats = self._detect_command_injection(query)
            threat_results.extend(cmd_threats)
            
            # Path traversal detection
            path_threats = self._detect_path_traversal(query)
            threat_results.extend(path_threats)
            
            # Determine overall threat level
            max_threat = SecurityThreatLevel.LOW
            all_errors = []
            all_warnings = []
            
            for threat in threat_results:
                if threat["level"] == SecurityThreatLevel.CRITICAL:
                    max_threat = SecurityThreatLevel.CRITICAL
                    all_errors.append(threat["description"])
                elif threat["level"] == SecurityThreatLevel.HIGH:
                    if max_threat in [SecurityThreatLevel.LOW, SecurityThreatLevel.MEDIUM]:
                        max_threat = SecurityThreatLevel.HIGH
                    all_errors.append(threat["description"])
                elif threat["level"] == SecurityThreatLevel.MEDIUM:
                    if max_threat == SecurityThreatLevel.LOW:
                        max_threat = SecurityThreatLevel.MEDIUM
                    all_warnings.append(threat["description"])
            
            # Sanitize the query
            sanitized_query = self._sanitize_query(query)
            
            # Final validation based on level
            is_valid = self._final_validation_check(max_threat, all_errors)
            
            return ValidationResult(
                is_valid=is_valid,
                threat_level=max_threat,
                errors=all_errors,
                warnings=all_warnings,
                sanitized_data=sanitized_query,
                metadata={
                    "original_length": len(query),
                    "sanitized_length": len(sanitized_query),
                    "threats_detected": len(threat_results),
                    "validation_level": self.validation_level.value
                }
            )
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return ValidationResult(
                is_valid=False,
                threat_level=SecurityThreatLevel.HIGH,
                errors=[f"Validation system error: {str(e)}"],
                sanitized_data=""
            )
    
    def validate_document_content(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """Validate document content for ingestion."""
        try:
            if not isinstance(content, str):
                return ValidationResult(
                    is_valid=False,
                    threat_level=SecurityThreatLevel.MEDIUM,
                    errors=["Document content must be a string"]
                )
            
            # Check for sensitive information
            sensitive_patterns = self._detect_sensitive_information(content)
            
            # Check for malicious content
            malicious_patterns = self._detect_malicious_content(content)
            
            # Check content quality
            quality_issues = self._assess_content_quality(content)
            
            errors = []
            warnings = []
            max_threat = SecurityThreatLevel.LOW
            
            # Process sensitive information detection
            for pattern in sensitive_patterns:
                if pattern["severity"] == "critical":
                    max_threat = SecurityThreatLevel.CRITICAL
                    errors.append(f"Critical sensitive data detected: {pattern['type']}")
                elif pattern["severity"] == "high":
                    if max_threat in [SecurityThreatLevel.LOW, SecurityThreatLevel.MEDIUM]:
                        max_threat = SecurityThreatLevel.HIGH
                    warnings.append(f"Sensitive data detected: {pattern['type']}")
            
            # Process malicious content detection
            for pattern in malicious_patterns:
                max_threat = SecurityThreatLevel.HIGH
                errors.append(f"Malicious content detected: {pattern}")
            
            # Process quality issues
            for issue in quality_issues:
                warnings.append(f"Content quality issue: {issue}")
            
            # Sanitize content
            sanitized_content = self._sanitize_document_content(content)
            
            is_valid = self._final_validation_check(max_threat, errors)
            
            return ValidationResult(
                is_valid=is_valid,
                threat_level=max_threat,
                errors=errors,
                warnings=warnings,
                sanitized_data=sanitized_content,
                metadata={
                    "original_length": len(content),
                    "sanitized_length": len(sanitized_content),
                    "sensitive_patterns": len(sensitive_patterns),
                    "malicious_patterns": len(malicious_patterns),
                    "quality_issues": len(quality_issues)
                }
            )
            
        except Exception as e:
            logger.error(f"Document validation failed: {e}")
            return ValidationResult(
                is_valid=False,
                threat_level=SecurityThreatLevel.HIGH,
                errors=[f"Document validation error: {str(e)}"]
            )
    
    def validate_api_parameters(self, params: Dict[str, Any]) -> ValidationResult:
        """Validate API parameters with type checking and bounds validation."""
        try:
            errors = []
            warnings = []
            sanitized_params = {}
            max_threat = SecurityThreatLevel.LOW
            
            # Define parameter validation rules
            validation_rules = {
                "query": {"type": str, "max_length": 10000, "required": True},
                "limit": {"type": int, "min": 1, "max": 1000, "default": 10},
                "offset": {"type": int, "min": 0, "max": 100000, "default": 0},
                "threshold": {"type": float, "min": 0.0, "max": 1.0, "default": 0.5},
                "user_id": {"type": str, "max_length": 255, "pattern": r"^[a-zA-Z0-9_-]+$"},
                "session_id": {"type": str, "max_length": 255, "pattern": r"^[a-zA-Z0-9_-]+$"},
                "filters": {"type": dict, "max_depth": 3},
                "sort_by": {"type": str, "allowed_values": ["relevance", "date", "popularity"]}
            }
            
            # Validate each parameter
            for param_name, value in params.items():
                if param_name not in validation_rules:
                    warnings.append(f"Unknown parameter: {param_name}")
                    continue
                
                rule = validation_rules[param_name]
                validation_result = self._validate_parameter(param_name, value, rule)
                
                if validation_result["is_valid"]:
                    sanitized_params[param_name] = validation_result["sanitized_value"]
                else:
                    errors.extend(validation_result["errors"])
                    if validation_result["threat_level"] == SecurityThreatLevel.HIGH:
                        max_threat = SecurityThreatLevel.HIGH
                    elif validation_result["threat_level"] == SecurityThreatLevel.MEDIUM and max_threat == SecurityThreatLevel.LOW:
                        max_threat = SecurityThreatLevel.MEDIUM
            
            # Check for required parameters
            for param_name, rule in validation_rules.items():
                if rule.get("required", False) and param_name not in params:
                    errors.append(f"Required parameter missing: {param_name}")
                elif param_name not in params and "default" in rule:
                    sanitized_params[param_name] = rule["default"]
            
            is_valid = len(errors) == 0
            
            return ValidationResult(
                is_valid=is_valid,
                threat_level=max_threat,
                errors=errors,
                warnings=warnings,
                sanitized_data=sanitized_params,
                metadata={
                    "parameters_validated": len(params),
                    "parameters_sanitized": len(sanitized_params),
                    "unknown_parameters": len([p for p in params if p not in validation_rules])
                }
            )
            
        except Exception as e:
            logger.error(f"Parameter validation failed: {e}")
            return ValidationResult(
                is_valid=False,
                threat_level=SecurityThreatLevel.HIGH,
                errors=[f"Parameter validation error: {str(e)}"]
            )
    
    def _load_threat_patterns(self) -> Dict[str, List[str]]:
        """Load threat detection patterns."""
        return {
            "sql_injection": [
                r"(?i)(\bUNION\b.*\bSELECT\b)",
                r"(?i)(\bSELECT\b.*\bFROM\b)",
                r"(?i)(\bINSERT\b.*\bINTO\b)",
                r"(?i)(\bUPDATE\b.*\bSET\b)",
                r"(?i)(\bDELETE\b.*\bFROM\b)",
                r"(?i)(\bDROP\b.*\bTABLE\b)",
                r"(?i)(\b(OR|AND)\b.*\b1\s*=\s*1\b)",
                r"(?i)(\'\s*(OR|AND)\s*\'\s*=\s*\')",
                r"(?i)(\;\s*(SELECT|INSERT|UPDATE|DELETE|DROP))"
            ],
            "xss": [
                r"(?i)(<script[^>]*>)",
                r"(?i)(javascript:)",
                r"(?i)(on\w+\s*=)",
                r"(?i)(<iframe[^>]*>)",
                r"(?i)(<object[^>]*>)",
                r"(?i)(<embed[^>]*>)",
                r"(?i)(eval\s*\()",
                r"(?i)(expression\s*\()"
            ],
            "command_injection": [
                r"(\||\&|\;)\s*(ls|cat|pwd|id|whoami|uname)",
                r"(\$\(|\`)(.*?)(\)|\`)",
                r"(\.\.\/|\.\.\\)",
                r"(\|\s*(curl|wget|nc|netcat))",
                r"(\&\&|\|\|)\s*(rm|del|format)"
            ],
            "path_traversal": [
                r"(\.\.\/|\.\.\\)",
                r"(%2e%2e%2f|%2e%2e%5c)",
                r"(\/etc\/|\\windows\\)",
                r"(\/proc\/|\/sys\/)",
                r"(\.\.%2f|\.\.%5c)"
            ]
        }
    
    def _load_sanitization_rules(self) -> Dict[str, Any]:
        """Load sanitization rules."""
        return {
            "remove_patterns": [
                r"<script[^>]*>.*?</script>",
                r"<iframe[^>]*>.*?</iframe>",
                r"javascript:",
                r"on\w+\s*="
            ],
            "replace_patterns": [
                (r"[<>\"\'&]", ""),  # Remove HTML/XML special chars
                (r"\x00", ""),  # Remove null bytes
                (r"[\r\n\t]+", " ")  # Replace whitespace with single space
            ],
            "max_lengths": {
                "query": 10000,
                "content": 1000000,
                "parameter": 1000
            }
        }
    
    def _detect_threats(self, text: str) -> List[Dict[str, Any]]:
        """Detect various security threats in text."""
        threats = []
        
        for threat_type, patterns in self.threat_patterns.items():
            for pattern in patterns:
                try:
                    if re.search(pattern, text):
                        threats.append({
                            "type": threat_type,
                            "pattern": pattern,
                            "level": self._get_threat_level(threat_type),
                            "description": f"{threat_type.replace('_', ' ').title()} pattern detected"
                        })
                except re.error:
                    logger.warning(f"Invalid regex pattern: {pattern}")
        
        return threats
    
    def _detect_sql_injection(self, text: str) -> List[Dict[str, Any]]:
        """Detect SQL injection attempts."""
        threats = []
        sql_keywords = ["SELECT", "INSERT", "UPDATE", "DELETE", "DROP", "UNION", "WHERE", "FROM"]
        
        text_upper = text.upper()
        sql_keyword_count = sum(1 for keyword in sql_keywords if keyword in text_upper)
        
        if sql_keyword_count >= 2:
            threats.append({
                "type": "sql_injection",
                "level": SecurityThreatLevel.HIGH,
                "description": f"Multiple SQL keywords detected ({sql_keyword_count})"
            })
        
        # Check for SQL comment patterns
        if re.search(r"(--|\#|\/\*)", text):
            threats.append({
                "type": "sql_injection",
                "level": SecurityThreatLevel.MEDIUM,
                "description": "SQL comment patterns detected"
            })
        
        return threats
    
    def _detect_xss_attempts(self, text: str) -> List[Dict[str, Any]]:
        """Detect XSS attempts."""
        threats = []
        
        # Check for script tags
        if re.search(r"(?i)<script", text):
            threats.append({
                "type": "xss",
                "level": SecurityThreatLevel.HIGH,
                "description": "Script tag detected"
            })
        
        # Check for event handlers
        if re.search(r"(?i)on\w+\s*=", text):
            threats.append({
                "type": "xss",
                "level": SecurityThreatLevel.MEDIUM,
                "description": "Event handler detected"
            })
        
        # Check for javascript: protocol
        if re.search(r"(?i)javascript:", text):
            threats.append({
                "type": "xss",
                "level": SecurityThreatLevel.HIGH,
                "description": "JavaScript protocol detected"
            })
        
        return threats
    
    def _detect_command_injection(self, text: str) -> List[Dict[str, Any]]:
        """Detect command injection attempts."""
        threats = []
        
        # Check for command separators with common commands
        dangerous_patterns = [
            r"(\||\&|\;)\s*(ls|cat|pwd|id|whoami|uname|ps|kill)",
            r"(\$\(|\`).*(\)|\`)",
            r"(\&\&|\|\|)\s*(rm|del|format|shutdown)"
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, text):
                threats.append({
                    "type": "command_injection",
                    "level": SecurityThreatLevel.HIGH,
                    "description": "Command injection pattern detected"
                })
        
        return threats
    
    def _detect_path_traversal(self, text: str) -> List[Dict[str, Any]]:
        """Detect path traversal attempts."""
        threats = []
        
        # Check for directory traversal patterns
        traversal_patterns = [
            r"\.\.\/",
            r"\.\.\\",
            r"%2e%2e%2f",
            r"%2e%2e%5c"
        ]
        
        for pattern in traversal_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                threats.append({
                    "type": "path_traversal",
                    "level": SecurityThreatLevel.MEDIUM,
                    "description": "Path traversal pattern detected"
                })
        
        # Check for sensitive system paths
        sensitive_paths = [
            r"\/etc\/passwd",
            r"\/etc\/shadow",
            r"\\windows\\system32",
            r"\/proc\/",
            r"\/sys\/"
        ]
        
        for path in sensitive_paths:
            if re.search(path, text, re.IGNORECASE):
                threats.append({
                    "type": "path_traversal",
                    "level": SecurityThreatLevel.HIGH,
                    "description": "Sensitive system path detected"
                })
        
        return threats
    
    def _detect_sensitive_information(self, content: str) -> List[Dict[str, Any]]:
        """Detect sensitive information in content."""
        patterns = []
        
        # API keys and tokens
        if re.search(r"(api[_-]?key|token|secret)[\"\':\s]*[\"\']\w{20,}[\"\']\b", content, re.IGNORECASE):
            patterns.append({"type": "api_key", "severity": "critical"})
        
        # Passwords
        if re.search(r"(password|pwd)[\"\':\s]*[\"\']\w{6,}[\"\']\b", content, re.IGNORECASE):
            patterns.append({"type": "password", "severity": "critical"})
        
        # Email addresses
        if re.search(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", content):
            patterns.append({"type": "email", "severity": "medium"})
        
        # Credit card numbers (simplified)
        if re.search(r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b", content):
            patterns.append({"type": "credit_card", "severity": "high"})
        
        # Social Security Numbers (US format)
        if re.search(r"\b\d{3}-\d{2}-\d{4}\b", content):
            patterns.append({"type": "ssn", "severity": "high"})
        
        return patterns
    
    def _detect_malicious_content(self, content: str) -> List[str]:
        """Detect malicious content patterns."""
        malicious_patterns = []
        
        # Check for executable code patterns
        if re.search(r"(?i)(eval|exec|system|shell_exec)\s*\(", content):
            malicious_patterns.append("Executable code pattern")
        
        # Check for network communication patterns
        if re.search(r"(?i)(curl|wget|fetch|XMLHttpRequest)", content):
            malicious_patterns.append("Network communication pattern")
        
        # Check for file system access patterns
        if re.search(r"(?i)(file_get_contents|fopen|readfile|include|require)", content):
            malicious_patterns.append("File system access pattern")
        
        return malicious_patterns
    
    def _assess_content_quality(self, content: str) -> List[str]:
        """Assess content quality issues."""
        issues = []
        
        # Check for excessive repetition
        words = content.split()
        if len(words) > 10:
            unique_words = set(words)
            repetition_ratio = len(unique_words) / len(words)
            if repetition_ratio < 0.3:
                issues.append("Excessive word repetition")
        
        # Check for proper text structure
        if len(content) > 100 and not re.search(r"[.!?]", content):
            issues.append("No sentence punctuation")
        
        # Check for excessive special characters
        special_char_count = len(re.findall(r"[^a-zA-Z0-9\s]", content))
        if special_char_count > len(content) * 0.5:
            issues.append("Excessive special characters")
        
        return issues
    
    def _sanitize_query(self, query: str) -> str:
        """Sanitize user query."""
        sanitized = query
        
        # Remove potential script tags
        sanitized = re.sub(r"<script[^>]*>.*?</script>", "", sanitized, flags=re.IGNORECASE | re.DOTALL)
        
        # Remove potential SQL injection patterns
        sanitized = re.sub(r"(?i)(\bUNION\b.*\bSELECT\b)", "", sanitized)
        sanitized = re.sub(r"(?i)(\;\s*(SELECT|INSERT|UPDATE|DELETE|DROP))", "", sanitized)
        
        # Remove control characters
        sanitized = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", sanitized)
        
        # Normalize whitespace
        sanitized = re.sub(r"\s+", " ", sanitized).strip()
        
        # Limit length
        if len(sanitized) > 10000:
            sanitized = sanitized[:10000]
        
        return sanitized
    
    def _sanitize_document_content(self, content: str) -> str:
        """Sanitize document content."""
        sanitized = content
        
        # Remove potential API keys and secrets
        sanitized = re.sub(r"(api[_-]?key|token|secret)[\"\':\s]*[\"\']\w{20,}[\"\']\b", 
                          "[REDACTED]", sanitized, flags=re.IGNORECASE)
        
        # Remove potential passwords
        sanitized = re.sub(r"(password|pwd)[\"\':\s]*[\"\']\w{6,}[\"\']\b", 
                          "[REDACTED]", sanitized, flags=re.IGNORECASE)
        
        # Remove credit card numbers
        sanitized = re.sub(r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b", 
                          "[REDACTED-CC]", sanitized)
        
        # Remove SSNs
        sanitized = re.sub(r"\b\d{3}-\d{2}-\d{4}\b", "[REDACTED-SSN]", sanitized)
        
        # Remove control characters
        sanitized = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", sanitized)
        
        return sanitized
    
    def _validate_parameter(self, name: str, value: Any, rule: Dict[str, Any]) -> Dict[str, Any]:
        """Validate individual parameter."""
        result = {
            "is_valid": True,
            "errors": [],
            "sanitized_value": value,
            "threat_level": SecurityThreatLevel.LOW
        }
        
        # Type validation
        expected_type = rule.get("type")
        if expected_type and not isinstance(value, expected_type):
            try:
                # Attempt type conversion
                if expected_type == int:
                    result["sanitized_value"] = int(value)
                elif expected_type == float:
                    result["sanitized_value"] = float(value)
                elif expected_type == str:
                    result["sanitized_value"] = str(value)
                else:
                    result["is_valid"] = False
                    result["errors"].append(f"Parameter {name} must be of type {expected_type.__name__}")
                    result["threat_level"] = SecurityThreatLevel.MEDIUM
                    return result
            except (ValueError, TypeError):
                result["is_valid"] = False
                result["errors"].append(f"Parameter {name} cannot be converted to {expected_type.__name__}")
                result["threat_level"] = SecurityThreatLevel.MEDIUM
                return result
        
        # String validations
        if isinstance(result["sanitized_value"], str):
            # Length validation
            max_length = rule.get("max_length")
            if max_length and len(result["sanitized_value"]) > max_length:
                result["sanitized_value"] = result["sanitized_value"][:max_length]
                result["errors"].append(f"Parameter {name} truncated to {max_length} characters")
            
            # Pattern validation
            pattern = rule.get("pattern")
            if pattern and not re.match(pattern, result["sanitized_value"]):
                result["is_valid"] = False
                result["errors"].append(f"Parameter {name} does not match required pattern")
                result["threat_level"] = SecurityThreatLevel.MEDIUM
            
            # Allowed values validation
            allowed_values = rule.get("allowed_values")
            if allowed_values and result["sanitized_value"] not in allowed_values:
                result["is_valid"] = False
                result["errors"].append(f"Parameter {name} must be one of: {allowed_values}")
        
        # Numeric validations
        if isinstance(result["sanitized_value"], (int, float)):
            # Range validation
            min_val = rule.get("min")
            max_val = rule.get("max")
            
            if min_val is not None and result["sanitized_value"] < min_val:
                result["sanitized_value"] = min_val
                result["errors"].append(f"Parameter {name} clamped to minimum value {min_val}")
            
            if max_val is not None and result["sanitized_value"] > max_val:
                result["sanitized_value"] = max_val
                result["errors"].append(f"Parameter {name} clamped to maximum value {max_val}")
        
        # Dictionary validation
        if isinstance(result["sanitized_value"], dict):
            max_depth = rule.get("max_depth", 5)
            if self._dict_depth(result["sanitized_value"]) > max_depth:
                result["is_valid"] = False
                result["errors"].append(f"Parameter {name} exceeds maximum depth {max_depth}")
                result["threat_level"] = SecurityThreatLevel.MEDIUM
        
        return result
    
    def _dict_depth(self, d: Dict[str, Any], depth: int = 0) -> int:
        """Calculate dictionary depth."""
        if not isinstance(d, dict) or not d:
            return depth
        return max(self._dict_depth(v, depth + 1) for v in d.values())
    
    def _get_threat_level(self, threat_type: str) -> SecurityThreatLevel:
        """Get threat level for threat type."""
        threat_levels = {
            "sql_injection": SecurityThreatLevel.HIGH,
            "xss": SecurityThreatLevel.HIGH,
            "command_injection": SecurityThreatLevel.CRITICAL,
            "path_traversal": SecurityThreatLevel.MEDIUM
        }
        return threat_levels.get(threat_type, SecurityThreatLevel.LOW)
    
    def _final_validation_check(self, threat_level: SecurityThreatLevel, errors: List[str]) -> bool:
        """Perform final validation check based on validation level."""
        if self.validation_level == ValidationLevel.PARANOID:
            return threat_level == SecurityThreatLevel.LOW and len(errors) == 0
        elif self.validation_level == ValidationLevel.STRICT:
            return threat_level in [SecurityThreatLevel.LOW, SecurityThreatLevel.MEDIUM] and len(errors) == 0
        elif self.validation_level == ValidationLevel.STANDARD:
            return threat_level != SecurityThreatLevel.CRITICAL
        else:  # BASIC
            return threat_level != SecurityThreatLevel.CRITICAL or len(errors) < 3


class RobustErrorHandler:
    """Comprehensive error handling and recovery system."""
    
    def __init__(self):
        self.error_handlers = {}
        self.recovery_strategies = {}
        self.error_metrics = defaultdict(int)
        self.circuit_breakers = {}
        self._lock = threading.Lock()
        
    def register_error_handler(self, error_type: type, handler: Callable[[Exception, ErrorContext], Any]):
        """Register custom error handler for specific error types."""
        self.error_handlers[error_type] = handler
    
    def register_recovery_strategy(self, component: str, strategy: Callable[[Exception, ErrorContext], Any]):
        """Register recovery strategy for specific components."""
        self.recovery_strategies[component] = strategy
    
    @contextmanager
    def error_context(self, component: str, operation: str, **kwargs):
        """Context manager for error handling with automatic recovery."""
        context = ErrorContext(
            component=component,
            operation=operation,
            user_id=kwargs.get("user_id"),
            session_id=kwargs.get("session_id"),
            request_id=kwargs.get("request_id")
        )
        
        try:
            yield context
        except Exception as e:
            self._handle_error(e, context)
            raise
    
    def _handle_error(self, error: Exception, context: ErrorContext):
        """Handle error with appropriate strategy."""
        with self._lock:
            # Record error metrics
            error_key = f"{context.component}.{type(error).__name__}"
            self.error_metrics[error_key] += 1
            
            # Add stack trace to context
            context.stack_trace = traceback.format_exc()
            
            # Log error
            logger.error(
                f"Error in {context.component}.{context.operation}: {str(error)}",
                extra={
                    "component": context.component,
                    "operation": context.operation,
                    "error_type": type(error).__name__,
                    "user_id": context.user_id,
                    "session_id": context.session_id,
                    "request_id": context.request_id
                }
            )
            
            # Try specific error handler
            error_type = type(error)
            if error_type in self.error_handlers:
                try:
                    self.error_handlers[error_type](error, context)
                except Exception as handler_error:
                    logger.error(f"Error handler failed: {handler_error}")
            
            # Try component recovery strategy
            if context.component in self.recovery_strategies:
                try:
                    self.recovery_strategies[context.component](error, context)
                except Exception as recovery_error:
                    logger.error(f"Recovery strategy failed: {recovery_error}")
            
            # Update circuit breaker
            self._update_circuit_breaker(context.component, False)
    
    def _update_circuit_breaker(self, component: str, success: bool):
        """Update circuit breaker state."""
        if component not in self.circuit_breakers:
            self.circuit_breakers[component] = {
                "failures": 0,
                "last_failure": None,
                "state": "closed"  # closed, open, half-open
            }
        
        breaker = self.circuit_breakers[component]
        
        if success:
            breaker["failures"] = 0
            breaker["state"] = "closed"
        else:
            breaker["failures"] += 1
            breaker["last_failure"] = datetime.now()
            
            # Open circuit after 5 failures
            if breaker["failures"] >= 5:
                breaker["state"] = "open"
    
    def is_circuit_open(self, component: str) -> bool:
        """Check if circuit breaker is open for component."""
        if component not in self.circuit_breakers:
            return False
        
        breaker = self.circuit_breakers[component]
        
        if breaker["state"] == "open":
            # Check if we should try half-open
            if breaker["last_failure"]:
                time_since_failure = datetime.now() - breaker["last_failure"]
                if time_since_failure > timedelta(minutes=5):
                    breaker["state"] = "half-open"
                    return False
            return True
        
        return False
    
    def get_error_metrics(self) -> Dict[str, Any]:
        """Get error metrics and statistics."""
        with self._lock:
            return {
                "error_counts": dict(self.error_metrics),
                "circuit_breakers": self.circuit_breakers.copy(),
                "total_errors": sum(self.error_metrics.values()),
                "components_with_errors": len(set(
                    key.split('.')[0] for key in self.error_metrics.keys()
                ))
            }


# Global instances
_robust_validator = None
_robust_error_handler = None

def get_robust_validator(validation_level: ValidationLevel = ValidationLevel.STANDARD) -> RobustValidator:
    """Get global robust validator instance."""
    global _robust_validator
    if _robust_validator is None:
        _robust_validator = RobustValidator(validation_level)
    return _robust_validator

def get_robust_error_handler() -> RobustErrorHandler:
    """Get global robust error handler instance."""
    global _robust_error_handler
    if _robust_error_handler is None:
        _robust_error_handler = RobustErrorHandler()
    return _robust_error_handler

# Utility functions for external use
def validate_query(query: str, context: Optional[Dict[str, Any]] = None) -> ValidationResult:
    """Validate user query with comprehensive security checks."""
    validator = get_robust_validator()
    return validator.validate_query_input(query, context)

def validate_document(content: str, metadata: Optional[Dict[str, Any]] = None) -> ValidationResult:
    """Validate document content for ingestion."""
    validator = get_robust_validator()
    return validator.validate_document_content(content, metadata)

def validate_api_params(params: Dict[str, Any]) -> ValidationResult:
    """Validate API parameters."""
    validator = get_robust_validator()
    return validator.validate_api_parameters(params)