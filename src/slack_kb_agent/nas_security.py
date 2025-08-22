"""Security and validation layer for Neural Architecture Search system.

This module provides comprehensive security validation, input sanitization,
and protection against malicious architecture configurations.
"""

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np

from .exceptions import (
    SecurityViolationError,
    ValidationError,
    ResourceLimitExceededError
)

logger = logging.getLogger(__name__)


class SecurityLevel(Enum):
    """Security validation levels."""
    PERMISSIVE = "permissive"
    STANDARD = "standard"
    STRICT = "strict"
    PARANOID = "paranoid"


class ThreatCategory(Enum):
    """Categories of security threats."""
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    CODE_INJECTION = "code_injection"
    DATA_EXFILTRATION = "data_exfiltration"
    CONFIGURATION_TAMPERING = "configuration_tampering"
    DENIAL_OF_SERVICE = "denial_of_service"


@dataclass
class SecurityViolation:
    """Represents a security violation found during validation."""
    threat_category: ThreatCategory
    severity: str  # "low", "medium", "high", "critical"
    description: str
    field_path: str
    value: Any
    recommended_action: str


@dataclass
class ValidationResult:
    """Result of security validation."""
    is_valid: bool
    violations: List[SecurityViolation]
    sanitized_config: Optional[Dict[str, Any]] = None
    risk_score: float = 0.0  # 0-100 scale


class NASSecurityValidator:
    """Security validator for NAS configurations."""
    
    def __init__(self, security_level: SecurityLevel = SecurityLevel.STANDARD):
        self.security_level = security_level
        self.max_violations = self._get_max_violations()
        self.resource_limits = self._get_resource_limits()
        self.blacklisted_patterns = self._get_blacklisted_patterns()
        self.whitelist_only_fields = self._get_whitelist_only_fields()
        
        logger.info(f"Initialized NAS security validator with {security_level.value} level")
    
    def validate_architecture_config(self, config: Dict[str, Any]) -> ValidationResult:
        """Validate architecture configuration for security issues."""
        violations = []
        sanitized_config = config.copy()
        
        try:
            # Input sanitization
            violations.extend(self._validate_input_sanitization(config))
            
            # Resource limits validation
            violations.extend(self._validate_resource_limits(config))
            
            # Configuration tampering detection
            violations.extend(self._validate_configuration_integrity(config))
            
            # Code injection prevention
            violations.extend(self._validate_code_injection(config))
            
            # Data exfiltration prevention
            violations.extend(self._validate_data_access(config))
            
            # Apply sanitization if violations found but not critical
            if violations and not self._has_critical_violations(violations):
                sanitized_config = self._sanitize_config(config, violations)
            
            # Calculate risk score
            risk_score = self._calculate_risk_score(violations)
            
            # Determine if configuration is acceptable
            is_valid = self._is_configuration_acceptable(violations, risk_score)
            
            return ValidationResult(
                is_valid=is_valid,
                violations=violations,
                sanitized_config=sanitized_config if is_valid else None,
                risk_score=risk_score
            )
            
        except Exception as e:
            logger.error(f"Security validation failed: {e}")
            return ValidationResult(
                is_valid=False,
                violations=[SecurityViolation(
                    threat_category=ThreatCategory.CONFIGURATION_TAMPERING,
                    severity="critical",
                    description=f"Validation process failed: {str(e)}",
                    field_path="__root__",
                    value=None,
                    recommended_action="Reject configuration and investigate"
                )],
                risk_score=100.0
            )
    
    def _validate_input_sanitization(self, config: Dict[str, Any]) -> List[SecurityViolation]:
        """Validate input sanitization and detect malicious patterns."""
        violations = []
        
        def check_value(key_path: str, value: Any) -> None:
            if isinstance(value, str):
                # Check for suspicious patterns
                for pattern, threat in self.blacklisted_patterns.items():
                    if re.search(pattern, value, re.IGNORECASE):
                        violations.append(SecurityViolation(
                            threat_category=ThreatCategory.CODE_INJECTION,
                            severity="high",
                            description=f"Suspicious pattern detected: {pattern}",
                            field_path=key_path,
                            value=value,
                            recommended_action="Remove or sanitize the suspicious content"
                        ))
                
                # Check string length
                if len(value) > 10000:  # 10KB limit for strings
                    violations.append(SecurityViolation(
                        threat_category=ThreatCategory.DENIAL_OF_SERVICE,
                        severity="medium",
                        description="Excessively long string value",
                        field_path=key_path,
                        value=f"String of length {len(value)}",
                        recommended_action="Truncate to reasonable length"
                    ))
            
            elif isinstance(value, dict):
                for k, v in value.items():
                    check_value(f"{key_path}.{k}", v)
            
            elif isinstance(value, list):
                for i, v in enumerate(value):
                    check_value(f"{key_path}[{i}]", v)
        
        for key, value in config.items():
            check_value(key, value)
        
        return violations
    
    def _validate_resource_limits(self, config: Dict[str, Any]) -> List[SecurityViolation]:
        """Validate resource consumption limits."""
        violations = []
        
        # Check parameter count
        estimated_params = self._estimate_parameter_count(config)
        if estimated_params > self.resource_limits["max_parameters"]:
            violations.append(SecurityViolation(
                threat_category=ThreatCategory.RESOURCE_EXHAUSTION,
                severity="high",
                description=f"Parameter count {estimated_params:,} exceeds limit",
                field_path="parameter_count",
                value=estimated_params,
                recommended_action="Reduce model size parameters"
            ))
        
        # Check memory usage
        estimated_memory_gb = self._estimate_memory_usage(config)
        if estimated_memory_gb > self.resource_limits["max_memory_gb"]:
            violations.append(SecurityViolation(
                threat_category=ThreatCategory.RESOURCE_EXHAUSTION,
                severity="high",
                description=f"Memory usage {estimated_memory_gb:.1f}GB exceeds limit",
                field_path="memory_usage",
                value=estimated_memory_gb,
                recommended_action="Reduce batch size or model dimensions"
            ))
        
        # Check numerical ranges
        numerical_limits = {
            "num_layers": (1, 1000),
            "hidden_size": (32, 16384),
            "num_heads": (1, 128),
            "max_sequence_length": (1, 32768),
            "batch_size": (1, 2048),
            "vocab_size": (100, 1000000)
        }
        
        for field, (min_val, max_val) in numerical_limits.items():
            if field in config:
                value = config[field]
                if not isinstance(value, (int, float)) or value < min_val or value > max_val:
                    violations.append(SecurityViolation(
                        threat_category=ThreatCategory.CONFIGURATION_TAMPERING,
                        severity="medium",
                        description=f"{field} value {value} outside valid range [{min_val}, {max_val}]",
                        field_path=field,
                        value=value,
                        recommended_action=f"Set {field} within valid range"
                    ))
        
        return violations
    
    def _validate_configuration_integrity(self, config: Dict[str, Any]) -> List[SecurityViolation]:
        """Validate configuration integrity and detect tampering."""
        violations = []
        
        # Check for required fields
        required_fields = {"architecture_type", "num_layers", "hidden_size"}
        missing_fields = required_fields - set(config.keys())
        
        if missing_fields:
            violations.append(SecurityViolation(
                threat_category=ThreatCategory.CONFIGURATION_TAMPERING,
                severity="high",
                description=f"Missing required fields: {missing_fields}",
                field_path="__required_fields__",
                value=list(missing_fields),
                recommended_action="Provide all required configuration fields"
            ))
        
        # Check for unknown fields (only in strict/paranoid mode)
        if self.security_level in [SecurityLevel.STRICT, SecurityLevel.PARANOID]:
            allowed_fields = self.whitelist_only_fields
            unknown_fields = set(config.keys()) - allowed_fields
            
            if unknown_fields:
                severity = "medium" if self.security_level == SecurityLevel.STRICT else "high"
                violations.append(SecurityViolation(
                    threat_category=ThreatCategory.CONFIGURATION_TAMPERING,
                    severity=severity,
                    description=f"Unknown configuration fields: {unknown_fields}",
                    field_path="__unknown_fields__",
                    value=list(unknown_fields),
                    recommended_action="Remove unknown fields or update whitelist"
                ))
        
        # Validate configuration consistency
        if "num_heads" in config and "hidden_size" in config:
            if config["hidden_size"] % config["num_heads"] != 0:
                violations.append(SecurityViolation(
                    threat_category=ThreatCategory.CONFIGURATION_TAMPERING,
                    severity="medium",
                    description="hidden_size must be divisible by num_heads",
                    field_path="hidden_size,num_heads",
                    value=(config["hidden_size"], config["num_heads"]),
                    recommended_action="Adjust hidden_size or num_heads for divisibility"
                ))
        
        return violations
    
    def _validate_code_injection(self, config: Dict[str, Any]) -> List[SecurityViolation]:
        """Validate against code injection attempts."""
        violations = []
        
        # Check for function/method calls in string values
        dangerous_patterns = [
            r'__[a-zA-Z_]+__',  # Python dunder methods
            r'eval\s*\(',       # eval function
            r'exec\s*\(',       # exec function
            r'import\s+',       # import statements
            r'subprocess\.',    # subprocess calls
            r'os\.',           # os module calls
            r'sys\.',          # sys module calls
            r'open\s*\(',      # file operations
        ]
        
        def check_injection(key_path: str, value: Any) -> None:
            if isinstance(value, str):
                for pattern in dangerous_patterns:
                    if re.search(pattern, value, re.IGNORECASE):
                        violations.append(SecurityViolation(
                            threat_category=ThreatCategory.CODE_INJECTION,
                            severity="critical",
                            description=f"Potential code injection pattern: {pattern}",
                            field_path=key_path,
                            value=value,
                            recommended_action="Remove or escape dangerous code patterns"
                        ))
            
            elif isinstance(value, dict):
                for k, v in value.items():
                    check_injection(f"{key_path}.{k}", v)
            
            elif isinstance(value, list):
                for i, v in enumerate(value):
                    check_injection(f"{key_path}[{i}]", v)
        
        for key, value in config.items():
            check_injection(key, value)
        
        return violations
    
    def _validate_data_access(self, config: Dict[str, Any]) -> List[SecurityViolation]:
        """Validate against data exfiltration attempts."""
        violations = []
        
        # Check for suspicious file paths
        suspicious_paths = [
            r'/etc/',
            r'/proc/',
            r'/sys/',
            r'C:\\Windows',
            r'\\\\[^\\]+\\',  # UNC paths
            r'\.\./',         # Directory traversal
            r'~/',           # Home directory
        ]
        
        def check_paths(key_path: str, value: Any) -> None:
            if isinstance(value, str):
                for path_pattern in suspicious_paths:
                    if re.search(path_pattern, value, re.IGNORECASE):
                        violations.append(SecurityViolation(
                            threat_category=ThreatCategory.DATA_EXFILTRATION,
                            severity="high",
                            description=f"Suspicious file path pattern: {path_pattern}",
                            field_path=key_path,
                            value=value,
                            recommended_action="Remove or validate file path access"
                        ))
        
        # Check all string values for suspicious paths
        def recursive_check(obj: Any, path: str = "") -> None:
            if isinstance(obj, dict):
                for k, v in obj.items():
                    recursive_check(v, f"{path}.{k}" if path else k)
            elif isinstance(obj, list):
                for i, v in enumerate(obj):
                    recursive_check(v, f"{path}[{i}]")
            elif isinstance(obj, str):
                check_paths(path, obj)
        
        recursive_check(config)
        
        return violations
    
    def _sanitize_config(self, config: Dict[str, Any], violations: List[SecurityViolation]) -> Dict[str, Any]:
        """Sanitize configuration by fixing non-critical violations."""
        sanitized = config.copy()
        
        for violation in violations:
            if violation.severity in ["low", "medium"] and violation.field_path in sanitized:
                # Apply sanitization based on violation type
                if violation.threat_category == ThreatCategory.RESOURCE_EXHAUSTION:
                    # Clamp values to safe ranges
                    if "parameter" in violation.field_path.lower():
                        # Reduce parameters by scaling down dimensions
                        if "hidden_size" in sanitized:
                            sanitized["hidden_size"] = min(sanitized["hidden_size"], 2048)
                        if "num_layers" in sanitized:
                            sanitized["num_layers"] = min(sanitized["num_layers"], 48)
                
                elif violation.threat_category == ThreatCategory.CODE_INJECTION:
                    # Remove or escape suspicious content
                    if isinstance(sanitized[violation.field_path], str):
                        # Simple sanitization: remove dangerous patterns
                        sanitized_value = re.sub(r'[<>"\']', '', str(violation.value))
                        sanitized[violation.field_path] = sanitized_value
        
        return sanitized
    
    def _calculate_risk_score(self, violations: List[SecurityViolation]) -> float:
        """Calculate overall risk score from violations."""
        if not violations:
            return 0.0
        
        severity_weights = {
            "low": 10.0,
            "medium": 25.0,
            "high": 50.0,
            "critical": 100.0
        }
        
        total_score = 0.0
        for violation in violations:
            weight = severity_weights.get(violation.severity, 25.0)
            total_score += weight
        
        # Cap at 100.0 and apply security level multiplier
        base_score = min(total_score, 100.0)
        
        level_multipliers = {
            SecurityLevel.PERMISSIVE: 0.7,
            SecurityLevel.STANDARD: 1.0,
            SecurityLevel.STRICT: 1.3,
            SecurityLevel.PARANOID: 1.5
        }
        
        multiplier = level_multipliers.get(self.security_level, 1.0)
        return min(base_score * multiplier, 100.0)
    
    def _is_configuration_acceptable(self, violations: List[SecurityViolation], risk_score: float) -> bool:
        """Determine if configuration is acceptable based on security level."""
        has_critical = self._has_critical_violations(violations)
        
        if has_critical:
            return False
        
        risk_thresholds = {
            SecurityLevel.PERMISSIVE: 80.0,
            SecurityLevel.STANDARD: 60.0,
            SecurityLevel.STRICT: 40.0,
            SecurityLevel.PARANOID: 20.0
        }
        
        threshold = risk_thresholds.get(self.security_level, 60.0)
        return risk_score <= threshold
    
    def _has_critical_violations(self, violations: List[SecurityViolation]) -> bool:
        """Check if any violations are critical."""
        return any(v.severity == "critical" for v in violations)
    
    def _get_max_violations(self) -> int:
        """Get maximum allowed violations based on security level."""
        limits = {
            SecurityLevel.PERMISSIVE: 100,
            SecurityLevel.STANDARD: 50,
            SecurityLevel.STRICT: 20,
            SecurityLevel.PARANOID: 5
        }
        return limits.get(self.security_level, 50)
    
    def _get_resource_limits(self) -> Dict[str, Any]:
        """Get resource limits based on security level."""
        base_limits = {
            "max_parameters": 10_000_000_000,  # 10B parameters
            "max_memory_gb": 256.0,           # 256GB memory
            "max_flops": 1e15,                # 1 petaFLOP
            "max_layers": 1000,
            "max_sequence_length": 32768
        }
        
        # Scale limits based on security level
        scale_factors = {
            SecurityLevel.PERMISSIVE: 2.0,
            SecurityLevel.STANDARD: 1.0,
            SecurityLevel.STRICT: 0.5,
            SecurityLevel.PARANOID: 0.25
        }
        
        scale = scale_factors.get(self.security_level, 1.0)
        return {k: v * scale for k, v in base_limits.items()}
    
    def _get_blacklisted_patterns(self) -> Dict[str, ThreatCategory]:
        """Get blacklisted patterns based on security level."""
        base_patterns = {
            r'<script[^>]*>': ThreatCategory.CODE_INJECTION,
            r'javascript:': ThreatCategory.CODE_INJECTION,
            r'data:text/html': ThreatCategory.CODE_INJECTION,
            r'vbscript:': ThreatCategory.CODE_INJECTION,
            r'onload\s*=': ThreatCategory.CODE_INJECTION,
            r'onerror\s*=': ThreatCategory.CODE_INJECTION,
            r'eval\s*\(': ThreatCategory.CODE_INJECTION,
            r'exec\s*\(': ThreatCategory.CODE_INJECTION,
            r'__import__': ThreatCategory.CODE_INJECTION,
            r'\.\.\/': ThreatCategory.DATA_EXFILTRATION,
            r'\\.\\.\\': ThreatCategory.DATA_EXFILTRATION,
        }
        
        if self.security_level in [SecurityLevel.STRICT, SecurityLevel.PARANOID]:
            # Add more restrictive patterns
            base_patterns.update({
                r'function\s*\(': ThreatCategory.CODE_INJECTION,
                r'=>': ThreatCategory.CODE_INJECTION,
                r'require\s*\(': ThreatCategory.CODE_INJECTION,
                r'module\.': ThreatCategory.CODE_INJECTION,
                r'process\.': ThreatCategory.CODE_INJECTION,
            })
        
        return base_patterns
    
    def _get_whitelist_only_fields(self) -> Set[str]:
        """Get whitelisted configuration fields."""
        return {
            "architecture_type", "num_layers", "hidden_size", "num_heads",
            "max_sequence_length", "batch_size", "vocab_size", "use_residual",
            "use_attention", "use_layer_norm", "use_mixed_precision",
            "dropout_rate", "activation_function", "kernel_sizes", "strides",
            "channels", "learning_rate", "weight_decay", "gradient_clipping",
            "optimizer", "scheduler", "warmup_steps", "max_steps"
        }
    
    def _estimate_parameter_count(self, config: Dict[str, Any]) -> int:
        """Estimate parameter count for resource validation."""
        layers = config.get("num_layers", 12)
        hidden_size = config.get("hidden_size", 768)
        vocab_size = config.get("vocab_size", 50000)
        
        # Simplified estimation
        embedding_params = vocab_size * hidden_size
        layer_params = layers * (4 * hidden_size * hidden_size + 2 * hidden_size)
        output_params = hidden_size * vocab_size
        
        return embedding_params + layer_params + output_params
    
    def _estimate_memory_usage(self, config: Dict[str, Any]) -> float:
        """Estimate memory usage in GB."""
        params = self._estimate_parameter_count(config)
        seq_length = config.get("max_sequence_length", 512)
        batch_size = config.get("batch_size", 32)
        hidden_size = config.get("hidden_size", 768)
        
        # Parameter memory (mixed precision)
        param_memory_gb = params * 2 / (1024**3)  # 2 bytes per parameter
        
        # Activation memory
        activation_memory_gb = batch_size * seq_length * hidden_size * 4 / (1024**3)
        
        # Gradient memory
        gradient_memory_gb = param_memory_gb  # Same as parameters
        
        return param_memory_gb + activation_memory_gb + gradient_memory_gb


class NASSecurityAuditor:
    """Security auditor for NAS system operations."""
    
    def __init__(self):
        self.audit_log: List[Dict[str, Any]] = []
        
    def log_security_event(
        self,
        event_type: str,
        severity: str,
        description: str,
        context: Dict[str, Any]
    ) -> None:
        """Log security event for audit trail."""
        event = {
            "timestamp": time.time(),
            "event_type": event_type,
            "severity": severity,
            "description": description,
            "context": context,
            "event_hash": self._generate_event_hash(event_type, description, context)
        }
        
        self.audit_log.append(event)
        logger.info(f"Security event logged: {event_type} - {description}")
    
    def _generate_event_hash(self, event_type: str, description: str, context: Dict[str, Any]) -> str:
        """Generate hash for event integrity verification."""
        content = f"{event_type}:{description}:{json.dumps(context, sort_keys=True)}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def get_security_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get security summary for specified time period."""
        cutoff_time = time.time() - (hours * 3600)
        recent_events = [e for e in self.audit_log if e["timestamp"] >= cutoff_time]
        
        severity_counts = {}
        event_type_counts = {}
        
        for event in recent_events:
            severity = event["severity"]
            event_type = event["event_type"]
            
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
            event_type_counts[event_type] = event_type_counts.get(event_type, 0) + 1
        
        return {
            "time_period_hours": hours,
            "total_events": len(recent_events),
            "severity_breakdown": severity_counts,
            "event_type_breakdown": event_type_counts,
            "high_risk_events": [e for e in recent_events if e["severity"] in ["high", "critical"]]
        }


# Factory functions
def get_security_validator(security_level: SecurityLevel = SecurityLevel.STANDARD) -> NASSecurityValidator:
    """Get configured security validator instance."""
    return NASSecurityValidator(security_level)


def get_security_auditor() -> NASSecurityAuditor:
    """Get security auditor instance."""
    return NASSecurityAuditor()


# Demo usage
def demo_security_validation():
    """Demonstrate security validation capabilities."""
    validator = get_security_validator(SecurityLevel.STANDARD)
    
    # Test legitimate configuration
    good_config = {
        "architecture_type": "transformer",
        "num_layers": 12,
        "hidden_size": 768,
        "num_heads": 12,
        "max_sequence_length": 512,
        "batch_size": 32,
        "vocab_size": 50000,
        "use_mixed_precision": True
    }
    
    print("🔒 Security Validation Demo")
    print("\n✅ Testing legitimate configuration...")
    result = validator.validate_architecture_config(good_config)
    print(f"Valid: {result.is_valid}, Risk Score: {result.risk_score:.1f}")
    print(f"Violations: {len(result.violations)}")
    
    # Test malicious configuration
    bad_config = {
        "architecture_type": "transformer",
        "num_layers": 50000,  # Resource exhaustion
        "hidden_size": 999999,  # Resource exhaustion
        "activation_function": "eval('malicious_code')",  # Code injection
        "data_path": "/etc/passwd",  # Data exfiltration
        "custom_script": "<script>alert('xss')</script>",  # XSS
        "unknown_field": "suspicious_value"  # Unknown field
    }
    
    print("\n❌ Testing malicious configuration...")
    result = validator.validate_architecture_config(bad_config)
    print(f"Valid: {result.is_valid}, Risk Score: {result.risk_score:.1f}")
    print(f"Violations: {len(result.violations)}")
    
    for violation in result.violations[:3]:  # Show first 3
        print(f"  - {violation.severity.upper()}: {violation.description}")


if __name__ == "__main__":
    import time
    demo_security_validation()