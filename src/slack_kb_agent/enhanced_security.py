"""
Enhanced security features for Slack KB Agent.
Implements comprehensive security measures, input validation, and threat detection.
"""

import logging
import re
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class ThreatLevel(Enum):
    """Threat severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SecurityEvent:
    """Represents a security event for logging and analysis."""
    event_type: str
    threat_level: ThreatLevel
    timestamp: datetime
    source_ip: Optional[str] = None
    user_id: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class SecurityFilter:
    """Advanced security filtering and threat detection."""

    # Common attack patterns
    SQL_INJECTION_PATTERNS = [
        r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION)\b)",
        r"(--|\/\*|\*\/)",
        r"(\b(OR|AND)\s+\d+\s*=\s*\d+)",
        r"('|\")(\s)*(OR|AND)(\s)*('|\")",
    ]

    XSS_PATTERNS = [
        r"<script[^>]*>.*?</script>",
        r"javascript:",
        r"vbscript:",
        r"on\w+\s*=",
        r"<iframe[^>]*>.*?</iframe>",
    ]

    COMMAND_INJECTION_PATTERNS = [
        r"[;&|`$(){}[\]\\]",
        r"\b(cat|ls|rm|mv|cp|chmod|chown|ps|kill|wget|curl)\b",
        r"\.\.\/",
        r"/etc/passwd",
        r"/bin/",
    ]

    SENSITIVE_DATA_PATTERNS = [
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # Email
        r"\b\d{3}-\d{2}-\d{4}\b",  # SSN
        r"\b4[0-9]{12}(?:[0-9]{3})?\b",  # Credit card (Visa)
        r"\b5[1-5][0-9]{14}\b",  # Credit card (MasterCard)
        r"(?i)(password|passwd|pwd|secret|token|key)\s*[:=]\s*\S+",
    ]

    def __init__(self):
        self.security_events: List[SecurityEvent] = []
        self.blocked_ips: Set[str] = set()
        self.rate_limits: Dict[str, List[datetime]] = {}

    def scan_for_threats(self, content: str, source_context: Optional[Dict[str, Any]] = None) -> List[SecurityEvent]:
        """
        Scan content for security threats.
        
        Args:
            content: Content to scan
            source_context: Additional context about the source
            
        Returns:
            List of detected security events
        """
        events = []

        # SQL Injection detection
        for pattern in self.SQL_INJECTION_PATTERNS:
            if re.search(pattern, content, re.IGNORECASE):
                events.append(SecurityEvent(
                    event_type="sql_injection_attempt",
                    threat_level=ThreatLevel.HIGH,
                    timestamp=datetime.utcnow(),
                    details={"pattern": pattern, "content_sample": content[:100]}
                ))

        # XSS detection
        for pattern in self.XSS_PATTERNS:
            if re.search(pattern, content, re.IGNORECASE):
                events.append(SecurityEvent(
                    event_type="xss_attempt",
                    threat_level=ThreatLevel.HIGH,
                    timestamp=datetime.utcnow(),
                    details={"pattern": pattern, "content_sample": content[:100]}
                ))

        # Command injection detection
        for pattern in self.COMMAND_INJECTION_PATTERNS:
            if re.search(pattern, content, re.IGNORECASE):
                events.append(SecurityEvent(
                    event_type="command_injection_attempt",
                    threat_level=ThreatLevel.CRITICAL,
                    timestamp=datetime.utcnow(),
                    details={"pattern": pattern, "content_sample": content[:100]}
                ))

        # Sensitive data detection
        for pattern in self.SENSITIVE_DATA_PATTERNS:
            if re.search(pattern, content, re.IGNORECASE):
                events.append(SecurityEvent(
                    event_type="sensitive_data_detected",
                    threat_level=ThreatLevel.MEDIUM,
                    timestamp=datetime.utcnow(),
                    details={"pattern": pattern, "content_sample": "[REDACTED]"}
                ))

        # Record events
        self.security_events.extend(events)

        return events

    def sanitize_content(self, content: str) -> str:
        """
        Sanitize content by removing or escaping dangerous elements.
        
        Args:
            content: Content to sanitize
            
        Returns:
            Sanitized content
        """
        if not content:
            return content

        # Remove script tags
        content = re.sub(r"<script[^>]*>.*?</script>", "", content, flags=re.IGNORECASE | re.DOTALL)

        # Remove dangerous event handlers
        content = re.sub(r"on\w+\s*=\s*[\"'][^\"']*[\"']", "", content, flags=re.IGNORECASE)

        # Escape HTML special characters
        content = content.replace("<", "&lt;").replace(">", "&gt;")
        content = content.replace("\"", "&quot;").replace("'", "&#x27;")

        # Remove potential command injection characters
        dangerous_chars = [";", "&", "|", "`", "$"]
        for char in dangerous_chars:
            content = content.replace(char, "")

        return content

    def check_rate_limit(self, identifier: str, max_requests: int = 100,
                        time_window: int = 3600) -> bool:
        """
        Check if identifier exceeds rate limit.
        
        Args:
            identifier: Unique identifier (IP, user ID, etc.)
            max_requests: Maximum requests in time window
            time_window: Time window in seconds
            
        Returns:
            True if within rate limit, False if exceeded
        """
        now = datetime.utcnow()
        cutoff = now - timedelta(seconds=time_window)

        # Initialize or clean old entries
        if identifier not in self.rate_limits:
            self.rate_limits[identifier] = []

        self.rate_limits[identifier] = [
            timestamp for timestamp in self.rate_limits[identifier]
            if timestamp > cutoff
        ]

        # Check current rate
        if len(self.rate_limits[identifier]) >= max_requests:
            self.security_events.append(SecurityEvent(
                event_type="rate_limit_exceeded",
                threat_level=ThreatLevel.MEDIUM,
                timestamp=now,
                details={"identifier": identifier, "request_count": len(self.rate_limits[identifier])}
            ))
            return False

        # Record this request
        self.rate_limits[identifier].append(now)
        return True

    def mask_sensitive_data(self, content: str) -> str:
        """
        Mask sensitive data in content for safe logging/storage.
        
        Args:
            content: Content potentially containing sensitive data
            
        Returns:
            Content with sensitive data masked
        """
        # Mask emails
        content = re.sub(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "[EMAIL_REDACTED]",
            content
        )

        # Mask SSNs
        content = re.sub(r"\b\d{3}-\d{2}-\d{4}\b", "[SSN_REDACTED]", content)

        # Mask credit cards
        content = re.sub(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b", "[CARD_REDACTED]", content)

        # Mask passwords/tokens
        content = re.sub(
            r"(?i)(password|passwd|pwd|secret|token|key)\s*[:=]\s*\S+",
            r"\1=[REDACTED]",
            content
        )

        return content


class AccessControl:
    """Implements role-based access control and permissions."""

    def __init__(self):
        self.user_roles: Dict[str, Set[str]] = {}
        self.role_permissions: Dict[str, Set[str]] = {
            "admin": {"read", "write", "delete", "manage_users", "view_analytics"},
            "moderator": {"read", "write", "delete", "view_analytics"},
            "user": {"read", "write"},
            "readonly": {"read"},
        }
        self.resource_restrictions: Dict[str, Set[str]] = {}

    def assign_role(self, user_id: str, role: str) -> bool:
        """Assign role to user."""
        if role not in self.role_permissions:
            logger.warning(f"Attempted to assign unknown role: {role}")
            return False

        if user_id not in self.user_roles:
            self.user_roles[user_id] = set()

        self.user_roles[user_id].add(role)
        logger.info(f"Assigned role {role} to user {user_id}")
        return True

    def check_permission(self, user_id: str, permission: str, resource: Optional[str] = None) -> bool:
        """
        Check if user has specific permission.
        
        Args:
            user_id: User identifier
            permission: Permission to check
            resource: Optional resource identifier
            
        Returns:
            True if user has permission, False otherwise
        """
        user_roles = self.user_roles.get(user_id, set())

        # Check if any user role has the required permission
        for role in user_roles:
            role_perms = self.role_permissions.get(role, set())
            if permission in role_perms:
                # Check resource-specific restrictions
                if resource and resource in self.resource_restrictions:
                    if user_id not in self.resource_restrictions[resource]:
                        continue
                return True

        return False

    def restrict_resource(self, resource: str, allowed_users: List[str]):
        """Restrict access to specific resource."""
        self.resource_restrictions[resource] = set(allowed_users)


class EncryptionManager:
    """Handles encryption and decryption of sensitive data."""

    def __init__(self, master_key: Optional[str] = None):
        self.master_key = master_key or self._generate_key()

    def _generate_key(self) -> str:
        """Generate a secure random key."""
        return secrets.token_hex(32)

    def encrypt_data(self, data: str) -> str:
        """
        Encrypt sensitive data.
        
        Args:
            data: Data to encrypt
            
        Returns:
            Encrypted data as hex string
        """
        # Simple XOR encryption for demo (use proper encryption in production)
        key_bytes = self.master_key.encode()
        data_bytes = data.encode()

        encrypted = bytearray()
        for i, byte in enumerate(data_bytes):
            encrypted.append(byte ^ key_bytes[i % len(key_bytes)])

        return encrypted.hex()

    def decrypt_data(self, encrypted_data: str) -> str:
        """
        Decrypt encrypted data.
        
        Args:
            encrypted_data: Encrypted data as hex string
            
        Returns:
            Decrypted data
        """
        try:
            key_bytes = self.master_key.encode()
            encrypted_bytes = bytes.fromhex(encrypted_data)

            decrypted = bytearray()
            for i, byte in enumerate(encrypted_bytes):
                decrypted.append(byte ^ key_bytes[i % len(key_bytes)])

            return decrypted.decode()
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            return ""


class SecurityAuditor:
    """Performs security audits and generates reports."""

    def __init__(self, security_filter: SecurityFilter):
        self.security_filter = security_filter

    def generate_security_report(self, time_range: int = 86400) -> Dict[str, Any]:
        """
        Generate comprehensive security report.
        
        Args:
            time_range: Time range in seconds for report
            
        Returns:
            Security report dictionary
        """
        cutoff = datetime.utcnow() - timedelta(seconds=time_range)
        recent_events = [
            event for event in self.security_filter.security_events
            if event.timestamp > cutoff
        ]

        # Categorize events by type and threat level
        event_types = {}
        threat_levels = {}

        for event in recent_events:
            event_types[event.event_type] = event_types.get(event.event_type, 0) + 1
            threat_levels[event.threat_level.value] = threat_levels.get(event.threat_level.value, 0) + 1

        # Calculate security score (0-100)
        total_events = len(recent_events)
        critical_events = threat_levels.get("critical", 0)
        high_events = threat_levels.get("high", 0)

        if total_events == 0:
            security_score = 100
        else:
            # Deduct points based on threat severity
            deductions = (critical_events * 20) + (high_events * 10) + (total_events * 2)
            security_score = max(0, 100 - deductions)

        return {
            "report_timestamp": datetime.utcnow().isoformat(),
            "time_range_hours": time_range / 3600,
            "total_events": total_events,
            "security_score": security_score,
            "events_by_type": event_types,
            "events_by_threat_level": threat_levels,
            "blocked_ips": list(self.security_filter.blocked_ips),
            "recommendations": self._generate_recommendations(recent_events)
        }

    def _generate_recommendations(self, events: List[SecurityEvent]) -> List[str]:
        """Generate security recommendations based on events."""
        recommendations = []

        event_types = {event.event_type for event in events}

        if "sql_injection_attempt" in event_types:
            recommendations.append("Implement stricter input validation for database queries")

        if "xss_attempt" in event_types:
            recommendations.append("Review and enhance content sanitization")

        if "command_injection_attempt" in event_types:
            recommendations.append("Restrict system command execution and validate inputs")

        if "rate_limit_exceeded" in event_types:
            recommendations.append("Consider implementing additional rate limiting measures")

        if "sensitive_data_detected" in event_types:
            recommendations.append("Implement data loss prevention measures")

        return recommendations


# Global security instances
security_filter = SecurityFilter()
access_control = AccessControl()
encryption_manager = EncryptionManager()
security_auditor = SecurityAuditor(security_filter)


def secure_function(required_permission: str = "read"):
    """
    Decorator to secure functions with permission checks.
    
    Args:
        required_permission: Permission required to execute function
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Extract user_id from kwargs or args
            user_id = kwargs.get('user_id') or (args[0] if args else None)

            if not user_id:
                raise PermissionError("User ID required for security check")

            if not access_control.check_permission(user_id, required_permission):
                raise PermissionError(f"User {user_id} lacks permission: {required_permission}")

            return func(*args, **kwargs)
        return wrapper
    return decorator


def get_security_status() -> Dict[str, Any]:
    """Get current security system status."""
    return {
        "security_filter": {
            "total_events": len(security_filter.security_events),
            "blocked_ips": len(security_filter.blocked_ips),
            "active_rate_limits": len(security_filter.rate_limits)
        },
        "access_control": {
            "total_users": len(access_control.user_roles),
            "total_roles": len(access_control.role_permissions),
            "restricted_resources": len(access_control.resource_restrictions)
        },
        "latest_report": security_auditor.generate_security_report(3600)  # Last hour
    }
