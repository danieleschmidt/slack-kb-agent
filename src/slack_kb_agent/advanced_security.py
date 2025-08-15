"""Advanced security monitoring and threat detection system.

This module provides comprehensive security features including:
- Real-time threat detection and analysis
- Advanced input validation and sanitization
- Security event monitoring and alerting
- Anomaly detection for user behavior
- Rate limiting and abuse prevention
- Security audit logging
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import logging
import re
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class ThreatLevel(Enum):
    """Security threat severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AttackType(Enum):
    """Types of security attacks detected."""
    INJECTION = "injection"               # SQL/Command injection attempts
    XSS = "xss"                          # Cross-site scripting
    TRAVERSAL = "traversal"              # Path traversal attacks
    BRUTE_FORCE = "brute_force"          # Brute force attempts
    RATE_LIMIT = "rate_limit"            # Rate limiting violations
    MALFORMED_INPUT = "malformed_input"   # Suspicious input patterns
    PRIVILEGE_ESCALATION = "privilege_escalation"  # Unauthorized access attempts
    DATA_EXFILTRATION = "data_exfiltration"       # Suspicious data access
    ANOMALOUS_BEHAVIOR = "anomalous_behavior"     # Unusual user patterns


class SecurityEventType(Enum):
    """Types of security events to monitor."""
    AUTHENTICATION_FAILURE = "auth_failure"
    AUTHORIZATION_FAILURE = "authz_failure"
    SUSPICIOUS_QUERY = "suspicious_query"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    MALICIOUS_INPUT = "malicious_input"
    PRIVILEGE_VIOLATION = "privilege_violation"
    DATA_ACCESS_ANOMALY = "data_access_anomaly"
    SYSTEM_COMPROMISE = "system_compromise"


@dataclass
class SecurityEvent:
    """Security event data structure."""
    event_id: str
    event_type: SecurityEventType
    threat_level: ThreatLevel
    attack_type: Optional[AttackType]
    user_id: Optional[str]
    ip_address: Optional[str]
    user_agent: Optional[str]
    query: str
    timestamp: float
    metadata: Dict[str, Any]
    blocked: bool                    # Whether the request was blocked
    confidence_score: float          # 0-1 confidence in threat assessment
    indicators: List[str]           # Specific threat indicators found


@dataclass
class UserBehaviorProfile:
    """User behavior profile for anomaly detection."""
    user_id: str
    query_patterns: Dict[str, int]   # Common query patterns
    access_times: List[float]        # Historical access timestamps
    request_frequency: float         # Requests per hour average
    typical_sources: Set[str]        # Usual IP addresses
    privilege_level: str             # User privilege level
    anomaly_score: float             # Current anomaly score
    last_updated: float


class ThreatDetector:
    """Advanced threat detection system."""

    def __init__(self):
        # Injection patterns
        self.injection_patterns = [
            # SQL injection
            r"(?i)(union\s+select|select\s+.*from|drop\s+table|insert\s+into)",
            r"(?i)(or\s+1\s*=\s*1|and\s+1\s*=\s*1|\'\s*or\s*\'\s*=\s*\')",
            r"(?i)(exec\s*\(|execute\s*\(|sp_executesql)",

            # Command injection
            r"(?i)(\|\s*ls|\|\s*cat|\|\s*wget|\|\s*curl)",
            r"(?i)(rm\s+-rf|del\s+/|format\s+c:)",
            r"(?i)(;.*rm|;.*del|;.*format)",

            # Code injection
            r"(?i)(eval\s*\(|exec\s*\(|system\s*\()",
            r"(?i)(__import__|getattr|setattr|delattr)",
        ]

        # XSS patterns
        self.xss_patterns = [
            r"(?i)<script[^>]*>.*?</script>",
            r"(?i)javascript\s*:",
            r"(?i)on(load|error|click|mouseover)\s*=",
            r"(?i)<iframe[^>]*>.*?</iframe>",
            r"(?i)expression\s*\(",
        ]

        # Path traversal patterns
        self.traversal_patterns = [
            r"\.\.[\\/]",
            r"(?i)[\\/](etc[\\/]passwd|windows[\\/]system32)",
            r"(?i)file\s*:\s*[\\/]",
            r"%2e%2e%2f",  # URL encoded ../
            r"%c0%af",     # Unicode bypass attempts
        ]

        # Sensitive data patterns
        self.sensitive_patterns = [
            r"\b([A-Za-z0-9+/]{4})*([A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?\b",  # Base64
            r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b",  # UUIDs
            r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b",  # IP addresses
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # Email addresses
            r"\b(password|token|key|secret|api_key)\s*[:=]\s*['\"]?[A-Za-z0-9+/]{8,}['\"]?",  # Credentials
        ]

        # Suspicious query patterns
        self.suspicious_patterns = [
            r"(?i)(show\s+tables|describe\s+\w+|information_schema)",
            r"(?i)(benchmark\s*\(|sleep\s*\(|waitfor\s+delay)",
            r"(?i)(load_file\s*\(|into\s+outfile|into\s+dumpfile)",
            r"(?i)(concat\s*\(.*char\s*\(|unhex\s*\()",
        ]

        # Rate limiting thresholds
        self.rate_limits = {
            'queries_per_minute': 60,
            'queries_per_hour': 1000,
            'failed_attempts_per_minute': 5,
            'unique_queries_per_minute': 30,
        }

    def analyze_query_threat(self, query: str, user_id: Optional[str] = None,
                           ip_address: Optional[str] = None,
                           user_agent: Optional[str] = None) -> Tuple[ThreatLevel, List[str], float]:
        """Analyze a query for security threats."""
        indicators = []
        threat_score = 0.0

        # Check for injection attempts
        injection_score = self._check_injection_attempts(query, indicators)
        threat_score += injection_score * 0.4

        # Check for XSS attempts
        xss_score = self._check_xss_attempts(query, indicators)
        threat_score += xss_score * 0.3

        # Check for path traversal
        traversal_score = self._check_traversal_attempts(query, indicators)
        threat_score += traversal_score * 0.2

        # Check for suspicious patterns
        suspicious_score = self._check_suspicious_patterns(query, indicators)
        threat_score += suspicious_score * 0.1

        # Determine threat level
        if threat_score >= 0.8:
            threat_level = ThreatLevel.CRITICAL
        elif threat_score >= 0.6:
            threat_level = ThreatLevel.HIGH
        elif threat_score >= 0.3:
            threat_level = ThreatLevel.MEDIUM
        else:
            threat_level = ThreatLevel.LOW

        return threat_level, indicators, threat_score

    def _check_injection_attempts(self, query: str, indicators: List[str]) -> float:
        """Check for injection attack patterns."""
        score = 0.0

        for pattern in self.injection_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            if matches:
                score += 0.3 * len(matches)
                indicators.append(f"Injection pattern detected: {pattern}")

        # Additional injection indicators
        dangerous_chars = ['\'', '"', ';', '--', '/*', '*/']
        char_count = sum(query.count(char) for char in dangerous_chars)
        if char_count > 3:
            score += 0.2
            indicators.append(f"High count of dangerous characters: {char_count}")

        return min(score, 1.0)

    def _check_xss_attempts(self, query: str, indicators: List[str]) -> float:
        """Check for XSS attack patterns."""
        score = 0.0

        for pattern in self.xss_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            if matches:
                score += 0.4 * len(matches)
                indicators.append(f"XSS pattern detected: {pattern}")

        # Check for encoded XSS attempts
        encoded_patterns = ['%3Cscript', '%3E', 'javascript%3A']
        for pattern in encoded_patterns:
            if pattern.lower() in query.lower():
                score += 0.3
                indicators.append(f"Encoded XSS attempt: {pattern}")

        return min(score, 1.0)

    def _check_traversal_attempts(self, query: str, indicators: List[str]) -> float:
        """Check for path traversal patterns."""
        score = 0.0

        for pattern in self.traversal_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            if matches:
                score += 0.5 * len(matches)
                indicators.append(f"Path traversal pattern detected: {pattern}")

        return min(score, 1.0)

    def _check_suspicious_patterns(self, query: str, indicators: List[str]) -> float:
        """Check for other suspicious patterns."""
        score = 0.0

        for pattern in self.suspicious_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            if matches:
                score += 0.2 * len(matches)
                indicators.append(f"Suspicious pattern detected: {pattern}")

        # Check for sensitive data patterns
        for pattern in self.sensitive_patterns:
            matches = re.findall(pattern, query)
            if matches:
                score += 0.1 * len(matches)
                indicators.append(f"Potential sensitive data pattern: {pattern}")

        return min(score, 1.0)


class UserBehaviorAnalyzer:
    """Analyze user behavior for anomaly detection."""

    def __init__(self, max_profiles: int = 10000):
        self.user_profiles: Dict[str, UserBehaviorProfile] = {}
        self.max_profiles = max_profiles
        self.baseline_update_interval = 3600  # Update baselines hourly

    def analyze_user_behavior(self, user_id: str, query: str,
                            ip_address: Optional[str] = None) -> Tuple[float, List[str]]:
        """Analyze user behavior and return anomaly score and indicators."""
        if not user_id:
            return 0.5, ["No user ID provided"]

        profile = self._get_or_create_profile(user_id)
        anomalies = []
        anomaly_score = 0.0

        # Update profile with current activity
        self._update_profile(profile, query, ip_address)

        # Analyze various behavioral aspects

        # 1. Query pattern analysis
        pattern_score = self._analyze_query_patterns(profile, query, anomalies)
        anomaly_score += pattern_score * 0.3

        # 2. Access time analysis
        time_score = self._analyze_access_times(profile, anomalies)
        anomaly_score += time_score * 0.2

        # 3. Request frequency analysis
        frequency_score = self._analyze_request_frequency(profile, anomalies)
        anomaly_score += frequency_score * 0.3

        # 4. IP address analysis
        ip_score = self._analyze_ip_patterns(profile, ip_address, anomalies)
        anomaly_score += ip_score * 0.2

        # Update profile anomaly score
        profile.anomaly_score = anomaly_score

        return anomaly_score, anomalies

    def _get_or_create_profile(self, user_id: str) -> UserBehaviorProfile:
        """Get existing profile or create new one."""
        if user_id in self.user_profiles:
            return self.user_profiles[user_id]

        # Create new profile
        profile = UserBehaviorProfile(
            user_id=user_id,
            query_patterns=defaultdict(int),
            access_times=[],
            request_frequency=0.0,
            typical_sources=set(),
            privilege_level='user',
            anomaly_score=0.0,
            last_updated=time.time()
        )

        # Manage profile limit
        if len(self.user_profiles) >= self.max_profiles:
            # Remove oldest profile
            oldest_user = min(self.user_profiles.keys(),
                            key=lambda u: self.user_profiles[u].last_updated)
            del self.user_profiles[oldest_user]

        self.user_profiles[user_id] = profile
        return profile

    def _update_profile(self, profile: UserBehaviorProfile, query: str,
                       ip_address: Optional[str]):
        """Update user profile with current activity."""
        current_time = time.time()

        # Update query patterns
        query_signature = self._generate_query_signature(query)
        profile.query_patterns[query_signature] += 1

        # Update access times (keep last 100)
        profile.access_times.append(current_time)
        if len(profile.access_times) > 100:
            profile.access_times = profile.access_times[-100:]

        # Update request frequency
        if len(profile.access_times) >= 2:
            time_span = profile.access_times[-1] - profile.access_times[0]
            if time_span > 0:
                profile.request_frequency = len(profile.access_times) / (time_span / 3600)  # Per hour

        # Update typical sources
        if ip_address:
            profile.typical_sources.add(ip_address)
            # Keep only recent sources (limit to 10)
            if len(profile.typical_sources) > 10:
                profile.typical_sources.pop()  # Remove arbitrary element

        profile.last_updated = current_time

    def _generate_query_signature(self, query: str) -> str:
        """Generate signature for query pattern matching."""
        # Normalize query for pattern analysis
        normalized = re.sub(r'\b\d+\b', 'NUMBER', query.lower())
        normalized = re.sub(r'\b[a-f0-9]{8,}\b', 'HASH', normalized)
        normalized = re.sub(r'\s+', ' ', normalized).strip()

        # Generate hash of normalized query
        return hashlib.md5(normalized.encode()).hexdigest()[:8]

    def _analyze_query_patterns(self, profile: UserBehaviorProfile, query: str,
                               anomalies: List[str]) -> float:
        """Analyze query patterns for anomalies."""
        query_signature = self._generate_query_signature(query)

        # Check if this is a completely new pattern
        if query_signature not in profile.query_patterns:
            # New pattern is somewhat suspicious
            anomalies.append("Completely new query pattern for user")
            return 0.3

        # Check frequency of this pattern
        pattern_frequency = profile.query_patterns[query_signature]
        total_queries = sum(profile.query_patterns.values())

        if total_queries > 0:
            pattern_ratio = pattern_frequency / total_queries

            # Very rare patterns (< 5% of queries) are suspicious
            if pattern_ratio < 0.05:
                anomalies.append(f"Rare query pattern (only {pattern_ratio:.1%} of queries)")
                return 0.4

        return 0.0

    def _analyze_access_times(self, profile: UserBehaviorProfile,
                             anomalies: List[str]) -> float:
        """Analyze access time patterns."""
        if len(profile.access_times) < 5:
            return 0.0  # Not enough data

        current_time = time.time()
        current_hour = datetime.fromtimestamp(current_time).hour

        # Analyze historical access hour distribution
        historical_hours = [datetime.fromtimestamp(t).hour for t in profile.access_times[:-1]]
        hour_counts = defaultdict(int)
        for hour in historical_hours:
            hour_counts[hour] += 1

        # Check if current hour is unusual for this user
        if hour_counts[current_hour] == 0 and len(historical_hours) > 20:
            anomalies.append(f"Access at unusual hour ({current_hour}:00)")
            return 0.6

        return 0.0

    def _analyze_request_frequency(self, profile: UserBehaviorProfile,
                                  anomalies: List[str]) -> float:
        """Analyze request frequency patterns."""
        if len(profile.access_times) < 10:
            return 0.0  # Not enough data

        # Calculate recent request frequency (last 10 minutes)
        current_time = time.time()
        recent_requests = [t for t in profile.access_times if current_time - t < 600]
        recent_frequency = len(recent_requests) / 10 * 60  # Per hour

        # Compare with historical average
        historical_frequency = profile.request_frequency

        if historical_frequency > 0:
            frequency_ratio = recent_frequency / historical_frequency

            # Unusual frequency spike
            if frequency_ratio > 3.0:
                anomalies.append(f"Request frequency spike: {frequency_ratio:.1f}x normal")
                return 0.7

            # Unusual frequency drop (might indicate compromise)
            if frequency_ratio < 0.2:
                anomalies.append(f"Request frequency drop: {frequency_ratio:.1f}x normal")
                return 0.3

        return 0.0

    def _analyze_ip_patterns(self, profile: UserBehaviorProfile,
                            ip_address: Optional[str], anomalies: List[str]) -> float:
        """Analyze IP address patterns."""
        if not ip_address or not profile.typical_sources:
            return 0.0

        # Check if IP is completely new
        if ip_address not in profile.typical_sources:
            anomalies.append(f"Access from new IP address: {ip_address}")

            # Try to determine if IP is from similar network
            try:
                current_ip = ipaddress.ip_address(ip_address)

                for typical_ip_str in profile.typical_sources:
                    try:
                        typical_ip = ipaddress.ip_address(typical_ip_str)

                        # Check if in same /24 subnet
                        if current_ip.version == typical_ip.version:
                            if current_ip.version == 4:
                                # IPv4 /24 comparison
                                if str(current_ip).rsplit('.', 1)[0] == str(typical_ip).rsplit('.', 1)[0]:
                                    return 0.2  # Same subnet, less suspicious
                            else:
                                # IPv6 /64 comparison (simplified)
                                if str(current_ip)[:19] == str(typical_ip)[:19]:
                                    return 0.2
                    except ValueError:
                        continue

                # Completely different network
                return 0.6

            except ValueError:
                # Invalid IP format
                return 0.4

        return 0.0


class SecurityEventLogger:
    """Log and manage security events."""

    def __init__(self, log_file: Optional[str] = None, max_events: int = 10000):
        self.log_file = Path(log_file) if log_file else Path("data/security_events.log")
        self.max_events = max_events
        self.recent_events: deque = deque(maxlen=max_events)

        # Event counters for monitoring
        self.event_counters = defaultdict(int)
        self.hourly_counters = defaultdict(lambda: defaultdict(int))

        # Ensure log directory exists
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def log_event(self, event: SecurityEvent):
        """Log a security event."""
        # Add to recent events
        self.recent_events.append(event)

        # Update counters
        self.event_counters[event.event_type.value] += 1
        current_hour = int(time.time() // 3600)
        self.hourly_counters[current_hour][event.event_type.value] += 1

        # Write to log file
        self._write_to_log(event)

        # Check for alert conditions
        self._check_alert_conditions(event)

    def _write_to_log(self, event: SecurityEvent):
        """Write event to log file."""
        try:
            log_entry = {
                'timestamp': datetime.fromtimestamp(event.timestamp).isoformat(),
                'event_id': event.event_id,
                'event_type': event.event_type.value,
                'threat_level': event.threat_level.value,
                'attack_type': event.attack_type.value if event.attack_type else None,
                'user_id': event.user_id,
                'ip_address': event.ip_address,
                'query': event.query,
                'blocked': event.blocked,
                'confidence_score': event.confidence_score,
                'indicators': event.indicators,
                'metadata': event.metadata
            }

            with open(self.log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')

        except Exception as e:
            logger.error(f"Failed to write security event to log: {e}")

    def _check_alert_conditions(self, event: SecurityEvent):
        """Check if event should trigger alerts."""
        # Critical events always alert
        if event.threat_level == ThreatLevel.CRITICAL:
            self._trigger_alert(f"CRITICAL security event: {event.event_type.value}", event)

        # Check for attack patterns
        current_hour = int(time.time() // 3600)
        hour_counts = self.hourly_counters[current_hour]

        # Multiple high-severity events
        high_severity_count = (
            hour_counts[SecurityEventType.SUSPICIOUS_QUERY.value] +
            hour_counts[SecurityEventType.MALICIOUS_INPUT.value] +
            hour_counts[SecurityEventType.PRIVILEGE_VIOLATION.value]
        )

        if high_severity_count >= 10:
            self._trigger_alert(f"High frequency security events: {high_severity_count} in last hour", event)

        # Brute force detection
        if hour_counts[SecurityEventType.AUTHENTICATION_FAILURE.value] >= 20:
            self._trigger_alert("Potential brute force attack detected", event)

    def _trigger_alert(self, message: str, event: SecurityEvent):
        """Trigger security alert."""
        logger.warning(f"SECURITY ALERT: {message} - Event ID: {event.event_id}")

        # In a real system, this would integrate with alerting systems
        # (email, Slack, PagerDuty, etc.)

    def get_recent_events(self, limit: int = 100,
                         threat_level: Optional[ThreatLevel] = None) -> List[SecurityEvent]:
        """Get recent security events."""
        events = list(self.recent_events)

        if threat_level:
            events = [e for e in events if e.threat_level == threat_level]

        return events[-limit:]

    def get_event_statistics(self, hours: int = 24) -> Dict[str, Any]:
        """Get security event statistics."""
        current_hour = int(time.time() // 3600)
        start_hour = current_hour - hours

        stats = {
            'total_events': len(self.recent_events),
            'events_by_type': dict(self.event_counters),
            'hourly_breakdown': {},
            'threat_level_distribution': defaultdict(int),
            'most_targeted_users': defaultdict(int),
            'top_attack_sources': defaultdict(int)
        }

        # Calculate hourly breakdown
        for hour in range(start_hour, current_hour + 1):
            if hour in self.hourly_counters:
                stats['hourly_breakdown'][hour] = dict(self.hourly_counters[hour])

        # Analyze recent events for additional stats
        for event in self.recent_events:
            stats['threat_level_distribution'][event.threat_level.value] += 1

            if event.user_id:
                stats['most_targeted_users'][event.user_id] += 1

            if event.ip_address:
                stats['top_attack_sources'][event.ip_address] += 1

        # Convert to regular dicts and get top entries
        stats['most_targeted_users'] = dict(
            sorted(stats['most_targeted_users'].items(),
                  key=lambda x: x[1], reverse=True)[:10]
        )

        stats['top_attack_sources'] = dict(
            sorted(stats['top_attack_sources'].items(),
                  key=lambda x: x[1], reverse=True)[:10]
        )

        return stats


class AdvancedSecuritySystem:
    """Comprehensive security monitoring and protection system."""

    def __init__(self, log_file: Optional[str] = None):
        self.threat_detector = ThreatDetector()
        self.behavior_analyzer = UserBehaviorAnalyzer()
        self.event_logger = SecurityEventLogger(log_file)

        # Rate limiting tracking
        self.rate_limiter = defaultdict(lambda: defaultdict(list))

    def analyze_security_threat(self, query: str, user_id: Optional[str] = None,
                               ip_address: Optional[str] = None,
                               user_agent: Optional[str] = None) -> SecurityEvent:
        """Comprehensive security analysis of a query."""
        event_id = f"sec_{int(time.time())}_{hashlib.md5(query.encode()).hexdigest()[:8]}"
        current_time = time.time()

        # Threat detection
        threat_level, threat_indicators, threat_score = self.threat_detector.analyze_query_threat(
            query, user_id, ip_address, user_agent
        )

        # Behavioral analysis
        behavior_score, behavior_indicators = self.behavior_analyzer.analyze_user_behavior(
            user_id or "anonymous", query, ip_address
        )

        # Rate limit checking
        rate_limit_exceeded, rate_indicators = self._check_rate_limits(
            user_id, ip_address, query
        )

        # Combine all indicators
        all_indicators = threat_indicators + behavior_indicators + rate_indicators

        # Determine overall threat assessment
        combined_score = max(threat_score, behavior_score)
        if rate_limit_exceeded:
            combined_score = max(combined_score, 0.8)

        # Determine if request should be blocked
        should_block = (
            threat_level == ThreatLevel.CRITICAL or
            combined_score >= 0.8 or
            rate_limit_exceeded
        )

        # Determine attack type
        attack_type = self._determine_attack_type(threat_indicators, behavior_indicators, rate_indicators)

        # Determine event type
        if rate_limit_exceeded:
            event_type = SecurityEventType.RATE_LIMIT_EXCEEDED
        elif threat_score >= 0.6:
            event_type = SecurityEventType.MALICIOUS_INPUT
        elif behavior_score >= 0.6:
            event_type = SecurityEventType.DATA_ACCESS_ANOMALY
        elif combined_score >= 0.3:
            event_type = SecurityEventType.SUSPICIOUS_QUERY
        else:
            return None  # No security event to log

        # Create security event
        event = SecurityEvent(
            event_id=event_id,
            event_type=event_type,
            threat_level=threat_level,
            attack_type=attack_type,
            user_id=user_id,
            ip_address=ip_address,
            user_agent=user_agent,
            query=query,
            timestamp=current_time,
            metadata={
                'threat_score': threat_score,
                'behavior_score': behavior_score,
                'combined_score': combined_score,
                'rate_limit_exceeded': rate_limit_exceeded
            },
            blocked=should_block,
            confidence_score=combined_score,
            indicators=all_indicators
        )

        # Log the event
        self.event_logger.log_event(event)

        return event

    def _check_rate_limits(self, user_id: Optional[str], ip_address: Optional[str],
                          query: str) -> Tuple[bool, List[str]]:
        """Check various rate limits."""
        current_time = time.time()
        indicators = []
        exceeded = False

        # Rate limit by user ID
        if user_id:
            user_requests = self.rate_limiter[f"user:{user_id}"]

            # Clean old entries
            minute_ago = current_time - 60
            hour_ago = current_time - 3600

            user_requests['queries'] = [t for t in user_requests['queries'] if t > minute_ago]
            user_requests['hourly'] = [t for t in user_requests['hourly'] if t > hour_ago]

            # Add current request
            user_requests['queries'].append(current_time)
            user_requests['hourly'].append(current_time)

            # Check limits
            if len(user_requests['queries']) > self.threat_detector.rate_limits['queries_per_minute']:
                indicators.append(f"User rate limit exceeded: {len(user_requests['queries'])} queries/minute")
                exceeded = True

            if len(user_requests['hourly']) > self.threat_detector.rate_limits['queries_per_hour']:
                indicators.append(f"User hourly limit exceeded: {len(user_requests['hourly'])} queries/hour")
                exceeded = True

        # Rate limit by IP address
        if ip_address:
            ip_requests = self.rate_limiter[f"ip:{ip_address}"]

            minute_ago = current_time - 60
            ip_requests['queries'] = [t for t in ip_requests['queries'] if t > minute_ago]
            ip_requests['queries'].append(current_time)

            if len(ip_requests['queries']) > self.threat_detector.rate_limits['queries_per_minute']:
                indicators.append(f"IP rate limit exceeded: {len(ip_requests['queries'])} queries/minute")
                exceeded = True

        return exceeded, indicators

    def _determine_attack_type(self, threat_indicators: List[str],
                             behavior_indicators: List[str],
                             rate_indicators: List[str]) -> Optional[AttackType]:
        """Determine the type of attack based on indicators."""
        all_indicators = " ".join(threat_indicators + behavior_indicators + rate_indicators).lower()

        if any(term in all_indicators for term in ['injection', 'sql', 'union', 'select']):
            return AttackType.INJECTION
        elif any(term in all_indicators for term in ['xss', 'script', 'javascript']):
            return AttackType.XSS
        elif any(term in all_indicators for term in ['traversal', 'path', '..']):
            return AttackType.TRAVERSAL
        elif any(term in all_indicators for term in ['rate limit', 'frequency']):
            return AttackType.RATE_LIMIT
        elif any(term in all_indicators for term in ['anomaly', 'unusual', 'suspicious']):
            return AttackType.ANOMALOUS_BEHAVIOR
        elif any(term in all_indicators for term in ['malformed', 'suspicious pattern']):
            return AttackType.MALFORMED_INPUT
        else:
            return None

    def get_security_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive security dashboard data."""
        stats = self.event_logger.get_event_statistics()
        recent_events = self.event_logger.get_recent_events(20)

        # Calculate additional metrics
        current_time = time.time()
        hour_ago = current_time - 3600

        recent_critical = [e for e in recent_events
                          if e.threat_level == ThreatLevel.CRITICAL and e.timestamp > hour_ago]

        return {
            'summary': {
                'total_events': stats['total_events'],
                'events_last_hour': sum(stats['hourly_breakdown'].get(int(current_time // 3600), {}).values()),
                'critical_events_last_hour': len(recent_critical),
                'blocked_requests': len([e for e in recent_events if e.blocked]),
            },
            'threat_levels': stats['threat_level_distribution'],
            'attack_types': {
                attack_type.value: len([e for e in recent_events if e.attack_type == attack_type])
                for attack_type in AttackType
            },
            'most_targeted_users': stats['most_targeted_users'],
            'top_attack_sources': stats['top_attack_sources'],
            'recent_high_severity': [
                {
                    'event_id': e.event_id,
                    'type': e.event_type.value,
                    'threat_level': e.threat_level.value,
                    'user_id': e.user_id,
                    'ip_address': e.ip_address,
                    'blocked': e.blocked,
                    'timestamp': e.timestamp
                }
                for e in recent_events
                if e.threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]
            ][-10:]  # Last 10 high-severity events
        }

    def is_request_blocked(self, query: str, user_id: Optional[str] = None,
                          ip_address: Optional[str] = None,
                          user_agent: Optional[str] = None) -> Tuple[bool, Optional[SecurityEvent]]:
        """Check if request should be blocked and return security event."""
        event = self.analyze_security_threat(query, user_id, ip_address, user_agent)

        if event:
            return event.blocked, event

        return False, None


# Convenience functions
def create_security_system(log_file: Optional[str] = None) -> AdvancedSecuritySystem:
    """Create and return an advanced security system."""
    return AdvancedSecuritySystem(log_file)


def analyze_query_security(query: str, user_id: Optional[str] = None) -> Tuple[bool, Dict[str, Any]]:
    """Quick security analysis of a query."""
    detector = ThreatDetector()
    threat_level, indicators, score = detector.analyze_query_threat(query, user_id)

    return score >= 0.6, {
        'threat_level': threat_level.value,
        'threat_score': score,
        'indicators': indicators,
        'should_block': score >= 0.8
    }
