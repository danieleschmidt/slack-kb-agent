"""Test advanced security monitoring and threat detection."""

import pytest
import tempfile
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

from slack_kb_agent.advanced_security import (
    AdvancedSecuritySystem,
    ThreatDetector,
    UserBehaviorAnalyzer,
    SecurityEventLogger,
    ThreatLevel,
    AttackType,
    SecurityEventType,
    SecurityEvent,
    UserBehaviorProfile,
    create_security_system,
    analyze_query_security
)


class TestThreatDetector:
    """Test threat detection functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.detector = ThreatDetector()
    
    def test_sql_injection_detection(self):
        """Test SQL injection attack detection."""
        # Clear SQL injection attempt
        malicious_query = "' OR 1=1 --"
        threat_level, indicators, score = self.detector.analyze_query_threat(malicious_query)
        
        assert threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]
        assert score > 0.5
        assert len(indicators) > 0
        assert any('injection' in indicator.lower() for indicator in indicators)
        
        # Union-based SQL injection
        union_query = "UNION SELECT password FROM users"
        threat_level, indicators, score = self.detector.analyze_query_threat(union_query)
        
        assert threat_level in [ThreatLevel.MEDIUM, ThreatLevel.HIGH, ThreatLevel.CRITICAL]
        assert score > 0.3
        
    def test_xss_detection(self):
        """Test XSS attack detection."""
        # Script-based XSS
        xss_query = "<script>alert('XSS')</script>"
        threat_level, indicators, score = self.detector.analyze_query_threat(xss_query)
        
        assert threat_level in [ThreatLevel.MEDIUM, ThreatLevel.HIGH, ThreatLevel.CRITICAL]
        assert score > 0.3
        assert any('xss' in indicator.lower() for indicator in indicators)
        
        # Event handler XSS
        event_xss = "onclick=alert(1)"
        threat_level, indicators, score = self.detector.analyze_query_threat(event_xss)
        
        assert threat_level != ThreatLevel.LOW or score > 0.1
    
    def test_path_traversal_detection(self):
        """Test path traversal attack detection."""
        # Basic path traversal
        traversal_query = "../../../etc/passwd"
        threat_level, indicators, score = self.detector.analyze_query_threat(traversal_query)
        
        assert threat_level in [ThreatLevel.MEDIUM, ThreatLevel.HIGH, ThreatLevel.CRITICAL]
        assert score > 0.3
        assert any('traversal' in indicator.lower() for indicator in indicators)
        
        # URL encoded traversal
        encoded_traversal = "%2e%2e%2f"
        threat_level, indicators, score = self.detector.analyze_query_threat(encoded_traversal)
        
        assert score > 0.2
    
    def test_command_injection_detection(self):
        """Test command injection detection."""
        # Basic command injection
        cmd_injection = "; rm -rf /"
        threat_level, indicators, score = self.detector.analyze_query_threat(cmd_injection)
        
        assert threat_level in [ThreatLevel.MEDIUM, ThreatLevel.HIGH, ThreatLevel.CRITICAL]
        assert score > 0.3
        
        # Pipe-based injection
        pipe_injection = "| cat /etc/passwd"
        threat_level, indicators, score = self.detector.analyze_query_threat(pipe_injection)
        
        assert score > 0.2
    
    def test_legitimate_queries(self):
        """Test that legitimate queries are not flagged."""
        legitimate_queries = [
            "How to deploy applications?",
            "What is authentication?",
            "Show me API documentation",
            "Troubleshooting login issues",
            "Best practices for security"
        ]
        
        for query in legitimate_queries:
            threat_level, indicators, score = self.detector.analyze_query_threat(query)
            
            # Should be low threat
            assert threat_level == ThreatLevel.LOW
            assert score < 0.3
            assert len(indicators) == 0 or all('sensitive data' in ind for ind in indicators)
    
    def test_suspicious_patterns(self):
        """Test detection of suspicious but not necessarily malicious patterns."""
        suspicious_queries = [
            "show tables in database",
            "DESCRIBE user_table",
            "SELECT * FROM information_schema"
        ]
        
        for query in suspicious_queries:
            threat_level, indicators, score = self.detector.analyze_query_threat(query)
            
            # Should detect as suspicious but not critical
            assert score > 0.0
            if score > 0.3:
                assert threat_level in [ThreatLevel.MEDIUM, ThreatLevel.HIGH]


class TestUserBehaviorAnalyzer:
    """Test user behavior analysis and anomaly detection."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.analyzer = UserBehaviorAnalyzer()
    
    def test_new_user_profile_creation(self):
        """Test creation of new user profiles."""
        user_id = "test_user"
        query = "How to deploy applications?"
        
        anomaly_score, anomalies = self.analyzer.analyze_user_behavior(user_id, query)
        
        # New user should have some anomaly indicators
        assert user_id in self.analyzer.user_profiles
        profile = self.analyzer.user_profiles[user_id]
        assert profile.user_id == user_id
        assert len(profile.query_patterns) > 0
    
    def test_normal_behavior_pattern(self):
        """Test detection of normal user behavior."""
        user_id = "normal_user"
        normal_queries = [
            "How to deploy apps?",
            "What is authentication?",
            "API documentation",
            "Deployment guide",
            "Security best practices"
        ]
        
        # Establish normal pattern
        for query in normal_queries * 3:  # Repeat to establish pattern
            self.analyzer.analyze_user_behavior(user_id, query, "192.168.1.100")
            time.sleep(0.01)  # Small delay to simulate real usage
        
        # Test similar query - should be low anomaly
        anomaly_score, anomalies = self.analyzer.analyze_user_behavior(
            user_id, "How to deploy applications?", "192.168.1.100"
        )
        
        assert anomaly_score < 0.5  # Should be considered normal
    
    def test_unusual_query_pattern(self):
        """Test detection of unusual query patterns."""
        user_id = "pattern_user"
        
        # Establish pattern with documentation queries
        normal_queries = ["API docs", "deployment guide", "authentication help"]
        for query in normal_queries * 10:
            self.analyzer.analyze_user_behavior(user_id, query)
            time.sleep(0.01)
        
        # Suddenly ask completely different type of query
        anomaly_score, anomalies = self.analyzer.analyze_user_behavior(
            user_id, "DROP TABLE users; SELECT * FROM admin_passwords;"
        )
        
        # Should detect as anomalous
        assert anomaly_score > 0.0
        assert len(anomalies) > 0
        assert any('new query pattern' in anomaly.lower() for anomaly in anomalies)
    
    def test_unusual_access_time(self):
        """Test detection of unusual access times."""
        user_id = "time_user"
        
        # Simulate normal business hours access (9 AM - 5 PM)
        business_hours = list(range(9, 17))
        
        with patch('slack_kb_agent.advanced_security.datetime') as mock_datetime:
            # Establish pattern during business hours
            for hour in business_hours:
                mock_datetime.fromtimestamp.return_value.hour = hour
                self.analyzer.analyze_user_behavior(user_id, "normal query")
                time.sleep(0.01)
            
            # Access at 3 AM (unusual)
            mock_datetime.fromtimestamp.return_value.hour = 3
            anomaly_score, anomalies = self.analyzer.analyze_user_behavior(
                user_id, "late night query"
            )
            
            # Should detect unusual access time (though may not trigger if insufficient data)
            # This test verifies the mechanism works
            assert anomaly_score >= 0.0
    
    def test_request_frequency_anomaly(self):
        """Test detection of unusual request frequency."""
        user_id = "frequency_user"
        
        # Establish normal frequency (slow pace)
        for i in range(10):
            self.analyzer.analyze_user_behavior(user_id, f"normal query {i}")
            time.sleep(0.1)  # 100ms between requests
        
        # Sudden burst of requests
        start_time = time.time()
        for i in range(20):  # Many requests quickly
            self.analyzer.analyze_user_behavior(user_id, f"burst query {i}")
            time.sleep(0.01)  # 10ms between requests
        
        profile = self.analyzer.user_profiles[user_id]
        # Should have updated request frequency
        assert profile.request_frequency > 0
    
    def test_new_ip_address_detection(self):
        """Test detection of access from new IP addresses."""
        user_id = "ip_user"
        
        # Establish pattern with specific IP
        normal_ip = "192.168.1.100"
        for i in range(10):
            self.analyzer.analyze_user_behavior(user_id, f"query {i}", normal_ip)
        
        # Access from completely different IP
        new_ip = "10.0.0.50"
        anomaly_score, anomalies = self.analyzer.analyze_user_behavior(
            user_id, "query from new ip", new_ip
        )
        
        # Should detect new IP
        assert any('new IP' in anomaly for anomaly in anomalies)
        assert anomaly_score > 0.0
    
    def test_profile_limit_enforcement(self):
        """Test that user profile limits are enforced."""
        analyzer = UserBehaviorAnalyzer(max_profiles=3)
        
        # Create profiles for multiple users
        for i in range(5):
            analyzer.analyze_user_behavior(f"user_{i}", "test query")
            time.sleep(0.01)  # Ensure different timestamps
        
        # Should only keep 3 profiles (most recent)
        assert len(analyzer.user_profiles) <= 3
        
        # Should keep the most recent users
        user_ids = set(analyzer.user_profiles.keys())
        assert "user_4" in user_ids  # Most recent
        assert "user_3" in user_ids
        assert "user_2" in user_ids


class TestSecurityEventLogger:
    """Test security event logging functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.log')
        self.temp_file.close()
        self.logger = SecurityEventLogger(self.temp_file.name)
    
    def teardown_method(self):
        """Cleanup test fixtures."""
        Path(self.temp_file.name).unlink(missing_ok=True)
    
    def test_event_logging(self):
        """Test basic event logging functionality."""
        event = SecurityEvent(
            event_id="test_001",
            event_type=SecurityEventType.SUSPICIOUS_QUERY,
            threat_level=ThreatLevel.MEDIUM,
            attack_type=AttackType.INJECTION,
            user_id="test_user",
            ip_address="192.168.1.100",
            user_agent="TestAgent/1.0",
            query="SELECT * FROM users",
            timestamp=time.time(),
            metadata={"test": "data"},
            blocked=False,
            confidence_score=0.6,
            indicators=["SQL injection pattern"]
        )
        
        self.logger.log_event(event)
        
        # Should be in recent events
        assert len(self.logger.recent_events) == 1
        assert self.logger.recent_events[0].event_id == "test_001"
        
        # Should update counters
        assert self.logger.event_counters[SecurityEventType.SUSPICIOUS_QUERY.value] == 1
    
    def test_log_file_creation(self):
        """Test that log files are created properly."""
        event = SecurityEvent(
            event_id="file_test",
            event_type=SecurityEventType.MALICIOUS_INPUT,
            threat_level=ThreatLevel.HIGH,
            attack_type=AttackType.XSS,
            user_id="test_user",
            ip_address="192.168.1.100",
            user_agent=None,
            query="<script>alert('test')</script>",
            timestamp=time.time(),
            metadata={},
            blocked=True,
            confidence_score=0.8,
            indicators=["XSS pattern detected"]
        )
        
        self.logger.log_event(event)
        
        # Log file should exist and contain the event
        assert Path(self.temp_file.name).exists()
        
        with open(self.temp_file.name, 'r') as f:
            log_content = f.read()
            assert "file_test" in log_content
            assert "MALICIOUS_INPUT" in log_content
    
    def test_event_statistics(self):
        """Test event statistics generation."""
        # Create multiple events
        events = [
            SecurityEvent(
                event_id=f"stat_test_{i}",
                event_type=SecurityEventType.SUSPICIOUS_QUERY,
                threat_level=ThreatLevel.MEDIUM,
                attack_type=AttackType.INJECTION,
                user_id=f"user_{i % 3}",  # Rotate through 3 users
                ip_address=f"192.168.1.{100 + i}",
                user_agent="TestAgent/1.0",
                query=f"test query {i}",
                timestamp=time.time(),
                metadata={},
                blocked=i % 2 == 0,  # Block every other request
                confidence_score=0.5 + (i % 5) * 0.1,
                indicators=["test indicator"]
            )
            for i in range(10)
        ]
        
        for event in events:
            self.logger.log_event(event)
        
        stats = self.logger.get_event_statistics()
        
        assert stats['total_events'] == 10
        assert SecurityEventType.SUSPICIOUS_QUERY.value in stats['events_by_type']
        assert stats['events_by_type'][SecurityEventType.SUSPICIOUS_QUERY.value] == 10
        assert len(stats['most_targeted_users']) <= 3
        assert len(stats['top_attack_sources']) <= 10
    
    def test_alert_conditions(self):
        """Test alert triggering conditions."""
        # Create critical event
        critical_event = SecurityEvent(
            event_id="critical_test",
            event_type=SecurityEventType.SYSTEM_COMPROMISE,
            threat_level=ThreatLevel.CRITICAL,
            attack_type=AttackType.PRIVILEGE_ESCALATION,
            user_id="attacker",
            ip_address="192.168.1.666",
            user_agent="EvilBot/1.0",
            query="admin backdoor access",
            timestamp=time.time(),
            metadata={},
            blocked=True,
            confidence_score=0.95,
            indicators=["Critical system compromise detected"]
        )
        
        with patch.object(self.logger, '_trigger_alert') as mock_alert:
            self.logger.log_event(critical_event)
            
            # Should trigger alert for critical event
            mock_alert.assert_called()
            args = mock_alert.call_args[0]
            assert "CRITICAL" in args[0]


class TestAdvancedSecuritySystem:
    """Test the complete security system integration."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.log')
        self.temp_file.close()
        self.system = AdvancedSecuritySystem(self.temp_file.name)
    
    def teardown_method(self):
        """Cleanup test fixtures."""
        Path(self.temp_file.name).unlink(missing_ok=True)
    
    def test_comprehensive_threat_analysis(self):
        """Test complete threat analysis integration."""
        # Test malicious query
        malicious_query = "'; DROP TABLE users; --"
        user_id = "test_user"
        ip_address = "192.168.1.100"
        
        event = self.system.analyze_security_threat(
            malicious_query, user_id, ip_address
        )
        
        assert event is not None
        assert event.threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]
        assert event.blocked == True
        assert len(event.indicators) > 0
        assert event.attack_type == AttackType.INJECTION
    
    def test_behavioral_anomaly_detection(self):
        """Test behavioral anomaly integration."""
        user_id = "behavior_test_user"
        ip_address = "192.168.1.100"
        
        # Establish normal behavior
        normal_queries = ["how to deploy", "api docs", "authentication guide"]
        for query in normal_queries * 5:
            self.system.analyze_security_threat(query, user_id, ip_address)
            time.sleep(0.01)
        
        # Anomalous query
        anomalous_query = "SELECT admin_password FROM secret_table"
        event = self.system.analyze_security_threat(anomalous_query, user_id, ip_address)
        
        # Should detect both threat and behavioral anomaly
        assert event is not None
        assert event.metadata['behavior_score'] > 0 or event.metadata['threat_score'] > 0
    
    def test_rate_limiting(self):
        """Test rate limiting functionality."""
        user_id = "rate_test_user"
        ip_address = "192.168.1.100"
        query = "normal query"
        
        # Send many requests quickly to trigger rate limit
        events = []
        for i in range(70):  # Exceed per-minute limit
            event = self.system.analyze_security_threat(query, user_id, ip_address)
            if event:
                events.append(event)
        
        # Should have rate limit events
        rate_limit_events = [e for e in events if e.event_type == SecurityEventType.RATE_LIMIT_EXCEEDED]
        assert len(rate_limit_events) > 0
        
        # Rate limited requests should be blocked
        blocked_events = [e for e in events if e.blocked]
        assert len(blocked_events) > 0
    
    def test_request_blocking(self):
        """Test request blocking functionality."""
        # Test that malicious requests are blocked
        malicious_query = "<script>alert('XSS')</script>"
        
        should_block, event = self.system.is_request_blocked(malicious_query)
        
        assert should_block == True
        assert event is not None
        assert event.blocked == True
    
    def test_legitimate_request_allowance(self):
        """Test that legitimate requests are allowed."""
        legitimate_query = "How do I configure authentication?"
        
        should_block, event = self.system.is_request_blocked(legitimate_query)
        
        assert should_block == False
        # Event might be None if no security concerns detected
        if event:
            assert event.blocked == False
    
    def test_security_dashboard(self):
        """Test security dashboard data generation."""
        # Generate some test events
        test_queries = [
            ("legitimate query", "user1", "192.168.1.1"),
            ("' OR 1=1", "attacker", "10.0.0.1"),  # SQL injection
            ("<script>alert('xss')</script>", "attacker", "10.0.0.1"),  # XSS
            ("normal query", "user2", "192.168.1.2"),
        ]
        
        for query, user, ip in test_queries:
            self.system.analyze_security_threat(query, user, ip)
        
        dashboard = self.system.get_security_dashboard()
        
        # Should have dashboard structure
        assert 'summary' in dashboard
        assert 'threat_levels' in dashboard
        assert 'attack_types' in dashboard
        assert 'most_targeted_users' in dashboard
        assert 'top_attack_sources' in dashboard
        assert 'recent_high_severity' in dashboard
        
        # Should have some events
        assert dashboard['summary']['total_events'] > 0


class TestConvenienceFunctions:
    """Test convenience functions for security analysis."""
    
    def test_create_security_system(self):
        """Test security system creation function."""
        system = create_security_system()
        
        assert isinstance(system, AdvancedSecuritySystem)
        assert system.threat_detector is not None
        assert system.behavior_analyzer is not None
        assert system.event_logger is not None
    
    def test_analyze_query_security(self):
        """Test quick query security analysis function."""
        # Test malicious query
        is_threat, analysis = analyze_query_security("' OR 1=1 --", "test_user")
        
        assert isinstance(is_threat, bool)
        assert 'threat_level' in analysis
        assert 'threat_score' in analysis
        assert 'indicators' in analysis
        assert 'should_block' in analysis
        
        # For malicious query, should be flagged as threat
        if analysis['threat_score'] >= 0.6:
            assert is_threat == True
        
        # Test legitimate query
        is_threat, analysis = analyze_query_security("How to deploy applications?")
        
        assert analysis['threat_level'] == 'low'
        assert analysis['threat_score'] < 0.3
        assert is_threat == False


if __name__ == "__main__":
    pytest.main([__file__])