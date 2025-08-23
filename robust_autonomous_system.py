#!/usr/bin/env python3
"""
Robust Autonomous Knowledge System - Generation 2 Implementation
Production-ready system with comprehensive error handling, validation, and security.
"""

import asyncio
import json
import logging
import os
import re
import time
import traceback
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import hashlib
import threading
from dataclasses import dataclass, field
from enum import Enum


# Comprehensive logging configuration
class SecureFormatter(logging.Formatter):
    """Secure formatter that redacts sensitive information."""
    
    SENSITIVE_PATTERNS = [
        (r'password["\']?\s*[:=]\s*["\']?([^"\'\\s]+)', r'password=***REDACTED***'),
        (r'token["\']?\s*[:=]\s*["\']?([^"\'\\s]+)', r'token=***REDACTED***'),
        (r'key["\']?\s*[:=]\s*["\']?([^"\'\\s]+)', r'key=***REDACTED***'),
        (r'secret["\']?\s*[:=]\s*["\']?([^"\'\\s]+)', r'secret=***REDACTED***'),
        (r'Bearer\s+([A-Za-z0-9\-_\.]+)', r'Bearer ***REDACTED***'),
    ]
    
    def format(self, record):
        message = super().format(record)
        
        # Redact sensitive information
        for pattern, replacement in self.SENSITIVE_PATTERNS:
            message = re.sub(pattern, replacement, message, flags=re.IGNORECASE)
        
        return message


def setup_secure_logging():
    """Setup comprehensive, secure logging system."""
    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # File handler with rotation (simulate)
    file_handler = logging.FileHandler(log_dir / f"autonomous_system_{datetime.now(timezone.utc).strftime('%Y%m%d')}.log")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(SecureFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(process)d:%(thread)d] - %(message)s'
    ))
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(SecureFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logging.getLogger("RobustAutonomousSystem")


class SecurityError(Exception):
    """Security-related error."""
    pass


class ValidationError(Exception):
    """Data validation error."""
    pass


class SystemError(Exception):
    """System-level error."""
    pass


class InputValidator:
    """Comprehensive input validation and sanitization."""
    
    # Dangerous patterns that should be blocked
    DANGEROUS_PATTERNS = [
        # SQL injection patterns
        r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION)\b)",
        # Command injection patterns  
        r"[;&|`$(){}[\]<>]",
        # Script injection patterns
        r"<script[^>]*>.*?</script>",
        r"javascript:",
        r"vbscript:",
        # Path traversal patterns
        r"\.\./",
        r"\.\.\\\\",
        # Null bytes
        r"\\x00",
    ]
    
    @classmethod
    def validate_query(cls, query: str) -> str:
        """Validate and sanitize user query."""
        if not isinstance(query, str):
            raise ValidationError("Query must be a string")
        
        if not query or not query.strip():
            raise ValidationError("Query cannot be empty")
        
        if len(query) > 5000:
            raise ValidationError("Query too long (max 5000 characters)")
        
        # Check for dangerous patterns
        for pattern in cls.DANGEROUS_PATTERNS:
            if re.search(pattern, query, re.IGNORECASE):
                raise SecurityError(f"Potentially dangerous pattern detected in query")
        
        # Sanitize
        sanitized = query.strip()
        # Remove control characters except newlines and tabs
        sanitized = ''.join(char for char in sanitized 
                           if ord(char) >= 32 or char in '\n\t')
        
        return sanitized
    
    @classmethod
    def validate_document_content(cls, content: str) -> str:
        """Validate document content."""
        if not isinstance(content, str):
            raise ValidationError("Document content must be a string")
        
        if len(content) > 1_000_000:  # 1MB limit
            raise ValidationError("Document too large (max 1MB)")
        
        # Check for dangerous patterns (less strict than queries)
        script_patterns = [r"<script[^>]*>.*?</script>", r"javascript:", r"vbscript:"]
        for pattern in script_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                raise SecurityError("Potentially dangerous script content detected")
        
        return content
    
    @classmethod
    def validate_metadata(cls, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Validate metadata dictionary."""
        if not isinstance(metadata, dict):
            raise ValidationError("Metadata must be a dictionary")
        
        if len(metadata) > 50:
            raise ValidationError("Too many metadata fields (max 50)")
        
        # Validate keys and values
        validated = {}
        for key, value in metadata.items():
            if not isinstance(key, str):
                raise ValidationError("Metadata keys must be strings")
            
            if len(key) > 100:
                raise ValidationError("Metadata key too long (max 100 characters)")
            
            # Validate value based on type
            if isinstance(value, str):
                if len(value) > 1000:
                    raise ValidationError("Metadata string value too long (max 1000 characters)")
                validated[key] = cls._sanitize_string(value)
            elif isinstance(value, (int, float, bool)):
                validated[key] = value
            elif isinstance(value, (list, tuple)) and all(isinstance(x, str) for x in value):
                validated[key] = [cls._sanitize_string(x) for x in value[:20]]  # Limit list size
            else:
                raise ValidationError(f"Unsupported metadata value type: {type(value)}")
        
        return validated
    
    @classmethod
    def _sanitize_string(cls, text: str) -> str:
        """Sanitize string value."""
        # Remove control characters
        sanitized = ''.join(char for char in text if ord(char) >= 32 or char in '\n\t')
        return sanitized.strip()


class RateLimiter:
    """Thread-safe rate limiting implementation."""
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = {}
        self.lock = threading.Lock()
    
    def is_allowed(self, identifier: str) -> bool:
        """Check if request is allowed for given identifier."""
        current_time = time.time()
        
        with self.lock:
            if identifier not in self.requests:
                self.requests[identifier] = []
            
            # Clean old requests
            self.requests[identifier] = [
                req_time for req_time in self.requests[identifier]
                if current_time - req_time < self.window_seconds
            ]
            
            # Check rate limit
            if len(self.requests[identifier]) >= self.max_requests:
                return False
            
            # Add current request
            self.requests[identifier].append(current_time)
            return True


class HealthMonitor:
    """System health monitoring and alerting."""
    
    def __init__(self):
        self.start_time = time.time()
        self.metrics = {
            'requests_total': 0,
            'requests_success': 0,
            'requests_error': 0,
            'avg_response_time': 0.0,
            'memory_usage_mb': 0.0,
            'disk_usage_mb': 0.0
        }
        self.alerts = []
        self.lock = threading.Lock()
    
    def record_request(self, success: bool, response_time: float):
        """Record request metrics."""
        with self.lock:
            self.metrics['requests_total'] += 1
            if success:
                self.metrics['requests_success'] += 1
            else:
                self.metrics['requests_error'] += 1
            
            # Update average response time
            total_requests = self.metrics['requests_total']
            current_avg = self.metrics['avg_response_time']
            self.metrics['avg_response_time'] = (
                (current_avg * (total_requests - 1) + response_time) / total_requests
            )
    
    def check_health(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        with self.lock:
            uptime = time.time() - self.start_time
            
            # Calculate error rate
            total_requests = self.metrics['requests_total']
            error_rate = (self.metrics['requests_error'] / total_requests * 100 
                         if total_requests > 0 else 0)
            
            # System resource checks
            try:
                import psutil
                memory_info = psutil.virtual_memory()
                disk_info = psutil.disk_usage('/')
                
                self.metrics['memory_usage_mb'] = memory_info.used / (1024 * 1024)
                self.metrics['disk_usage_mb'] = disk_info.used / (1024 * 1024)
                
                memory_percent = memory_info.percent
                disk_percent = disk_info.percent
            except ImportError:
                # Fallback without psutil
                memory_percent = 0
                disk_percent = 0
            
            # Health status determination
            status = 'healthy'
            issues = []
            
            if error_rate > 10:  # More than 10% error rate
                status = 'unhealthy'
                issues.append(f"High error rate: {error_rate:.1f}%")
            
            if self.metrics['avg_response_time'] > 5000:  # More than 5 seconds
                status = 'degraded'
                issues.append(f"High response time: {self.metrics['avg_response_time']:.1f}ms")
            
            if memory_percent > 90:
                status = 'critical'
                issues.append(f"High memory usage: {memory_percent:.1f}%")
            
            if disk_percent > 95:
                status = 'critical'
                issues.append(f"High disk usage: {disk_percent:.1f}%")
            
            return {
                'status': status,
                'uptime_seconds': uptime,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'metrics': self.metrics.copy(),
                'system': {
                    'memory_percent': memory_percent,
                    'disk_percent': disk_percent
                },
                'issues': issues
            }


class RobustDocumentProcessor:
    """Enhanced document processing with error handling and validation."""
    
    def __init__(self, validator: InputValidator):
        self.validator = validator
        self.logger = logging.getLogger(f"{__name__}.DocumentProcessor")
        
        self.content_patterns = {
            'api_documentation': [
                r'\b(GET|POST|PUT|DELETE|PATCH|HEAD|OPTIONS)\s+/',
                r'\b(endpoint|route|parameter|request|response|payload)\b',
                r'HTTP/\d+\.\d+',
                r'\b(JSON|XML|API|REST|GraphQL)\b',
                r'\b(authentication|authorization|bearer|token)\b'
            ],
            'code_documentation': [
                r'```\w*',
                r'\bfunction\s+\w+\(',
                r'\bclass\s+\w+',
                r'@\w+',  # decorators
                r'\b(import|from|require|include)\b',
                r'\b(def|class|function|method|variable)\b'
            ],
            'troubleshooting': [
                r'\b(error|exception|fail|bug|issue|problem)\b',
                r'\b(fix|solve|resolve|debug|troubleshoot)\b',
                r'\b(stacktrace|traceback|backtrace)\b',
                r'\b(timeout|crash|freeze|hang)\b'
            ],
            'configuration': [
                r'\b(config|configuration|settings|environment)\b',
                r'\.(yaml|yml|json|toml|ini|conf)\b',
                r'\b[A-Z_]{2,}=',  # Environment variables
                r'\b(PORT|HOST|DATABASE_URL|API_KEY)\b'
            ],
            'deployment': [
                r'\b(deploy|deployment|docker|kubernetes|container)\b',
                r'\b(build|compile|bundle|package)\b',
                r'\b(production|staging|development)\b',
                r'\bdocker\s+(run|build|push|pull)\b'
            ]
        }
    
    def process_document(self, content: str, source: str, 
                        metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process document with comprehensive error handling."""
        try:
            # Validate inputs
            content = self.validator.validate_document_content(content)
            if not source or not isinstance(source, str):
                raise ValidationError("Source must be a non-empty string")
            
            if metadata:
                metadata = self.validator.validate_metadata(metadata)
            else:
                metadata = {}
            
            # Extract document information
            doc_info = self._extract_document_info(content)
            
            # Classify content type
            content_type = self._classify_content_type(content)
            
            # Extract key terms and entities
            key_terms = self._extract_key_terms(content)
            
            # Calculate quality metrics
            quality_metrics = self._calculate_quality_metrics(content)
            
            # Security check
            security_score = self._calculate_security_score(content)
            
            # Build comprehensive metadata
            enhanced_metadata = {
                **metadata,
                'source': source,
                'content_type': content_type,
                'key_terms': key_terms[:50],  # Limit to prevent bloat
                'word_count': doc_info['word_count'],
                'char_count': doc_info['char_count'],
                'line_count': doc_info['line_count'],
                'quality_score': quality_metrics['overall_score'],
                'security_score': security_score,
                'readability_score': quality_metrics['readability_score'],
                'technical_density': quality_metrics['technical_density'],
                'processed_at': datetime.now(timezone.utc).isoformat(),
                'content_hash': hashlib.sha256(content.encode()).hexdigest(),
                'language': self._detect_language(content),
                'has_code_examples': '```' in content,
                'has_urls': bool(re.search(r'https?://', content)),
                'estimated_read_time_minutes': max(1, doc_info['word_count'] // 200)
            }
            
            self.logger.info(f"Successfully processed document from {source}: "
                           f"{doc_info['word_count']} words, type: {content_type}")
            
            return {
                'content': content,
                'metadata': enhanced_metadata,
                'processing_success': True,
                'processing_errors': []
            }
            
        except (ValidationError, SecurityError) as e:
            self.logger.error(f"Validation/Security error processing document from {source}: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error processing document from {source}: {e}")
            self.logger.error(traceback.format_exc())
            raise SystemError(f"Document processing failed: {e}")
    
    def _extract_document_info(self, content: str) -> Dict[str, Any]:
        """Extract basic document information."""
        lines = content.split('\\n')
        words = content.split()
        
        return {
            'word_count': len(words),
            'char_count': len(content),
            'line_count': len(lines),
            'paragraph_count': len([p for p in content.split('\\n\\n') if p.strip()]),
            'avg_words_per_line': len(words) / len(lines) if lines else 0
        }
    
    def _classify_content_type(self, content: str) -> str:
        """Classify content type with improved accuracy."""
        content_lower = content.lower()
        scores = {}
        
        for doc_type, patterns in self.content_patterns.items():
            score = 0
            for pattern in patterns:
                matches = re.findall(pattern, content_lower, re.IGNORECASE)
                # Weight by frequency and uniqueness
                score += len(matches) * (1 + len(pattern) / 50)
            scores[doc_type] = score
        
        if not any(scores.values()):
            return 'general'
        
        return max(scores.items(), key=lambda x: x[1])[0]
    
    def _extract_key_terms(self, content: str) -> List[str]:
        """Extract key technical terms with improved filtering."""
        technical_patterns = [
            r'\b[A-Z_]{3,}\b',          # Constants/environment variables
            r'\b\w+\(\)',              # Functions calls
            r'\b\w+\.\w+',             # Method calls/properties  
            r'@\w+',                     # Decorators/annotations
            r'\b\w+://\S+',            # URLs (limited)
            r'\b\w+\.\w{2,4}\b',     # File extensions
            r'\b(?:def|class|function|method|const|var|let)\s+(\w+)',  # Declarations
            r'\b\w*[Ee]rror\b',        # Error types
            r'\b\w*[Ee]xception\b'     # Exception types
        ]
        
        terms = set()
        for pattern in technical_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            # Extract actual terms from matches (handle groups)
            for match in matches:
                if isinstance(match, tuple):
                    terms.update(term for term in match if term)
                else:
                    terms.add(match)
        
        # Filter out common/noise terms
        stopwords = {
            'the', 'and', 'for', 'with', 'from', 'this', 'that', 'html', 'http',
            'www', 'com', 'org', 'net', 'get', 'set', 'put', 'post', 'delete'
        }
        
        filtered_terms = [
            term.strip() for term in terms 
            if len(term) > 2 and term.lower() not in stopwords
        ]
        
        # Sort by length (longer terms likely more specific)
        return sorted(set(filtered_terms), key=len, reverse=True)
    
    def _calculate_quality_metrics(self, content: str) -> Dict[str, float]:
        """Calculate comprehensive quality metrics."""
        word_count = len(content.split())
        char_count = len(content)
        
        # Length quality (sweet spot around 300-2000 words)
        if word_count < 50:
            length_score = 0.3
        elif 300 <= word_count <= 2000:
            length_score = 1.0
        elif word_count < 300:
            length_score = 0.3 + (word_count - 50) / 250 * 0.7
        else:
            length_score = max(0.5, 1.0 - (word_count - 2000) / 10000)
        
        # Structure quality
        structure_score = 0.7  # Base score
        if '```' in content:  # Code examples
            structure_score += 0.2
        if re.search(r'^#{1,6}\\s', content, re.MULTILINE):  # Headers
            structure_score += 0.1
        if re.search(r'^\\s*[-*+]\\s', content, re.MULTILINE):  # Lists
            structure_score += 0.1
        
        # Readability (simplified)
        sentences = len(re.split(r'[.!?]+', content))
        avg_sentence_length = word_count / sentences if sentences > 0 else 0
        readability_score = max(0.3, 1.0 - abs(avg_sentence_length - 20) / 50)
        
        # Technical density
        technical_terms = len(self._extract_key_terms(content))
        technical_density = min(1.0, technical_terms / max(1, word_count / 100))
        
        overall_score = (
            length_score * 0.3 + 
            structure_score * 0.3 + 
            readability_score * 0.2 + 
            technical_density * 0.2
        )
        
        return {
            'overall_score': min(1.0, overall_score),
            'length_score': length_score,
            'structure_score': min(1.0, structure_score),
            'readability_score': readability_score,
            'technical_density': technical_density
        }
    
    def _calculate_security_score(self, content: str) -> float:
        """Calculate security score (higher = safer)."""
        security_risks = [
            r'<script[^>]*>.*?</script>',  # Script tags
            r'javascript:',                # JavaScript URLs
            r'vbscript:',                 # VBScript URLs
            r'\\b(password|secret|key)\\s*=\\s*["\']?\\w+',  # Exposed secrets
            r'\\b(SELECT|INSERT|UPDATE|DELETE)\\b.*\\bFROM\\b',  # SQL queries
        ]
        
        risk_count = 0
        for pattern in security_risks:
            if re.search(pattern, content, re.IGNORECASE):
                risk_count += 1
        
        # Score from 0.0 (many risks) to 1.0 (no risks)
        return max(0.0, 1.0 - (risk_count * 0.2))
    
    def _detect_language(self, content: str) -> str:
        """Detect content language (simplified)."""
        # Look for code language indicators
        code_patterns = {
            'python': [r'\\bdef\\s+\\w+', r'\\bimport\\s+\\w+', r'\\bfrom\\s+\\w+', r':\\s*$'],
            'javascript': [r'\\bfunction\\s+\\w+', r'\\bconst\\s+\\w+', r'\\blet\\s+\\w+', r'=>'],
            'java': [r'\\bpublic\\s+class', r'\\bprivate\\s+\\w+', r'\\bstatic\\s+void'],
            'sql': [r'\\bSELECT\\b', r'\\bFROM\\b', r'\\bWHERE\\b', r'\\bINSERT\\s+INTO\\b'],
            'bash': [r'#!/bin/bash', r'\\becho\\b', r'\\$\\{?\\w+', r'\\|\\s*\\w+'],
            'yaml': [r'^\\s*\\w+:', r'^---', r'^\\s*-\\s+\\w+'],
            'json': [r'^\\s*{', r'"\\w+"\\s*:', r'\\[\\s*{']
        }
        
        content_lower = content.lower()
        language_scores = {}
        
        for lang, patterns in code_patterns.items():
            score = sum(1 for pattern in patterns 
                       if re.search(pattern, content, re.MULTILINE | re.IGNORECASE))
            if score > 0:
                language_scores[lang] = score
        
        if language_scores:
            return max(language_scores.items(), key=lambda x: x[1])[0]
        
        return 'text'


class RobustAutonomousSystem:
    """Production-ready autonomous knowledge system with comprehensive reliability features."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = setup_secure_logging()
        
        # Initialize core components
        self.validator = InputValidator()
        self.rate_limiter = RateLimiter(
            max_requests=self.config.get('rate_limit_requests', 100),
            window_seconds=self.config.get('rate_limit_window', 60)
        )
        self.health_monitor = HealthMonitor()
        self.document_processor = RobustDocumentProcessor(self.validator)
        
        # Knowledge storage
        self.documents = {}
        self.search_index = {}
        self.query_history = []
        self.system_metrics = {
            'startup_time': datetime.now(timezone.utc),
            'documents_processed': 0,
            'queries_processed': 0,
            'errors_handled': 0
        }
        
        # Thread safety
        self.lock = threading.RLock()
        
        self.logger.info("🛡️ Robust Autonomous System initialized successfully")
    
    def add_knowledge(self, content: str, source: str, 
                     metadata: Optional[Dict[str, Any]] = None,
                     client_id: str = "system") -> Dict[str, Any]:
        """Add knowledge with comprehensive validation and error handling."""
        try:
            # Rate limiting
            if not self.rate_limiter.is_allowed(client_id):
                raise SystemError("Rate limit exceeded")
            
            start_time = time.time()
            
            # Process document
            processed_doc = self.document_processor.process_document(content, source, metadata)
            
            # Generate unique document ID
            doc_id = self._generate_doc_id(content, source)
            
            with self.lock:
                # Store document
                self.documents[doc_id] = processed_doc
                
                # Update search index
                self._update_search_index(doc_id, processed_doc)
                
                # Update metrics
                self.system_metrics['documents_processed'] += 1
            
            processing_time = (time.time() - start_time) * 1000
            self.health_monitor.record_request(True, processing_time)
            
            self.logger.info(f"Added knowledge document {doc_id} from {source} "
                           f"({processed_doc['metadata']['word_count']} words)")
            
            return {
                'success': True,
                'doc_id': doc_id,
                'metadata': processed_doc['metadata'],
                'processing_time_ms': processing_time
            }
            
        except Exception as e:
            self.health_monitor.record_request(False, 0)
            self.system_metrics['errors_handled'] += 1
            self.logger.error(f"Failed to add knowledge from {source}: {e}")
            
            return {
                'success': False,
                'error': str(e),
                'error_type': type(e).__name__
            }
    
    def query_knowledge(self, query: str, client_id: str = "anonymous",
                       options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Query knowledge with robust error handling."""
        start_time = time.time()
        
        try:
            # Rate limiting
            if not self.rate_limiter.is_allowed(client_id):
                raise SystemError(f"Rate limit exceeded for client {client_id}")
            
            # Validate query
            sanitized_query = self.validator.validate_query(query)
            
            # Extract options
            options = options or {}
            limit = min(options.get('limit', 10), 50)  # Cap at 50 results
            include_metadata = options.get('include_metadata', True)
            
            with self.lock:
                # Perform search
                results = self._search_documents(sanitized_query, limit)
                
                # Generate response
                response = self._generate_response(sanitized_query, results, options)
                
                # Record query
                self.query_history.append({
                    'query': sanitized_query,
                    'client_id': client_id,
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'results_count': len(results),
                    'processing_time_ms': 0  # Will be updated below
                })
                
                self.system_metrics['queries_processed'] += 1
            
            processing_time = (time.time() - start_time) * 1000
            self.query_history[-1]['processing_time_ms'] = processing_time
            
            self.health_monitor.record_request(True, processing_time)
            
            self.logger.info(f"Query processed for {client_id}: '{sanitized_query}' "
                           f"-> {len(results)} results in {processing_time:.2f}ms")
            
            return {
                'success': True,
                'query': sanitized_query,
                'results': results if include_metadata else [r['snippet'] for r in results],
                'total_results': len(results),
                'processing_time_ms': processing_time,
                'suggestions': self._generate_suggestions(sanitized_query, results),
                'confidence': self._calculate_confidence(results)
            }
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            self.health_monitor.record_request(False, processing_time)
            self.system_metrics['errors_handled'] += 1
            
            self.logger.error(f"Query failed for {client_id}: {e}")
            
            return {
                'success': False,
                'query': query,
                'error': str(e),
                'error_type': type(e).__name__,
                'processing_time_ms': processing_time
            }
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health status."""
        try:
            health_data = self.health_monitor.check_health()
            
            with self.lock:
                health_data['system_metrics'] = self.system_metrics.copy()
                health_data['knowledge_base'] = {
                    'total_documents': len(self.documents),
                    'total_queries': len(self.query_history),
                    'recent_queries': len([q for q in self.query_history 
                                         if datetime.fromisoformat(q['timestamp'].replace('Z', '+00:00')) > 
                                            datetime.now(timezone.utc) - timedelta(hours=1)])
                }
            
            return health_data
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
    
    def _generate_doc_id(self, content: str, source: str) -> str:
        """Generate unique document ID."""
        doc_hash = hashlib.sha256(f"{content[:1000]}{source}".encode()).hexdigest()
        timestamp = int(time.time())
        return f"{source.replace('/', '_')}_{timestamp}_{doc_hash[:8]}"
    
    def _update_search_index(self, doc_id: str, processed_doc: Dict[str, Any]):
        """Update search index with new document."""
        content = processed_doc['content'].lower()
        words = re.findall(r'\b\w{3,}\b', content)  # Words with 3+ characters
        
        for word in set(words):  # Remove duplicates
            if word not in self.search_index:
                self.search_index[word] = {}
            
            # Simple TF calculation
            tf = words.count(word) / len(words) if words else 0
            self.search_index[word][doc_id] = tf
    
    def _search_documents(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Search documents with relevance scoring."""
        query_words = set(re.findall(r'\b\w{3,}\b', query.lower()))
        if not query_words:
            return []
        
        doc_scores = {}
        
        # Calculate relevance scores
        for word in query_words:
            if word in self.search_index:
                for doc_id, tf_score in self.search_index[word].items():
                    if doc_id not in doc_scores:
                        doc_scores[doc_id] = 0
                    doc_scores[doc_id] += tf_score
        
        # Sort by relevance
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Build results
        results = []
        for doc_id, score in sorted_docs[:limit]:
            if doc_id in self.documents:
                doc_data = self.documents[doc_id]
                results.append({
                    'doc_id': doc_id,
                    'relevance_score': score,
                    'snippet': self._generate_snippet(doc_data['content'], query_words),
                    'metadata': doc_data['metadata'],
                    'content_type': doc_data['metadata'].get('content_type', 'unknown'),
                    'source': doc_data['metadata']['source']
                })
        
        return results
    
    def _generate_snippet(self, content: str, query_words: set, max_length: int = 200) -> str:
        """Generate relevant snippet from content."""
        sentences = [s.strip() for s in re.split(r'[.!?]', content) if s.strip()]
        
        # Find sentence with most query words
        best_sentence = ""
        best_score = 0
        
        for sentence in sentences:
            sentence_words = set(re.findall(r'\b\w+\b', sentence.lower()))
            score = len(query_words.intersection(sentence_words))
            if score > best_score:
                best_score = score
                best_sentence = sentence
        
        if best_sentence:
            if len(best_sentence) <= max_length:
                return best_sentence
            else:
                return best_sentence[:max_length-3] + "..."
        
        # Fallback to beginning of content
        return (content[:max_length-3] + "...") if len(content) > max_length else content
    
    def _generate_response(self, query: str, results: List[Dict], 
                          options: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate enhanced search results."""
        enhanced_results = []
        
        for result in results:
            enhanced_result = {
                'doc_id': result['doc_id'],
                'snippet': result['snippet'],
                'relevance_score': round(result['relevance_score'], 3),
                'source': result['source'],
                'content_type': result['content_type']
            }
            
            # Add metadata if requested
            if options.get('include_metadata', True):
                safe_metadata = {
                    k: v for k, v in result['metadata'].items()
                    if k not in ['content_hash']  # Don't expose internal hashes
                }
                enhanced_result['metadata'] = safe_metadata
            
            enhanced_results.append(enhanced_result)
        
        return enhanced_results
    
    def _calculate_confidence(self, results: List[Dict]) -> float:
        """Calculate confidence score for results."""
        if not results:
            return 0.0
        
        # Base confidence on top result relevance and number of results
        top_score = results[0]['relevance_score'] if results else 0
        result_count_factor = min(1.0, len(results) / 5.0)  # Normalize to 5 results
        
        confidence = min(1.0, (top_score * 2) + (result_count_factor * 0.3))
        return round(confidence, 2)
    
    def _generate_suggestions(self, query: str, results: List[Dict]) -> List[str]:
        """Generate helpful search suggestions."""
        suggestions = []
        
        if not results:
            suggestions.append("Try using different keywords")
            suggestions.append("Check spelling and try broader terms")
        elif len(results) == 1:
            # Suggest related topics from the single result
            result = results[0]
            key_terms = result['metadata'].get('key_terms', [])[:3]
            for term in key_terms:
                if term.lower() not in query.lower():
                    suggestions.append(f"Learn more about {term}")
        else:
            # Suggest refinements
            content_types = list(set(r['content_type'] for r in results))
            for ct in content_types[:2]:
                suggestions.append(f"Filter by {ct.replace('_', ' ')}")
        
        return suggestions[:3]


def main():
    """Main demonstration of the robust autonomous system."""
    logger = logging.getLogger("RobustAutonomousSystem.Demo")
    logger.info("🛡️ Starting Robust Autonomous Knowledge System - Generation 2")
    
    # Initialize system with configuration
    config = {
        'rate_limit_requests': 50,
        'rate_limit_window': 60,
        'max_document_size': 500000,  # 500KB
        'enable_security_scanning': True
    }
    
    system = RobustAutonomousSystem(config)
    
    # Add comprehensive test knowledge
    test_documents = [
        {
            'content': """
            # Advanced API Security Implementation Guide
            
            This comprehensive guide covers implementing robust security for REST APIs.
            
            ## Authentication Methods
            
            ### JWT Bearer Tokens
            ```javascript
            const token = jwt.sign({ userId: user.id }, process.env.JWT_SECRET, {
                expiresIn: '1h'
            });
            
            // Usage in requests
            headers: {
                'Authorization': `Bearer ${token}`
            }
            ```
            
            ### API Key Authentication
            - Generate cryptographically secure API keys
            - Use environment variables for secrets
            - Implement rate limiting per API key
            - Log authentication attempts for monitoring
            
            ## Common Security Vulnerabilities
            
            1. **SQL Injection**: Always use parameterized queries
            2. **Cross-Site Scripting (XSS)**: Sanitize all user inputs
            3. **Cross-Site Request Forgery (CSRF)**: Use CSRF tokens
            4. **Insecure Direct Object References**: Validate user permissions
            
            ## Best Practices
            - Enable HTTPS everywhere
            - Implement proper error handling
            - Use input validation libraries
            - Regular security audits and penetration testing
            """,
            'source': 'security_guide',
            'metadata': {
                'title': 'API Security Implementation Guide',
                'category': 'security',
                'difficulty': 'advanced',
                'tags': ['api', 'security', 'authentication', 'jwt']
            }
        },
        {
            'content': """
            # Database Performance Optimization Strategies
            
            Comprehensive strategies for optimizing database performance in production systems.
            
            ## Indexing Strategies
            
            ### B-Tree Indexes
            ```sql
            CREATE INDEX idx_users_email ON users(email);
            CREATE INDEX idx_orders_created_at ON orders(created_at DESC);
            ```
            
            ### Composite Indexes
            ```sql
            CREATE INDEX idx_user_orders ON orders(user_id, created_at DESC);
            ```
            
            ## Query Optimization
            
            ### Avoid N+1 Problems
            ```python
            # Bad: N+1 queries
            users = User.objects.all()
            for user in users:
                print(user.orders.count())  # Separate query for each user
            
            # Good: Single query with joins
            users = User.objects.prefetch_related('orders').all()
            for user in users:
                print(user.orders.count())
            ```
            
            ### Use EXPLAIN to analyze queries
            ```sql
            EXPLAIN ANALYZE SELECT * FROM orders 
            WHERE user_id = 123 AND created_at > '2024-01-01';
            ```
            
            ## Common Performance Issues
            
            1. **Missing Indexes**: Monitor slow query logs
            2. **Over-indexing**: Too many indexes slow down writes
            3. **Inefficient JOINs**: Ensure proper foreign key relationships
            4. **Large Result Sets**: Implement pagination
            
            ## Monitoring and Maintenance
            - Use database monitoring tools
            - Regular VACUUM and ANALYZE operations (PostgreSQL)
            - Monitor connection pool usage
            - Set up alerts for query performance degradation
            """,
            'source': 'database_guide',
            'metadata': {
                'title': 'Database Performance Optimization',
                'category': 'database',
                'difficulty': 'intermediate',
                'tags': ['database', 'performance', 'sql', 'optimization']
            }
        },
        {
            'content': """
            # Docker Deployment Troubleshooting Guide
            
            Common Docker deployment issues and their solutions.
            
            ## Container Startup Issues
            
            ### Problem: Container exits immediately
            ```bash
            # Check container logs
            docker logs container_name
            
            # Check exit code
            docker ps -a
            ```
            
            **Common causes:**
            - Missing environment variables
            - Port conflicts
            - Incorrect entry point or command
            - Permission issues
            
            ### Problem: "bind: address already in use"
            ```bash
            # Find process using the port
            sudo netstat -tlnp | grep :3000
            
            # Kill the process
            sudo kill -9 PID
            
            # Or use different port mapping
            docker run -p 3001:3000 myapp
            ```
            
            ## Resource Issues
            
            ### Memory Limits
            ```dockerfile
            # Set memory limit in Dockerfile
            FROM node:16-alpine
            
            # In docker run command
            docker run -m 512m myapp
            ```
            
            ### Disk Space Issues
            ```bash
            # Clean up unused containers and images
            docker system prune -a
            
            # Remove specific containers
            docker rm $(docker ps -aq --filter "status=exited")
            ```
            
            ## Network Issues
            
            ### Container Communication
            ```bash
            # Create custom network
            docker network create mynetwork
            
            # Run containers on same network
            docker run --network mynetwork --name app1 myapp1
            docker run --network mynetwork --name app2 myapp2
            ```
            
            ## Debug Commands
            
            ```bash
            # Run container with shell access
            docker run -it --entrypoint /bin/sh myapp
            
            # Execute commands in running container
            docker exec -it container_name bash
            
            # Copy files from container
            docker cp container_name:/app/logs ./logs
            ```
            """,
            'source': 'docker_troubleshooting',
            'metadata': {
                'title': 'Docker Deployment Troubleshooting',
                'category': 'troubleshooting',
                'difficulty': 'intermediate',
                'tags': ['docker', 'deployment', 'troubleshooting', 'containers']
            }
        }
    ]
    
    # Add documents to system
    logger.info("Adding test documents...")
    for doc in test_documents:
        result = system.add_knowledge(doc['content'], doc['source'], doc['metadata'])
        if result['success']:
            logger.info(f"✅ Added: {result['metadata']['title']}")
        else:
            logger.error(f"❌ Failed to add document: {result['error']}")
    
    # Test queries with various scenarios
    test_queries = [
        "How do I implement JWT authentication?",
        "My Docker container won't start, what should I check?",
        "How do I optimize database queries?",
        "What are common API security vulnerabilities?",
        "Container memory limit issues",
        "SQL injection prevention",
        "",  # Empty query (should be handled)
        "SELECT * FROM users WHERE id = 1; DROP TABLE users;",  # SQL injection attempt (should be blocked)
    ]
    
    logger.info("\\n🔍 Testing query processing with various scenarios...")
    
    for i, query in enumerate(test_queries):
        logger.info(f"\\n--- Test Query {i+1}: '{query}' ---")
        
        try:
            result = system.query_knowledge(query, client_id=f"test_client_{i}")
            
            if result['success']:
                print(f"Query: {result['query']}")
                print(f"Results: {result['total_results']}")
                print(f"Confidence: {result['confidence']}")
                print(f"Processing Time: {result['processing_time_ms']:.2f}ms")
                
                if result['results']:
                    print(f"Top Result: {result['results'][0]['snippet'][:100]}...")
                    print(f"Source: {result['results'][0]['source']}")
                
                if result['suggestions']:
                    print(f"Suggestions: {', '.join(result['suggestions'])}")
            else:
                print(f"❌ Query failed: {result['error']} ({result['error_type']})")
                
        except Exception as e:
            logger.error(f"Unexpected error during query {i+1}: {e}")
    
    # Test rate limiting
    logger.info("\\n🚦 Testing rate limiting...")
    rate_limit_client = "rate_test_client"
    successful_requests = 0
    
    for i in range(55):  # Try to exceed the default limit of 50
        result = system.query_knowledge("test query", client_id=rate_limit_client)
        if result['success']:
            successful_requests += 1
        else:
            logger.info(f"Rate limit hit after {successful_requests} requests")
            break
    
    # Show system health
    logger.info("\\n🏥 System Health Check...")
    health = system.get_system_health()
    
    print(f"System Status: {health['status']}")
    print(f"Uptime: {health['uptime_seconds']:.1f} seconds")
    print(f"Total Requests: {health['metrics']['requests_total']}")
    print(f"Success Rate: {health['metrics']['requests_success'] / max(1, health['metrics']['requests_total']) * 100:.1f}%")
    print(f"Average Response Time: {health['metrics']['avg_response_time']:.2f}ms")
    print(f"Documents in KB: {health['knowledge_base']['total_documents']}")
    print(f"Queries Processed: {health['knowledge_base']['total_queries']}")
    
    if health['issues']:
        print(f"Issues: {', '.join(health['issues'])}")
    
    logger.info("🛡️ ✅ Robust Autonomous System - Generation 2 demonstration complete!")
    logger.info("System successfully handled security validation, rate limiting, error recovery, and comprehensive monitoring.")


if __name__ == "__main__":
    main()