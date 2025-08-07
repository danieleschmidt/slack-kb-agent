"""Test content curation system functionality."""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from slack_kb_agent.content_curation import (
    ContentCurationSystem,
    ContentQualityAssessor,
    ContentTypeClassifier,
    KnowledgeGapDetector,
    ContentMetrics,
    ContentQuality,
    ContentType,
    KnowledgeGap,
    KnowledgeGapAnalysis,
    CuratedContent
)
from slack_kb_agent.models import Document


class TestContentQualityAssessor:
    """Test content quality assessment functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.assessor = ContentQualityAssessor()
    
    def test_readability_assessment(self):
        """Test readability scoring."""
        # Good readability - simple sentences
        simple_content = "This is easy to read. Short sentences work well. Clear and simple."
        readable_score = self.assessor._assess_readability(simple_content)
        assert readable_score > 0.5
        
        # Poor readability - complex sentences
        complex_content = "This is an extraordinarily complicated and unnecessarily convoluted sentence that contains many subordinate clauses and complex terminology that makes it extremely difficult for the average reader to comprehend and understand."
        complex_score = self.assessor._assess_readability(complex_content)
        assert complex_score < readable_score
    
    def test_completeness_assessment(self):
        """Test content completeness scoring."""
        # Complete content with structure
        complete_content = """
        # How to Deploy Applications
        
        ## What is deployment?
        Deployment is the process...
        
        ## Why deploy?
        Benefits include:
        - Scalability
        - Reliability
        
        ## How to deploy:
        1. First step
        2. Second step
        
        ```bash
        docker build -t app .
        ```
        
        Example: Here's a sample deployment
        """
        
        complete_score = self.assessor._assess_completeness(complete_content)
        assert complete_score > 0.7
        
        # Incomplete content
        incomplete_content = "Deploy with docker"
        incomplete_score = self.assessor._assess_completeness(incomplete_content)
        assert incomplete_score < complete_score
    
    def test_freshness_assessment(self):
        """Test content freshness scoring."""
        # Current content
        current_content = "Updated in 2024. New features include..."
        current_score = self.assessor._assess_freshness(
            Document(content=current_content, source="test")
        )
        assert current_score > 0.5
        
        # Outdated content
        outdated_content = "This was deprecated in 2018. Old version 0.1..."
        outdated_score = self.assessor._assess_freshness(
            Document(content=outdated_content, source="test")
        )
        assert outdated_score < current_score
    
    def test_accuracy_confidence_assessment(self):
        """Test accuracy confidence scoring."""
        # High confidence content
        confident_content = "This is verified and documented. Official recommendation is..."
        confident_score = self.assessor._assess_accuracy_confidence(confident_content)
        assert confident_score > 0.7
        
        # Low confidence content
        uncertain_content = "This might work. Probably the right approach. Not sure about..."
        uncertain_score = self.assessor._assess_accuracy_confidence(uncertain_content)
        assert uncertain_score < confident_score
    
    def test_technical_depth_assessment(self):
        """Test technical depth scoring."""
        # High technical depth
        technical_content = """
        ```python
        def authenticate(api_key: str) -> bool:
            return validate_jwt_token(api_key)
        ```
        
        The AuthenticationService class implements OAuth 2.0 protocol...
        Database schema includes user_tokens table with indexes...
        """
        
        technical_score = self.assessor._assess_technical_depth(technical_content)
        assert technical_score > 0.5
        
        # Low technical depth
        simple_content = "Authentication is important for security."
        simple_score = self.assessor._assess_technical_depth(simple_content)
        assert simple_score < technical_score
    
    def test_structure_quality_assessment(self):
        """Test structure quality scoring."""
        # Well-structured content
        structured_content = """
        # Main Title
        
        ## Section 1
        Content here.
        
        ## Section 2
        - Point 1
        - Point 2
        
        1. Step one
        2. Step two
        
        ```code
        example
        ```
        """
        
        structured_score = self.assessor._assess_structure_quality(structured_content)
        assert structured_score > 0.5
        
        # Poorly structured content (wall of text)
        unstructured_content = "This is all one big paragraph with no structure or formatting or breaks or lists or headers or anything to help organize the information and make it easier to read and understand."
        unstructured_score = self.assessor._assess_structure_quality(unstructured_content)
        assert unstructured_score < structured_score
    
    def test_overall_quality_assessment(self):
        """Test complete quality assessment."""
        # High-quality document
        high_quality_doc = Document(
            content="""
            # Complete Guide to Docker Deployment
            
            ## Overview
            This guide provides step-by-step instructions for deploying applications with Docker.
            
            ## Prerequisites
            - Docker installed
            - Basic command line knowledge
            
            ## Steps
            1. Create Dockerfile
            2. Build image
            3. Run container
            
            ```dockerfile
            FROM node:16
            COPY . /app
            WORKDIR /app
            RUN npm install
            CMD ["npm", "start"]
            ```
            
            ## Example
            Here's a complete example...
            """,
            source="docker-guide.md"
        )
        
        metrics = self.assessor.assess_content_quality(high_quality_doc)
        assert metrics.overall_quality_score() > 0.6
        assert 0 <= metrics.readability_score <= 1
        assert 0 <= metrics.completeness_score <= 1
        assert 0 <= metrics.accuracy_confidence <= 1


class TestContentTypeClassifier:
    """Test content type classification."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.classifier = ContentTypeClassifier()
    
    def test_tutorial_classification(self):
        """Test tutorial content classification."""
        tutorial_content = """
        # Step-by-step Tutorial: Getting Started with React
        
        In this tutorial, we'll walk through creating your first React application.
        
        ## Step 1: Setup
        First, install Node.js...
        
        ## Step 2: Create Project
        Next, run the following command...
        """
        
        doc = Document(content=tutorial_content, source="react-tutorial.md")
        content_type, confidence = self.classifier.classify_content_type(doc)
        
        assert content_type == ContentType.TUTORIAL
        assert confidence > 0.0
    
    def test_reference_classification(self):
        """Test reference documentation classification."""
        reference_content = """
        # API Reference: User Management
        
        ## Authentication Methods
        
        ### login(username, password)
        **Parameters:**
        - username (string): User's login name
        - password (string): User's password
        
        **Returns:** Authentication token
        
        ### logout()
        Invalidates the current session.
        """
        
        doc = Document(content=reference_content, source="api-ref.md")
        content_type, confidence = self.classifier.classify_content_type(doc)
        
        assert content_type == ContentType.REFERENCE
        assert confidence > 0.0
    
    def test_troubleshooting_classification(self):
        """Test troubleshooting content classification."""
        troubleshooting_content = """
        # Troubleshooting: Login Issues
        
        ## Problem: User cannot login
        
        ### Symptoms
        - Error message: "Invalid credentials"
        - Login form not working
        
        ### Solution
        1. Check username and password
        2. Clear browser cache
        3. Reset password if needed
        
        This should fix the login problem.
        """
        
        doc = Document(content=troubleshooting_content, source="login-troubleshooting.md")
        content_type, confidence = self.classifier.classify_content_type(doc)
        
        assert content_type == ContentType.TROUBLESHOOTING
        assert confidence > 0.0
    
    def test_faq_classification(self):
        """Test FAQ content classification."""
        faq_content = """
        # Frequently Asked Questions
        
        Q: How do I reset my password?
        A: Click on the "Forgot Password" link on the login page.
        
        Q: Can I change my username?
        A: Yes, go to Profile Settings and update your username.
        
        Q: What browsers are supported?
        A: We support Chrome, Firefox, Safari, and Edge.
        """
        
        doc = Document(content=faq_content, source="faq.md")
        content_type, confidence = self.classifier.classify_content_type(doc)
        
        assert content_type == ContentType.FAQ
        assert confidence > 0.0
    
    def test_code_example_classification(self):
        """Test code example classification."""
        code_content = """
        # Authentication Code Examples
        
        Here are some code samples for implementing authentication:
        
        ```python
        import jwt
        
        def create_token(user_id):
            payload = {'user_id': user_id}
            return jwt.encode(payload, secret_key, algorithm='HS256')
        
        def verify_token(token):
            try:
                payload = jwt.decode(token, secret_key, algorithms=['HS256'])
                return payload['user_id']
            except jwt.InvalidTokenError:
                return None
        ```
        
        Example usage:
        ```python
        token = create_token(123)
        user_id = verify_token(token)
        ```
        """
        
        doc = Document(content=code_content, source="auth-examples.md")
        content_type, confidence = self.classifier.classify_content_type(doc)
        
        assert content_type == ContentType.CODE_EXAMPLE
        assert confidence > 0.0


class TestKnowledgeGapDetector:
    """Test knowledge gap detection."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.detector = KnowledgeGapDetector()
    
    def test_topic_coverage_analysis(self):
        """Test topic coverage analysis."""
        documents = [
            Document(content="How to use authentication with JWT tokens", source="auth.md"),
            Document(content="Database queries with PostgreSQL", source="db.md"),
            Document(content="Authentication best practices", source="auth2.md"),
            # Missing: deployment, monitoring, security, etc.
        ]
        
        coverage = self.detector._analyze_topic_coverage(documents)
        
        # Should detect authentication and database topics
        assert coverage['authentication'] >= 2  # Two auth documents
        assert coverage['database'] >= 1  # One database document
        
        # Some topics should have zero coverage
        assert coverage['monitoring'] == 0
    
    def test_missing_topic_gap_detection(self):
        """Test detection of completely missing topics."""
        documents = [
            Document(content="Authentication guide", source="auth.md"),
            # Missing all other topics
        ]
        
        gaps = self.detector.analyze_knowledge_gaps(documents)
        
        # Should detect multiple missing topics
        missing_topic_gaps = [gap for gap in gaps if gap.gap_type == KnowledgeGap.MISSING_TOPIC]
        assert len(missing_topic_gaps) > 0
        
        # Should have high severity for missing topics
        high_severity_gaps = [gap for gap in missing_topic_gaps if gap.severity >= 0.8]
        assert len(high_severity_gaps) > 0
    
    def test_incomplete_coverage_gap_detection(self):
        """Test detection of insufficient topic coverage."""
        # Create documents with uneven coverage
        documents = []
        
        # Lots of auth documents
        for i in range(10):
            documents.append(Document(
                content=f"Authentication guide {i} with JWT and OAuth",
                source=f"auth{i}.md"
            ))
        
        # Only one deployment document
        documents.append(Document(
            content="Basic deployment with docker",
            source="deploy.md"
        ))
        
        gaps = self.detector.analyze_knowledge_gaps(documents)
        
        # Should detect incomplete coverage for deployment
        incomplete_gaps = [gap for gap in gaps if gap.gap_type == KnowledgeGap.INCOMPLETE_COVERAGE]
        deployment_gaps = [gap for gap in incomplete_gaps if 'deployment' in gap.topic]
        
        assert len(deployment_gaps) > 0
    
    def test_outdated_content_detection(self):
        """Test detection of outdated content."""
        from datetime import datetime, timedelta
        
        # Create old document
        old_doc = Document(
            content="This feature was deprecated in 2015. Use legacy API version 0.1.",
            source="old-guide.md"
        )
        # Simulate old timestamp
        old_doc.created_at = (datetime.now() - timedelta(days=400)).isoformat()
        
        # Create current document
        current_doc = Document(
            content="Updated guide for 2024. Use new API version 2.0.",
            source="new-guide.md"
        )
        current_doc.created_at = datetime.now().isoformat()
        
        documents = [old_doc, current_doc]
        gaps = self.detector.analyze_knowledge_gaps(documents)
        
        # Should detect outdated content
        outdated_gaps = [gap for gap in gaps if gap.gap_type == KnowledgeGap.OUTDATED_INFO]
        assert len(outdated_gaps) > 0
    
    def test_query_based_gap_detection(self):
        """Test gap detection based on unresolved queries."""
        query_analytics = {
            'low_satisfaction_queries': [
                'How to setup microservices?',
                'Microservices architecture guide',
                'Best practices for microservices',
                'Microservices deployment strategies',
                # Multiple similar queries about missing topic
            ]
        }
        
        documents = [
            Document(content="Basic API guide", source="api.md")
            # No microservices content
        ]
        
        gaps = self.detector.analyze_knowledge_gaps(documents, query_analytics)
        
        # Should detect missing microservices topic
        query_gaps = [gap for gap in gaps if gap.gap_type == KnowledgeGap.MISSING_TOPIC]
        microservices_gaps = [gap for gap in query_gaps if 'microservice' in gap.topic]
        
        assert len(microservices_gaps) > 0


class TestContentCurationSystem:
    """Test the complete content curation system."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.json')
        self.temp_file.close()
        self.system = ContentCurationSystem(self.temp_file.name)
    
    def teardown_method(self):
        """Cleanup test fixtures."""
        Path(self.temp_file.name).unlink(missing_ok=True)
    
    def test_single_document_curation(self):
        """Test curating a single document."""
        doc = Document(
            content="""
            # Complete Docker Tutorial
            
            ## Introduction
            Docker is a containerization platform that allows you to package applications.
            
            ## Step-by-step Guide
            1. Install Docker
            2. Create Dockerfile
            3. Build image
            4. Run container
            
            ```dockerfile
            FROM node:16
            COPY . /app
            CMD ["npm", "start"]
            ```
            
            ## Example
            Here's a complete example of dockerizing a Node.js application.
            """,
            source="docker-tutorial.md"
        )
        
        curated = self.system.curate_document(doc)
        
        assert curated.content_type == ContentType.TUTORIAL
        assert curated.quality in [ContentQuality.GOOD, ContentQuality.EXCELLENT]
        assert curated.metrics.overall_quality_score() > 0.5
        assert len(curated.tags) > 0
        assert 'docker' in curated.tags or 'tutorial' in curated.tags
    
    def test_document_collection_curation(self):
        """Test curating a collection of documents."""
        documents = [
            Document(
                content="# API Reference\n\nFunction: authenticate(user, pass)\nReturns: token",
                source="api-ref.md"
            ),
            Document(
                content="# Troubleshooting Login\n\nProblem: Cannot login\nSolution: Reset password",
                source="login-troubleshooting.md"
            ),
            Document(
                content="Outdated guide from 2010. Use old methods.",
                source="old-guide.md"
            ),
            Document(
                content="""
                # Complete Deployment Guide
                
                ## Overview
                This comprehensive guide covers all aspects of deployment.
                
                ## Prerequisites
                - Docker installed
                - Kubernetes cluster
                
                ## Step-by-step Process
                1. Build application
                2. Create Docker image
                3. Deploy to Kubernetes
                
                ```yaml
                apiVersion: apps/v1
                kind: Deployment
                metadata:
                  name: my-app
                spec:
                  replicas: 3
                  selector:
                    matchLabels:
                      app: my-app
                  template:
                    metadata:
                      labels:
                        app: my-app
                    spec:
                      containers:
                      - name: my-app
                        image: my-app:latest
                        ports:
                        - containerPort: 8080
                ```
                
                ## Best Practices
                - Use proper resource limits
                - Implement health checks
                - Monitor deployments
                
                ## Troubleshooting
                Common issues and solutions...
                """,
                source="comprehensive-deploy.md"
            )
        ]
        
        curated_docs = self.system.curate_document_collection(documents)
        
        assert len(curated_docs) == 4
        
        # Check that different content types were detected
        content_types = {content.content_type for content in curated_docs.values()}
        assert ContentType.REFERENCE in content_types
        assert ContentType.TROUBLESHOOTING in content_types
        
        # Check quality distribution
        quality_levels = {content.quality for content in curated_docs.values()}
        assert ContentQuality.POOR in quality_levels  # Old guide should be poor
        assert ContentQuality.EXCELLENT in quality_levels or ContentQuality.GOOD in quality_levels  # Comprehensive guide
    
    def test_high_quality_content_identification(self):
        """Test identification of high-quality content."""
        # Add some documents first
        documents = [
            Document(
                content="""
                # Comprehensive Authentication Guide
                
                ## Overview
                This guide provides complete coverage of authentication strategies.
                
                ## What is Authentication?
                Authentication is the process of verifying identity...
                
                ## Why Use Authentication?
                Benefits include:
                - Security
                - User management
                - Access control
                
                ## How to Implement
                Step-by-step implementation:
                1. Choose authentication method
                2. Implement validation
                3. Secure token storage
                
                ```python
                import jwt
                
                def authenticate_user(username, password):
                    # Verify credentials
                    if verify_password(username, password):
                        return generate_token(username)
                    return None
                ```
                
                ## Examples
                Here are complete working examples...
                
                ## Best Practices
                - Use HTTPS
                - Implement rate limiting
                - Regular security audits
                """,
                source="auth-comprehensive.md"
            ),
            Document(
                content="Basic info",  # Poor quality
                source="basic.md"
            )
        ]
        
        self.system.curate_document_collection(documents)
        high_quality = self.system.get_high_quality_content(limit=5)
        
        assert len(high_quality) > 0
        # The comprehensive guide should be in high quality content
        high_quality_sources = {content.document.source for content in high_quality}
        assert "auth-comprehensive.md" in high_quality_sources
    
    def test_content_needing_improvement(self):
        """Test identification of content needing improvement."""
        poor_doc = Document(
            content="Old guide. Doesn't work. TODO: fix this.",
            source="poor-guide.md"
        )
        
        self.system.curate_document(poor_doc)
        needs_improvement = self.system.get_content_needing_improvement(limit=5)
        
        assert len(needs_improvement) > 0
        
        # Should have suggestions for improvement
        for content in needs_improvement:
            assert len(content.suggested_improvements) > 0
    
    def test_knowledge_gap_detection_integration(self):
        """Test knowledge gap detection in the system."""
        # Documents covering only a few topics
        documents = [
            Document(
                content="Authentication with JWT tokens and OAuth2",
                source="auth.md"
            )
            # Missing: deployment, monitoring, database, etc.
        ]
        
        self.system.curate_document_collection(documents)
        gaps = self.system.get_knowledge_gaps(min_severity=0.5)
        
        assert len(gaps) > 0
        
        # Should identify missing topics
        gap_topics = {gap.topic for gap in gaps}
        # Should detect missing deployment, monitoring, etc.
        assert any('deploy' in topic or 'monitor' in topic for topic in gap_topics)
    
    def test_content_priority_suggestions(self):
        """Test content priority suggestions."""
        # Create mixed quality content with gaps
        documents = [
            Document(content="Poor auth guide. TODO: improve", source="auth-poor.md"),
            Document(content="Good API reference documentation", source="api-good.md"),
            # Missing: deployment, monitoring
        ]
        
        self.system.curate_document_collection(documents)
        priorities = self.system.suggest_content_priorities()
        
        assert len(priorities) > 0
        
        # Should suggest improvements and additions
        priority_text = " ".join(priorities).lower()
        assert any(keyword in priority_text for keyword in ['critical', 'improve', 'add'])
    
    def test_content_by_type_filtering(self):
        """Test filtering content by type."""
        documents = [
            Document(
                content="# Step-by-step tutorial for beginners",
                source="tutorial.md"
            ),
            Document(
                content="# API Reference: function definitions",
                source="reference.md"
            ),
            Document(
                content="# Troubleshooting common errors",
                source="troubleshooting.md"
            )
        ]
        
        self.system.curate_document_collection(documents)
        
        tutorials = self.system.get_content_by_type(ContentType.TUTORIAL)
        references = self.system.get_content_by_type(ContentType.REFERENCE)
        troubleshooting = self.system.get_content_by_type(ContentType.TROUBLESHOOTING)
        
        assert len(tutorials) >= 1
        assert len(references) >= 1  
        assert len(troubleshooting) >= 1
    
    def test_content_tag_generation(self):
        """Test automatic tag generation."""
        doc = Document(
            content="""
            # Advanced Python API Development
            
            This guide covers advanced concepts for building APIs with Python.
            
            ```python
            import fastapi
            from fastapi import FastAPI
            
            app = FastAPI()
            ```
            
            Authentication, deployment, and monitoring are covered.
            """,
            source="python-api-advanced.md"
        )
        
        curated = self.system.curate_document(doc)
        
        # Should generate relevant tags
        assert 'python' in curated.tags
        assert 'advanced' in curated.tags
        assert 'api' in curated.tags


if __name__ == "__main__":
    pytest.main([__file__])