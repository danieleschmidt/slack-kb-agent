"""Test multi-modal search capabilities."""

import pytest
from unittest.mock import patch, MagicMock

from slack_kb_agent.multimodal_search import (
    MultiModalSearchEngine,
    CodeAnalyzer,
    DocumentStructureAnalyzer,
    MultiModalDocument,
    ContentModality,
    create_multimodal_search_engine,
    analyze_documents_multimodal
)
from slack_kb_agent.models import Document


class TestCodeAnalyzer:
    """Test code analysis functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.analyzer = CodeAnalyzer()
    
    def test_python_language_detection(self):
        """Test Python language detection."""
        python_code = """
        def authenticate_user(username, password):
            import hashlib
            from datetime import datetime
            
            class UserAuthenticator:
                def __init__(self):
                    pass
                    
                def verify_credentials(self, user, pwd):
                    return True
        """
        
        features = self.analyzer.analyze_code(python_code)
        
        assert 'python' in features['language']
        assert len(features['functions']) >= 2
        assert len(features['classes']) >= 1
        assert len(features['imports']) >= 2
    
    def test_javascript_language_detection(self):
        """Test JavaScript language detection."""
        js_code = """
        function authenticateUser(username, password) {
            const jwt = require('jsonwebtoken');
            let isValid = false;
            var token = generateToken();
            
            return {
                valid: isValid,
                token: token
            };
        }
        
        const config = {
            secret: 'key'
        };
        """
        
        features = self.analyzer.analyze_code(js_code)
        
        assert 'javascript' in features['language']
        assert 'authenticateUser' in features['functions']
    
    def test_java_language_detection(self):
        """Test Java language detection."""
        java_code = """
        import java.util.List;
        import java.security.MessageDigest;
        
        public class AuthenticationService {
            public static void main(String[] args) {
                System.out.println("Auth service started");
            }
            
            public boolean authenticate(String username, String password) {
                return true;
            }
        }
        """
        
        features = self.analyzer.analyze_code(java_code)
        
        assert 'java' in features['language']
        assert len(features['imports']) >= 2
        assert 'AuthenticationService' in features['classes']
    
    def test_sql_language_detection(self):
        """Test SQL language detection."""
        sql_code = """
        SELECT users.id, users.username, profiles.email
        FROM users 
        JOIN profiles ON users.id = profiles.user_id
        WHERE users.active = true;
        
        INSERT INTO user_sessions (user_id, session_token, created_at)
        VALUES (1, 'abc123', NOW());
        
        CREATE TABLE user_permissions (
            id SERIAL PRIMARY KEY,
            user_id INTEGER REFERENCES users(id),
            permission_name VARCHAR(255)
        );
        
        ALTER TABLE users ADD COLUMN last_login TIMESTAMP;
        """
        
        features = self.analyzer.analyze_code(sql_code)
        
        assert 'sql' in features['language']
    
    def test_function_extraction(self):
        """Test function name extraction."""
        multi_lang_code = """
        # Python functions
        def process_data(input_data):
            pass
            
        def validate_input():
            return True
        
        // JavaScript functions
        function handleRequest(req, res) {
            return response;
        }
        
        const processResponse = function(data) {
            return processed;
        };
        
        // Java methods
        public int calculateSum(int a, int b) {
            return a + b;
        }
        """
        
        features = self.analyzer.analyze_code(multi_lang_code)
        
        functions = features['functions']
        assert 'process_data' in functions
        assert 'validate_input' in functions
        assert 'handleRequest' in functions
        assert 'calculateSum' in functions
    
    def test_class_extraction(self):
        """Test class name extraction."""
        class_code = """
        class UserManager:
            def __init__(self):
                pass
        
        class AuthenticationService {
            private String secret;
        }
        
        interface UserRepository {
            User findById(int id);
        }
        """
        
        features = self.analyzer.analyze_code(class_code)
        
        classes = features['classes']
        assert 'UserManager' in classes
        assert 'AuthenticationService' in classes
        assert 'UserRepository' in classes
    
    def test_import_extraction(self):
        """Test import statement extraction."""
        import_code = """
        import os
        import json
        from datetime import datetime
        from flask import Flask, request
        
        const jwt = require('jsonwebtoken');
        const bcrypt = require('bcrypt');
        
        import java.util.List;
        import javax.servlet.http.HttpServlet;
        
        #include <stdio.h>
        #include <stdlib.h>
        """
        
        features = self.analyzer.analyze_code(import_code)
        
        imports = features['imports']
        assert 'os' in imports
        assert 'datetime' in imports
        assert 'jsonwebtoken' in imports
        assert 'java.util.List' in imports
        assert 'stdio.h' in imports
    
    def test_complexity_calculation(self):
        """Test code complexity calculation."""
        complex_code = """
        def complex_function(data):
            if data is None:
                return None
            
            for item in data:
                try:
                    if item.type == 'user':
                        while item.is_active:
                            process_user(item)
                            if item.needs_update:
                                update_item(item)
                        else:
                            deactivate_user(item)
                    elif item.type == 'admin':
                        process_admin(item)
                except Exception as e:
                    handle_error(e)
            
            return data
        
        class DataProcessor:
            def __init__(self):
                pass
        """
        
        features = self.analyzer.analyze_code(complex_code)
        
        # Should detect various complexity indicators
        complexity = features['complexity_score']
        assert complexity > 0
        assert isinstance(complexity, float)
    
    def test_documentation_ratio(self):
        """Test documentation ratio calculation."""
        documented_code = """
        \"\"\"
        This module provides authentication services.
        It includes user validation and token management.
        \"\"\"
        
        def authenticate_user(username, password):
            \"\"\"
            Authenticate a user with username and password.
            
            Args:
                username (str): The user's username
                password (str): The user's password
                
            Returns:
                bool: True if authentication successful
            \"\"\"
            # Check if username exists
            if not username:
                return False
            
            # Validate password strength
            if len(password) < 8:
                return False
                
            # Hash password and compare
            return verify_password(username, password)
        """
        
        features = self.analyzer.analyze_code(documented_code)
        
        doc_ratio = features['documentation_ratio']
        assert doc_ratio > 0.2  # Should have decent documentation ratio
    
    def test_code_block_extraction(self):
        """Test markdown code block extraction."""
        markdown_with_code = """
        # Authentication Guide
        
        Here's how to implement authentication:
        
        ```python
        import jwt
        
        def create_token(user_id):
            return jwt.encode({'user_id': user_id}, 'secret', algorithm='HS256')
        ```
        
        For JavaScript, use this:
        
        ```javascript
        function createToken(userId) {
            return jwt.sign({userId: userId}, 'secret');
        }
        ```
        
        Example without language specification:
        
        ```
        curl -X POST /api/auth -d '{"username":"user","password":"pass"}'
        ```
        """
        
        features = self.analyzer.analyze_code(markdown_with_code)
        
        code_blocks = features['code_blocks']
        assert len(code_blocks) == 3
        
        # Check language detection in code blocks
        languages = [block['language'] for block in code_blocks]
        assert 'python' in languages
        assert 'javascript' in languages
        assert 'unknown' in languages


class TestDocumentStructureAnalyzer:
    """Test document structure analysis."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.analyzer = DocumentStructureAnalyzer()
    
    def test_markdown_format_detection(self):
        """Test markdown format detection."""
        markdown_content = """
        # Main Title
        
        ## Section 1
        This is content
        
        ### Subsection
        More content here
        """
        
        structure = self.analyzer.analyze_structure(markdown_content, "test.md")
        
        assert structure['format'] == 'markdown'
    
    def test_html_format_detection(self):
        """Test HTML format detection."""
        html_content = """
        <html>
        <head><title>Test</title></head>
        <body>
        <h1>Main Title</h1>
        <p>Content here</p>
        </body>
        </html>
        """
        
        structure = self.analyzer.analyze_structure(html_content, "test.html")
        
        assert structure['format'] == 'html'
    
    def test_json_format_detection(self):
        """Test JSON format detection."""
        json_content = """
        {
            "title": "Test Document",
            "sections": [
                {"name": "Introduction", "content": "..."},
                {"name": "Details", "content": "..."}
            ]
        }
        """
        
        structure = self.analyzer.analyze_structure(json_content, "test.json")
        
        assert structure['format'] == 'json'
    
    def test_section_extraction(self):
        """Test section/heading extraction."""
        sectioned_content = """
        # Main Title
        Introduction content
        
        ## Getting Started
        Setup instructions
        
        ### Prerequisites
        Required software
        
        ## Advanced Topics
        Complex concepts
        
        ### Performance Optimization
        Speed improvements
        
        #### Caching Strategies
        Different cache types
        """
        
        structure = self.analyzer.analyze_structure(sectioned_content)
        
        sections = structure['sections']
        assert len(sections) >= 5
        
        # Check heading levels
        levels = [section['level'] for section in sections]
        assert 1 in levels  # H1
        assert 2 in levels  # H2
        assert 3 in levels  # H3
        assert 4 in levels  # H4
    
    def test_list_extraction(self):
        """Test list extraction."""
        list_content = """
        # Features
        
        ## Bullet Lists
        - Feature 1
        - Feature 2
        * Another feature
        + One more feature
        
        ## Numbered Lists
        1. First step
        2. Second step
        3. Third step
        
        ## Mixed Content
        Some text here
        - Bullet point
        1. Numbered item
        """
        
        structure = self.analyzer.analyze_structure(list_content)
        
        lists = structure['lists']
        assert lists['bullet_lists'] >= 4
        assert lists['numbered_lists'] >= 3
        assert lists['total_lists'] >= 7
    
    def test_link_extraction(self):
        """Test link extraction."""
        link_content = """
        # Resources
        
        Check out the [official documentation](https://example.com/docs)
        and the [API reference](https://api.example.com/reference).
        
        Also visit https://github.com/example/repo for source code.
        
        Download from https://releases.example.com/latest
        """
        
        structure = self.analyzer.analyze_structure(link_content)
        
        links = structure['links']
        assert len(links) >= 4
        
        # Check different link types
        link_types = [link['type'] for link in links]
        assert 'markdown' in link_types
        assert 'plain' in link_types
    
    def test_image_extraction(self):
        """Test image reference extraction."""
        image_content = """
        # Documentation
        
        ![Architecture Diagram](images/architecture.png)
        
        ![User Flow](https://example.com/images/user-flow.svg)
        
        ![Screenshot](./screenshots/login.jpg)
        """
        
        structure = self.analyzer.analyze_structure(image_content)
        
        images = structure['images']
        assert len(images) == 3
        
        # Check image alt texts
        alt_texts = [img['alt'] for img in images]
        assert 'Architecture Diagram' in alt_texts
        assert 'User Flow' in alt_texts
        assert 'Screenshot' in alt_texts
    
    def test_table_detection(self):
        """Test table detection."""
        table_content = """
        # API Endpoints
        
        | Method | Endpoint | Description |
        |--------|----------|-------------|
        | GET    | /users   | Get all users |
        | POST   | /users   | Create user |
        | PUT    | /users/1 | Update user |
        
        ## Status Codes
        
        | Code | Meaning |
        |------|---------|
        | 200  | OK      |
        | 404  | Not Found |
        """
        
        structure = self.analyzer.analyze_structure(table_content)
        
        table_count = structure['tables']
        assert table_count >= 2
    
    def test_heading_hierarchy_analysis(self):
        """Test heading hierarchy analysis."""
        hierarchical_content = """
        # Main Document Title
        
        ## Section A
        Content A
        
        ### Subsection A.1
        Details A.1
        
        ### Subsection A.2
        Details A.2
        
        ## Section B
        Content B
        
        ### Subsection B.1
        Details B.1
        
        #### Sub-subsection B.1.1
        Very detailed content
        """
        
        structure = self.analyzer.analyze_structure(hierarchical_content)
        
        hierarchy = structure['headings_hierarchy']
        assert hierarchy['total'] >= 6
        assert hierarchy['max_depth'] == 4  # H1 through H4
        assert hierarchy['min_depth'] == 1  # H1
        assert 0 <= hierarchy['structure_score'] <= 1
    
    def test_readability_features(self):
        """Test readability feature extraction."""
        readable_content = """
        # Welcome to Our Service
        
        Our service provides amazing features. You can use it easily.
        It works great for everyone!
        
        ## Getting Started
        
        First, sign up for an account. Then, configure your settings.
        Finally, start using the service.
        
        ## Advanced Usage
        
        Power users can access advanced features. These include automation
        and custom integrations. Contact support for help.
        """
        
        structure = self.analyzer.analyze_structure(readable_content)
        
        readability = structure['readability_features']
        assert readability['words'] > 0
        assert readability['sentences'] > 0
        assert readability['paragraphs'] > 0
        assert readability['avg_words_per_sentence'] > 0
        assert readability['avg_sentences_per_paragraph'] > 0


class TestMultiModalSearchEngine:
    """Test the multi-modal search engine."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.engine = MultiModalSearchEngine()
    
    def test_code_document_analysis(self):
        """Test analysis of code documents."""
        code_doc = Document(
            content="""
            # Authentication Module
            
            ```python
            import jwt
            from datetime import datetime
            
            class AuthService:
                def __init__(self, secret_key):
                    self.secret_key = secret_key
                
                def generate_token(self, user_id):
                    payload = {
                        'user_id': user_id,
                        'exp': datetime.utcnow() + timedelta(hours=24)
                    }
                    return jwt.encode(payload, self.secret_key, algorithm='HS256')
                
                def verify_token(self, token):
                    try:
                        payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
                        return payload['user_id']
                    except jwt.ExpiredSignatureError:
                        return None
            ```
            
            ## Usage Example
            
            ```python
            auth = AuthService('your-secret-key')
            token = auth.generate_token(123)
            user_id = auth.verify_token(token)
            ```
            """,
            source="auth_module.md"
        )
        
        multimodal_doc = self.engine.analyze_document(code_doc)
        
        assert multimodal_doc.modality == ContentModality.CODE
        assert 'python' in multimodal_doc.content_features['language']
        assert 'AuthService' in multimodal_doc.content_features['classes']
        assert 'generate_token' in multimodal_doc.content_features['functions']
        assert 'lang:python' in multimodal_doc.semantic_tags
    
    def test_documentation_analysis(self):
        """Test analysis of documentation."""
        doc_content = Document(
            content="""
            # API Documentation
            
            ## Overview
            This API provides user management functionality.
            
            ## Endpoints
            
            ### GET /users
            Retrieve all users in the system.
            
            ### POST /users
            Create a new user account.
            
            ## Authentication
            All endpoints require authentication using JWT tokens.
            
            ## Rate Limiting
            API calls are limited to 1000 requests per hour.
            """,
            source="api_docs.md"
        )
        
        multimodal_doc = self.engine.analyze_document(doc_content)
        
        assert multimodal_doc.modality == ContentModality.DOCUMENT
        assert multimodal_doc.structure_metadata['format'] == 'markdown'
        assert multimodal_doc.structure_metadata['headings_hierarchy']['total'] >= 5
        assert 'format:markdown' in multimodal_doc.semantic_tags
    
    def test_text_content_analysis(self):
        """Test analysis of plain text content."""
        text_doc = Document(
            content="""
            Welcome to our platform! We provide excellent customer service
            and innovative solutions for your business needs.
            
            Our team is dedicated to helping you succeed. Contact us today
            to learn more about our services and pricing options.
            
            We serve clients worldwide and have offices in major cities.
            """,
            source="welcome.txt"
        )
        
        multimodal_doc = self.engine.analyze_document(text_doc)
        
        assert multimodal_doc.modality == ContentModality.TEXT
        assert multimodal_doc.content_features['word_count'] > 0
        assert multimodal_doc.content_features['sentence_count'] > 0
        assert 'length:medium' in multimodal_doc.semantic_tags or 'length:short' in multimodal_doc.semantic_tags
    
    def test_multimodal_search(self):
        """Test multi-modal search functionality."""
        # Create documents of different modalities
        code_doc = self.engine.analyze_document(Document(
            content="""
            ```python
            def authenticate_user(username, password):
                return validate_credentials(username, password)
            ```
            """,
            source="auth_code.md"
        ))
        
        doc_doc = self.engine.analyze_document(Document(
            content="""
            # Authentication Guide
            
            ## How to authenticate users
            Use the authentication API to verify user credentials.
            """,
            source="auth_guide.md"
        ))
        
        text_doc = self.engine.analyze_document(Document(
            content="Authentication is important for security and user management.",
            source="auth_info.txt"
        ))
        
        documents = [code_doc, doc_doc, text_doc]
        
        # Search for authentication-related content
        results = self.engine.search("authentication user login", documents, max_results=10)
        
        assert len(results) >= 3
        
        # Check that results are sorted by relevance
        scores = [score for doc, score in results]
        assert scores == sorted(scores, reverse=True)
        
        # All documents should have some relevance to authentication query
        for doc, score in results:
            assert score > 0.1
    
    def test_code_relevance_scoring(self):
        """Test code-specific relevance scoring."""
        code_doc = self.engine.analyze_document(Document(
            content="""
            ```python
            def authenticate_user(username, password):
                import hashlib
                hashed = hashlib.md5(password.encode()).hexdigest()
                return verify_hash(username, hashed)
            
            class UserAuthenticator:
                def login(self, credentials):
                    return self.authenticate_user(credentials.username, credentials.password)
            ```
            """,
            source="auth.py"
        ))
        
        # Query matching function name
        results = self.engine.search("authenticate_user", [code_doc])
        assert len(results) > 0
        assert results[0][1] > 0.5  # High relevance for exact function match
        
        # Query matching class name
        results = self.engine.search("UserAuthenticator", [code_doc])
        assert len(results) > 0
        assert results[0][1] > 0.5  # High relevance for exact class match
        
        # Query matching language
        results = self.engine.search("python", [code_doc])
        assert len(results) > 0
        assert results[0][1] > 0.3  # Good relevance for language match
    
    def test_document_structure_relevance(self):
        """Test document structure-based relevance scoring."""
        structured_doc = self.engine.analyze_document(Document(
            content="""
            # Database Management Guide
            
            ## Introduction
            This guide covers database operations.
            
            ## Database Setup
            Instructions for setting up your database.
            
            ## Query Optimization
            Tips for optimizing database queries.
            
            [Database Tutorial](https://example.com/db-tutorial)
            [Performance Guide](https://example.com/performance)
            """,
            source="db_guide.md"
        ))
        
        # Query matching heading
        results = self.engine.search("database setup", [structured_doc])
        assert len(results) > 0
        assert results[0][1] > 0.4  # Good relevance for heading match
        
        # Query matching link text
        results = self.engine.search("performance guide", [structured_doc])
        assert len(results) > 0
        assert results[0][1] > 0.3  # Decent relevance for link match
    
    def test_semantic_tag_relevance(self):
        """Test semantic tag-based relevance scoring."""
        tagged_doc = self.engine.analyze_document(Document(
            content="""
            ```javascript
            function complexCalculation(data) {
                // Very complex algorithm here
                for (let i = 0; i < data.length; i++) {
                    if (data[i].type === 'special') {
                        processSpecialData(data[i]);
                    }
                }
            }
            ```
            """,
            source="complex_js.md"
        ))
        
        # Should have language and complexity tags
        assert 'lang:javascript' in tagged_doc.semantic_tags
        complexity_tags = [tag for tag in tagged_doc.semantic_tags if tag.startswith('complexity:')]
        assert len(complexity_tags) > 0
        
        # Query matching semantic tags
        results = self.engine.search("javascript", [tagged_doc])
        assert len(results) > 0
        assert results[0][1] > 0.2  # Some relevance through tags


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_create_multimodal_search_engine(self):
        """Test multimodal search engine creation."""
        engine = create_multimodal_search_engine()
        
        assert isinstance(engine, MultiModalSearchEngine)
        assert isinstance(engine.code_analyzer, CodeAnalyzer)
        assert isinstance(engine.structure_analyzer, DocumentStructureAnalyzer)
    
    def test_analyze_documents_multimodal(self):
        """Test batch document analysis."""
        documents = [
            Document(content="# Tutorial\nStep by step guide", source="tutorial.md"),
            Document(content="```python\ndef hello():\n    pass\n```", source="code.py"),
            Document(content="Simple text content here", source="text.txt")
        ]
        
        analyzed_docs = analyze_documents_multimodal(documents)
        
        assert len(analyzed_docs) == 3
        
        # Check that different modalities were detected
        modalities = {doc.modality for doc in analyzed_docs}
        assert ContentModality.DOCUMENT in modalities
        assert ContentModality.CODE in modalities
        assert ContentModality.TEXT in modalities


if __name__ == "__main__":
    pytest.main([__file__])