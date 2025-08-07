"""Multi-modal search capabilities for diverse content types.

This module provides advanced search capabilities across different content modalities:
- Text and semantic search
- Code and syntax-aware search  
- Image and visual content search (metadata-based)
- Document structure and format-aware search
- Cross-modal relevance scoring
"""

from __future__ import annotations

import re
import logging
import hashlib
from typing import Dict, List, Optional, Tuple, Set, Any, Union
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from .models import Document

logger = logging.getLogger(__name__)


class ContentModality(Enum):
    """Types of content modalities supported."""
    TEXT = "text"
    CODE = "code"
    IMAGE = "image"
    DOCUMENT = "document"
    STRUCTURED_DATA = "structured_data"
    MULTIMEDIA = "multimedia"


@dataclass
class MultiModalDocument:
    """Extended document with multi-modal capabilities."""
    document: Document
    modality: ContentModality
    content_features: Dict[str, Any]
    structure_metadata: Dict[str, Any]
    semantic_tags: Set[str]
    relevance_scores: Dict[str, float]


class CodeAnalyzer:
    """Analyze and extract features from code content."""
    
    def __init__(self):
        # Programming language patterns
        self.language_patterns = {
            'python': [
                r'def\s+\w+\s*\(',
                r'class\s+\w+\s*[:\(]',
                r'import\s+\w+',
                r'from\s+\w+\s+import'
            ],
            'javascript': [
                r'function\s+\w+\s*\(',
                r'const\s+\w+\s*=',
                r'let\s+\w+\s*=',
                r'var\s+\w+\s*='
            ],
            'java': [
                r'public\s+class\s+\w+',
                r'public\s+static\s+void\s+main',
                r'import\s+[\w\.]+;'
            ],
            'sql': [
                r'SELECT\s+.+\s+FROM',
                r'INSERT\s+INTO',
                r'CREATE\s+TABLE',
                r'ALTER\s+TABLE'
            ]
        }
    
    def analyze_code(self, content: str) -> Dict[str, Any]:
        """Analyze code content and extract features."""
        features = {
            'language': self._detect_language(content),
            'functions': self._extract_functions(content),
            'classes': self._extract_classes(content),
            'imports': self._extract_imports(content),
            'complexity_score': self._calculate_complexity(content),
            'documentation_ratio': self._calculate_doc_ratio(content),
            'code_blocks': self._extract_code_blocks(content)
        }
        
        return features
    
    def _detect_language(self, content: str) -> List[str]:
        """Detect programming languages in content."""
        detected_languages = []
        
        for language, patterns in self.language_patterns.items():
            pattern_matches = sum(
                1 for pattern in patterns 
                if re.search(pattern, content, re.IGNORECASE)
            )
            
            if pattern_matches >= 2:  # Require multiple pattern matches
                detected_languages.append(language)
        
        # Check for code blocks with language specification
        code_block_langs = re.findall(r'```(\w+)', content)
        detected_languages.extend(code_block_langs)
        
        return list(set(detected_languages))
    
    def _extract_functions(self, content: str) -> List[str]:
        """Extract function names from code."""
        function_patterns = [
            r'def\s+(\w+)\s*\(',  # Python
            r'function\s+(\w+)\s*\(',  # JavaScript
            r'public\s+\w+\s+(\w+)\s*\(',  # Java methods
            r'(\w+)\s*:\s*function\s*\(',  # JavaScript object methods
        ]
        
        functions = []
        for pattern in function_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            functions.extend(matches)
        
        return list(set(functions))
    
    def _extract_classes(self, content: str) -> List[str]:
        """Extract class names from code."""
        class_patterns = [
            r'class\s+(\w+)\s*[:\(]',  # Python/Java
            r'class\s+(\w+)\s*\{',  # Java/C++
            r'interface\s+(\w+)\s*\{',  # Java interfaces
        ]
        
        classes = []
        for pattern in class_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            classes.extend(matches)
        
        return list(set(classes))
    
    def _extract_imports(self, content: str) -> List[str]:
        """Extract import statements and dependencies."""
        import_patterns = [
            r'import\s+([\w\.]+)',  # Python/Java
            r'from\s+([\w\.]+)\s+import',  # Python
            r'require\s*\(\s*["\']([^"\']+)["\']',  # Node.js
            r'#include\s*<([^>]+)>',  # C/C++
        ]
        
        imports = []
        for pattern in import_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            imports.extend(matches)
        
        return list(set(imports))
    
    def _calculate_complexity(self, content: str) -> float:
        """Calculate code complexity score."""
        complexity_indicators = [
            r'if\s*\(',
            r'for\s*\(',
            r'while\s*\(',
            r'switch\s*\(',
            r'try\s*\{',
            r'catch\s*\(',
            r'def\s+\w+',
            r'function\s+\w+',
            r'class\s+\w+'
        ]
        
        total_complexity = sum(
            len(re.findall(pattern, content, re.IGNORECASE))
            for pattern in complexity_indicators
        )
        
        # Normalize by content length
        lines = len(content.split('\n'))
        return total_complexity / max(lines, 1)
    
    def _calculate_doc_ratio(self, content: str) -> float:
        """Calculate documentation to code ratio."""
        # Count comment lines
        comment_patterns = [
            r'#.*',  # Python/Shell comments
            r'//.*',  # C++/Java/JavaScript comments
            r'/\*.*?\*/',  # Multi-line comments
            r'""".*?"""',  # Python docstrings
            r"'''.*?'''",  # Python docstrings
        ]
        
        comment_lines = 0
        for pattern in comment_patterns:
            matches = re.findall(pattern, content, re.DOTALL)
            comment_lines += sum(match.count('\n') + 1 for match in matches)
        
        total_lines = len(content.split('\n'))
        return comment_lines / max(total_lines, 1)
    
    def _extract_code_blocks(self, content: str) -> List[Dict[str, str]]:
        """Extract code blocks from markdown content."""
        code_block_pattern = r'```(\w*)\s*\n?(.*?)\n?```'
        matches = re.findall(code_block_pattern, content, re.DOTALL)
        
        return [
            {'language': lang or 'unknown', 'code': code.strip()}
            for lang, code in matches
        ]


class DocumentStructureAnalyzer:
    """Analyze document structure and format."""
    
    def analyze_structure(self, content: str, filename: Optional[str] = None) -> Dict[str, Any]:
        """Analyze document structure and extract metadata."""
        structure = {
            'format': self._detect_format(content, filename),
            'sections': self._extract_sections(content),
            'lists': self._extract_lists(content),
            'links': self._extract_links(content),
            'images': self._extract_images(content),
            'tables': self._extract_tables(content),
            'headings_hierarchy': self._analyze_headings(content),
            'word_count': len(content.split()),
            'readability_features': self._extract_readability_features(content)
        }
        
        return structure
    
    def _detect_format(self, content: str, filename: Optional[str] = None) -> str:
        """Detect document format."""
        if filename:
            ext = Path(filename).suffix.lower()
            format_map = {
                '.md': 'markdown',
                '.rst': 'restructuredtext', 
                '.txt': 'plaintext',
                '.html': 'html',
                '.xml': 'xml',
                '.json': 'json',
                '.yaml': 'yaml',
                '.yml': 'yaml'
            }
            if ext in format_map:
                return format_map[ext]
        
        # Content-based detection
        if re.search(r'^#{1,6}\s+', content, re.MULTILINE):
            return 'markdown'
        elif '<html' in content.lower() or '<body' in content.lower():
            return 'html'
        elif content.strip().startswith('{') and content.strip().endswith('}'):
            return 'json'
        else:
            return 'plaintext'
    
    def _extract_sections(self, content: str) -> List[Dict[str, Any]]:
        """Extract document sections."""
        sections = []
        
        # Markdown headings
        heading_pattern = r'^(#{1,6})\s+(.+)$'
        matches = re.finditer(heading_pattern, content, re.MULTILINE)
        
        for match in matches:
            level = len(match.group(1))
            title = match.group(2).strip()
            sections.append({
                'level': level,
                'title': title,
                'position': match.start()
            })
        
        return sections
    
    def _extract_lists(self, content: str) -> Dict[str, int]:
        """Extract list information."""
        # Bullet lists
        bullet_lists = len(re.findall(r'^\s*[-*+]\s+', content, re.MULTILINE))
        
        # Numbered lists
        numbered_lists = len(re.findall(r'^\s*\d+\.\s+', content, re.MULTILINE))
        
        return {
            'bullet_lists': bullet_lists,
            'numbered_lists': numbered_lists,
            'total_lists': bullet_lists + numbered_lists
        }
    
    def _extract_links(self, content: str) -> List[Dict[str, str]]:
        """Extract links from content."""
        links = []
        
        # Markdown links
        md_links = re.findall(r'\[([^\]]+)\]\(([^)]+)\)', content)
        for text, url in md_links:
            links.append({'text': text, 'url': url, 'type': 'markdown'})
        
        # Plain URLs
        url_pattern = r'https?://[^\s<>"\']+[^\s<>"\'.,)]'
        urls = re.findall(url_pattern, content)
        for url in urls:
            links.append({'text': url, 'url': url, 'type': 'plain'})
        
        return links
    
    def _extract_images(self, content: str) -> List[Dict[str, str]]:
        """Extract image references."""
        images = []
        
        # Markdown images
        md_images = re.findall(r'!\[([^\]]*)\]\(([^)]+)\)', content)
        for alt_text, url in md_images:
            images.append({'alt': alt_text, 'url': url, 'type': 'markdown'})
        
        return images
    
    def _extract_tables(self, content: str) -> int:
        """Count tables in content."""
        # Markdown tables (look for table rows)
        table_rows = re.findall(r'\|.*\|', content)
        
        # Estimate number of tables (group consecutive rows)
        if not table_rows:
            return 0
        
        # Simple heuristic: count separator rows
        separator_rows = len(re.findall(r'\|\s*:?-+:?\s*\|', content))
        return max(separator_rows, 1) if table_rows else 0
    
    def _analyze_headings(self, content: str) -> Dict[str, Any]:
        """Analyze heading structure hierarchy."""
        headings = []
        heading_pattern = r'^(#{1,6})\s+(.+)$'
        
        for match in re.finditer(heading_pattern, content, re.MULTILINE):
            level = len(match.group(1))
            title = match.group(2).strip()
            headings.append({'level': level, 'title': title})
        
        if not headings:
            return {'total': 0, 'max_depth': 0, 'structure_score': 0.0}
        
        levels = [h['level'] for h in headings]
        return {
            'total': len(headings),
            'max_depth': max(levels),
            'min_depth': min(levels),
            'structure_score': self._calculate_structure_score(levels),
            'headings': headings[:10]  # First 10 headings
        }
    
    def _calculate_structure_score(self, levels: List[int]) -> float:
        """Calculate document structure quality score."""
        if len(levels) < 2:
            return 0.5
        
        # Good structure has logical progression (1, 2, 3, etc.)
        logical_progression = 0
        for i in range(1, len(levels)):
            diff = levels[i] - levels[i-1]
            if -2 <= diff <= 1:  # Acceptable level changes
                logical_progression += 1
        
        return logical_progression / (len(levels) - 1)
    
    def _extract_readability_features(self, content: str) -> Dict[str, float]:
        """Extract readability features."""
        sentences = len(re.findall(r'[.!?]+', content))
        words = len(content.split())
        paragraphs = len(re.findall(r'\n\s*\n', content)) + 1
        
        return {
            'avg_words_per_sentence': words / max(sentences, 1),
            'avg_sentences_per_paragraph': sentences / max(paragraphs, 1),
            'words': words,
            'sentences': sentences,
            'paragraphs': paragraphs
        }


class MultiModalSearchEngine:
    """Multi-modal search engine that handles different content types."""
    
    def __init__(self):
        self.code_analyzer = CodeAnalyzer()
        self.structure_analyzer = DocumentStructureAnalyzer()
        self.modality_weights = {
            ContentModality.TEXT: 1.0,
            ContentModality.CODE: 0.8,
            ContentModality.DOCUMENT: 0.9,
            ContentModality.STRUCTURED_DATA: 0.7
        }
    
    def analyze_document(self, document: Document) -> MultiModalDocument:
        """Analyze document and determine its modality and features."""
        modality = self._determine_modality(document)
        
        # Extract content features based on modality
        if modality == ContentModality.CODE:
            content_features = self.code_analyzer.analyze_code(document.content)
        else:
            content_features = self._extract_text_features(document.content)
        
        # Always analyze structure
        structure_metadata = self.structure_analyzer.analyze_structure(
            document.content, 
            getattr(document, 'filename', None)
        )
        
        # Generate semantic tags
        semantic_tags = self._generate_semantic_tags(document, content_features, structure_metadata)
        
        return MultiModalDocument(
            document=document,
            modality=modality,
            content_features=content_features,
            structure_metadata=structure_metadata,
            semantic_tags=semantic_tags,
            relevance_scores={}
        )
    
    def search(self, query: str, documents: List[MultiModalDocument], 
               max_results: int = 10) -> List[Tuple[MultiModalDocument, float]]:
        """Perform multi-modal search across documents."""
        results = []
        
        for doc in documents:
            relevance_score = self._calculate_relevance(query, doc)
            if relevance_score > 0.1:  # Minimum relevance threshold
                results.append((doc, relevance_score))
        
        # Sort by relevance score
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:max_results]
    
    def _determine_modality(self, document: Document) -> ContentModality:
        """Determine the primary modality of a document."""
        content = document.content
        
        # Check for code indicators
        code_indicators = [
            r'```\w*\n.*?\n```',  # Code blocks
            r'def\s+\w+\s*\(',     # Python functions
            r'function\s+\w+\s*\(', # JavaScript functions
            r'class\s+\w+\s*[{:]',  # Class definitions
            r'import\s+[\w\.]+',    # Import statements
        ]
        
        code_matches = sum(
            len(re.findall(pattern, content, re.DOTALL | re.IGNORECASE))
            for pattern in code_indicators
        )
        
        # Check content composition
        total_lines = len(content.split('\n'))
        code_ratio = code_matches / max(total_lines, 1)
        
        if code_ratio > 0.1 or code_matches > 5:
            return ContentModality.CODE
        
        # Check for structured document indicators
        structure_indicators = [
            r'^#{1,6}\s+',  # Markdown headings
            r'\|.*\|',      # Tables
            r'^\s*[-*+]\s+', # Lists
        ]
        
        structure_matches = sum(
            len(re.findall(pattern, content, re.MULTILINE))
            for pattern in structure_indicators
        )
        
        if structure_matches > 3:
            return ContentModality.DOCUMENT
        
        # Default to text
        return ContentModality.TEXT
    
    def _extract_text_features(self, content: str) -> Dict[str, Any]:
        """Extract features from text content."""
        features = {
            'word_count': len(content.split()),
            'sentence_count': len(re.findall(r'[.!?]+', content)),
            'paragraph_count': len(re.findall(r'\n\s*\n', content)) + 1,
            'questions': len(re.findall(r'\?', content)),
            'exclamations': len(re.findall(r'!', content)),
            'technical_terms': self._extract_technical_terms(content),
            'topic_keywords': self._extract_topic_keywords(content)
        }
        
        return features
    
    def _extract_technical_terms(self, content: str) -> List[str]:
        """Extract technical terminology from content."""
        # Common technical term patterns
        technical_patterns = [
            r'\b[A-Z][a-z]*(?:[A-Z][a-z]*)+\b',  # CamelCase
            r'\b[a-z]+_[a-z_]+\b',               # snake_case
            r'\b[A-Z]{2,}\b',                    # ACRONYMS
            r'\b\w+\.[a-z]{2,4}\b',              # Domain names, file extensions
        ]
        
        terms = set()
        for pattern in technical_patterns:
            matches = re.findall(pattern, content)
            terms.update(matches)
        
        # Filter common words
        common_words = {'THE', 'AND', 'OR', 'BUT', 'FOR', 'WITH', 'FROM', 'TO'}
        return [term for term in terms if term.upper() not in common_words]
    
    def _extract_topic_keywords(self, content: str) -> List[str]:
        """Extract topic-relevant keywords."""
        # Common topic domains
        domain_keywords = {
            'authentication': ['auth', 'login', 'password', 'token', 'jwt', 'oauth'],
            'database': ['db', 'sql', 'query', 'table', 'index', 'schema'],
            'api': ['endpoint', 'rest', 'graphql', 'request', 'response'],
            'deployment': ['deploy', 'build', 'ci', 'cd', 'docker', 'kubernetes'],
            'monitoring': ['log', 'metric', 'alert', 'dashboard', 'observability'],
            'security': ['encrypt', 'decrypt', 'ssl', 'tls', 'vulnerability'],
        }
        
        content_lower = content.lower()
        found_topics = []
        
        for topic, keywords in domain_keywords.items():
            if any(keyword in content_lower for keyword in keywords):
                found_topics.append(topic)
        
        return found_topics
    
    def _generate_semantic_tags(self, document: Document, content_features: Dict[str, Any], 
                               structure_metadata: Dict[str, Any]) -> Set[str]:
        """Generate semantic tags for the document."""
        tags = set()
        
        # Add modality-based tags
        if 'language' in content_features:
            languages = content_features['language']
            tags.update(f"lang:{lang}" for lang in languages)
        
        if 'topic_keywords' in content_features:
            tags.update(f"topic:{topic}" for topic in content_features['topic_keywords'])
        
        # Add structure-based tags
        doc_format = structure_metadata.get('format', 'unknown')
        tags.add(f"format:{doc_format}")
        
        # Add complexity tags
        if 'complexity_score' in content_features:
            complexity = content_features['complexity_score']
            if complexity > 0.5:
                tags.add("complexity:high")
            elif complexity > 0.2:
                tags.add("complexity:medium")
            else:
                tags.add("complexity:low")
        
        # Add length tags
        word_count = structure_metadata.get('word_count', 0)
        if word_count > 1000:
            tags.add("length:long")
        elif word_count > 200:
            tags.add("length:medium")
        else:
            tags.add("length:short")
        
        return tags
    
    def _calculate_relevance(self, query: str, doc: MultiModalDocument) -> float:
        """Calculate relevance score for a document given a query."""
        query_lower = query.lower()
        content_lower = doc.document.content.lower()
        
        # Base text relevance
        text_relevance = self._calculate_text_relevance(query_lower, content_lower)
        
        # Modality-specific scoring
        modality_bonus = 0.0
        
        if doc.modality == ContentModality.CODE:
            modality_bonus = self._calculate_code_relevance(query_lower, doc.content_features)
        elif doc.modality == ContentModality.DOCUMENT:
            modality_bonus = self._calculate_document_relevance(query_lower, doc.structure_metadata)
        
        # Semantic tag matching
        tag_relevance = self._calculate_tag_relevance(query_lower, doc.semantic_tags)
        
        # Combine scores with modality weighting
        modality_weight = self.modality_weights.get(doc.modality, 0.5)
        
        total_relevance = (
            text_relevance * 0.6 +
            modality_bonus * 0.3 +
            tag_relevance * 0.1
        ) * modality_weight
        
        return min(total_relevance, 1.0)
    
    def _calculate_text_relevance(self, query: str, content: str) -> float:
        """Calculate basic text relevance score."""
        query_terms = query.split()
        
        if not query_terms:
            return 0.0
        
        # Count term matches
        term_matches = 0
        for term in query_terms:
            if term in content:
                term_matches += 1
        
        # Calculate relevance as ratio of matched terms
        basic_relevance = term_matches / len(query_terms)
        
        # Boost for exact phrase matches
        if query in content:
            basic_relevance += 0.2
        
        return min(basic_relevance, 1.0)
    
    def _calculate_code_relevance(self, query: str, content_features: Dict[str, Any]) -> float:
        """Calculate code-specific relevance."""
        relevance = 0.0
        
        # Check if query matches programming language
        languages = content_features.get('language', [])
        for lang in languages:
            if lang.lower() in query:
                relevance += 0.3
        
        # Check if query matches function names
        functions = content_features.get('functions', [])
        for func in functions:
            if func.lower() in query:
                relevance += 0.2
        
        # Check if query matches class names
        classes = content_features.get('classes', [])
        for cls in classes:
            if cls.lower() in query:
                relevance += 0.2
        
        return min(relevance, 1.0)
    
    def _calculate_document_relevance(self, query: str, structure_metadata: Dict[str, Any]) -> float:
        """Calculate document structure-specific relevance."""
        relevance = 0.0
        
        # Check section headings
        headings_info = structure_metadata.get('headings_hierarchy', {})
        headings = headings_info.get('headings', [])
        
        for heading in headings:
            heading_text = heading.get('title', '').lower()
            if any(term in heading_text for term in query.split()):
                relevance += 0.3
        
        # Check links
        links = structure_metadata.get('links', [])
        for link in links:
            link_text = link.get('text', '').lower()
            if any(term in link_text for term in query.split()):
                relevance += 0.1
        
        return min(relevance, 1.0)
    
    def _calculate_tag_relevance(self, query: str, tags: Set[str]) -> float:
        """Calculate semantic tag relevance."""
        relevance = 0.0
        
        for tag in tags:
            tag_parts = tag.split(':')
            if len(tag_parts) == 2:
                tag_type, tag_value = tag_parts
                if tag_value.lower() in query or any(term in tag_value.lower() for term in query.split()):
                    relevance += 0.2
        
        return min(relevance, 1.0)


# Convenience functions
def create_multimodal_search_engine() -> MultiModalSearchEngine:
    """Create a multi-modal search engine."""
    return MultiModalSearchEngine()


def analyze_documents_multimodal(documents: List[Document]) -> List[MultiModalDocument]:
    """Analyze a list of documents for multi-modal search."""
    engine = MultiModalSearchEngine()
    return [engine.analyze_document(doc) for doc in documents]