#!/usr/bin/env python3
"""
Enhanced Autonomous Knowledge System - Generation 1 Implementation
Advanced self-improving knowledge management system with intelligent routing.
"""

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Configure logging for autonomous operation
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('autonomous_system.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("AutonomousSystem")


class IntelligentDocumentProcessor:
    """Advanced document processing with automatic categorization."""
    
    def __init__(self):
        self.content_patterns = {
            'api_documentation': [
                r'\b(GET|POST|PUT|DELETE)\s+/',
                r'\b(endpoint|route|parameter|request|response)\b',
                r'HTTP/\d+\.\d+',
                r'\b(JSON|XML|API)\b'
            ],
            'code_documentation': [
                r'```\w+',
                r'\bfunction\s+\w+\(',
                r'\bclass\s+\w+',
                r'@\w+',  # decorators
                r'\b(import|from|require)\b'
            ],
            'troubleshooting': [
                r'\b(error|exception|fail|bug|issue)\b',
                r'\b(fix|solve|resolve|debug)\b',
                r'\b(stacktrace|traceback)\b'
            ],
            'configuration': [
                r'\b(config|settings|environment)\b',
                r'\.(yaml|yml|json|toml|ini)\b',
                r'\b(PORT|HOST|DATABASE_URL)\b'
            ]
        }
    
    def classify_content(self, content: str) -> str:
        """Classify content type based on pattern matching."""
        content_lower = content.lower()
        
        scores = {}
        for doc_type, patterns in self.content_patterns.items():
            score = sum(1 for pattern in patterns 
                       if len(re.findall(pattern, content_lower, re.IGNORECASE)) > 0)
            scores[doc_type] = score
            
        if not any(scores.values()):
            return 'general'
            
        return max(scores.items(), key=lambda x: x[1])[0]
    
    def extract_key_terms(self, content: str) -> List[str]:
        """Extract key technical terms from content."""
        import re
        
        # Technical term patterns
        technical_patterns = [
            r'\b[A-Z_]{2,}\b',  # Constants/environment variables
            r'\b\w+\(\)',       # Functions
            r'\b\w+\.\w+',      # Method calls/properties
            r'@\w+',            # Decorators/annotations
            r'\b\w+://\S+',     # URLs
            r'\b\w+\.\w{2,4}\b' # File extensions
        ]
        
        terms = set()
        for pattern in technical_patterns:
            matches = re.findall(pattern, content)
            terms.update(match.strip() for match in matches if len(match) > 2)
        
        # Remove common words
        stopwords = {'the', 'and', 'for', 'with', 'from', 'this', 'that'}
        return [term for term in terms if term.lower() not in stopwords][:20]


class AdaptiveKnowledgeIndex:
    """Self-optimizing knowledge index with usage-based ranking."""
    
    def __init__(self):
        self.documents = {}
        self.query_history = []
        self.document_usage = {}
        self.knowledge_gaps = []
        self.last_optimization = datetime.utcnow()
    
    def add_document(self, doc_id: str, content: str, metadata: Dict[str, Any]):
        """Add document with automatic preprocessing."""
        processor = IntelligentDocumentProcessor()
        
        enhanced_metadata = {
            **metadata,
            'content_type': processor.classify_content(content),
            'key_terms': processor.extract_key_terms(content),
            'word_count': len(content.split()),
            'indexed_at': datetime.utcnow().isoformat(),
            'usage_count': 0,
            'quality_score': self._calculate_quality_score(content)
        }
        
        self.documents[doc_id] = {
            'content': content,
            'metadata': enhanced_metadata,
            'search_index': self._create_search_index(content)
        }
        
        logger.info(f"Indexed document {doc_id} with type: {enhanced_metadata['content_type']}")
    
    def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Advanced search with relevance scoring."""
        query_terms = set(query.lower().split())
        results = []
        
        for doc_id, doc_data in self.documents.items():
            score = self._calculate_relevance_score(
                query_terms, 
                doc_data['search_index'],
                doc_data['metadata']
            )
            
            if score > 0:
                results.append({
                    'doc_id': doc_id,
                    'content': doc_data['content'][:500] + '...',
                    'metadata': doc_data['metadata'],
                    'relevance_score': score,
                    'snippet': self._generate_snippet(doc_data['content'], query_terms)
                })
                
                # Track usage for optimization
                self.document_usage[doc_id] = self.document_usage.get(doc_id, 0) + 1
        
        # Sort by relevance and usage
        results.sort(key=lambda x: (x['relevance_score'], 
                                   self.document_usage.get(x['doc_id'], 0)), 
                    reverse=True)
        
        self.query_history.append({
            'query': query,
            'timestamp': datetime.utcnow().isoformat(),
            'results_count': len(results)
        })
        
        return results[:limit]
    
    def _calculate_quality_score(self, content: str) -> float:
        """Calculate document quality score."""
        word_count = len(content.split())
        
        # Base score on length (sweet spot around 200-1000 words)
        if word_count < 50:
            length_score = 0.5
        elif 200 <= word_count <= 1000:
            length_score = 1.0
        else:
            length_score = max(0.6, 1.0 - (word_count - 1000) / 10000)
        
        # Code/structured content bonus
        structure_score = 1.0
        if '```' in content or '{' in content:
            structure_score = 1.2
        
        return min(1.0, length_score * structure_score)
    
    def _create_search_index(self, content: str) -> Dict[str, int]:
        """Create inverted index for content."""
        import re
        
        words = re.findall(r'\b\w+\b', content.lower())
        word_freq = {}
        
        for word in words:
            if len(word) > 2:  # Skip very short words
                word_freq[word] = word_freq.get(word, 0) + 1
        
        return word_freq
    
    def _calculate_relevance_score(self, query_terms: set, search_index: Dict[str, int], 
                                 metadata: Dict[str, Any]) -> float:
        """Calculate relevance score for document."""
        base_score = 0
        total_words = sum(search_index.values())
        
        for term in query_terms:
            if term.lower() in search_index:
                # TF-IDF-like scoring
                term_freq = search_index[term.lower()]
                score = (term_freq / total_words) * 10
                base_score += score
        
        # Boost for recent and high-quality documents
        quality_boost = metadata.get('quality_score', 1.0)
        usage_boost = 1.0 + (self.document_usage.get(metadata.get('doc_id'), 0) * 0.1)
        
        return base_score * quality_boost * usage_boost
    
    def _generate_snippet(self, content: str, query_terms: set) -> str:
        """Generate relevant snippet from content."""
        sentences = content.split('.')
        
        for sentence in sentences:
            if any(term.lower() in sentence.lower() for term in query_terms):
                return sentence.strip()[:200] + '...'
        
        return content[:200] + '...'
    
    def optimize_index(self):
        """Periodically optimize the search index."""
        if datetime.utcnow() - self.last_optimization < timedelta(hours=1):
            return
        
        # Remove low-quality, unused documents
        docs_to_remove = []
        for doc_id, doc_data in self.documents.items():
            quality = doc_data['metadata'].get('quality_score', 0)
            usage = self.document_usage.get(doc_id, 0)
            
            if quality < 0.3 and usage == 0:
                docs_to_remove.append(doc_id)
        
        for doc_id in docs_to_remove:
            del self.documents[doc_id]
            logger.info(f"Removed low-quality document: {doc_id}")
        
        self.last_optimization = datetime.utcnow()
        logger.info("Search index optimization completed")


class AutonomousLearningSystem:
    """Self-improving system that learns from interactions."""
    
    def __init__(self):
        self.knowledge_index = AdaptiveKnowledgeIndex()
        self.conversation_context = {}
        self.learning_patterns = {}
        self.improvement_suggestions = []
    
    def process_query(self, query: str, user_context: Optional[Dict] = None) -> Dict[str, Any]:
        """Process user query with continuous learning."""
        start_time = time.time()
        
        # Search for relevant documents
        search_results = self.knowledge_index.search(query)
        
        # Generate intelligent response
        response = self._generate_response(query, search_results, user_context)
        
        # Learn from this interaction
        self._learn_from_interaction(query, search_results, response)
        
        processing_time = (time.time() - start_time) * 1000
        
        return {
            'query': query,
            'response': response,
            'sources': [r['doc_id'] for r in search_results[:3]],
            'confidence': self._calculate_confidence(search_results),
            'processing_time_ms': processing_time,
            'suggestions': self._generate_suggestions(query, search_results)
        }
    
    def _generate_response(self, query: str, search_results: List[Dict], 
                         user_context: Optional[Dict]) -> str:
        """Generate intelligent response based on search results."""
        if not search_results:
            return self._handle_no_results(query)
        
        # Combine top results into coherent response
        top_result = search_results[0]
        response_parts = []
        
        # Add main answer
        response_parts.append(f"Based on our knowledge base:\n\n{top_result['snippet']}")
        
        # Add additional context if available
        if len(search_results) > 1:
            related = search_results[1]
            response_parts.append(f"\n\nRelated information: {related['snippet']}")
        
        # Add source attribution
        sources = [f"• {r['metadata'].get('title', r['doc_id'])}" 
                  for r in search_results[:3]]
        response_parts.append(f"\n\nSources:\n" + "\n".join(sources))
        
        return "\n".join(response_parts)
    
    def _handle_no_results(self, query: str) -> str:
        """Handle queries with no matching results."""
        # Record knowledge gap for future improvement
        self.knowledge_index.knowledge_gaps.append({
            'query': query,
            'timestamp': datetime.utcnow().isoformat(),
            'category': 'no_results'
        })
        
        return (f"I couldn't find specific information about '{query}' in our knowledge base. "
                f"This has been noted as a potential knowledge gap for improvement.")
    
    def _learn_from_interaction(self, query: str, results: List[Dict], response: str):
        """Learn patterns from user interactions."""
        # Track query patterns
        query_type = self._classify_query_type(query)
        
        if query_type not in self.learning_patterns:
            self.learning_patterns[query_type] = {
                'count': 0,
                'avg_results': 0,
                'common_terms': {}
            }
        
        pattern = self.learning_patterns[query_type]
        pattern['count'] += 1
        pattern['avg_results'] = (pattern['avg_results'] + len(results)) / 2
        
        # Learn common terms for this query type
        for term in query.lower().split():
            if len(term) > 2:
                pattern['common_terms'][term] = pattern['common_terms'].get(term, 0) + 1
    
    def _classify_query_type(self, query: str) -> str:
        """Classify query type for learning purposes."""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['how', 'what', 'where', 'when', 'why']):
            return 'question'
        elif any(word in query_lower for word in ['error', 'problem', 'not working', 'fail']):
            return 'troubleshooting'  
        elif any(word in query_lower for word in ['show', 'list', 'find', 'search']):
            return 'search'
        else:
            return 'general'
    
    def _calculate_confidence(self, results: List[Dict]) -> float:
        """Calculate confidence score for response."""
        if not results:
            return 0.0
        
        top_score = results[0]['relevance_score']
        num_results = len(results)
        
        # Base confidence on top result relevance and number of supporting results
        confidence = min(1.0, (top_score / 10) + (num_results / 20))
        
        return round(confidence, 2)
    
    def _generate_suggestions(self, query: str, results: List[Dict]) -> List[str]:
        """Generate helpful query suggestions."""
        suggestions = []
        
        if results:
            # Suggest related topics based on key terms
            top_result = results[0]
            key_terms = top_result['metadata'].get('key_terms', [])
            
            for term in key_terms[:3]:
                suggestions.append(f"Learn more about {term}")
        
        # Suggest broader or narrower searches
        query_words = query.split()
        if len(query_words) > 3:
            suggestions.append(f"Try a simpler search: {' '.join(query_words[:2])}")
        else:
            suggestions.append(f"Try a more specific search with additional terms")
        
        return suggestions[:3]
    
    def add_knowledge(self, content: str, source: str, metadata: Optional[Dict] = None):
        """Add new knowledge to the system."""
        doc_id = f"{source}_{hash(content) % 1000000}"
        full_metadata = metadata or {}
        full_metadata.update({
            'source': source,
            'added_at': datetime.utcnow().isoformat()
        })
        
        self.knowledge_index.add_document(doc_id, content, full_metadata)
        logger.info(f"Added knowledge from {source}: {len(content)} characters")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            'documents_count': len(self.knowledge_index.documents),
            'queries_processed': len(self.knowledge_index.query_history),
            'knowledge_gaps': len(self.knowledge_index.knowledge_gaps),
            'learning_patterns': len(self.learning_patterns),
            'last_optimization': self.knowledge_index.last_optimization.isoformat(),
            'system_uptime_hours': (datetime.utcnow() - datetime.utcnow().replace(hour=0, minute=0, second=0)).total_seconds() / 3600
        }


def main():
    """Main function demonstrating the autonomous system."""
    logger.info("🚀 Starting Enhanced Autonomous Knowledge System - Generation 1")
    
    # Initialize the autonomous learning system
    system = AutonomousLearningSystem()
    
    # Add some sample knowledge
    sample_docs = [
        {
            'content': '''
            # API Authentication Guide
            
            Our API uses Bearer token authentication. To authenticate:
            
            1. Obtain your API key from the dashboard
            2. Include it in the Authorization header:
               `Authorization: Bearer YOUR_API_KEY`
            3. All requests must be made over HTTPS
            
            Example:
            ```bash
            curl -H "Authorization: Bearer your-key" https://api.example.com/users
            ```
            
            For troubleshooting authentication issues, check that:
            - Your API key is valid and not expired
            - You're using the correct header format
            - Your account has the necessary permissions
            ''',
            'source': 'api_docs',
            'metadata': {'title': 'API Authentication Guide', 'category': 'authentication'}
        },
        {
            'content': '''
            # Deployment Process
            
            To deploy the application:
            
            1. Build the Docker image: `docker build -t myapp .`
            2. Run tests: `docker run myapp npm test`
            3. Push to registry: `docker push registry.com/myapp`
            4. Deploy to production: `kubectl apply -f deployment.yaml`
            
            Common deployment issues:
            - Port conflicts: Check if port 3000 is available
            - Environment variables: Ensure all required vars are set
            - Database connectivity: Verify DATABASE_URL is correct
            ''',
            'source': 'deployment_guide',
            'metadata': {'title': 'Deployment Process', 'category': 'deployment'}
        }
    ]
    
    # Add sample knowledge
    for doc in sample_docs:
        system.add_knowledge(doc['content'], doc['source'], doc['metadata'])
    
    # Demonstrate query processing
    test_queries = [
        "How do I authenticate with the API?",
        "My deployment is failing, what should I check?",
        "What is the Bearer token format?",
        "Database connection error during deployment"
    ]
    
    logger.info("Processing test queries...")
    
    for query in test_queries:
        logger.info(f"\n--- Processing Query: '{query}' ---")
        result = system.process_query(query)
        
        print(f"\nQuery: {result['query']}")
        print(f"Confidence: {result['confidence']}")
        print(f"Processing Time: {result['processing_time_ms']:.2f}ms")
        print(f"Response:\n{result['response']}")
        print(f"Suggestions: {', '.join(result['suggestions'])}")
    
    # Show system status
    status = system.get_system_status()
    logger.info(f"\n--- System Status ---")
    logger.info(f"Documents: {status['documents_count']}")
    logger.info(f"Queries Processed: {status['queries_processed']}")
    logger.info(f"Knowledge Gaps: {status['knowledge_gaps']}")
    
    logger.info("✅ Generation 1 autonomous system demonstration complete!")


if __name__ == "__main__":
    import re
    main()