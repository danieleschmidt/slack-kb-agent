"""Advanced API server with REST and GraphQL endpoints for knowledge base access."""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, Query, Path, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import strawberry
from strawberry.fastapi import GraphQLRouter

from .knowledge_base import KnowledgeBase
from .query_processor import QueryProcessor
from .repositories.document_repository import DocumentRepository
from .repositories.analytics_repository import AnalyticsRepository
from .advanced_algorithms import IntelligentQueryRouter, KnowledgeGapAnalyzer
from .models import Document, DocumentType, SourceType
from .auth import get_auth_middleware
from .rate_limiting import get_rate_limiter
from .monitoring import get_global_metrics
from .exceptions import KnowledgeBaseError, ValidationError


logger = logging.getLogger(__name__)
security = HTTPBearer()


# Pydantic models for API
class QueryRequest(BaseModel):
    """Request model for knowledge base queries."""
    
    query: str = Field(..., min_length=1, max_length=1000, description="Search query")
    limit: Optional[int] = Field(10, ge=1, le=100, description="Maximum number of results")
    include_sources: bool = Field(True, description="Include source information in response")
    user_context: Optional[Dict[str, Any]] = Field(None, description="User context for personalization")
    filters: Optional[Dict[str, Any]] = Field(None, description="Search filters")


class QueryResponse(BaseModel):
    """Response model for knowledge base queries."""
    
    query: str
    results: List[Dict[str, Any]]
    total_results: int
    response_time_ms: float
    suggested_queries: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DocumentRequest(BaseModel):
    """Request model for adding documents."""
    
    content: str = Field(..., min_length=1, description="Document content")
    source: str = Field(..., description="Document source")
    title: Optional[str] = Field(None, description="Document title")
    doc_type: Optional[str] = Field(None, description="Document type")
    source_type: Optional[str] = Field(None, description="Source type")
    tags: List[str] = Field(default_factory=list, description="Document tags")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    priority: int = Field(1, ge=1, le=5, description="Document priority")


class AnalyticsResponse(BaseModel):
    """Response model for analytics data."""
    
    period: str
    total_queries: int
    successful_queries: int
    success_rate: float
    avg_response_time_ms: Optional[float]
    popular_queries: List[Dict[str, Any]]
    knowledge_gaps: List[Dict[str, Any]]
    generated_at: str


class HealthResponse(BaseModel):
    """Response model for health checks."""
    
    status: str
    timestamp: str
    version: str
    components: Dict[str, Dict[str, Any]]
    uptime_seconds: float


# GraphQL types
@strawberry.type
class DocumentType:
    """GraphQL document type."""
    
    id: str
    content: str
    source: str
    title: Optional[str]
    author: Optional[str]
    url: Optional[str]
    doc_type: str
    source_type: str
    priority: int
    tags: List[str]
    created_at: str
    updated_at: Optional[str]


@strawberry.type
class SearchResult:
    """GraphQL search result type."""
    
    document: DocumentType
    score: float
    snippet: str
    explanation: Optional[str]


@strawberry.type
class QueryResult:
    """GraphQL query result type."""
    
    query: str
    results: List[SearchResult]
    total_results: int
    response_time_ms: float
    suggestions: List[str]


@strawberry.input
class QueryInput:
    """GraphQL query input type."""
    
    query: str
    limit: Optional[int] = 10
    filters: Optional[str] = None  # JSON string
    user_context: Optional[str] = None  # JSON string


class APIServer:
    """Advanced API server for knowledge base access."""
    
    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        query_processor: QueryProcessor,
        enable_graphql: bool = True,
        enable_auth: bool = True,
        enable_rate_limiting: bool = True
    ):
        self.knowledge_base = knowledge_base
        self.query_processor = query_processor
        self.document_repository = DocumentRepository()
        self.analytics_repository = AnalyticsRepository()
        self.query_router = IntelligentQueryRouter()
        self.gap_analyzer = KnowledgeGapAnalyzer()
        
        # Create FastAPI app
        self.app = FastAPI(
            title="Slack KB Agent API",
            description="Advanced knowledge base API with intelligent search and analytics",
            version="1.7.2",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        # Add middleware
        self._setup_middleware()
        
        # Setup authentication and rate limiting
        self.auth_middleware = get_auth_middleware() if enable_auth else None
        self.rate_limiter = get_rate_limiter() if enable_rate_limiting else None
        
        # Setup routes
        self._setup_routes()
        
        # Setup GraphQL if enabled
        if enable_graphql:
            self._setup_graphql()
        
        # Metrics
        self.metrics = get_global_metrics()
        self.start_time = datetime.utcnow()
    
    def _setup_middleware(self):
        """Setup FastAPI middleware."""
        # CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure appropriately for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Request timing middleware
        @self.app.middleware("http")
        async def add_process_time_header(request, call_next):
            start_time = datetime.utcnow()
            response = await call_next(request)
            process_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            response.headers["X-Process-Time"] = str(process_time)
            return response
    
    def _setup_routes(self):
        """Setup REST API routes."""
        
        @self.app.get("/health", response_model=HealthResponse)
        async def health_check():
            """Health check endpoint."""
            components = {
                "knowledge_base": {
                    "status": "healthy",
                    "documents": len(self.knowledge_base.documents)
                },
                "database": {
                    "status": "healthy",  # Would check actual DB connection
                    "connection": "active"
                },
                "cache": {
                    "status": "healthy",
                    "hit_rate": 0.85  # Would get from actual cache
                }
            }
            
            uptime = (datetime.utcnow() - self.start_time).total_seconds()
            
            return HealthResponse(
                status="healthy",
                timestamp=datetime.utcnow().isoformat(),
                version="1.7.2",
                components=components,
                uptime_seconds=uptime
            )
        
        @self.app.post("/search", response_model=QueryResponse)
        async def search_knowledge_base(
            request: QueryRequest,
            auth: Optional[HTTPAuthorizationCredentials] = Depends(security)
        ):
            """Search the knowledge base with intelligent query processing."""
            start_time = datetime.utcnow()
            
            try:
                # Authenticate if required
                user_id = "anonymous"
                if self.auth_middleware and auth:
                    user_context = await self.auth_middleware.authenticate(auth.credentials)
                    user_id = user_context.get("user_id", "anonymous")
                
                # Rate limiting
                if self.rate_limiter:
                    await self.rate_limiter.check_rate_limit(user_id)
                
                # Route query intelligently
                routing = self.query_router.route_query(request.query, request.user_context or {})
                
                # Perform search
                results = await self._search_knowledge_base(
                    query=request.query,
                    limit=request.limit or 10,
                    user_context=request.user_context or {},
                    routing=routing
                )
                
                # Calculate response time
                response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
                
                # Record analytics
                await self.analytics_repository.record_event(
                    event_type="query",
                    user_id=user_id,
                    query=request.query,
                    response_time_ms=response_time,
                    result_count=len(results),
                    success=True
                )
                
                # Generate suggestions
                suggestions = await self._generate_query_suggestions(request.query, results)
                
                return QueryResponse(
                    query=request.query,
                    results=results,
                    total_results=len(results),
                    response_time_ms=response_time,
                    suggested_queries=suggestions,
                    metadata={
                        "routing": routing,
                        "user_id": user_id
                    }
                )
                
            except Exception as e:
                logger.error(f"Search failed: {e}")
                response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
                
                # Record failure
                await self.analytics_repository.record_event(
                    event_type="query",
                    user_id=user_id,
                    query=request.query,
                    response_time_ms=response_time,
                    result_count=0,
                    success=False,
                    error_message=str(e)
                )
                
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/documents")
        async def add_document(
            request: DocumentRequest,
            auth: Optional[HTTPAuthorizationCredentials] = Depends(security)
        ):
            """Add a new document to the knowledge base."""
            try:
                # Authenticate if required
                if self.auth_middleware and auth:
                    user_context = await self.auth_middleware.authenticate(auth.credentials)
                    # Check permissions for document creation
                
                # Create document
                document = Document(
                    content=request.content,
                    source=request.source,
                    title=request.title,
                    doc_type=DocumentType(request.doc_type) if request.doc_type else DocumentType.TEXT,
                    source_type=SourceType(request.source_type) if request.source_type else SourceType.MANUAL_ENTRY,
                    tags=request.tags,
                    metadata=request.metadata,
                    priority=request.priority
                )
                
                # Add to knowledge base
                doc_id = self.knowledge_base.add_document(document)
                
                # Add to database
                await self.document_repository.create_from_document(document)
                
                return {"document_id": doc_id, "status": "created"}
                
            except Exception as e:
                logger.error(f"Document creation failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/analytics", response_model=AnalyticsResponse)
        async def get_analytics(
            period: str = Query("7d", description="Analytics period (7d, 30d, 90d)"),
            auth: Optional[HTTPAuthorizationCredentials] = Depends(security)
        ):
            """Get analytics data for the specified period."""
            try:
                # Authenticate if required
                if self.auth_middleware and auth:
                    user_context = await self.auth_middleware.authenticate(auth.credentials)
                    # Check permissions for analytics access
                
                # Parse period
                days_map = {"7d": 7, "30d": 30, "90d": 90}
                days = days_map.get(period, 7)
                
                # Get analytics data
                dashboard_data = await self.analytics_repository.get_usage_dashboard_data()
                popular_queries = await self.analytics_repository.get_popular_queries(days=days)
                knowledge_gaps = self.gap_analyzer.identify_knowledge_gaps(days_window=days)
                
                # Format response
                current_stats = dashboard_data["current_week"]
                
                return AnalyticsResponse(
                    period=period,
                    total_queries=current_stats["total_queries"],
                    successful_queries=current_stats["successful_queries"],
                    success_rate=current_stats["success_rate"],
                    avg_response_time_ms=current_stats["avg_response_time"],
                    popular_queries=popular_queries,
                    knowledge_gaps=[{
                        "topic": gap.topic,
                        "frequency": gap.frequency,
                        "priority_score": gap.priority_score,
                        "suggested_sources": gap.suggested_sources
                    } for gap in knowledge_gaps],
                    generated_at=datetime.utcnow().isoformat()
                )
                
            except Exception as e:
                logger.error(f"Analytics retrieval failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/documents/{doc_id}")
        async def get_document(
            doc_id: str = Path(..., description="Document ID"),
            auth: Optional[HTTPAuthorizationCredentials] = Depends(security)
        ):
            """Get a specific document by ID."""
            try:
                # Authenticate if required
                if self.auth_middleware and auth:
                    user_context = await self.auth_middleware.authenticate(auth.credentials)
                
                # Get document from knowledge base
                document = self.knowledge_base.get_document(doc_id)
                if not document:
                    raise HTTPException(status_code=404, detail="Document not found")
                
                return {
                    "id": doc_id,
                    "content": document.content,
                    "source": document.source,
                    "title": document.title,
                    "metadata": document.metadata,
                    "created_at": document.created_at.isoformat() if document.created_at else None
                }
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Document retrieval failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.delete("/documents/{doc_id}")
        async def delete_document(
            doc_id: str = Path(..., description="Document ID"),
            auth: Optional[HTTPAuthorizationCredentials] = Depends(security)
        ):
            """Delete a document by ID."""
            try:
                # Authenticate if required
                if self.auth_middleware and auth:
                    user_context = await self.auth_middleware.authenticate(auth.credentials)
                    # Check permissions for document deletion
                
                # Remove from knowledge base
                success = self.knowledge_base.remove_document(doc_id)
                if not success:
                    raise HTTPException(status_code=404, detail="Document not found")
                
                return {"status": "deleted"}
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Document deletion failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/suggestions")
        async def get_query_suggestions(
            q: str = Query(..., description="Partial query for suggestions"),
            limit: int = Query(5, ge=1, le=20, description="Number of suggestions")
        ):
            """Get query suggestions based on popular queries and patterns."""
            try:
                suggestions = await self._get_query_autocomplete(q, limit)
                return {"suggestions": suggestions}
                
            except Exception as e:
                logger.error(f"Suggestion generation failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
    
    def _setup_graphql(self):
        """Setup GraphQL endpoint."""
        
        @strawberry.type
        class Query:
            """GraphQL query root."""
            
            @strawberry.field
            async def search(self, input: QueryInput) -> QueryResult:
                """Search knowledge base via GraphQL."""
                try:
                    # Parse optional JSON fields
                    filters = json.loads(input.filters) if input.filters else {}
                    user_context = json.loads(input.user_context) if input.user_context else {}
                    
                    # Route query
                    routing = self.query_router.route_query(input.query, user_context)
                    
                    # Perform search
                    results = await self._search_knowledge_base(
                        query=input.query,
                        limit=input.limit or 10,
                        user_context=user_context,
                        routing=routing
                    )
                    
                    # Convert to GraphQL types
                    search_results = []
                    for result in results:
                        doc_data = result["document"]
                        document = DocumentType(
                            id=result["id"],
                            content=doc_data["content"],
                            source=doc_data["source"],
                            title=doc_data.get("title"),
                            author=doc_data.get("author"),
                            url=doc_data.get("url"),
                            doc_type=doc_data["doc_type"],
                            source_type=doc_data["source_type"],
                            priority=doc_data["priority"],
                            tags=doc_data["tags"],
                            created_at=doc_data["created_at"],
                            updated_at=doc_data.get("updated_at")
                        )
                        
                        search_results.append(SearchResult(
                            document=document,
                            score=result["score"],
                            snippet=result.get("snippet", ""),
                            explanation=result.get("explanation")
                        ))
                    
                    return QueryResult(
                        query=input.query,
                        results=search_results,
                        total_results=len(results),
                        response_time_ms=0.0,  # Would calculate actual time
                        suggestions=[]
                    )
                    
                except Exception as e:
                    logger.error(f"GraphQL search failed: {e}")
                    raise Exception(str(e))
        
        schema = strawberry.Schema(query=Query)
        graphql_app = GraphQLRouter(schema)
        self.app.include_router(graphql_app, prefix="/graphql")
    
    async def _search_knowledge_base(
        self,
        query: str,
        limit: int,
        user_context: Dict[str, Any],
        routing: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Perform intelligent knowledge base search."""
        
        # Choose search strategy based on routing
        strategy = routing.get("suggested_search_strategy", "hybrid_weighted")
        
        if strategy == "keyword_exact":
            results = self.knowledge_base.search(query, limit=limit)
        elif strategy == "semantic_deep":
            results = self.knowledge_base.search_semantic(query, limit=limit)
        else:  # hybrid_weighted or multi_step_reasoning
            results = self.knowledge_base.search_hybrid(query, limit=limit)
        
        # Format results for API response
        formatted_results = []
        for result in results:
            formatted_results.append({
                "id": result.metadata.get("doc_id", "unknown"),
                "document": {
                    "content": result.content,
                    "source": result.source,
                    "title": result.metadata.get("title"),
                    "author": result.metadata.get("author"),
                    "url": result.metadata.get("url"),
                    "doc_type": result.metadata.get("doc_type", "text"),
                    "source_type": result.metadata.get("source_type", "unknown"),
                    "priority": result.metadata.get("priority", 1),
                    "tags": result.metadata.get("tags", []),
                    "created_at": result.metadata.get("created_at"),
                    "updated_at": result.metadata.get("updated_at")
                },
                "score": getattr(result, 'score', 0.0),
                "snippet": result.content[:200] + "..." if len(result.content) > 200 else result.content,
                "explanation": f"Found via {strategy} search"
            })
        
        return formatted_results
    
    async def _generate_query_suggestions(
        self,
        query: str,
        results: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate query suggestions based on results and patterns."""
        suggestions = []
        
        # Get popular queries similar to current query
        popular = await self.analytics_repository.get_popular_queries(limit=10)
        
        # Simple similarity-based suggestions
        query_words = set(query.lower().split())
        for popular_query in popular:
            popular_words = set(popular_query["query"].lower().split())
            overlap = len(query_words.intersection(popular_words))
            if overlap > 0 and popular_query["query"] != query:
                suggestions.append(popular_query["query"])
        
        return suggestions[:5]  # Return top 5 suggestions
    
    async def _get_query_autocomplete(self, partial: str, limit: int) -> List[str]:
        """Get query autocomplete suggestions."""
        # Get popular queries that start with or contain the partial query
        popular = await self.analytics_repository.get_popular_queries(limit=50)
        
        suggestions = []
        partial_lower = partial.lower()
        
        for query_data in popular:
            query = query_data["query"]
            if (query.lower().startswith(partial_lower) or 
                partial_lower in query.lower()) and query != partial:
                suggestions.append(query)
                
                if len(suggestions) >= limit:
                    break
        
        return suggestions


async def create_api_server(
    knowledge_base: KnowledgeBase,
    query_processor: QueryProcessor,
    **kwargs
) -> APIServer:
    """Factory function to create API server."""
    return APIServer(knowledge_base, query_processor, **kwargs)


# Server lifecycle management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage server lifecycle events."""
    # Startup
    logger.info("Starting API server...")
    yield
    # Shutdown
    logger.info("Shutting down API server...")


def create_app() -> FastAPI:
    """Create FastAPI application with all configurations."""
    # This would be called by a proper application factory
    # For now, return a basic app for testing
    app = FastAPI(
        title="Slack KB Agent API",
        description="Advanced knowledge base API",
        version="1.7.2",
        lifespan=lifespan
    )
    
    @app.get("/")
    async def root():
        return {"message": "Slack KB Agent API", "version": "1.7.2"}
    
    return app