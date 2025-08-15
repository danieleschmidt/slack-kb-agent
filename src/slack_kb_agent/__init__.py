"""Slack Knowledge Base Agent package."""

__version__ = "1.7.2"

from . import cli
from .analytics import UsageAnalytics
from .auth import AuthConfig, AuthMiddleware, BasicAuthenticator, get_auth_middleware
from .autonomous_sdlc import AutonomousSDLC, QualityGate, SDLCPhase, get_autonomous_sdlc
from .escalation import SlackNotifier
from .ingestion import (
    BatchIngester,
    ContentProcessor,
    FileIngester,
    GitHubIngester,
    SlackHistoryIngester,
    WebDocumentationCrawler,
)
from .knowledge_base import KnowledgeBase
from .llm import LLMConfig, LLMResponse, ResponseGenerator, get_response_generator
from .models import Document
from .monitoring import (
    HealthChecker,
    MetricsCollector,
    MonitoredKnowledgeBase,
    MonitoringConfig,
    PerformanceTracker,
    StructuredLogger,
    get_global_metrics,
    setup_monitoring,
    start_monitoring_server,
)
from .password_hash import PasswordHasher
from .quantum_task_planner import (
    QuantumTask,
    QuantumTaskPlanner,
    TaskPriority,
    TaskState,
    create_dependent_tasks,
    create_entangled_task_pair,
    create_simple_task,
    get_quantum_planner,
)
from .query_processor import Query, QueryProcessor
from .rate_limiting import (
    RateLimitConfig,
    RateLimiter,
    UserRateLimiter,
    get_rate_limiter,
    get_user_rate_limiter,
)
from .real_time import RealTimeUpdater
from .resilience import (
    BackoffStrategy,
    BulkheadConfig,
    BulkheadIsolation,
    CircuitBreaker,
    HealthMonitor,
    HealthStatus,
    ResilientExecutor,
    RetryConfig,
    get_bulkhead,
    get_circuit_breaker,
    get_health_monitor,
    get_resilient_executor,
)
from .slack_bot import SlackBotServer, create_bot_from_env, is_slack_bot_available
from .smart_routing import RoutingEngine, TeamMember, load_team_profiles
from .sources import (
    BaseSource,
    CodeRepositorySource,
    FileSource,
    GitHubIssueSource,
    SlackHistorySource,
)
from .utils import add
from .validation import (
    InputValidator,
    ValidationConfig,
    sanitize_query,
    validate_slack_input,
)
from .vector_search import VectorSearchEngine, is_vector_search_available

__all__ = [
    "add",
    "Document",
    "BaseSource",
    "FileSource",
    "GitHubIssueSource",
    "CodeRepositorySource",
    "SlackHistorySource",
    "KnowledgeBase",
    "VectorSearchEngine",
    "is_vector_search_available",
    "SlackBotServer",
    "create_bot_from_env",
    "is_slack_bot_available",
    "FileIngester",
    "GitHubIngester",
    "WebDocumentationCrawler",
    "SlackHistoryIngester",
    "ContentProcessor",
    "BatchIngester",
    "MetricsCollector",
    "HealthChecker",
    "PerformanceTracker",
    "StructuredLogger",
    "MonitoredKnowledgeBase",
    "MonitoringConfig",
    "setup_monitoring",
    "get_global_metrics",
    "start_monitoring_server",
    "Query",
    "QueryProcessor",
    "RealTimeUpdater",
    "RoutingEngine",
    "TeamMember",
    "load_team_profiles",
    "SlackNotifier",
    "UsageAnalytics",
    "AuthConfig",
    "AuthMiddleware",
    "BasicAuthenticator",
    "PasswordHasher",
    "get_auth_middleware",
    "InputValidator",
    "ValidationConfig",
    "sanitize_query",
    "validate_slack_input",
    "RateLimiter",
    "RateLimitConfig",
    "UserRateLimiter",
    "get_rate_limiter",
    "get_user_rate_limiter",
    "LLMConfig",
    "ResponseGenerator",
    "LLMResponse",
    "get_response_generator",
    "QuantumTaskPlanner",
    "QuantumTask",
    "TaskState",
    "TaskPriority",
    "get_quantum_planner",
    "create_simple_task",
    "create_dependent_tasks",
    "create_entangled_task_pair",
    "AutonomousSDLC",
    "SDLCPhase",
    "QualityGate",
    "get_autonomous_sdlc",
    "ResilientExecutor",
    "CircuitBreaker",
    "BulkheadIsolation",
    "HealthMonitor",
    "RetryConfig",
    "BulkheadConfig",
    "BackoffStrategy",
    "HealthStatus",
    "get_resilient_executor",
    "get_circuit_breaker",
    "get_bulkhead",
    "get_health_monitor",
    "cli",
]
