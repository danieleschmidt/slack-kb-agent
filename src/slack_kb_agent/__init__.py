"""Slack Knowledge Base Agent package."""

__version__ = "1.3.0"

from .utils import add
from .models import Document
from .sources import (
    BaseSource,
    CodeRepositorySource,
    FileSource,
    GitHubIssueSource,
    SlackHistorySource,
)
from .knowledge_base import KnowledgeBase
from .vector_search import VectorSearchEngine, is_vector_search_available
from .slack_bot import SlackBotServer, create_bot_from_env, is_slack_bot_available
from .ingestion import (
    FileIngester,
    GitHubIngester, 
    WebDocumentationCrawler,
    SlackHistoryIngester,
    ContentProcessor,
    BatchIngester
)
from .query_processor import Query, QueryProcessor
from .real_time import RealTimeUpdater
from .smart_routing import RoutingEngine, TeamMember, load_team_profiles
from .escalation import SlackNotifier
from .analytics import UsageAnalytics
from . import cli

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
    "Query",
    "QueryProcessor",
    "RealTimeUpdater",
    "RoutingEngine",
    "TeamMember",
    "load_team_profiles",
    "SlackNotifier",
    "UsageAnalytics",
    "cli",
]
