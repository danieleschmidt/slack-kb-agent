"""Slack Knowledge Base Agent package."""

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
from .query_processor import Query, QueryProcessor
from .real_time import RealTimeUpdater
from .smart_routing import RoutingEngine, TeamMember, load_team_profiles
from .escalation import SlackNotifier

__all__ = [
    "add",
    "Document",
    "BaseSource",
    "FileSource",
    "GitHubIssueSource",
    "CodeRepositorySource",
    "SlackHistorySource",
    "KnowledgeBase",
    "Query",
    "QueryProcessor",
    "RealTimeUpdater",
    "RoutingEngine",
    "TeamMember",
    "load_team_profiles",
    "SlackNotifier",
]
