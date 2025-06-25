"""Query processing and contextual question answering."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .knowledge_base import KnowledgeBase
from .models import Document
from .smart_routing import RoutingEngine, TeamMember


@dataclass
class Query:
    """Represents a user query with optional metadata."""

    text: str
    user: Optional[str] = None
    channel: Optional[str] = None
    metadata: Dict[str, str] = field(default_factory=dict)


class QueryProcessor:
    """Handle search queries with team-specific context and escalation."""

    def __init__(
        self,
        kb: KnowledgeBase,
        terminology: Optional[Dict[str, str]] = None,
        *,
        routing: Optional["RoutingEngine"] = None,
        enable_escalation: bool = True,
    ) -> None:
        self.kb = kb
        # terminology maps slang or abbreviations to canonical terms
        self.terminology = {
            k.lower(): v.lower() for k, v in (terminology or {}).items()
        }
        self.routing = routing
        self.enable_escalation = enable_escalation

    def normalize(self, text: str) -> str:
        """Expand known terminology and return a normalized query string."""
        tokens = []
        for token in text.split():
            key = token.lower()
            tokens.append(self.terminology.get(key, token))
        return " ".join(tokens)

    def process_query(self, query: Query | str) -> List[Document]:
        """Return documents matching the normalized query."""
        if isinstance(query, Query):
            text = query.text
        else:
            text = query
        normalized = self.normalize(text)
        return self.kb.search(normalized)

    def search_and_route(
        self, query: Query | str
    ) -> Tuple[List[Document], List["TeamMember"]]:
        """Search the knowledge base and route if no results are found."""

        results = self.process_query(query)
        if results:
            return results, []

        if self.enable_escalation and self.routing is not None:
            text = query.text if isinstance(query, Query) else str(query)
            experts = self.routing.route(text)
            return [], experts

        return [], []
