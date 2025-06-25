"""Real-time learning utilities for updating the knowledge base."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

from .knowledge_base import KnowledgeBase
from .models import Document


class RealTimeUpdater:
    """Ingest new messages and update the knowledge base in real time."""

    def __init__(self, kb: KnowledgeBase) -> None:
        self.kb = kb

    def ingest_message(
        self, text: str, *, user: Optional[str] = None, ts: Optional[str] = None
    ) -> None:
        """Add a Slack message as a document to the knowledge base."""
        doc = Document(content=text, source="slack", metadata={"user": user, "ts": ts})
        self.kb.add_document(doc)

    def ingest_messages(self, messages: Iterable[Dict[str, Any]]) -> None:
        """Add multiple Slack messages to the knowledge base."""
        for msg in messages:
            self.ingest_message(
                msg.get("text", ""), user=msg.get("user"), ts=msg.get("ts")
            )
