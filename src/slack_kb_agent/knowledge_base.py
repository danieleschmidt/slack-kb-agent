"""In-memory knowledge base supporting multiple data sources."""

from __future__ import annotations

from typing import List
import json
from dataclasses import asdict
from pathlib import Path

from .models import Document
from .sources import BaseSource


class KnowledgeBase:
    """Aggregate documents from various sources and provide simple search."""

    def __init__(self) -> None:
        self.sources: List[BaseSource] = []
        self.documents: List[Document] = []

    def add_source(self, source: BaseSource) -> None:
        """Register a new data source."""
        self.sources.append(source)

    def index(self) -> None:
        """Load all documents from registered sources."""
        for source in self.sources:
            self.documents.extend(source.load())

    def add_document(self, document: Document) -> None:
        """Add a single document to the knowledge base."""
        self.documents.append(document)

    def add_documents(self, documents: List[Document]) -> None:
        """Add multiple documents to the knowledge base."""
        self.documents.extend(documents)

    def search(self, query: str) -> List[Document]:
        """Return documents containing the query string."""
        q = query.lower()
        return [doc for doc in self.documents if q in doc.content.lower()]

    # Persistence helpers -------------------------------------------------

    def to_dict(self) -> dict[str, list[dict]]:
        """Return a serializable representation of all documents."""
        return {"documents": [asdict(d) for d in self.documents]}

    @classmethod
    def from_dict(cls, data: dict[str, list[dict]]) -> "KnowledgeBase":
        """Create a knowledge base from a dictionary."""
        kb = cls()
        for item in data.get("documents", []):
            if not isinstance(item, dict):
                continue
            kb.add_document(Document(**item))
        return kb

    def save(self, path: str | Path) -> None:
        """Persist documents to ``path`` as JSON."""
        Path(path).write_text(json.dumps(self.to_dict()), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "KnowledgeBase":
        """Load documents from ``path`` and return a new knowledge base."""
        try:
            text = Path(path).read_text(encoding="utf-8")
        except OSError:
            return cls()
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return cls()
        if not isinstance(data, dict):
            return cls()
        return cls.from_dict(data)
