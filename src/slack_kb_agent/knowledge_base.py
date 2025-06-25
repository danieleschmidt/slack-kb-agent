"""In-memory knowledge base supporting multiple data sources."""

from __future__ import annotations

from typing import List

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
