"""Data source definitions for the knowledge base."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from .models import Document


class BaseSource(ABC):
    """Abstract base class for all data sources."""

    name: str

    @abstractmethod
    def load(self) -> List[Document]:
        """Load documents from the source."""
        raise NotImplementedError


class FileSource(BaseSource):
    """Load documents from a collection of text files."""

    def __init__(self, paths: Sequence[str | Path], name: str = "docs") -> None:
        self.paths = [Path(p) for p in paths]
        self.name = name

    def load(self) -> List[Document]:
        documents: List[Document] = []
        for path in self.paths:
            try:
                content = Path(path).read_text(encoding="utf-8")
            except OSError:
                continue
            documents.append(
                Document(
                    content=content, source=self.name, metadata={"path": str(path)}
                )
            )
        return documents


class GitHubIssueSource(BaseSource):
    """Load documents from GitHub issue metadata."""

    def __init__(self, issues: Iterable[Dict[str, Any]], name: str = "github") -> None:
        self.issues = list(issues)
        self.name = name

    def load(self) -> List[Document]:
        documents: List[Document] = []
        for issue in self.issues:
            title = issue.get("title", "")
            body = issue.get("body", "")
            content = f"{title}\n\n{body}".strip()
            documents.append(
                Document(
                    content=content,
                    source=self.name,
                    metadata={"id": issue.get("id"), "number": issue.get("number")},
                )
            )
        return documents


class CodeRepositorySource(BaseSource):
    """Index source code files from a repository."""

    def __init__(
        self,
        root: str | Path,
        extensions: Sequence[str] = (".py", ".md"),
        name: str = "code",
    ) -> None:
        self.root = Path(root)
        self.extensions = tuple(extensions)
        self.name = name

    def load(self) -> List[Document]:
        documents: List[Document] = []
        if not self.root.exists():
            return documents

        for path in self.root.rglob("*"):
            if path.suffix in self.extensions and path.is_file():
                try:
                    content = path.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    continue
                documents.append(
                    Document(
                        content=content,
                        source=self.name,
                        metadata={"path": str(path.relative_to(self.root))},
                    )
                )
        return documents


class SlackHistorySource(BaseSource):
    """Load Slack messages provided as a list of dictionaries."""

    def __init__(self, messages: Iterable[Dict[str, Any]], name: str = "slack") -> None:
        self.messages = list(messages)
        self.name = name

    def load(self) -> List[Document]:
        documents: List[Document] = []
        for msg in self.messages:
            text = msg.get("text", "")
            documents.append(
                Document(
                    content=text,
                    source=self.name,
                    metadata={"user": msg.get("user"), "ts": msg.get("ts")},
                )
            )
        return documents
