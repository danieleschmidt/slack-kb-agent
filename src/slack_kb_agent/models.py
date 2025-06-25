from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class Document:
    """A piece of content stored in the knowledge base."""

    content: str
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)
