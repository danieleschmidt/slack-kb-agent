"""Simple usage analytics for tracking query frequency."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
import json
from typing import List, Tuple


@dataclass
class UsageAnalytics:
    """Collect and report query statistics."""

    counts: Counter[str] = field(default_factory=Counter)
    user_counts: Counter[str] = field(default_factory=Counter)
    channel_counts: Counter[str] = field(default_factory=Counter)

    def record_query(
        self, query: str, *, user: str | None = None, channel: str | None = None
    ) -> None:
        """Record a user query for analytics."""
        normalized = query.lower().strip()
        if normalized:
            self.counts[normalized] += 1
            if user:
                self.user_counts[user] += 1
            if channel:
                self.channel_counts[channel] += 1

    def top_queries(self, n: int = 5) -> List[Tuple[str, int]]:
        """Return the ``n`` most common queries."""
        return self.counts.most_common(n)

    def top_users(self, n: int = 5) -> List[Tuple[str, int]]:
        """Return the ``n`` most active users."""
        return self.user_counts.most_common(n)

    def top_channels(self, n: int = 5) -> List[Tuple[str, int]]:
        """Return the ``n`` most active channels."""
        return self.channel_counts.most_common(n)

    def reset(self) -> None:
        """Clear all recorded statistics."""
        self.counts.clear()
        self.user_counts.clear()
        self.channel_counts.clear()

    # Persistence helpers -------------------------------------------------

    def to_dict(self) -> dict[str, dict[str, int]]:
        """Return a serializable representation of analytics data."""
        return {
            "counts": dict(self.counts),
            "user_counts": dict(self.user_counts),
            "channel_counts": dict(self.channel_counts),
        }

    @classmethod
    def from_dict(cls, data: dict[str, dict[str, int]]) -> "UsageAnalytics":
        """Create a :class:`UsageAnalytics` instance from a dictionary."""
        ua = cls()
        ua.counts.update(data.get("counts", {}))
        ua.user_counts.update(data.get("user_counts", {}))
        ua.channel_counts.update(data.get("channel_counts", {}))
        return ua

    def save(self, path: str | Path) -> None:
        """Persist analytics data to ``path`` as JSON."""
        Path(path).write_text(json.dumps(self.to_dict()), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "UsageAnalytics":
        """Load analytics data from ``path`` returning a new instance."""
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
