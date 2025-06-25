"""Smart routing utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class TeamMember:
    """Represents a team member with areas of expertise."""

    id: str
    name: str
    expertise: List[str] = field(default_factory=list)


def load_team_profiles(path: str | Path) -> List[TeamMember]:
    """Load team member profiles from a JSON file.

    Each profile should be a mapping with keys ``id``, ``name`` and ``expertise``.
    Returns an empty list if the file cannot be read or parsed.
    """

    try:
        data = Path(path).read_text(encoding="utf-8")
    except OSError:
        return []

    try:
        items = json.loads(data)
    except json.JSONDecodeError:
        return []

    members: List[TeamMember] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        members.append(
            TeamMember(
                id=str(item.get("id", "")),
                name=str(item.get("name", "")),
                expertise=list(item.get("expertise", []) or []),
            )
        )
    return members


class RoutingEngine:
    """Map queries to team members based on expertise keywords."""

    def __init__(self, members: List[TeamMember]) -> None:
        self.members = members

    def route(self, query: str) -> List[TeamMember]:
        """Return members whose expertise appears in the query string."""
        q = query.lower()
        matched: List[TeamMember] = []
        for member in self.members:
            for topic in member.expertise:
                if topic.lower() in q:
                    matched.append(member)
                    break
        return matched
