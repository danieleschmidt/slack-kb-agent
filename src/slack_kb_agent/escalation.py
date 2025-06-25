"""Utilities for notifying experts about escalated queries."""

from __future__ import annotations

import json
import logging
from typing import Callable, Iterable

from .smart_routing import TeamMember


class SlackNotifier:
    """Send escalation messages to team members via Slack."""

    def __init__(
        self,
        token: str,
        *,
        sender: Callable[[str, str], None] | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.token = token
        self.sender = sender or self._default_sender
        self.logger = logger or logging.getLogger(__name__)

    def _default_sender(self, member_id: str, text: str) -> None:
        from urllib import request

        payload = json.dumps({"channel": member_id, "text": text}).encode("utf-8")
        req = request.Request(
            "https://slack.com/api/chat.postMessage",
            data=payload,
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with request.urlopen(req, timeout=5) as resp:  # nosec B310
            resp.read()

    def notify(self, member_id: str, text: str) -> bool:
        """Send an escalation message to the given member."""
        try:
            self.sender(member_id, text)
        except Exception as exc:  # pragma: no cover - defensive
            self.logger.error("Slack notification failed: %s", exc)
            return False
        self.logger.info("Escalation message sent to %s", member_id)
        return True

    def notify_all(self, members: Iterable[TeamMember], text: str) -> None:
        """Send an escalation message to all provided members."""
        for member in members:
            self.notify(member.id, text)
