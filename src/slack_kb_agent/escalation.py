"""Utilities for notifying experts about escalated queries."""

from __future__ import annotations

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
        import requests
        from requests.exceptions import (
            ConnectionError,
            HTTPError,
            RequestException,
            Timeout,
        )

        payload = {"channel": member_id, "text": text}
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
            "User-Agent": "slack-kb-agent/1.0"  # Proper user agent
        }

        try:
            response = requests.post(
                "https://slack.com/api/chat.postMessage",
                json=payload,
                headers=headers,
                timeout=5,
                verify=True  # Explicit SSL certificate verification
            )
            response.raise_for_status()  # Raise exception for HTTP errors
        except (ConnectionError, Timeout) as exc:
            raise RuntimeError(f"Network error sending Slack notification: {exc}") from exc
        except HTTPError as exc:
            raise RuntimeError(f"HTTP error sending Slack notification: {exc}") from exc
        except RequestException as exc:
            raise RuntimeError(f"Request error sending Slack notification: {exc}") from exc

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
