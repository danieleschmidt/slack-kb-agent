import json
from pathlib import Path


from slack_kb_agent.smart_routing import (
    TeamMember,
    load_team_profiles,
    RoutingEngine,
)
from slack_kb_agent.escalation import SlackNotifier
from slack_kb_agent.knowledge_base import KnowledgeBase
from slack_kb_agent.query_processor import QueryProcessor


def test_loading_valid_profiles(tmp_path: Path):
    profiles = [
        {"id": "U1", "name": "Alice", "expertise": ["backend", "database"]},
        {"id": "U2", "name": "Bob", "expertise": ["frontend"]},
    ]
    path = tmp_path / "profiles.json"
    path.write_text(json.dumps(profiles), encoding="utf-8")

    members = load_team_profiles(path)
    assert len(members) == 2
    assert all(isinstance(m, TeamMember) for m in members)
    assert members[0].name == "Alice"
    assert "database" in members[0].expertise


def test_invalid_profile_path(tmp_path: Path):
    path = tmp_path / "missing.json"
    members = load_team_profiles(path)
    assert members == []


def test_routing_match_found():
    members = [
        TeamMember(id="U1", name="Alice", expertise=["backend", "database"]),
        TeamMember(id="U2", name="Bob", expertise=["frontend"]),
    ]
    engine = RoutingEngine(members)
    result = engine.route("Need help with backend stuff")
    assert result == [members[0]]


def test_routing_no_match():
    members = [
        TeamMember(id="U1", name="Alice", expertise=["backend"]),
        TeamMember(id="U2", name="Bob", expertise=["frontend"]),
    ]
    engine = RoutingEngine(members)
    result = engine.route("devops question")
    assert result == []


def test_escalate_on_no_results():
    kb = KnowledgeBase()
    members = [
        TeamMember(id="U1", name="Alice", expertise=["backend"]),
        TeamMember(id="U2", name="Bob", expertise=["frontend"]),
    ]
    router = RoutingEngine(members)
    processor = QueryProcessor(kb, routing=router)

    docs, experts = processor.search_and_route("backend issue")

    assert docs == []
    assert experts == [members[0]]


def test_escalation_disabled():
    kb = KnowledgeBase()
    members = [
        TeamMember(id="U1", name="Alice", expertise=["backend"]),
    ]
    router = RoutingEngine(members)
    processor = QueryProcessor(kb, routing=router, enable_escalation=False)

    docs, experts = processor.search_and_route("backend issue")

    assert docs == []
    assert experts == []


def test_slack_message_sent(caplog):
    calls = []

    def sender(member_id: str, text: str) -> None:
        calls.append((member_id, text))

    notifier = SlackNotifier(token="xoxb-test", sender=sender)
    member = TeamMember(id="U1", name="Alice", expertise=["backend"])

    with caplog.at_level("INFO"):
        result = notifier.notify(member.id, "Need help")

    assert result is True
    assert calls == [(member.id, "Need help")]


def test_api_failure(caplog):
    def sender(member_id: str, text: str) -> None:
        raise RuntimeError("oops")

    notifier = SlackNotifier(token="xoxb-test", sender=sender)

    with caplog.at_level("ERROR"):
        result = notifier.notify("U1", "Hello")

    assert result is False
    assert "oops" in caplog.text
