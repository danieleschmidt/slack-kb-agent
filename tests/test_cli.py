from slack_kb_agent.knowledge_base import KnowledgeBase
from slack_kb_agent.models import Document
from slack_kb_agent import cli


def test_cli_search(tmp_path, capsys):
    kb = KnowledgeBase()
    kb.add_document(Document(content="hello world", source="manual"))
    kb_path = tmp_path / "kb.json"
    ua_path = tmp_path / "ua.json"
    kb.save(kb_path)

    cli.main(["hello", "--kb", str(kb_path), "--analytics", str(ua_path)])

    out = capsys.readouterr().out
    assert "hello world" in out
    assert ua_path.exists()
