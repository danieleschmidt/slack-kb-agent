from slack_kb_agent.knowledge_base import KnowledgeBase
from slack_kb_agent.models import Document


def test_save_and_load(tmp_path):
    kb = KnowledgeBase()
    kb.add_document(Document(content="hi", source="manual"))
    path = tmp_path / "kb.json"
    kb.save(path)

    loaded = KnowledgeBase.load(path)
    assert len(loaded.documents) == 1
    assert loaded.documents[0].content == "hi"


def test_load_missing_file(tmp_path):
    path = tmp_path / "missing.json"
    kb = KnowledgeBase.load(path)
    assert kb.documents == []

