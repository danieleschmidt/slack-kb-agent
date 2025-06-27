from slack_kb_agent.analytics import UsageAnalytics
from slack_kb_agent.query_processor import QueryProcessor, Query
from slack_kb_agent.knowledge_base import KnowledgeBase


def test_record_and_top_queries():
    ua = UsageAnalytics()
    ua.record_query("Hello")
    ua.record_query("Hello")
    ua.record_query("World")

    top = ua.top_queries(2)
    assert top[0] == ("hello", 2)
    assert ("world", 1) in top


def test_reset_clears_counts():
    ua = UsageAnalytics()
    ua.record_query("something")
    ua.reset()
    assert ua.top_queries() == []


def test_top_users():
    ua = UsageAnalytics()
    ua.record_query("hi", user="U1")
    ua.record_query("yo", user="U1")
    ua.record_query("yo", user="U2")

    assert ua.top_users(1) == [("U1", 2)]


def test_top_channels():
    ua = UsageAnalytics()
    ua.record_query("hi", channel="C1")
    ua.record_query("yo", channel="C1")
    ua.record_query("yo", channel="C2")

    assert ua.top_channels(1) == [("C1", 2)]


def test_processor_records_queries():
    kb = KnowledgeBase()
    ua = UsageAnalytics()
    qp = QueryProcessor(kb, analytics=ua)
    qp.process_query(Query(text="Hello", user="U1", channel="C1"))

    assert ua.top_queries() == [("hello", 1)]
    assert ua.top_users() == [("U1", 1)]
    assert ua.top_channels() == [("C1", 1)]


def test_save_and_load(tmp_path):
    ua = UsageAnalytics()
    ua.record_query("hi", user="U1", channel="C1")
    path = tmp_path / "ua.json"
    ua.save(path)

    loaded = UsageAnalytics.load(path)
    assert loaded.top_queries() == [("hi", 1)]
    assert loaded.top_users() == [("U1", 1)]
    assert loaded.top_channels() == [("C1", 1)]
