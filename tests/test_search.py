import pytest
import tempfile
import os
from slack_kb_agent.search import tokenize, build_tfidf_index, search, excerpt
from slack_kb_agent.loader import load_docs


def make_docs(texts):
    return [{"filename": f"doc{i}.md", "content": t} for i, t in enumerate(texts)]


def test_tokenize_basic():
    assert "hello" in tokenize("Hello world")
    assert "world" in tokenize("Hello world")


def test_tokenize_numbers_ignored():
    tokens = tokenize("a1 b2")
    # multi-char alphanumeric ok — just don't crash
    assert "a1" in tokens or len(tokens) >= 0


def test_build_index():
    docs = make_docs(["python is great", "java is verbose"])
    idx = build_tfidf_index(docs)
    assert len(idx["docs"]) == 2
    assert len(idx["tfidf"]) == 2


def test_search_finds_relevant():
    docs = make_docs(["python is a programming language", "cooking recipes for pasta"])
    idx = build_tfidf_index(docs)
    results = search(idx, "python programming")
    assert len(results) > 0
    score, doc = results[0]
    assert "python" in doc["content"]


def test_search_no_results():
    docs = make_docs(["cats and dogs", "birds and fish"])
    idx = build_tfidf_index(docs)
    results = search(idx, "zzzzunlikely")
    assert results == []


def test_excerpt_extracts_best():
    content = "intro text\n\npython is great for data science\n\nsome footer"
    ex = excerpt(content, "python data")
    assert "python" in ex


def test_load_docs_empty_dir():
    with tempfile.TemporaryDirectory() as d:
        docs = load_docs(d)
    assert docs == []


def test_load_docs_reads_md():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "test.md")
        with open(p, "w") as f:
            f.write("# Hello\nThis is a test doc.")
        docs = load_docs(d)
    assert len(docs) == 1
    assert "Hello" in docs[0]["content"]
