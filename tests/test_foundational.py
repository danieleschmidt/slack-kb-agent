from slack_kb_agent import add


def test_success():
    assert add(1, 2) == 3


def test_edge_case_null_input():
    assert add(None, None) == 0
