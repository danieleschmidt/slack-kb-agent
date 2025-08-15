from slack_kb_agent import add


def test_success():
    assert add(1, 2) == 3


def test_edge_case_null_input():
    assert add(None, None) == 0


def test_version_constant():
    from slack_kb_agent import __version__

    assert __version__ == "1.7.2"
