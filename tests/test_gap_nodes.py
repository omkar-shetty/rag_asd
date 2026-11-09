from src.agent import mcp_client, nodes

QUESTION = "What is the earliest age autism can be diagnosed?"


def test_propose_without_web_results_has_empty_urls():
    out = nodes.propose_gap_node({"question": QUESTION})
    assert out["proposed_gap"] == {"question": QUESTION, "urls": []}
    assert out["gap_status"] == "proposed"


def test_propose_ignores_web_results():
    # gaps only come from needs_direct, so web URLs are never part of the payload
    out = nodes.propose_gap_node({"question": QUESTION,
                                  "web_results": ["https://example.com"]})
    assert out["proposed_gap"]["urls"] == []


def test_log_gap_success(monkeypatch):
    calls = []
    monkeypatch.setattr(mcp_client, "log_corpus_gap",
                        lambda question, urls: calls.append((question, urls)))

    state = {"proposed_gap": {"question": QUESTION, "urls": []}}
    out = nodes.log_gap_node(state)

    assert out == {"gap_status": "logged"}
    assert calls == [(QUESTION, [])]


def test_log_gap_failure_is_swallowed(monkeypatch):
    def boom(question, urls):
        raise mcp_client.ToolCallError("actions server down")

    monkeypatch.setattr(mcp_client, "log_corpus_gap", boom)

    state = {"proposed_gap": {"question": QUESTION, "urls": []}}
    out = nodes.log_gap_node(state)  # must not raise

    assert out == {"gap_status": "failed"}


def test_log_gap_sends_the_proposed_payload_not_the_state_question(monkeypatch):
    calls = []
    monkeypatch.setattr(mcp_client, "log_corpus_gap",
                        lambda question, urls: calls.append((question, urls)))

    state = {
        "question": "something the user typed later",
        "proposed_gap": {"question": QUESTION, "urls": ["https://www.cdc.gov/autism"]},
    }
    nodes.log_gap_node(state)

    assert calls == [(QUESTION, ["https://www.cdc.gov/autism"])]