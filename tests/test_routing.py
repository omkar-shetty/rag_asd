# tests/test_routing.py
from langgraph.graph import END

from src.agent.routing import (
    route_after_direct_answer,
    route_after_propose_gap,
    route_after_quality_gate,
)
from src.constants import Constants


def test_quality_gate_route_passes_decision_through():
    assert route_after_quality_gate({"routing_decision": "needs_web"}) == "needs_web"

def test_no_gap_proposed_without_mcp(monkeypatch):
    monkeypatch.setattr(Constants, "USE_MCP", False)
    assert route_after_direct_answer({}) == END

def test_gap_proposed_with_mcp(monkeypatch):
    monkeypatch.setattr(Constants, "USE_MCP", True)
    assert route_after_direct_answer({}) == "propose_gap"

def test_required_mode_never_logs_yet(monkeypatch):
    monkeypatch.setattr(Constants, "GAP_APPROVAL", "required")
    assert route_after_propose_gap({}) == END

def test_auto_mode_logs(monkeypatch):
    monkeypatch.setattr(Constants, "GAP_APPROVAL", "auto")
    assert route_after_propose_gap({}) == "log_gap"