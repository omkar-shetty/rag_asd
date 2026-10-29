# src/agent/routing.py
from langgraph.graph import END

from src.agent.state import AgentState
from src.constants import Constants


def route_after_quality_gate(state: AgentState) -> str:
    return state["routing_decision"]

def route_after_direct_answer(state: AgentState) -> str:
    """Propose a gap only when the actions server is in use."""
    return "propose_gap" if Constants.USE_MCP else END

def route_after_propose_gap(state: AgentState) -> str:
    """Log immediately in auto mode; otherwise stop (until the approval step exists)."""
    return "log_gap" if Constants.GAP_APPROVAL == "auto" else END