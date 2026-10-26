from asyncio import graph

from langgraph.graph import StateGraph, END
from src.agent.state import AgentState
from src.agent.nodes import (
    corpus_search_node,
    quality_gate_node,
    answer_from_corpus_node,
    web_search_node,
    direct_answer_node,
    refuse_node,
    propose_gap_node,
    log_gap_node,
)
from src.constants import Constants

def route_after_quality_gate(state: AgentState) -> str:
    return state["routing_decision"]

def route_after_direct_answer(state: AgentState) -> str:
    """Propose a gap only when the actions server is in use."""
    return "propose_gap" if Constants.USE_MCP else END

def route_after_propose_gap(state: AgentState) -> str:
    """Log immediately in auto mode; otherwise stop (until the approval step exists)."""
    return "log_gap" if Constants.GAP_APPROVAL == "auto" else END

def build_graph():
    graph = StateGraph(AgentState)
    graph.set_entry_point("corpus_search")
    graph.add_node("corpus_search", corpus_search_node)
    graph.add_node("quality_gate", quality_gate_node)
    graph.add_node("answer_from_corpus", answer_from_corpus_node)
    graph.add_node("web_search", web_search_node)
    graph.add_node("direct_answer", direct_answer_node)
    graph.add_node("refuse", refuse_node)
    graph.add_node("propose_gap", propose_gap_node)
    graph.add_node("log_gap", log_gap_node)
    graph.add_edge("corpus_search", "quality_gate")
    graph.add_conditional_edges(
        "quality_gate",
        route_after_quality_gate,
        {
            "sufficient": "answer_from_corpus",
            "needs_web": "web_search",
            "needs_direct": "direct_answer",
            "out_of_scope": "refuse",
        },
    )
    graph.add_edge("answer_from_corpus", END)
    graph.add_edge("web_search", END)
    graph.add_conditional_edges(
    "direct_answer",
    route_after_direct_answer,
    {"propose_gap": "propose_gap", END: END},
    )
    graph.add_conditional_edges(
    "propose_gap",
    route_after_propose_gap,
    {"log_gap": "log_gap", END: END},
    )
    graph.add_edge("log_gap", END)
    graph.add_edge("propose_gap", log_gap_node)
    graph.add_edge("refuse", END)

    return graph.compile()

compiled_graph = build_graph()