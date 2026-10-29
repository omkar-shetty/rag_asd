"""Run one question through the graph and show the gap outcome.

Usage: python -m scripts.try_gap "your question"
"""

import sys

from src.actions_server.gaps import get_table, question_key
from src.agent.graph import compiled_graph

question = sys.argv[1]
state = compiled_graph.invoke({
    "question": question, "retrieved_docs": [], "routing_decision": None,
    "web_results": None, "answer": None, "source": None,
    "proposed_gap": None, "gap_status": None,
})

print("route:", state["routing_decision"], "| source:", state["source"])
print("gap_status:", state.get("gap_status"))
print("proposed_gap:", state.get("proposed_gap"))
row = get_table().get_item(Key={"question_hash": question_key(question)}).get("Item")
print("row in table:", row)