"""Runs the example questions through the graph and writes a comparable summary.

Run once per mode from the repo root, then diff the two files:
    python compare_paths.py direct.txt
    USE_MCP=1 python compare_paths.py mcp.txt
    diff direct.txt mcp.txt
"""

import hashlib
import sys

from dotenv import load_dotenv

load_dotenv()

from src.agent.graph import compiled_graph

QUESTIONS = [
    "What is sensory invalidation?",
    "What are the most recent FDA-approved treatments for ASD announced this year?",
    "What are common early signs of autism in toddlers?",
    "What's a good recipe for banana bread?",
]


def fingerprint(docs):
    # Short hash of the retrieved text: identical hash means identical passages.
    text = "".join(d.page_content for d in docs)
    return hashlib.md5(text.encode()).hexdigest()[:8]


lines = []
for question in QUESTIONS:
    result = compiled_graph.invoke({
        "question": question,
        "retrieved_docs": [],
        "routing_decision": None,
        "web_results": None,
        "answer": None,
        "source": None,
    })
    lines.append(question)
    lines.append(f"  route: {result['routing_decision']} | source: {result['source']}")
    lines.append(f"  corpus docs: {len(result['retrieved_docs'])} | fingerprint: {fingerprint(result['retrieved_docs'])}")
    lines.append(f"  web urls: {result['web_results']}")
    lines.append("")

with open(sys.argv[1], "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("wrote", sys.argv[1])