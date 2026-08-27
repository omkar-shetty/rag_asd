from src.agent.state import AgentState
from src.agent.nodes import corpus_search_node

state: AgentState = {
    "question": "What are common early signs of autism in toddlers?",
    "retrieved_docs": [],
    "routing_decision": None,
    "web_results": None,
    "answer": None,
    "source": None,
}

result = corpus_search_node(state)
print(f"Retrieved {len(result['retrieved_docs'])} docs")
for doc in result["retrieved_docs"]:
    print("---")
    print(doc.page_content[:200])