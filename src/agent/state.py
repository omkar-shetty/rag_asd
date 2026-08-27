from typing import Literal, TypedDict
from langchain_core.documents import Document
 
 
class AgentState(TypedDict):
    """Shared state passed between nodes in the agent graph.
 
    A messages list is not needed here as each node in the graph 
    has one job and runs once per question.
    """
 
    question: str   

    retrieved_docs: list[Document] # populated by corpus_search_node - empty list if nothing came back
 
    # set by quality_gate, read by the conditional edge to pick the next node
    routing_decision: Literal["out_of_scope", "sufficient", "needs_web", "needs_direct"] | None
 
    web_results: str | None # only populated if web_search_node actually runs
 
    answer: str | None # final user-facing text
 
    source: Literal["corpus", "web", "direct", "refused"] | None # which path actually produced the answer