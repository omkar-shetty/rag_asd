"""MCP server exposing rag_asd's retrieval and web search 
"""

from functools import cache

from mcp.server.mcpserver import MCPServer

from src.agent.web_search import tavily_client
from src.constants import Constants

mcp = MCPServer("rag-asd")


@cache
def get_retriever():
    # Built on first call, not at startup: loading the embedding model is
    # slow, and clients can time out waiting for the startup handshake.
    from src.agent.retriever import build_retriever
    return build_retriever()


@mcp.tool()
def corpus_search(query: str) -> list[dict]:
    """Search the curated ASD (Autism Spectrum Disorder) clinical and
    caregiver document corpus. Use for questions about symptoms, diagnosis,
    therapies, caregiver strategies, or research findings. Returns relevant
    passages with their source files, or an empty list if nothing matches."""
    docs = get_retriever().invoke(query)
    return [{"content": d.page_content, "source": d.metadata.get("source")} for d in docs]


@mcp.tool()
def web_search(query: str) -> list[dict]:
    """Search a fixed allowlist of trusted ASD sources (CDC, NIH, WHO and
    similar) for recent information the corpus may lack. Try corpus_search
    first for established knowledge."""
    response = tavily_client.search(
        query=query,
        include_domains=Constants.CURATED_DOMAINS,
        max_results=5,
    )
    return [
        {"title": r["title"], "url": r["url"], "content": r["content"]}
        for r in response["results"]
    ]


if __name__ == "__main__":
    mcp.run()
