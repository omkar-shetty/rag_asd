"""MCP server exposing rag_asd's retrieval and web search 
"""
import asyncio
import os
from functools import cache

from mcp.server.fastmcp import FastMCP  

from src.agent.web_search import tavily_client
from src.constants import Constants
from src.agent.retriever import build_retriever

mcp = FastMCP("rag-asd")  


@cache
def get_retriever():
    # Built on first call, not at startup: loading the embedding model is
    # slow, and clients can time out waiting for the startup handshake.
    return build_retriever()


@mcp.tool()
async def corpus_search(query: str) -> list[dict]:
    """...keep your existing docstring..."""
    docs = await asyncio.to_thread(lambda: get_retriever().invoke(query))
    return [{"content": d.page_content, "source": d.metadata.get("source")} for d in docs]


@mcp.tool()
async def web_search(query: str) -> list[dict]:
    """...keep your existing docstring..."""
    response = await asyncio.to_thread(
        tavily_client.search,
        query=query,
        include_domains=Constants.CURATED_DOMAINS,
        max_results=5,
    )
    return [
        {"title": r["title"], "url": r["url"], "content": r["content"]}
        for r in response["results"]
    ]


if __name__ == "__main__":
    mcp.run(transport=os.getenv("MCP_TRANSPORT", "stdio"))
