import asyncio
import json
from concurrent.futures import ThreadPoolExecutor

from langchain_core.documents import Document
from langchain_mcp_adapters.client import MultiServerMCPClient

from src.constants import Constants

class ToolCallError(Exception):
    """Raised when a tool call fails."""

TOOL_SERVERS = {"corpus_search":"rag_asd",
                "web_search":"rag_asd",
                "log_corpus_gap": "actions"}
_SERVERS = {
    "rag_asd": {"transport": "streamable_http", "url": Constants.MCP_SERVER_URL},
    "actions": {"transport": "streamable_http", "url": Constants.ACTIONS_SERVER_URL}
}
_client = MultiServerMCPClient(_SERVERS)

def _root_cause(e):
    while isinstance(e, BaseExceptionGroup) and e.exceptions:
        e = e.exceptions[0]
    return e

async def _call_tool(name: str, args: dict) -> list[dict]:
    server = TOOL_SERVERS.get(name)
    if server is None:
        raise ToolCallError(f"Unknown tool '{name}'")
    try:
        async with _client.session(server) as session:
            result = await session.call_tool(name, args)
    except Exception as e:
        cause = _root_cause(e)
        raise ToolCallError(
            f"{name}: call to server '{server}' failed: {type(cause).__name__}: {cause}"
        ) from e

    if result.isError:
        raise ToolCallError(f"{name}: {result.content[0].text}")

    return [json.loads(block.text) for block in result.content]


def _run(coro):
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


def corpus_search(query: str) -> list[Document]:
    items = _run(_call_tool("corpus_search", {"query": query}))
    return [Document(page_content=i["content"], metadata={"source": i["source"]}) for i in items]


def web_search(query: str) -> list[dict]:
    return _run(_call_tool("web_search", {"query": query}))


def log_corpus_gap(question: str, urls: list[str]) -> dict:
    return _run(_call_tool("log_corpus_gap", {"question": question, "urls": urls}))[0]