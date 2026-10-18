import asyncio
import json
from concurrent.futures import ThreadPoolExecutor

from langchain_core.documents import Document
from langchain_mcp_adapters.client import MultiServerMCPClient

from src.constants import Constants

_SERVERS = {
    "rag_asd": {"transport": "streamable_http", "url": Constants.MCP_SERVER_URL},
}
_client = MultiServerMCPClient(_SERVERS)


async def _call_tool(name: str, query: str) -> list[dict]:
    tools = {t.name: t for t in await _client.get_tools()}
    if name not in tools:
        raise ValueError(f"MCP server has no tool '{name}'. Available: {sorted(tools)}")
    blocks = await tools[name].ainvoke({"query": query})
    return [json.loads(b["text"]) for b in blocks]


def _run(coro):
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


def corpus_search(query: str) -> list[Document]:
    items = _run(_call_tool("corpus_search", query))
    return [Document(page_content=i["content"], metadata={"source": i["source"]}) for i in items]


def web_search(query: str) -> list[dict]:
    return _run(_call_tool("web_search", query))

