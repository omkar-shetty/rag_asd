"""Smoke test for the rag-asd MCP server.

Run from the repo root: python -m src.client_test
Starts the server as a subprocess, lists its tools, then calls each once.
"""

import asyncio
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

QUERY = "early signs of autism in toddlers"
TIMEOUT_SECONDS = 120  # first corpus_search is slow (embedding model load)


async def main():
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "src.mcp_server.server"],
    )

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            for tool in tools.tools:
                print(tool.name, "|", tool.inputSchema.get("required"))

            for tool in tools.tools:
                result = await session.call_tool(tool.name, {"query": QUERY})
                print(f"\n--- {tool.name} ---")
                print(str(result)[:600])


if __name__ == "__main__":
    asyncio.run(asyncio.wait_for(main(), timeout=TIMEOUT_SECONDS))