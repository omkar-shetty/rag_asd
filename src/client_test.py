import asyncio, sys
from pathlib import Path
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

PROJECT_ROOT = Path(__file__).resolve().parent.parent

async def main():
    params = StdioServerParameters(command=sys.executable, args=["-m", "src.mcp_server.server"], cwd=str(PROJECT_ROOT))
    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            tools = await s.list_tools()
            for name in ["corpus_search", "web_search"]:
                result = await s.call_tool(name, {"query": "early signs of autism in toddlers"})
                print(f"\n--- {name} ---")
                print(str(result)[:600])
asyncio.run(main())