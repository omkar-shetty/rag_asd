"""MCP server writing any observed corpus gaps to DynamoDB."""

import os
import asyncio
import logging
from botocore.exceptions import BotoCoreError, ClientError
from mcp.server.fastmcp import FastMCP  
from src.actions_server.gaps import get_table, log_gap

log = logging.getLogger(__name__)
mcp = FastMCP("rag-asd-actions", port=8002)

@mcp.tool()
async def log_corpus_gap(question: str, urls: list[str]) -> dict:
    """Record that a question could not be answered from the corpus. urls are 
    the web sources used instead, and may be empty. Repeated calls for the same 
    question increase a counter. Limits: question up to 1000 characters, up to 5 http(s) URLs."""

    try:
        return await asyncio.to_thread(log_gap, get_table(), question, urls)
    except (BotoCoreError, ClientError) as e:
        log.exception(f"Error occurred while logging corpus gap: {e}")
        raise RuntimeError("Could not record the gap; try again later.") from e

if __name__ == "__main__":
    mcp.run(transport=os.getenv("MCP_TRANSPORT", "stdio"))