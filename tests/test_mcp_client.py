"""Tests for the MCP client helper."""

import pytest
from langchain_mcp_adapters.client import MultiServerMCPClient

from src.agent import mcp_client
from src.agent.mcp_client import ToolCallError


def test_unknown_tool_name_is_rejected():
    with pytest.raises(ToolCallError, match="Unknown tool"):
        mcp_client._run(mcp_client._call_tool("no_such_tool", {}))


def test_unreachable_server_raises_tool_call_error(monkeypatch):
    dead = MultiServerMCPClient(
        {"actions": {"transport": "streamable_http", "url": "http://127.0.0.1:9/mcp"}}
    )
    monkeypatch.setattr(mcp_client, "_client", dead)
    with pytest.raises(ToolCallError, match="'actions' failed"):
        mcp_client.log_corpus_gap("q", ["https://example.org/a"])