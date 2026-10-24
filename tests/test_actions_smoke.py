"""Smoke test for action server functions."""

# DynamoDB Local and the actions server must be running.
# Skipped automatically when they are not.

import asyncio
import json
import socket
import uuid

import pytest
from langchain_mcp_adapters.client import MultiServerMCPClient

from src.actions_server.gaps import get_table, question_key
from src.constants import Constants

URL = "http://127.0.0.1:8002/mcp"


def _server_up():
    try:
        socket.create_connection(("127.0.0.1", 8002), timeout=1).close()
        return True
    except OSError:
        return False


pytestmark = pytest.mark.skipif(
    not (_server_up() and Constants.DYNAMODB_ENDPOINT),
    reason="needs the actions server on :8002 and DYNAMODB_ENDPOINT set",
)


async def _call(question, urls):
    client = MultiServerMCPClient({"actions": {"transport": "streamable_http", "url": URL}})
    async with client.session("actions") as session:
        return await session.call_tool(
            "log_corpus_gap", {"question": question, "urls": urls}
        )


def call(question, urls):
    return asyncio.run(_call(question, urls))


@pytest.fixture
def question():
    q = f"smoke test {uuid.uuid4()}"
    yield q
    get_table().delete_item(Key={"question_hash": question_key(q)})  # tidy up


def test_logging_twice_increments_count(question):
    first = call(question, ["https://example.org/a"])
    second = call(question, ["https://example.org/a"])
    assert not first.isError and not second.isError
    assert json.loads(first.content[0].text)["count"] == 1
    assert json.loads(second.content[0].text)["count"] == 2


def test_invalid_input_is_rejected_and_nothing_is_written(question):
    result = call(question, ["javascript:alert(1)"])
    assert result.isError
    assert "valid http" in result.content[0].text
    assert "Item" not in get_table().get_item(Key={"question_hash": question_key(question)})