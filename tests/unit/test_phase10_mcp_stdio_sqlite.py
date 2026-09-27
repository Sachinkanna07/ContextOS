"""Actual contextos-mcp subprocess against deterministic real SQLite wiring."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from uuid import UUID

import pytest
from mcp import Client, StdioServerParameters

from contextos.core.models import MemoryFilters
from contextos.storage.database import Database
from contextos.storage.memory_repo import SqliteMemoryRepository


def body(result):
    assert result.structured_content is not None
    return result.structured_content


@pytest.mark.asyncio
async def test_contextos_mcp_entrypoint_uses_real_sqlite_and_persists(tmp_path: Path):
    home, data_dir = tmp_path / "home", tmp_path / "data"
    config_dir = home / "AppData" / "Local" / "contextos"
    config_dir.mkdir(parents=True)
    (config_dir / "config.toml").write_text(
        "[daemon]\n"
        f'data_dir = "{data_dir.as_posix()}"\n'
        "[embedding]\nmodel = \"deterministic\"\n"
        "[mcp]\nenabled = true\nallow_read = true\nallow_write = true\nallow_telemetry = true\n",
        encoding="utf-8",
    )
    environment = {**os.environ, "HOME": str(home), "USERPROFILE": str(home)}
    parameters = StdioServerParameters(command=sys.executable, args=["-m", "contextos.mcp.server"], env=environment, cwd=str(Path.cwd()))
    async with Client(parameters, read_timeout_seconds=15) as client:
        listed = await client.list_tools()
        assert client.server_info.name == "ContextOS" and len(listed.tools) == 8
        remembered = body(await client.call_tool("contextos_remember", {"text": "I am working on Project Atlas. Project Atlas uses Ollama."}))
        assert remembered["ok"] and remembered["created_memory_ids"]
        assert body(await client.call_tool("contextos_search_memory", {"query": "Atlas Ollama"}))["result_count"] >= 1
        compiled = body(await client.call_tool("contextos_compile_context", {"query": "What does Project Atlas use?", "token_budget": 100}))
        assert compiled["ok"] and compiled["token_count"] <= 100
        assert body(await client.call_tool("contextos_graph_neighbors", {"entity": "Atlas"}))["ok"]
        assert body(await client.call_tool("contextos_current_state", {"property": "uses", "subject": "Project Atlas", "scope": "global"}))["ok"]
    database = Database(data_dir / "contextos.db")
    await database.initialize()
    try:
        memory = await SqliteMemoryRepository(database.connection()).get(UUID(remembered["created_memory_ids"][0]))
        assert memory is not None and memory.source_type == "mcp"
    finally:
        await database.close()
