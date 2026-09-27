"""Phase 10 MCP protocol and local STDIO boundary tests."""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import pytest
from mcp import Client, StdioServerParameters

from contextos.core.exceptions import SecretDetectedError
from contextos.core.models import CandidateMemory, IngestResult, RetrievalResult, TelemetrySummary
from contextos.mcp.server import MCPPermissions, ContextOSMCPApplication, create_mcp_server


def body(result):
    assert result.structured_content is not None
    return result.structured_content


class Retrieval:
    def __init__(self, fail: bool = False): self.fail = fail; self.calls = 0
    async def retrieve(self, request):
        self.calls += 1
        if self.fail: raise RuntimeError("E:/private/db.sqlite password=secret")
        return RetrievalResult(query=request.text)


class Telemetry:
    async def summary_today(self): return TelemetrySummary()


class Ingestion:
    def __init__(self, secret: bool = False): self.calls = 0; self.secret = secret
    async def ingest(self, request):
        self.calls += 1
        if self.secret: raise SecretDetectedError(["api_key"])
        return IngestResult(event_id=uuid4(), candidates=[CandidateMemory(content="User is learning C++17", evidence="User is learning C++17")])


class Temporal:
    def __init__(self, fail_after: int | None = None): self.accepted = 0; self.fail_after = fail_after
    async def accept(self, candidate, provenance_event_id):
        self.accepted += 1
        if self.fail_after is not None and self.accepted > self.fail_after: raise RuntimeError("injected failure")
        class Decision: outcome = type("O", (), {"value": "add_new"})()
        class Memory: id = uuid4()
        return type("Result", (), {"decision": Decision(), "memory": Memory()})()


def services(**overrides):
    values = {"retrieval": Retrieval(), "telemetry_query": Telemetry(), "ingestion": Ingestion(), "temporal": Temporal()}
    values.update(overrides)
    return values


@pytest.mark.asyncio
async def test_real_sdk_handshake_metadata_list_call_and_shutdown():
    server = create_mcp_server(services())
    async with Client(server) as client:
        listed = await client.list_tools()
        assert client.server_info.name == "ContextOS"
        assert {tool.name for tool in listed.tools} >= {"contextos_search_memory", "contextos_remember"}
        assert body(await client.call_tool("contextos_search_memory", {"query": "C++17"}))["ok"]


@pytest.mark.asyncio
async def test_sdk_validation_unknown_and_safe_internal_error():
    server = create_mcp_server(services(retrieval=Retrieval(fail=True)))
    async with Client(server) as client:
        invalid = body(await client.call_tool("contextos_search_memory", {"query": "x", "limit": -1}))
        assert invalid == {"ok": False, "error_code": "VALIDATION_ERROR"}
        failed = body(await client.call_tool("contextos_search_memory", {"query": "x"}))
        assert failed == {"ok": False, "error_code": "INTERNAL_ERROR"}
        unknown = await client.call_tool("no_such_contextos_tool", {})
        assert unknown.is_error
        assert "private" not in str(failed).lower()


@pytest.mark.asyncio
async def test_permissions_checked_before_service_execution():
    retrieval, ingestion = Retrieval(), Ingestion()
    denied = create_mcp_server(services(retrieval=retrieval, ingestion=ingestion), MCPPermissions(allow_read=False, allow_write=False, allow_telemetry=False))
    async with Client(denied) as client:
        assert body(await client.call_tool("contextos_search_memory", {"query": "x"}))["error_code"] == "PERMISSION_DENIED"
        assert body(await client.call_tool("contextos_remember", {"text": "User is learning C++17"}))["error_code"] == "PERMISSION_DENIED"
        assert body(await client.call_tool("contextos_telemetry_summary", {}))["error_code"] == "PERMISSION_DENIED"
    assert retrieval.calls == ingestion.calls == 0


@pytest.mark.asyncio
async def test_remember_uses_ingestion_then_temporal_acceptance():
    ingestion, temporal = Ingestion(), Temporal()
    server = create_mcp_server(services(ingestion=ingestion, temporal=temporal), MCPPermissions(allow_write=True))
    async with Client(server) as client:
        response = body(await client.call_tool("contextos_remember", {"text": "User is learning C++17"}))
    assert response["ok"] and response["result_count"] == 1
    assert ingestion.calls == temporal.accepted == 1


@pytest.mark.asyncio
async def test_remember_reports_partial_write_without_false_success():
    ingestion, temporal = Ingestion(), Temporal(fail_after=0)
    server = create_mcp_server(services(ingestion=ingestion, temporal=temporal), MCPPermissions(allow_write=True))
    async with Client(server) as client:
        response = body(await client.call_tool("contextos_remember", {"text": "User is learning C++17"}))
    assert response == {"ok": False, "error_code": "INTERNAL_ERROR", "created_memory_ids": [], "updated_memory_ids": [], "result_count": 0}


@pytest.mark.asyncio
async def test_privacy_rejection_and_telemetry_exclude_secret():
    secret = "sk-proj-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    server = create_mcp_server(services(ingestion=Ingestion(secret=True)), MCPPermissions(allow_write=True))
    async with Client(server) as client:
        response = body(await client.call_tool("contextos_remember", {"text": secret}))
    assert response == {"ok": False, "error_code": "PRIVACY_REJECTED"}
    records = server.contextos_telemetry.recent()
    assert secret not in str(records)
    assert secret not in str(records[0].values())


@pytest.mark.asyncio
async def test_stdio_subprocess_handshake_list_call_and_clean_shutdown():
    script = Path(__file__).with_name("mcp_stdio_server.py")
    params = StdioServerParameters(command=sys.executable, args=[str(script)])
    async with Client(params) as client:
        listed = await client.list_tools()
        assert client.server_info.name == "ContextOS"
        assert len(listed.tools) == 8
        assert body(await client.call_tool("contextos_search_memory", {"query": "safe"}))["ok"]


@pytest.mark.asyncio
@pytest.mark.parametrize("method,args", [("search", ("safe", 1, "hybrid", False)), ("graph_neighbors", ("safe", 1, 1, 1))])
async def test_cancelled_operations_propagate(method, args):
    checkpoint = __import__("asyncio").Event()
    release = __import__("asyncio").Event()
    class Blocking:
        async def retrieve(self, request):
            checkpoint.set(); await release.wait()
        async def expand(self, **kwargs):
            checkpoint.set(); await release.wait()
    app = ContextOSMCPApplication({"retrieval": Blocking(), "graph": Blocking()})
    task = __import__("asyncio").create_task(getattr(app, method)(*args))
    await checkpoint.wait()
    task.cancel()
    with pytest.raises(__import__("asyncio").CancelledError): await task
