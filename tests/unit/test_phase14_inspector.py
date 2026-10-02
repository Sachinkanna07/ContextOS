"""Inspector contract: one execution, bounded evidence, and no provider generation."""

from __future__ import annotations

import json

import pytest
from httpx import ASGITransport, AsyncClient
from typer.testing import CliRunner

from contextos.api.server import create_app, set_services
from contextos.cli.app import app
from contextos.config.settings import Settings
from contextos.core.enums import MemoryStatus
from contextos.core.models import Memory
from contextos.daemon.wiring import wire_services


@pytest.fixture
async def stack(tmp_path):
    services = await wire_services(
        Settings(daemon={"data_dir": tmp_path}, embedding={"model": "deterministic"})
    )
    set_services(services)
    try:
        async with AsyncClient(
            transport=ASGITransport(app=create_app()), base_url="http://localhost"
        ) as client:
            yield services, client
    finally:
        await services["database"].close()


@pytest.mark.asyncio
async def test_inspector_empty_is_content_free_and_never_generates(stack, monkeypatch):
    services, client = stack

    async def forbidden(*args, **kwargs):
        raise AssertionError("inspector called provider")

    monkeypatch.setattr(services["fake_provider"], "generate", forbidden)
    response = await client.post("/api/v1/inspect", json={"query": "private raw prompt"})
    assert response.status_code == 200
    data = response.json()
    assert data["provider_dispatch"]["state"] == "NOT_ATTEMPTED"
    assert data["candidates"] == []
    assert data["content"] is None
    assert "private raw prompt" not in response.text
    assert data["context_diff"]["candidate_tokens"] == 0
    assert data["context_diff"]["reduction_ratio"] == 0


@pytest.mark.asyncio
async def test_inspector_single_pass_decisions_and_opt_in(stack, monkeypatch):
    services, client = stack
    memory = await services["memory_repo"].create(
        Memory(
            content="Python tooling preference for Apollo project.",
            status=MemoryStatus.ACTIVE,
        )
    )
    await services["retrieval_index"].ensure_current()
    counts = {"retrieval": 0, "optimizer": 0, "compiler": 0}
    for service_key, method_name, count_key in (
        ("retrieval", "retrieve", "retrieval"),
        ("optimizer", "optimize", "optimizer"),
        ("compilation", "compile", "compiler"),
    ):
        service = services[service_key]
        original = getattr(service, method_name)
        if method_name == "optimize":

            def counted(*args, _original=original, _key=count_key, **kwargs):
                counts[_key] += 1
                return _original(*args, **kwargs)
        else:

            async def counted(*args, _original=original, _key=count_key, **kwargs):
                counts[_key] += 1
                return await _original(*args, **kwargs)

        monkeypatch.setattr(service, method_name, counted)
    payload = {"query": "Python Apollo", "target_memory_id": str(memory.id), "graph": False}
    response = await client.post("/api/v1/inspect", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert counts == {"retrieval": 1, "optimizer": 1, "compiler": 1}
    assert data["configuration"]["mode"] == "hybrid"
    assert data["requested_memory"]["memory_id"] == str(memory.id)
    assert data["context_diff"]["net_token_change"] == (
        data["context_diff"]["compiled_tokens"] - data["context_diff"]["candidate_tokens"]
    )
    assert "Python tooling preference" not in response.text
    opted = (await client.post("/api/v1/inspect", json={**payload, "include_content": True})).json()
    assert "Python tooling preference" in json.dumps(opted)


@pytest.mark.asyncio
async def test_inspector_comparison_requires_opt_in_and_no_ground_truth_claim(stack):
    services, client = stack
    await services["memory_repo"].create(
        Memory(content="Python builds Apollo.", status=MemoryStatus.ACTIVE)
    )
    await services["retrieval_index"].ensure_current()
    response = await client.post(
        "/api/v1/inspect", json={"query": "Python Apollo", "compare": True}
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["comparison"]["runs"]) == 4
    assert data["comparison"]["ground_truth_metrics"] is None
    assert data["provider_dispatch"]["state"] == "NOT_ATTEMPTED"
    assert (
        await client.post("/api/v1/inspect", json={"query": "Python Apollo", "limit": 101})
    ).status_code == 422


@pytest.mark.asyncio
async def test_target_model_identifier_is_not_echoed_in_structured_inspection(stack):
    _, client = stack
    marker = "qwen-sk-testprivatefixture12345678"
    response = await client.post("/api/v1/inspect", json={"query": "empty", "target_model": marker})
    assert response.status_code == 200
    assert marker not in response.text
    assert response.json()["configuration"]["target_model_requested"] is True


def test_inspector_cli_json_and_terminal_are_bounded(monkeypatch):
    payload = {
        "inspection_id": "12345678-1234-4234-8234-123456789012",
        "stages": [],
        "context_diff": {
            "candidate_tokens": 0,
            "optimized_tokens": 0,
            "compiled_tokens": 0,
            "tokens_removed": 0,
            "reduction_ratio": 0,
            "token_measurement_source": "approximated",
            "tokenizer": "fixture",
            "facts_emitted": 0,
            "facts_excluded": 0,
        },
        "provider_dispatch": {"state": "NOT_ATTEMPTED"},
        "candidates": [],
        "requested_memory": None,
        "comparison": None,
        "content": None,
    }

    class Reply:
        def json(self):
            return payload

    monkeypatch.setattr("contextos.cli.app._api", lambda *args, **kwargs: Reply())
    runner = CliRunner()
    assert runner.invoke(app, ["inspect", "query", "--json"]).exit_code == 0
    rendered = runner.invoke(app, ["inspect", "query"])
    assert rendered.exit_code == 0
    assert "NOT_ATTEMPTED" in rendered.stdout
