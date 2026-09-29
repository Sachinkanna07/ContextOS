"""Phase 12 terminal product: SQLite, API, CLI, and rendering boundaries."""

from __future__ import annotations

import asyncio
import os
from uuid import uuid4

import pytest
from httpx import ASGITransport, AsyncClient
from typer.testing import CliRunner

from contextos.api.server import create_app, set_services
from contextos.cli.app import app
from contextos.cli.dashboard import safe
from contextos.config.settings import Settings
from contextos.connectors.fake import FakeConnector
from contextos.core.enums import TokenMeasurementSource
from contextos.core.models import ModelInvocationTelemetry
from contextos.daemon.wiring import wire_services
from contextos.daemon.manager import is_running


@pytest.fixture
async def client(tmp_path):
    settings = Settings(daemon={"data_dir": tmp_path}, embedding={"model": "deterministic"})
    services = await wire_services(settings)
    set_services(services)
    async with AsyncClient(transport=ASGITransport(app=create_app()), base_url="http://localhost") as http:
        yield http, services
    await services["database"].close()


@pytest.mark.asyncio
async def test_empty_dashboard_and_connector_failure(client):
    http, services = client
    response = await http.get("/api/v1/dashboard")
    assert response.status_code == 200
    assert response.json()["memories"] == {"active": 0, "historical": 0, "expired": 0}
    assert response.json()["recent"] == []
    assert (await http.get("/api/v1/connectors")).json() == []
    assert (await http.post("/api/v1/connectors/missing/sync")).status_code == 404
    broken = FakeConnector("broken", [])
    broken.failure = ValueError("private source detail")
    services["connectors"].register(broken)
    failure = await http.post("/api/v1/connectors/broken/sync")
    assert failure.json()["status"] == "failed"
    assert "private source detail" not in failure.text
    assert (await http.get("/api/v1/connectors")).json()[0]["error_code"] == "VALIDATION_ERROR"


@pytest.mark.asyncio
async def test_provider_unavailable_is_bounded(client):
    http, services = client
    class Unavailable:
        async def list_models(self):
            raise ConnectionError("private endpoint")
    services["providers"].clear()
    services["providers"]["offline"] = Unavailable()
    response = await http.get("/api/v1/dashboard")
    assert response.status_code == 200
    assert response.json()["models"] == []
    assert "private endpoint" not in response.text


@pytest.mark.asyncio
async def test_configured_local_connector_syncs_through_pipeline(tmp_path):
    notes = tmp_path / "notes"
    notes.mkdir()
    (notes / "preference.txt").write_text("I prefer concise technical documentation.", encoding="utf-8")
    settings = Settings(daemon={"data_dir": tmp_path / "data"},
                        embedding={"model": "deterministic"},
                        connectors={"local_files": {"notes": [notes]}})
    services = await wire_services(settings)
    set_services(services)
    try:
        async with AsyncClient(transport=ASGITransport(app=create_app()), base_url="http://localhost") as http:
            assert (await http.get("/api/v1/connectors")).json()[0]["id"] == "notes"
            result = await http.post("/api/v1/connectors/notes/sync")
            assert result.status_code == 200
            assert result.json()["accepted"] >= 1
            assert (await http.get("/api/v1/dashboard")).json()["memories"]["active"] >= 1
    finally:
        await services["database"].close()


@pytest.mark.asyncio
async def test_telemetry_filters_and_separate_counts(client):
    http, services = client
    await services["telemetry_repo"].record(ModelInvocationTelemetry(
        invocation_id=uuid4(), provider_id="fake", model_id="model-a", is_local=True,
        candidate_context_tokens=100, compiled_context_tokens=40,
        context_tokens_avoided=60, reduction_ratio=.6,
        preflight_input_tokens=50, final_input_tokens=50, provider_input_tokens=48,
        context_token_measurement_source=TokenMeasurementSource.TOKENIZER_COUNTED,
        context_tokenizer="cl100k_base",
        retrieval_ms=2.5, compilation_ms=1.5, graph_expanded_count=2,
    ))
    data = (await http.get("/api/v1/dashboard", params={"model": "model-a"})).json()
    assert data["summary"]["weighted_reduction_ratio"] == .6
    assert data["recent"][0]["preflight_input_tokens"] == 50
    assert data["recent"][0]["provider_input_tokens"] == 48
    assert data["recent"][0]["graph_expanded_count"] == 2
    assert data["context_measurement_bases"] == [{"source": "tokenizer_counted", "tokenizer": "cl100k_base"}]
    assert (await http.get("/api/v1/dashboard", params={"model": "other"})).json()["recent"] == []


@pytest.mark.asyncio
async def test_remember_privacy_and_concurrent_dashboard(client):
    http, _ = client
    async def read():
        for _ in range(3):
            assert (await http.get("/api/v1/dashboard")).status_code == 200
    async def write():
        response = await http.post("/api/v1/remember", json={"text": "I prefer concise Python documentation."})
        assert response.status_code == 200
        assert response.json()["count"] >= 1
    await asyncio.gather(read(), write())
    assert (await http.get("/api/v1/dashboard")).json()["memories"]["active"] >= 1
    secret = "sk_test_" + "contextosfixture000000000000000000"
    response = await http.post("/api/v1/remember", json={"text": f"My key is {secret}"})
    assert secret not in response.text


def test_cli_commands_and_terminal_sanitization(monkeypatch):
    assert safe("private\x1b[31mRED\x1b]0;title\x07\x00") == "privateRED"
    class Reply:
        def json(self):
            return {"memories": {"active": 0, "historical": 0, "expired": 0},
                    "connectors": [], "models": [], "recent": [],
                    "summary": {"total_invocations": 0, "by_model": {},
                                "total_tokens_avoided": 0, "average_reduction_ratio": 0,
                                "weighted_reduction_ratio": 0}}
    monkeypatch.setattr("contextos.cli.app._api", lambda *a, **kw: Reply())
    runner = CliRunner()
    assert runner.invoke(app, ["stats"]).exit_code == 0
    assert runner.invoke(app, ["monitor", "--samples", "1"]).exit_code == 0
    assert runner.invoke(app, ["monitor", "--interval", "0.01", "--samples", "1"]).exit_code != 0


def test_cli_rendering_is_ascii_compatible(monkeypatch):
    class Reply:
        def json(self):
            return {"memories": {"active": 0, "historical": 0, "expired": 0},
                    "connectors": [], "models": [], "recent": [],
                    "summary": {"total_invocations": 0, "by_model": {},
                                "total_tokens_avoided": 0, "average_reduction_ratio": 0,
                                "weighted_reduction_ratio": 0}}
    monkeypatch.setattr("contextos.cli.app._api", lambda *a, **kw: Reply())
    runner = CliRunner()
    result = runner.invoke(app, ["monitor", "--samples", "1"])
    assert result.exit_code == 0
    assert "->" in result.stdout


def test_configuration_fails_closed():
    with pytest.raises(ValueError):
        Settings(daemon={"host": "0.0.0.0"})


def test_stale_pid_cannot_target_unrelated_python(tmp_path):
    settings = Settings(daemon={"data_dir": tmp_path})
    pid_file = tmp_path / "contextos.pid"
    pid_file.write_text(str(os.getpid()))
    assert is_running(settings) == (False, None)
    assert not pid_file.exists()
