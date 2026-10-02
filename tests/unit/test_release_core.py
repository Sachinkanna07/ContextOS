"""RC1 regressions for default core startup, offline accounting, and cleanup."""

from __future__ import annotations

import socket
import sys
from typing import TYPE_CHECKING

import pytest
import tiktoken
from httpx import ASGITransport, AsyncClient

import contextos.api.server as api_server
from contextos.config.settings import DaemonConfig, Settings, TokenCounterConfig
from contextos.core.enums import TokenMeasurementSource
from contextos.daemon.wiring import wire_services
from contextos.demo import run_demo
from contextos.embedding.sentence_transformers import SentenceTransformerEmbedding
from contextos.storage.database import Database

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


@pytest.fixture
def offline_assets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[list[str]]:
    """Fail on any tokenizer load or external socket; leave loopback diagnostics usable."""
    calls: list[str] = []
    monkeypatch.setenv("TIKTOKEN_CACHE_DIR", str(tmp_path / "empty-tokenizer-cache"))
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    monkeypatch.setitem(sys.modules, "torch", None)

    def download(encoding: str) -> None:
        calls.append(encoding)
        raise AssertionError("Offline core must not request tokenizer assets")

    connect = socket.socket.connect

    def guarded_connect(sock: socket.socket, address: tuple[str, int]) -> None:
        if address[0] not in {"127.0.0.1", "localhost", "::1"}:
            calls.append(address[0])
            raise AssertionError("Offline core must not use an external socket")
        connect(sock, address)

    monkeypatch.setattr(tiktoken, "get_encoding", download)
    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    yield calls


@pytest.mark.asyncio
async def test_default_core_real_wiring_is_offline(
    tmp_path: Path, offline_assets: list[str],
) -> None:
    settings = Settings(daemon=DaemonConfig(data_dir=tmp_path))
    assert settings.embedding.model == "deterministic"
    services = await wire_services(settings)
    previous = api_server._services.copy()
    try:
        assert services["token_counter"].measurement_source == TokenMeasurementSource.APPROXIMATED
        api_server.set_services(services)
        async with AsyncClient(
            transport=ASGITransport(app=api_server.create_app()), base_url="http://audit",
        ) as client:
            remembered = await client.post(
                "/api/v1/remember", json={"text": "I prefer concise technical answers."},
            )
            assert remembered.status_code == 200, remembered.text
            inspected = await client.post("/api/v1/inspect", json={"query": "technical answers"})
            assert inspected.status_code == 200, inspected.text
            assert inspected.json()["candidates"]
            assert inspected.json()["context_diff"]["token_measurement_source"] == "approximated"
            doctor = await client.post("/api/v1/doctor")
            assert doctor.json()["overall"] is True
            dashboard = await client.get("/api/v1/dashboard")
            assert dashboard.status_code == 200
        with pytest.raises(RuntimeError, match=r"contextos\[embeddings\]"):
            await SentenceTransformerEmbedding().embed_query("explicit optional embedding")
        assert not offline_assets
    finally:
        api_server._services.clear()
        api_server._services.update(previous)
        await services["database"].close()


@pytest.mark.asyncio
async def test_first_run_demo_needs_no_tokenizer_cache(
    tmp_path: Path, offline_assets: list[str], monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tempfile

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    result = await run_demo()
    assert result["token_measurement_source"] == "approximated"
    assert result["tokenizer"] == "deterministic-word-approximation"
    assert result["model"]["compiled_tokens"] > 0
    assert result["telemetry_invocations"] == 1
    assert not offline_assets
    assert not list(tmp_path.glob("contextos-demo-*"))


@pytest.mark.asyncio
async def test_failed_explicit_tokenizer_startup_closes_database(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    closed: list[Database] = []
    close = Database.close

    async def record_close(database: Database) -> None:
        await close(database)
        closed.append(database)

    def unavailable(encoding: str) -> None:
        raise RuntimeError("Explicit tokenizer asset unavailable")

    monkeypatch.setattr(Database, "close", record_close)
    monkeypatch.setattr(tiktoken, "get_encoding", unavailable)
    with pytest.raises(RuntimeError, match="Explicit tokenizer asset unavailable"):
        await wire_services(Settings(
            daemon=DaemonConfig(data_dir=tmp_path),
            token_counter=TokenCounterConfig(encoding="cl100k_base"),
        ))
    assert len(closed) == 1
    with pytest.raises(RuntimeError, match="not initialized"):
        closed[0].connection()
    (tmp_path / "contextos.db").unlink()
