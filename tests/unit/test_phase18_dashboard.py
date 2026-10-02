"""Provider/model telemetry stays separated by provider and measurement basis."""

from uuid import uuid4

import pytest
from httpx import ASGITransport, AsyncClient

from contextos.api.server import create_app, set_services
from contextos.config.settings import Settings
from contextos.connectors.fake import FakeConnector
from contextos.connectors.models import ConnectorItem
from contextos.core.enums import MemoryStatus, TokenMeasurementSource
from contextos.core.models import Memory, MemorySlot, ModelInvocationTelemetry
from contextos.daemon.wiring import wire_services


@pytest.mark.asyncio
async def test_dashboard_provider_model_groups_and_redacts_suspicious_identifiers(tmp_path):
    services = await wire_services(
        Settings(daemon={"data_dir": tmp_path}, embedding={"model": "deterministic"})
    )
    set_services(services)
    try:
        for provider, basis, tokenizer in (
            ("fake", TokenMeasurementSource.TOKENIZER_COUNTED, "cl100k_base"),
            ("sk-privatefixture123456789", TokenMeasurementSource.APPROXIMATED, "qwen-profile"),
        ):
            await services["telemetry_repo"].record(
                ModelInvocationTelemetry(
                    invocation_id=uuid4(),
                    provider_id=provider,
                    model_id="shared-model",
                    is_local=provider == "fake",
                    candidate_context_tokens=100,
                    compiled_context_tokens=40,
                    context_tokens_avoided=60,
                    reduction_ratio=0.6,
                    lexical_candidate_count=7,
                    dense_candidate_count=5,
                    selected_memory_count=3,
                    graph_expanded_count=2,
                    retrieval_ms=12.5,
                    compilation_ms=4.5,
                    provider_latency_ms=8.0,
                    context_token_measurement_source=basis,
                    context_tokenizer=tokenizer,
                )
            )
        async with AsyncClient(
            transport=ASGITransport(app=create_app()), base_url="http://localhost"
        ) as client:
            response = await client.get("/api/v1/dashboard", params={"model": "shared-model"})
            assert response.status_code == 200
            data = response.json()
            assert len(data["provider_models"]) == 2
            assert data["recent"][0]["lexical_candidate_count"] == 7
            assert data["recent"][0]["dense_candidate_count"] == 5
            assert data["recent"][0]["selected_memory_count"] == 3
            assert data["recent"][0]["graph_expanded_count"] == 2
            assert data["recent"][0]["provider_ms"] == 8.0
            assert {
                (row["provider_measurement_label"], row["context_measurement_label"])
                for row in data["recent"]
            } == {("MEASURED", "MEASURED"), ("MEASURED", "APPROXIMATED")}
            assert data["mcp"] == {
                "enabled": False, "transport": None, "read": False, "write": False,
            }
            assert {row["context_measurement_source"] for row in data["provider_models"]} == {
                "tokenizer_counted",
                "approximated",
            }
            assert "sk-privatefixture123456789" not in response.text
            filtered = (
                await client.get(
                    "/api/v1/dashboard",
                    params={
                        "model": "shared-model",
                        "provider": "fake",
                        "period": "today",
                    },
                )
            ).json()
            assert filtered["period"] == "today"
            assert len(filtered["provider_models"]) == 1
            assert filtered["provider_models"][0]["provider"] == "fake"
            assert filtered["context_measurement_bases"] == [
                {"source": "tokenizer_counted", "tokenizer": "cl100k_base"}
            ]
    finally:
        await services["database"].close()


@pytest.mark.asyncio
async def test_graph_dashboard_uses_projection_counts_and_bounded_paths(tmp_path):
    services = await wire_services(
        Settings(daemon={"data_dir": tmp_path}, embedding={"model": "deterministic"})
    )
    set_services(services)
    try:
        memory = await services["memory_repo"].create(
            Memory(
                content="Project Atlas uses Ollama",
                status=MemoryStatus.ACTIVE,
            )
        )
        await services["graph"].rebuild()
        async with AsyncClient(
            transport=ASGITransport(app=create_app()), base_url="http://localhost"
        ) as client:
            stats = (await client.get("/api/v1/graph/stats")).json()
            assert stats["nodes"] >= 2
            assert stats["edges"] >= 1
            assert stats["dirty"] is False
            shown = (await client.get(f"/api/v1/graph/show/{memory.id}")).json()
            assert shown["candidate_count"] >= 1
            assert len(shown["candidates"]) <= 10
            assert all(len(item["paths"]) <= 5 for item in shown["candidates"])
            assert (
                await client.get("/api/v1/graph/search", params={"entity": "x" * 129})
            ).status_code == 422
    finally:
        await services["database"].close()


@pytest.mark.asyncio
async def test_doctor_and_legacy_stats_report_evidence_without_private_path_or_fake_totals(
    tmp_path,
):
    services = await wire_services(
        Settings(daemon={"data_dir": tmp_path}, embedding={"model": "deterministic"})
    )
    set_services(services)
    try:
        async with AsyncClient(
            transport=ASGITransport(app=create_app()), base_url="http://localhost"
        ) as client:
            doctor = (await client.post("/api/v1/doctor")).json()
            assert doctor["overall"] is True
            for check in (
                "database_integrity", "schema", "tables", "indexes", "graph_projection",
                "telemetry", "connectors", "local_configuration", "privacy_scanner",
                "providers", "embedding_model", "index_consistency",
            ):
                assert doctor["checks"][check]["ok"] is True, check
            assert doctor["checks"]["local_configuration"]["data_directory_accessible"] is True
            assert doctor["checks"]["providers"]["available"] >= 1
            status = (await client.get("/api/v1/status")).json()
            assert str(tmp_path) not in str(status)
            stats = (await client.get("/api/v1/stats")).json()
            assert stats["total_compilations"] is None
            assert stats["average_compression_ratio"] is None
    finally:
        await services["database"].close()


@pytest.mark.asyncio
async def test_temporal_history_is_bounded_and_content_requires_opt_in(tmp_path):
    services = await wire_services(
        Settings(daemon={"data_dir": tmp_path}, embedding={"model": "deterministic"})
    )
    set_services(services)
    try:
        slot = MemorySlot(subject="user", property="response_style", scope="technical")
        previous = await services["memory_repo"].create(
            Memory(
                content="I prefer concise technical replies.",
                status=MemoryStatus.SUPERSEDED,
                slot=slot,
                source_uri="C:\\private\\fixture.txt",
            )
        )
        current = await services["memory_repo"].create(
            Memory(
                content="I prefer detailed technical replies.",
                status=MemoryStatus.ACTIVE,
                slot=slot,
            )
        )
        await services["retrieval_index"].ensure_current()
        async with AsyncClient(
            transport=ASGITransport(app=create_app()), base_url="http://localhost"
        ) as client:
            doctor = (await client.post("/api/v1/doctor")).json()
            assert doctor["checks"]["index_consistency"]["ok"] is True
            assert doctor["checks"]["index_consistency"]["indexable_memories"] == 2
            response = await client.get(f"/api/v1/temporal/history/{current.id}")
            assert response.status_code == 200
            assert len(response.json()["history"]) == 2
            assert str(previous.id) in response.text
            assert "concise technical replies" not in response.text
            assert "C:\\private" not in response.text
            shown = await client.get(
                f"/api/v1/temporal/history/{current.id}", params={"include_content": True}
            )
            assert "concise technical replies" in shown.text
            current_rows = (await client.get("/api/v1/temporal/current")).json()
            assert current_rows["memories"][0]["memory_id"] == str(current.id)
    finally:
        await services["database"].close()


@pytest.mark.asyncio
async def test_connector_dashboard_shows_tracked_count_without_source_uri(tmp_path):
    services = await wire_services(
        Settings(daemon={"data_dir": tmp_path}, embedding={"model": "deterministic"})
    )
    set_services(services)
    marker = "privatefixtureConnectorPath987654"
    try:
        services["connectors"].register(FakeConnector("notes", [ConnectorItem(
            external_id="item-1", source_type="fake", revision="1",
            source_uri=f"fake://{marker}/item-1",
            content="I prefer Rust for systems programming.",
        )]))
        result = await services["connectors"].sync("notes")
        assert result.accepted == 1
        async with AsyncClient(
            transport=ASGITransport(app=create_app()), base_url="http://localhost"
        ) as client:
            response = await client.get("/api/v1/dashboard")
            assert response.status_code == 200
            connector = response.json()["connectors"][0]
            assert connector["tracked_items"] == 1
            assert connector["cursor_present"] is True
            assert marker not in response.text
    finally:
        await services["database"].close()


@pytest.mark.asyncio
async def test_dashboard_discards_malformed_provider_discovery_entries(tmp_path):
    services = await wire_services(
        Settings(daemon={"data_dir": tmp_path}, embedding={"model": "deterministic"})
    )
    set_services(services)

    class MalformedProvider:
        async def list_models(self):
            return [object()]

    services["providers"] = {"malformed": MalformedProvider()}
    set_services(services)
    try:
        async with AsyncClient(
            transport=ASGITransport(app=create_app()), base_url="http://localhost"
        ) as client:
            response = await client.get("/api/v1/dashboard")
            assert response.status_code == 200
            assert response.json()["models"] == []
    finally:
        await services["database"].close()
