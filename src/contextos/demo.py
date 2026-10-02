"""Reproducible offline ContextOS walkthrough on a temporary database."""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from typing import Any
from uuid import UUID

from httpx import ASGITransport, AsyncClient

import contextos.api.server as api_server
from contextos.config.settings import DaemonConfig, EmbeddingConfig, Settings, TokenCounterConfig
from contextos.connectors.fake import FakeConnector
from contextos.connectors.models import ConnectorItem
from contextos.core.enums import SourceRole
from contextos.core.exceptions import SecretDetectedError
from contextos.core.models import IngestRequest
from contextos.daemon.wiring import wire_services
from contextos.services.explainability import ExplanationRequest
from contextos.services.inspection import InspectionRequest


async def run_demo() -> dict[str, Any]:
    """Exercise real services; synthetic fixture values are discarded on exit."""
    with tempfile.TemporaryDirectory(prefix="contextos-demo-") as folder:
        services = await wire_services(
            Settings(
                daemon=DaemonConfig(data_dir=Path(folder)),
                embedding=EmbeddingConfig(model="deterministic"),
                token_counter=TokenCounterConfig(encoding="deterministic"),
            )
        )
        try:

            async def remember(text: str) -> list[str]:
                ingested = await services["ingestion"].ingest(
                    IngestRequest(
                        content=text,
                        source_type="manual",
                        source_role=SourceRole.USER,
                    )
                )
                accepted = [
                    await services["temporal"].accept(
                        candidate, provenance_event_id=ingested.event_id
                    )
                    for candidate in ingested.candidates
                ]
                return [str(item.memory.id) for item in accepted]

            initial = await remember("I prefer concise answers for technical questions.")
            changed = await remember("I now prefer detailed answers for technical questions.")
            graph_ids = await remember("My project Atlas uses Ollama for local inference.")
            connector = FakeConnector(
                "demo_notes",
                [
                    ConnectorItem(
                        external_id="note-1",
                        source_type="fake",
                        source_uri="fake://demo/note-1",
                        revision="1",
                        content="I prefer concise technical documentation.",
                    )
                ],
            )
            services["connectors"].register(connector)
            first_sync = await services["connectors"].sync("demo_notes")
            second_sync = await services["connectors"].sync("demo_notes")
            await services["retrieval_index"].ensure_current()
            graph_nodes, graph_edges, _ = await services["graph"].rebuild()
            query = "detailed answers for technical questions"
            explained = await services["explainability"].explain(
                ExplanationRequest(
                    query=query,
                    graph=False,
                )
            )
            inspected = await services["inspector"].inspect(
                InspectionRequest(
                    query=query,
                    graph=False,
                    target_memory_id=UUID(changed[0]) if changed else None,
                )
            )
            graph_expansion = await services["graph"].expand(
                query_text="Atlas",
                seed_memory_ids=[UUID(graph_ids[0])] if graph_ids else [],
            )
            graph_memory = (
                await services["memory_repo"].get(UUID(graph_ids[0])) if graph_ids else None
            )
            asked = await services["model_service"].ask(
                query, target_provider="fake", target_model="fake-default", explain=True
            )
            rejected = False
            try:
                await services["ingestion"].ingest(
                    IngestRequest(
                        content="My test API key is sk-demoPrivateFixture1234567890",
                        source_type="manual",
                        source_role=SourceRole.USER,
                    )
                )
            except SecretDetectedError:
                rejected = True
            telemetry = await services["telemetry_query"].summary_range(
                provider_id="fake", model_id="fake-default"
            )
            old_memory = await services["memory_repo"].get(UUID(initial[0])) if initial else None
            new_memory = await services["memory_repo"].get(UUID(changed[0])) if changed else None
            dashboard_services = api_server._services.copy()
            try:
                api_server.set_services(services)
                async with AsyncClient(
                    transport=ASGITransport(app=api_server.create_app()),
                    base_url="http://localhost",
                ) as client:
                    dashboard_response = await client.get("/api/v1/dashboard")
                    dashboard_response.raise_for_status()
                    dashboard_data = dashboard_response.json()
            finally:
                api_server._services.clear()
                api_server._services.update(dashboard_services)
            if not (
                initial and changed and graph_ids and first_sync.accepted > 0
                and second_sync.unchanged > 0 and graph_edges > 0
                and graph_expansion.candidate_scores and explained.trace_id
                and inspected.candidates and inspected.context_diff["facts_emitted"] > 0
                and asked.compiled_context.included_memory_ids
                and asked.compiled_context.total_tokens > 0
                and asked.telemetry.context_tokens_avoided > 0
                and asked.telemetry.selected_memory_count > 0
                and asked.telemetry.compiled_fact_count > 0
                and asked.compiled_context.provenance_coverage > 0
            ):
                raise RuntimeError("Offline demo did not produce all required pipeline evidence")
            if asked.telemetry.provider_id != "fake" or telemetry.total_invocations != 1:
                raise RuntimeError(
                    "Offline demo did not invoke and record FakeProvider exactly once"
                )
            demo_connector = next(
                (item for item in dashboard_data["connectors"] if item["id"] == "demo_notes"),
                None,
            )
            if (
                dashboard_data["memories"]["active"] < 1
                or demo_connector is None
                or demo_connector["tracked_items"] < 1
                or not dashboard_data["recent"]
            ):
                raise RuntimeError(
                    "Offline demo dashboard is missing memory, connector, or telemetry data"
                )
            return {
                "label": "LOCAL SYNTHETIC OFFLINE DEMO",
                "token_measurement_source": services["token_counter"].measurement_source.value,
                "tokenizer": services["token_counter"].encoding_name,
                "initial_memory_ids": initial[:10],
                "changed_memory_ids": changed[:10],
                "temporal": {
                    "previous_status": old_memory.status.value if old_memory else None,
                    "current_status": new_memory.status.value if new_memory else None,
                },
                "connector_first": {"accepted": first_sync.accepted, "status": first_sync.status},
                "connector_second": {
                    "unchanged": second_sync.unchanged,
                    "status": second_sync.status,
                },
                "graph": {
                    "nodes": graph_nodes,
                    "edges": graph_edges,
                    "expansion_candidates": len(graph_expansion.candidate_scores),
                    "memory_status": graph_memory.status.value if graph_memory else None,
                    "seed_count": len(graph_expansion.seed_node_ids),
                    "visited": len(graph_expansion.visited_node_ids),
                },
                "explanation": {
                    "trace_id": explained.trace_id,
                    "provider_state": explained.provider_dispatch["state"],
                },
                "inspection": {
                    "inspection_id": inspected.inspection_id,
                    "candidates": len(inspected.candidates),
                    "context_diff": inspected.context_diff,
                },
                "model": {
                    "provider": asked.route_decision.selected_provider,
                    "model": asked.route_decision.selected_model,
                    "dispatch_state": asked.dispatch_evidence.state.value,
                    "compiled_memory_ids": len(asked.compiled_context.included_memory_ids),
                    "compiled_tokens": asked.compiled_context.total_tokens,
                    "tokens_avoided": asked.telemetry.context_tokens_avoided,
                    "provenance_coverage": asked.compiled_context.provenance_coverage,
                },
                "telemetry_invocations": telemetry.total_invocations,
                "dashboard": {
                    "active_memories": dashboard_data["memories"]["active"],
                    "connector_tracked_items": demo_connector["tracked_items"],
                    "recent_invocations": len(dashboard_data["recent"]),
                    "graph_projection_dirty": dashboard_data["graph"]["dirty"],
                },
                "privacy_secret_rejected": rejected,
                "ephemeral": True,
            }
        finally:
            await services["database"].close()


def main() -> None:
    print(json.dumps(asyncio.run(run_demo()), indent=2))


if __name__ == "__main__":
    main()
