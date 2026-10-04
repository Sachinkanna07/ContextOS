"""Tests for Phase 9 API routes: /api/v1/ask, /api/v1/models, /api/v1/telemetry/summary."""

from __future__ import annotations

from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from contextos.api.server import create_app, set_services
from contextos.config.settings import ProvidersConfig
from contextos.core.enums import RoutingPolicy
from contextos.embedding.deterministic import DeterministicEmbedding
from contextos.providers.fake import DeterministicFakeProvider
from contextos.services.compilation import QueryAwareContextCompiler
from contextos.services.graph import MemoryGraphService
from contextos.services.graph_retrieval import GraphAugmentedRetrievalEngine
from contextos.services.model_discovery import ModelDiscovery
from contextos.services.model_service import ContextOSModelService
from contextos.services.optimization import MemoryContextOptimizer
from contextos.services.retrieval import HybridRetrievalEngine
from contextos.services.retrieval_index import RetrievalIndexSynchronizer
from contextos.services.router import DeterministicModelRouter
from contextos.services.telemetry_query import TelemetryQueryService
from contextos.services.token_counter import TiktokenCounter
from contextos.storage.database import Database
from contextos.storage.graph_repo import SqliteGraphRepository
from contextos.storage.lexical.bm25 import BM25Index
from contextos.storage.memory_repo import SqliteMemoryRepository
from contextos.storage.relation_repo import SqliteRelationRepository
from contextos.storage.telemetry_repo import SqliteTelemetryRepository
from contextos.storage.vector.in_memory import InMemoryVectorStore


@pytest.fixture
async def test_client(tmp_path: Path):
    db_path = tmp_path / "api_test.db"
    db = Database(db_path)
    await db.initialize()

    conn = db.connection()
    memory_repo = SqliteMemoryRepository(conn)
    relation_repo = SqliteRelationRepository(conn)
    graph_repo = SqliteGraphRepository(conn)
    telemetry_repo = SqliteTelemetryRepository(conn)
    telemetry_query = TelemetryQueryService(telemetry_repo)

    token_counter = TiktokenCounter("cl100k_base")
    optimizer = MemoryContextOptimizer(token_counter=token_counter)
    compiler = QueryAwareContextCompiler(token_counter=token_counter)

    dim = 16
    embedding_service = DeterministicEmbedding(dim)
    lexical_index = BM25Index()
    vector_store = InMemoryVectorStore(dim)
    index_sync = RetrievalIndexSynchronizer(
        memory_repo=memory_repo,
        embedding_service=embedding_service,
        vector_store=vector_store,
        lexical_index=lexical_index,
    )
    base_retrieval = HybridRetrievalEngine(
        memory_repo=memory_repo,
        vector_store=vector_store,
        lexical_index=lexical_index,
        embedding_service=embedding_service,
        index_synchronizer=index_sync,
    )
    graph = MemoryGraphService(
        memory_repo=memory_repo,
        relation_repo=relation_repo,
        graph_repo=graph_repo,
    )
    retrieval = GraphAugmentedRetrievalEngine(
        base_engine=base_retrieval,
        graph_service=graph,
        memory_repo=memory_repo,
    )

    fake_provider = DeterministicFakeProvider(provider_id="fake")
    providers = {"fake": fake_provider}
    router = DeterministicModelRouter(
        default_provider_id="fake",
        default_model_id="fake-default",
        default_policy=RoutingPolicy.LOCAL_FIRST,
    )

    model_service = ContextOSModelService(
        retrieval_service=retrieval,
        optimizer=optimizer,
        compilation_service=compiler,
        router=router,
        providers=providers,
        telemetry_repo=telemetry_repo,
        token_counter=token_counter,
    )

    services = {
        "database": db,
        "memory_repo": memory_repo,
        "relation_repo": relation_repo,
        "graph_repo": graph_repo,
        "telemetry_repo": telemetry_repo,
        "telemetry_query": telemetry_query,
        "providers": providers,
        "provider_settings": ProvidersConfig(),
        "model_discovery": ModelDiscovery(),
        "router": router,
        "model_service": model_service,
    }
    set_services(services)

    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client, services

    await db.close()


@pytest.mark.asyncio
async def test_api_list_models(test_client):
    client, _ = test_client
    resp = await client.get("/api/v1/models")
    assert resp.status_code == 200
    models = resp.json()
    assert len(models) >= 1
    model_ids = [m["model_id"] for m in models]
    assert "fake-default" in model_ids


@pytest.mark.asyncio
async def test_api_ask_endpoint(test_client):
    client, services = test_client
    # Ingest a memory first
    from contextos.core.enums import MemoryStatus, MemoryType
    from contextos.core.models import Memory
    mem_repo = services["memory_repo"]
    await mem_repo.create(
        Memory(
            content="ContextOS Phase 9 introduces deterministic model routing.",
            type=MemoryType.FACT,
            status=MemoryStatus.ACTIVE,
        )
    )

    resp = await client.post(
        "/api/v1/ask",
        json={
            "query": "What does ContextOS Phase 9 introduce?",
            "provider": "fake",
            "model": "fake-default",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "response" in data
    assert "compiled_context" in data
    assert "route_decision" in data
    assert "telemetry" in data
    assert data["route_decision"]["selected_provider"] == "fake"
    assert data["route_decision"]["selected_model"] == "fake-default"
    assert data["telemetry"]["candidate_context_tokens"] > 0


@pytest.mark.asyncio
async def test_api_telemetry_summary_endpoint(test_client):
    client, _ = test_client
    resp = await client.get("/api/v1/telemetry/summary")
    assert resp.status_code == 200
    summary = resp.json()
    assert "total_invocations" in summary
    assert "total_tokens_avoided" in summary
    assert "by_provider" in summary
