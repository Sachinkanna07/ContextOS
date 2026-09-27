"""Real on-disk SQLite integration coverage for the Phase 10 MCP adapter."""

from __future__ import annotations

from pathlib import Path

import pytest
from mcp import Client

from contextos.core.enums import SecretDetectionMode
from contextos.core.models import TelemetrySummary
from contextos.embedding.deterministic import DeterministicEmbedding
from contextos.mcp.server import MCPPermissions, create_mcp_server
from contextos.services.compilation import QueryAwareContextCompiler
from contextos.services.extraction import RuleBasedMemoryExtractor
from contextos.services.graph import MemoryGraphService
from contextos.services.graph_retrieval import GraphAugmentedRetrievalEngine
from contextos.services.ingestion import IngestionPipeline
from contextos.services.optimization import MemoryContextOptimizer
from contextos.services.retrieval import HybridRetrievalEngine
from contextos.services.retrieval_index import RetrievalIndexSynchronizer
from contextos.services.secret_scanner import PatternSecretScanner
from contextos.services.telemetry_query import TelemetryQueryService
from contextos.services.temporal import TemporalMemoryService
from contextos.services.token_counter import DeterministicWordTokenCounter
from contextos.storage.database import Database
from contextos.storage.event_repo import SqliteEventRepository
from contextos.storage.graph_repo import SqliteGraphRepository
from contextos.storage.lexical.bm25 import BM25Index
from contextos.storage.memory_repo import SqliteMemoryRepository
from contextos.storage.relation_repo import SqliteRelationRepository
from contextos.storage.telemetry_repo import SqliteTelemetryRepository
from contextos.storage.vector.in_memory import InMemoryVectorStore


def content(result):
    assert result.structured_content is not None
    return result.structured_content


@pytest.fixture
async def sqlite_mcp(tmp_path: Path):
    db = Database(tmp_path / "phase10-mcp.db")
    await db.initialize()
    connection = db.connection()
    memory_repo = SqliteMemoryRepository(connection)
    event_repo = SqliteEventRepository(connection)
    relation_repo = SqliteRelationRepository(connection)
    graph_repo = SqliteGraphRepository(connection)
    telemetry_repo = SqliteTelemetryRepository(connection)
    embedding = DeterministicEmbedding(16)
    lexical, vector = BM25Index(), InMemoryVectorStore(16)
    index = RetrievalIndexSynchronizer(memory_repo=memory_repo, embedding_service=embedding, vector_store=vector, lexical_index=lexical)
    graph = MemoryGraphService(memory_repo=memory_repo, relation_repo=relation_repo, graph_repo=graph_repo)
    retrieval = GraphAugmentedRetrievalEngine(base_engine=HybridRetrievalEngine(memory_repo=memory_repo, vector_store=vector, lexical_index=lexical, embedding_service=embedding, index_synchronizer=index), graph_service=graph, memory_repo=memory_repo)
    token_counter = DeterministicWordTokenCounter()
    services = {
        "retrieval": retrieval,
        "optimizer": MemoryContextOptimizer(token_counter=token_counter),
        "compilation": QueryAwareContextCompiler(token_counter=token_counter),
        "graph": graph,
        "temporal": TemporalMemoryService(memory_repo),
        "ingestion": IngestionPipeline(secret_scanner=PatternSecretScanner(), memory_extractor=RuleBasedMemoryExtractor(), memory_repo=memory_repo, event_repo=event_repo, embedding_service=embedding, vector_store=vector, lexical_index=lexical, token_counter=token_counter, secret_detection_mode=SecretDetectionMode.STRICT),
        "telemetry_query": TelemetryQueryService(telemetry_repo),
    }
    yield create_mcp_server(services, MCPPermissions(allow_write=True)), memory_repo
    await db.close()


@pytest.mark.asyncio
async def test_real_sqlite_remember_search_compile_and_graph(sqlite_mcp):
    server, memory_repo = sqlite_mcp
    async with Client(server) as client:
        remembered = content(await client.call_tool("contextos_remember", {"text": "I am working on Project Atlas. Project Atlas uses Ollama."}))
        assert remembered["ok"] and remembered["created_memory_ids"]
        found = content(await client.call_tool("contextos_search_memory", {"query": "Atlas Ollama", "mode": "hybrid"}))
        assert found["ok"] and found["result_count"] >= 1
        compiled = content(await client.call_tool("contextos_compile_context", {"query": "What does Project Atlas use?", "token_budget": 100}))
        assert compiled["ok"] and compiled["token_count"] <= 100 and compiled["provenance_ids"]
        graph = content(await client.call_tool("contextos_graph_neighbors", {"entity": "Atlas", "max_hops": 1}))
        assert graph["ok"] and graph["graph_node_count"] >= 1
    assert await memory_repo.get(__import__("uuid").UUID(remembered["created_memory_ids"][0])) is not None


@pytest.mark.asyncio
async def test_real_sqlite_parallel_reads_remain_usable(sqlite_mcp):
    server, _ = sqlite_mcp
    async with Client(server) as client:
        await client.call_tool("contextos_remember", {"text": "I am working on Project Atlas. Project Atlas uses Ollama."})
        results = await __import__("asyncio").gather(*[client.call_tool("contextos_search_memory", {"query": "Atlas"}) for _ in range(8)])
        assert all(content(result)["ok"] for result in results)
        assert content(await client.call_tool("contextos_search_memory", {"query": "Atlas"}))["ok"]
