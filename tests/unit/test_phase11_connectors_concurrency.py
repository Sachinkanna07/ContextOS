"""Phase 11 connector concurrency and system integration test suite."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from contextos.connectors.fake import FakeConnector
from contextos.connectors.manager import ConnectorManager
from contextos.connectors.models import ConnectorItem, RetentionPolicy
from contextos.core.enums import SecretDetectionMode
from contextos.core.models import RetrievalQuery
from contextos.embedding.deterministic import DeterministicEmbedding
from contextos.mcp.server import ContextOSMCPApplication
from contextos.services.extraction import RuleBasedMemoryExtractor
from contextos.services.graph import MemoryGraphService
from contextos.services.graph_retrieval import GraphAugmentedRetrievalEngine
from contextos.services.ingestion import IngestionPipeline
from contextos.services.retrieval import HybridRetrievalEngine
from contextos.services.retrieval_index import RetrievalIndexSynchronizer
from contextos.services.secret_scanner import PatternSecretScanner
from contextos.services.temporal import TemporalMemoryService
from contextos.services.token_counter import DeterministicWordTokenCounter
from contextos.storage.connector_repo import SqliteConnectorRepository
from contextos.storage.database import Database
from contextos.storage.event_repo import SqliteEventRepository
from contextos.storage.graph_repo import SqliteGraphRepository
from contextos.storage.lexical.bm25 import BM25Index
from contextos.storage.memory_repo import SqliteMemoryRepository
from contextos.storage.relation_repo import SqliteRelationRepository
from contextos.storage.vector.in_memory import InMemoryVectorStore


@pytest.fixture
async def full_stack(tmp_path: Path):
    db_path = tmp_path / "conc_stack.db"
    db = Database(db_path)
    await db.initialize()
    conn = db.connection()

    memory_repo = SqliteMemoryRepository(conn)
    event_repo = SqliteEventRepository(conn)
    relation_repo = SqliteRelationRepository(conn)
    graph_repo = SqliteGraphRepository(conn)
    connector_repo = SqliteConnectorRepository(conn)

    embedding = DeterministicEmbedding(16)
    lexical = BM25Index()
    vector = InMemoryVectorStore(16)
    token_counter = DeterministicWordTokenCounter()

    index_sync = RetrievalIndexSynchronizer(
        memory_repo=memory_repo,
        embedding_service=embedding,
        vector_store=vector,
        lexical_index=lexical,
    )
    graph_service = MemoryGraphService(
        memory_repo=memory_repo,
        relation_repo=relation_repo,
        graph_repo=graph_repo,
    )
    base_retrieval = HybridRetrievalEngine(
        memory_repo=memory_repo,
        vector_store=vector,
        lexical_index=lexical,
        embedding_service=embedding,
        index_synchronizer=index_sync,
    )
    graph_retrieval = GraphAugmentedRetrievalEngine(
        base_engine=base_retrieval,
        graph_service=graph_service,
        memory_repo=memory_repo,
    )

    ingestion = IngestionPipeline(
        secret_scanner=PatternSecretScanner(),
        memory_extractor=RuleBasedMemoryExtractor(),
        memory_repo=memory_repo,
        event_repo=event_repo,
        embedding_service=embedding,
        vector_store=vector,
        lexical_index=lexical,
        token_counter=token_counter,
        secret_detection_mode=SecretDetectionMode.STRICT,
    )
    temporal = TemporalMemoryService(memory_repo)

    manager = ConnectorManager(
        state_repo=connector_repo,
        ingestion=ingestion,
        temporal=temporal,
        retention_policy=RetentionPolicy.KEEP_DERIVED_MEMORY,
        memory_repo=memory_repo,
    )

    yield {
        "db": db,
        "conn": conn,
        "manager": manager,
        "retrieval": graph_retrieval,
        "graph": graph_service,
        "memory_repo": memory_repo,
        "connector_repo": connector_repo,
    }

    await db.close()


@pytest.mark.asyncio
async def test_same_connector_concurrency_locking(full_stack):
    """Attack 11: Verify two simultaneous sync(connector_id) calls truly overlap and serialize using barriers."""
    manager = full_stack["manager"]

    task1_in_scan = asyncio.Event()
    release_task1 = asyncio.Event()

    class BarrierConnector(FakeConnector):
        def __init__(self, connector_id: str, items: list[ConnectorItem]):
            super().__init__(connector_id, items)
            self.scan_count = 0

        async def scan(self, cursor: str | None):
            self.scan_count += 1
            if self.scan_count == 1:
                task1_in_scan.set()
                await release_task1.wait()
            return self.items, str(len(self.items))

    items = [
        ConnectorItem(
            external_id=f"item_{i}",
            source_type="fake",
            source_uri=f"fake://{i}",
            content=f"I am working on Project Atlas item {i}.",
            revision="r1",
        )
        for i in range(10)
    ]
    connector = BarrierConnector("conn-same", items)
    manager.register(connector)

    # Launch sync 1 in background task
    sync1_task = asyncio.create_task(manager.sync("conn-same"))

    # Wait until sync 1 is holding the lock inside scan()
    await task1_in_scan.wait()
    assert manager._locks["conn-same"].locked() is True

    # Launch sync 2 in background task while sync 1 is blocked inside lock
    sync2_task = asyncio.create_task(manager.sync("conn-same"))
    await asyncio.sleep(0.01)

    # Sync 2 is blocked waiting for the lock
    assert not sync2_task.done()

    # Release sync 1
    release_task1.set()
    res1 = await sync1_task
    res2 = await sync2_task

    # One active sync processes items, second sync sees items as unchanged
    statuses = {res1.status, res2.status}
    assert statuses == {"success"}
    total_accepted = res1.accepted + res2.accepted
    total_unchanged = res1.unchanged + res2.unchanged

    assert total_accepted == 10
    assert total_unchanged == 10

    # Test lock bounding and unregister
    assert "conn-same" in manager._locks
    manager.unregister("conn-same")
    assert "conn-same" not in manager._locks
    assert "conn-same" not in manager._connectors


@pytest.mark.asyncio
async def test_different_connector_concurrency(full_stack):
    manager = full_stack["manager"]

    items_a = [
        ConnectorItem(
            external_id=f"a_{i}",
            source_type="fake",
            source_uri=f"fake://a_{i}",
            content=f"I am working on Project Alpha item {i}.",
            revision="r1",
        )
        for i in range(5)
    ]
    items_b = [
        ConnectorItem(
            external_id=f"b_{i}",
            source_type="fake",
            source_uri=f"fake://b_{i}",
            content=f"I am working on Project Beta item {i}.",
            revision="r1",
        )
        for i in range(5)
    ]

    conn_a = FakeConnector("conn-a", items_a)
    conn_b = FakeConnector("conn-b", items_b)
    manager.register(conn_a)
    manager.register(conn_b)

    res_a, res_b = await asyncio.gather(
        manager.sync("conn-a"),
        manager.sync("conn-b"),
    )

    assert res_a.status == "success" and res_a.accepted == 5
    assert res_b.status == "success" and res_b.accepted == 5


@pytest.mark.asyncio
async def test_connector_sync_concurrent_with_retrieval_and_graph(full_stack):
    manager = full_stack["manager"]
    retrieval = full_stack["retrieval"]
    graph = full_stack["graph"]

    items = [
        ConnectorItem(
            external_id=f"bg_{i}",
            source_type="fake",
            source_uri=f"fake://bg_{i}",
            content="I am working on Project Atlas. Project Atlas uses Ollama.",
            revision="r1",
        )
        for i in range(5)
    ]
    connector = FakeConnector("conn-bg", items)
    manager.register(connector)

    async def search_task():
        for _ in range(5):
            await retrieval.retrieve(RetrievalQuery(text="Atlas Ollama", k=5))
            await asyncio.sleep(0.001)

    async def graph_task():
        for _ in range(5):
            await graph.expand(query_text="Atlas", max_hops=1)
            await asyncio.sleep(0.001)

    sync_res, _, _ = await asyncio.gather(
        manager.sync("conn-bg"),
        search_task(),
        graph_task(),
    )

    assert sync_res.status == "success"
    assert sync_res.accepted >= 1


@pytest.mark.asyncio
async def test_connector_sync_concurrent_with_mcp_search(full_stack):
    manager = full_stack["manager"]
    retrieval = full_stack["retrieval"]

    services = {"retrieval": retrieval, "connectors": manager}
    mcp_app = ContextOSMCPApplication(services)

    items = [
        ConnectorItem(
            external_id=f"mcp_{i}",
            source_type="fake",
            source_uri=f"fake://mcp_{i}",
            content="I prefer Python for backend development.",
            revision="r1",
        )
        for i in range(5)
    ]
    connector = FakeConnector("conn-mcp", items)
    manager.register(connector)

    async def mcp_search_task():
        for _ in range(3):
            await mcp_app.search("Python backend", 5, "hybrid", False)
            await asyncio.sleep(0.001)

    sync_res, _ = await asyncio.gather(
        manager.sync("conn-mcp"),
        mcp_search_task(),
    )

    assert sync_res.status == "success"
    assert sync_res.accepted >= 1
