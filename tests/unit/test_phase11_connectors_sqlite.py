"""Phase 11 real SQLite connector integration test suite."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from contextos.connectors.fake import FakeConnector
from contextos.connectors.manager import ConnectorManager
from contextos.connectors.models import ConnectorItem, RetentionPolicy
from contextos.core.enums import MemoryStatus, SecretDetectionMode
from contextos.core.models import CompilationConfig, ContextBudget, RetrievalQuery
from contextos.embedding.deterministic import DeterministicEmbedding
from contextos.services.compilation import QueryAwareContextCompiler
from contextos.services.extraction import RuleBasedMemoryExtractor
from contextos.services.graph import MemoryGraphService
from contextos.services.graph_retrieval import GraphAugmentedRetrievalEngine
from contextos.services.ingestion import IngestionPipeline
from contextos.services.optimization import MemoryContextOptimizer
from contextos.services.retrieval import HybridRetrievalEngine
from contextos.services.retrieval_index import RetrievalIndexSynchronizer
from contextos.services.secret_scanner import PatternSecretScanner
from contextos.services.temporal import TemporalMemoryService
from contextos.services.token_counter import DeterministicWordTokenCounter
from contextos.storage.connector_repo import SqliteConnectorRepository
from contextos.storage.database import (
    MIGRATION_2_SQL,
    MIGRATION_3_SQL,
    MIGRATION_4_SQL,
    MIGRATION_5_SQL,
    SCHEMA_SQL,
    Database,
)
from contextos.storage.event_repo import SqliteEventRepository
from contextos.storage.graph_repo import SqliteGraphRepository
from contextos.storage.lexical.bm25 import BM25Index
from contextos.storage.memory_repo import SqliteMemoryRepository
from contextos.storage.relation_repo import SqliteRelationRepository
from contextos.storage.vector.in_memory import InMemoryVectorStore


@pytest.fixture
async def full_sqlite_stack(tmp_path: Path):
    db_path = tmp_path / "full_conn_test.db"
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

    secret_scanner = PatternSecretScanner()
    memory_extractor = RuleBasedMemoryExtractor()

    ingestion = IngestionPipeline(
        secret_scanner=secret_scanner,
        memory_extractor=memory_extractor,
        memory_repo=memory_repo,
        event_repo=event_repo,
        embedding_service=embedding,
        vector_store=vector,
        lexical_index=lexical,
        token_counter=token_counter,
        secret_detection_mode=SecretDetectionMode.STRICT,
    )
    temporal = TemporalMemoryService(memory_repo)

    graph_service = MemoryGraphService(
        memory_repo=memory_repo,
        relation_repo=relation_repo,
        graph_repo=graph_repo,
    )

    index_sync = RetrievalIndexSynchronizer(
        memory_repo=memory_repo,
        embedding_service=embedding,
        vector_store=vector,
        lexical_index=lexical,
    )
    base_retrieval = HybridRetrievalEngine(
        memory_repo=memory_repo,
        vector_store=vector,
        lexical_index=lexical,
        embedding_service=embedding,
        index_synchronizer=index_sync,
    )
    retrieval = GraphAugmentedRetrievalEngine(
        base_engine=base_retrieval,
        graph_service=graph_service,
        memory_repo=memory_repo,
    )

    optimizer = MemoryContextOptimizer(token_counter=token_counter)
    compiler = QueryAwareContextCompiler(token_counter=token_counter)

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
        "memory_repo": memory_repo,
        "event_repo": event_repo,
        "relation_repo": relation_repo,
        "graph_repo": graph_repo,
        "connector_repo": connector_repo,
        "secret_scanner": secret_scanner,
        "memory_extractor": memory_extractor,
        "ingestion": ingestion,
        "temporal": temporal,
        "graph_service": graph_service,
        "retrieval": retrieval,
        "optimizer": optimizer,
        "compiler": compiler,
        "manager": manager,
    }

    await db.close()


@pytest.mark.asyncio
async def test_real_sqlite_connector_ingestion_and_provenance(full_sqlite_stack):
    """Prove connector item is ingested with full provenance metadata."""
    manager = full_sqlite_stack["manager"]
    connector_repo = full_sqlite_stack["connector_repo"]
    memory_repo = full_sqlite_stack["memory_repo"]

    ts = datetime(2026, 9, 1, 10, 0, tzinfo=timezone.utc)
    items = [
        ConnectorItem(
            external_id="item_1",
            source_type="fake",
            source_uri="fake://item_1",
            content="I am working on Project Atlas.",
            revision="rev1",
            updated_at=ts,
        )
    ]
    connector = FakeConnector("conn-1", items)
    manager.register(connector)

    result = await manager.sync("conn-1")
    assert result.status == "success"
    assert result.accepted == 1
    assert result.scanned == 1

    # Verify provenance in connector_items table
    mids = await connector_repo.get_item_memory_ids("conn-1", "item_1")
    assert len(mids) == 1

    # Verify memory is created with connector source type and uri
    memory = await memory_repo.get(mids[0])
    assert memory is not None
    assert memory.source_type == "connector:fake"
    assert memory.source_uri == "fake://item_1"
    assert memory.status == MemoryStatus.ACTIVE


@pytest.mark.asyncio
async def test_no_direct_memory_bypass(full_sqlite_stack, monkeypatch):
    """Prove connectors NEVER directly persist arbitrary source content.

    Connector -> privacy -> extraction -> temporal -> persistence.
    """
    manager = full_sqlite_stack["manager"]
    memory_repo = full_sqlite_stack["memory_repo"]

    direct_create_called = False
    original_create = memory_repo.create

    async def monkey_create(*args, **kwargs):
        nonlocal direct_create_called
        direct_create_called = True
        return await original_create(*args, **kwargs)

    monkeypatch.setattr(memory_repo, "create", monkey_create)

    raw_noise = "Hey team, this is unstructured chatter! I am working on Project Beta. Random ending notes."
    items = [
        ConnectorItem(
            external_id="item_bypass",
            source_type="fake",
            source_uri="fake://bypass",
            content=raw_noise,
            revision="rev1",
        )
    ]
    connector = FakeConnector("conn-bypass", items)
    manager.register(connector)

    await manager.sync("conn-bypass")

    # Ingestion uses temporal.accept(), which manages lifecycle rather than direct un-gated memory_repo.create()
    assert direct_create_called is False

    # Also verify raw chatter preamble was NOT persisted directly into memory content!
    connector_repo = full_sqlite_stack["connector_repo"]
    mids = await connector_repo.get_item_memory_ids("conn-bypass", "item_bypass")
    assert len(mids) >= 1
    mem = await memory_repo.get(mids[0])
    assert mem is not None
    assert "Hey team, this is unstructured chatter" not in mem.content
    assert "Random ending notes" not in mem.content


@pytest.mark.asyncio
async def test_unchanged_incremental_sync_skips_expensive_stages(full_sqlite_stack, monkeypatch):
    """Prove first sync executes privacy/extraction/temporal; second unchanged sync executes NONE."""
    manager = full_sqlite_stack["manager"]
    scanner = full_sqlite_stack["secret_scanner"]
    extractor = full_sqlite_stack["memory_extractor"]
    temporal = full_sqlite_stack["temporal"]

    items = [
        ConnectorItem(
            external_id="item_idemp",
            source_type="fake",
            source_uri="fake://idemp",
            content="I am working on Project Gamma.",
            revision="rev1",
        )
    ]
    connector = FakeConnector("conn-idemp", items)
    manager.register(connector)

    # First sync
    res1 = await manager.sync("conn-idemp")
    assert res1.accepted == 1

    # Instrument invocation counts
    privacy_calls = 0
    extraction_calls = 0
    temporal_calls = 0

    orig_scan = scanner.scan
    def count_scan(text, *args, **kwargs):
        nonlocal privacy_calls
        privacy_calls += 1
        return orig_scan(text, *args, **kwargs)

    orig_extract = extractor.extract
    async def count_extract(text, *args, **kwargs):
        nonlocal extraction_calls
        extraction_calls += 1
        return await orig_extract(text, *args, **kwargs)

    orig_accept = temporal.accept
    async def count_accept(candidate, *args, **kwargs):
        nonlocal temporal_calls
        temporal_calls += 1
        return await orig_accept(candidate, *args, **kwargs)

    monkeypatch.setattr(scanner, "scan", count_scan)
    monkeypatch.setattr(extractor, "extract", count_extract)
    monkeypatch.setattr(temporal, "accept", count_accept)

    # Second sync — identical item
    res2 = await manager.sync("conn-idemp")
    assert res2.status == "success"
    assert res2.unchanged == 1
    assert res2.accepted == 0

    # Invariants: 0 privacy calls, 0 extraction calls, 0 temporal acceptance calls
    assert privacy_calls == 0
    assert extraction_calls == 0
    assert temporal_calls == 0


@pytest.mark.asyncio
async def test_cursor_safety_and_partial_sync_semantics(full_sqlite_stack):
    """Cursor safety: items 1,2 succeed, item 3 fails.

    Restart: item 3 must be reconsidered. Never advance beyond unsafe progress.
    """
    manager = full_sqlite_stack["manager"]
    connector_repo = full_sqlite_stack["connector_repo"]

    class FailingConnector(FakeConnector):
        def __init__(self, connector_id: str, items: list[ConnectorItem]):
            super().__init__(connector_id, items)
            self.calls = 0

        async def scan(self, cursor: str | None) -> tuple[list[ConnectorItem], str | None]:
            self.calls += 1
            return self.items, f"cursor_batch_{self.calls}"

    items = [
        ConnectorItem(external_id="ok_1", source_type="fake", source_uri="fake://1", content="I am working on Project Atlas.", revision="r1"),
        ConnectorItem(external_id="ok_2", source_type="fake", source_uri="fake://2", content="I prefer concise answers.", revision="r1"),
        ConnectorItem(external_id="fail_3", source_type="fake", source_uri="fake://3", content="TRIGGER_FAIL_ITEM", revision="r1"),
    ]

    fail_trigger = True
    orig_ingest = full_sqlite_stack["ingestion"].ingest

    async def conditional_ingest(req):
        if fail_trigger and "TRIGGER_FAIL_ITEM" in req.content:
            raise RuntimeError("transient item 3 failure")
        return await orig_ingest(req)

    full_sqlite_stack["ingestion"].ingest = conditional_ingest

    conn = FailingConnector("conn-safety", items)
    manager.register(conn)

    # Run 1: item 1 and 2 succeed, item 3 fails
    res1 = await manager.sync("conn-safety")
    assert res1.status == "partial"
    assert res1.failed == 1
    assert res1.accepted >= 2

    # Verify state cursor was NOT advanced to cursor_batch_1!
    state1 = await connector_repo.state("conn-safety")
    assert state1.cursor is None or state1.cursor != "cursor_batch_1"

    # Fix transient issue on item 3
    fail_trigger = False
    items[2].content = "I am working on Project Omega."

    # Run 2: restart sync; item 3 is reconsidered and succeeds
    res2 = await manager.sync("conn-safety")
    assert res2.status == "success"
    assert res2.unchanged >= 2  # items 1 and 2 skipped as unchanged
    assert res2.accepted >= 1   # item 3 now accepted!

    # Cursor now successfully advances
    state2 = await connector_repo.state("conn-safety")
    assert state2.cursor == "cursor_batch_2"


@pytest.mark.asyncio
async def test_cancellation_during_item_loop(full_sqlite_stack):
    """Cancellation during processing: committed items remain, unsafe item is not skipped, DB remains usable."""
    manager = full_sqlite_stack["manager"]
    connector_repo = full_sqlite_stack["connector_repo"]
    conn = full_sqlite_stack["conn"]

    items = [
        ConnectorItem(external_id="c_1", source_type="fake", source_uri="fake://c1", content="I am working on Project Atlas.", revision="r1"),
        ConnectorItem(external_id="c_2", source_type="fake", source_uri="fake://c2", content="CANCEL_TRIGGER", revision="r1"),
    ]

    orig_ingest = full_sqlite_stack["ingestion"].ingest

    async def cancelling_ingest(req):
        if "CANCEL_TRIGGER" in req.content:
            raise asyncio.CancelledError()
        return await orig_ingest(req)

    full_sqlite_stack["ingestion"].ingest = cancelling_ingest

    connector = FakeConnector("conn-cancel", items)
    manager.register(connector)

    with pytest.raises(asyncio.CancelledError):
        await manager.sync("conn-cancel")

    # Item 1 was committed before cancellation
    mids = await connector_repo.get_item_memory_ids("conn-cancel", "c_1")
    assert len(mids) >= 1

    # Item 2 was NOT committed
    mids_c2 = await connector_repo.get_item_memory_ids("conn-cancel", "c_2")
    assert len(mids_c2) == 0

    # State is cancelled and cursor not advanced
    state = await connector_repo.state("conn-cancel")
    assert state.status == "cancelled"

    # SQLite DB is fully usable
    async with conn.execute("SELECT COUNT(*) FROM memories") as cursor:
        row = await cursor.fetchone()
        assert row[0] >= 1


@pytest.mark.asyncio
async def test_retention_policy_keep_derived_vs_expire(tmp_path: Path):
    """Test KEEP_DERIVED_MEMORY preserves memories upon source deletion."""
    db_path = tmp_path / "keep_test.db"
    db = Database(db_path)
    await db.initialize()
    conn = db.connection()

    memory_repo = SqliteMemoryRepository(conn)
    event_repo = SqliteEventRepository(conn)
    connector_repo = SqliteConnectorRepository(conn)

    ingestion = IngestionPipeline(
        secret_scanner=PatternSecretScanner(),
        memory_extractor=RuleBasedMemoryExtractor(),
        memory_repo=memory_repo,
        event_repo=event_repo,
        embedding_service=DeterministicEmbedding(16),
        vector_store=InMemoryVectorStore(16),
        lexical_index=BM25Index(),
        token_counter=DeterministicWordTokenCounter(),
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

    items = [
        ConnectorItem(
            external_id="keep_1",
            source_type="fake",
            source_uri="fake://keep_1",
            content="I am working on Project Keeper.",
            revision="r1",
        )
    ]
    connector = FakeConnector("conn-keep", items)
    manager.register(connector)

    await manager.sync("conn-keep")
    mids = await connector_repo.get_item_memory_ids("conn-keep", "keep_1")
    assert len(mids) == 1

    # Mark deleted in source
    items[0] = ConnectorItem(
        external_id="keep_1",
        source_type="fake",
        source_uri="fake://keep_1",
        content="deleted",
        revision="r2",
        deleted=True,
    )
    connector.items = items

    res_del = await manager.sync("conn-keep")
    assert res_del.deleted == 1

    # Memory must remain ACTIVE under KEEP_DERIVED_MEMORY
    mem = await memory_repo.get(mids[0])
    assert mem.status == MemoryStatus.ACTIVE

    await db.close()


@pytest.mark.asyncio
async def test_retention_multi_source_provenance_shared_fact(tmp_path: Path):
    """IMPORTANT: do not expire a fact still independently supported by another source.

    Determine behavior from real provenance model rather than guessing.
    """
    db_path = tmp_path / "multi_src_test.db"
    db = Database(db_path)
    await db.initialize()
    conn = db.connection()

    memory_repo = SqliteMemoryRepository(conn)
    event_repo = SqliteEventRepository(conn)
    connector_repo = SqliteConnectorRepository(conn)

    ingestion = IngestionPipeline(
        secret_scanner=PatternSecretScanner(),
        memory_extractor=RuleBasedMemoryExtractor(),
        memory_repo=memory_repo,
        event_repo=event_repo,
        embedding_service=DeterministicEmbedding(16),
        vector_store=InMemoryVectorStore(16),
        lexical_index=BM25Index(),
        token_counter=DeterministicWordTokenCounter(),
        secret_detection_mode=SecretDetectionMode.STRICT,
    )
    temporal = TemporalMemoryService(memory_repo)

    manager = ConnectorManager(
        state_repo=connector_repo,
        ingestion=ingestion,
        temporal=temporal,
        retention_policy=RetentionPolicy.EXPIRE_ON_SOURCE_DELETE,
        memory_repo=memory_repo,
    )

    # Connector A has doc_a with fact "I am working on Project Atlas."
    item_a = ConnectorItem(
        external_id="doc_a",
        source_type="fake",
        source_uri="fake://doc_a",
        content="I am working on Project Atlas.",
        revision="r1",
    )
    conn_a = FakeConnector("conn-a", [item_a])
    manager.register(conn_a)
    await manager.sync("conn-a")

    mids = await connector_repo.get_item_memory_ids("conn-a", "doc_a")
    assert len(mids) == 1
    shared_mid = mids[0]

    # Connector B also references the exact same fact!
    item_b = ConnectorItem(
        external_id="doc_b",
        source_type="fake",
        source_uri="fake://doc_b",
        content="I am working on Project Atlas.",
        revision="r1",
    )
    conn_b = FakeConnector("conn-b", [item_b])
    manager.register(conn_b)
    await manager.sync("conn-b")

    # Manually associate shared_mid with conn-b item to model multi-source reference
    await connector_repo.save_item("conn-b", item_b, "hash_b", [shared_mid], deleted=False)

    # Active references count should be 2
    refs = await connector_repo.count_active_references(shared_mid)
    assert refs >= 2

    # Now delete source doc_a
    item_a_del = ConnectorItem(
        external_id="doc_a",
        source_type="fake",
        source_uri="fake://doc_a",
        content="deleted",
        revision="r2",
        deleted=True,
    )
    conn_a.items = [item_a_del]
    await manager.sync("conn-a")

    # Fact is still supported by doc_b! It must NOT be expired!
    mem_after_a_del = await memory_repo.get(shared_mid)
    assert mem_after_a_del.status != MemoryStatus.EXPIRED

    # Now delete source doc_b as well
    item_b_del = ConnectorItem(
        external_id="doc_b",
        source_type="fake",
        source_uri="fake://doc_b",
        content="deleted",
        revision="r2",
        deleted=True,
    )
    conn_b.items = [item_b_del]
    await manager.sync("conn-b")

    # Now all references are deleted; fact is expired!
    mem_after_b_del = await memory_repo.get(shared_mid)
    assert mem_after_b_del.status == MemoryStatus.EXPIRED

    await db.close()


@pytest.mark.asyncio
async def test_graph_retrieval_and_compiler_pipeline(full_sqlite_stack):
    """connector sync -> accepted memory -> graph dirty -> graph refresh -> retrievable -> compilable."""
    manager = full_sqlite_stack["manager"]
    graph_service = full_sqlite_stack["graph_service"]
    graph_repo = full_sqlite_stack["graph_repo"]
    retrieval = full_sqlite_stack["retrieval"]
    optimizer = full_sqlite_stack["optimizer"]
    compiler = full_sqlite_stack["compiler"]

    # 1. Sync connector item: "I am working on Project Atlas. Project Atlas uses Python."
    item = ConnectorItem(
        external_id="graph_doc_1",
        source_type="fake",
        source_uri="fake://graph1",
        content="I am working on Project Atlas. Project Atlas uses Python.",
        revision="r1",
    )
    connector = FakeConnector("conn-graph", [item])
    manager.register(connector)
    sync_res = await manager.sync("conn-graph")
    assert sync_res.accepted >= 1

    # 2. Graph projection is dirty after memory creation
    is_dirty = await graph_repo.source_is_dirty()
    assert is_dirty is True

    # 3. Graph refresh
    rebuilt = await graph_service.ensure_current()
    assert rebuilt is True
    assert await graph_repo.source_is_dirty() is False

    # 4. Retrieval finds accepted memory
    retrieval_res = await retrieval.retrieve(RetrievalQuery(text="Project Atlas Python", k=5))
    assert len(retrieval_res.memories) >= 1
    assert any("Project Atlas" in m.memory.content for m in retrieval_res.memories)

    # 5. Context optimizer selects it
    optimized = optimizer.optimize(
        query="Project Atlas Python",
        candidates=retrieval_res.memories,
        budget=ContextBudget(max_tokens=1000),
    )
    assert len(optimized.selected_memories) >= 1



    # 6. Context compiler compiles it
    compiled = await compiler.compile(
        "Project Atlas Python",
        retrieval_res.memories,
        CompilationConfig(budget=1000),
    )
    assert compiled.context_text != ""
    assert "Project Atlas" in compiled.context_text



    # 7. Update source via connector sync and verify expiration lifecycle semantics through real sync
    manager._retention_policy = RetentionPolicy.EXPIRE_ON_SOURCE_DELETE
    connector_repo = full_sqlite_stack["connector_repo"]
    memory_repo = full_sqlite_stack["memory_repo"]
    mids = await connector_repo.get_item_memory_ids("conn-graph", "graph_doc_1")
    assert len(mids) >= 1

    connector.items = [
        ConnectorItem(
            external_id="graph_doc_1",
            source_type="fake",
            source_uri="fake://graph1",
            content="deleted",
            revision="r2",
            deleted=True,
        )
    ]
    del_res = await manager.sync("conn-graph")
    assert del_res.deleted == 1

    # Memory transitioned to EXPIRED by connector sync
    for mid in mids:
        mem = await memory_repo.get(mid)
        assert mem is not None
        assert mem.status == MemoryStatus.EXPIRED

    # Graph dirty trigger fired on memory status update
    assert await graph_repo.source_is_dirty() is True
    await graph_service.ensure_current()

    # Expired memory is no longer eligible for active retrieval!
    retrieval_after = await retrieval.retrieve(RetrievalQuery(text="Project Atlas Python", k=5))
    assert len(retrieval_after.memories) == 0

    # Full lifecycle transition: EXPIRED -> HISTORICAL -> DELETED removes node from graph
    for mid in mids:
        mem = await memory_repo.get(mid)
        if mem:
            hist_mem = await memory_repo.update_status(mid, MemoryStatus.HISTORICAL, expected_version=mem.version)
            await memory_repo.update_status(mid, MemoryStatus.DELETED, expected_version=hist_mem.version)
    assert await graph_repo.source_is_dirty() is True
    await graph_service.ensure_current()
    nodes_after = await graph_service.find_entities("Project Atlas")
    assert len(nodes_after) == 0



@pytest.mark.asyncio
async def test_schema_v5_to_v6_real_database_migration(tmp_path: Path):
    """Create REAL Phase 10 schema-v5 DB with all data types, migrate to v6, prove old data survives."""
    db_path = tmp_path / "real_phase10_v5.db"

    # Step 1: Create a raw SQLite schema-v5 database using Phase 10 migrations
    raw_conn = sqlite3.connect(str(db_path))
    raw_conn.execute("PRAGMA foreign_keys = OFF;")

    # Apply v1 to v5
    raw_conn.executescript(
        "BEGIN IMMEDIATE;\n"
        + SCHEMA_SQL
        + "\nINSERT INTO schema_version (version, description) VALUES (1, 'Initial');\n"
        + MIGRATION_2_SQL
        + "\nINSERT INTO schema_version (version, description) VALUES (2, 'Temporal');\n"
        + MIGRATION_3_SQL
        + "\nINSERT INTO schema_version (version, description) VALUES (3, 'Graph');\n"
        + MIGRATION_4_SQL
        + "\nINSERT INTO schema_version (version, description) VALUES (4, 'Graph dirty');\n"
        + MIGRATION_5_SQL
        + "\nINSERT INTO schema_version (version, description) VALUES (5, 'Model invocations');\n"
        + "COMMIT;"
    )

    # Insert real sample data across all Phase 1-10 tables
    mem_id = str(uuid4())
    event_id = str(uuid4())
    rel_id = str(uuid4())
    target_mem_id = str(uuid4())
    node_id = str(uuid4())
    edge_id = str(uuid4())
    inv_id = str(uuid4())
    now_str = datetime.now(timezone.utc).isoformat()

    # Memories
    raw_conn.execute(
        "INSERT INTO memories(id, content, content_hash, status, type) VALUES(?, ?, ?, ?, ?)",
        (mem_id, "Phase 10 existing memory fact", "hash1", "active", "fact"),
    )
    raw_conn.execute(
        "INSERT INTO memories(id, content, content_hash, status, type) VALUES(?, ?, ?, ?, ?)",
        (target_mem_id, "Target memory fact", "hash2", "active", "fact"),
    )

    # Events
    raw_conn.execute(
        "INSERT INTO events(id, event_type, content, content_hash) VALUES(?, ?, ?, ?)",
        (event_id, "ingest", "Sanitized event content", "ev_hash"),
    )

    # Memory relations
    raw_conn.execute(
        "INSERT INTO memory_relations(id, source_memory_id, target_memory_id, relation_type) VALUES(?, ?, ?, ?)",
        (rel_id, mem_id, target_mem_id, "relates_to"),
    )

    # Graph nodes & edges
    raw_conn.execute(
        "INSERT INTO graph_nodes(id, node_type, canonical_key, label, created_at, updated_at) VALUES(?, ?, ?, ?, ?, ?)",
        (node_id, "project", "project:atlas", "Project Atlas", now_str, now_str),
    )
    raw_conn.execute(
        "INSERT INTO graph_edges(id, source_node_id, target_node_id, relation_type, confidence, created_at, updated_at) VALUES(?, ?, ?, ?, ?, ?, ?)",
        (edge_id, node_id, node_id, "self_loop", 1.0, now_str, now_str),
    )
    raw_conn.execute(
        "INSERT INTO graph_edge_supports(edge_id, memory_id, confidence, created_at) VALUES(?, ?, ?, ?)",
        (edge_id, mem_id, 1.0, now_str),
    )

    # Model invocations
    raw_conn.execute(
        "INSERT INTO model_invocations(id, invocation_id, provider_id, model_id, timestamp, routing_policy, routing_reason, selected_provider, selected_model, token_measurement_source) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (str(uuid4()), inv_id, "fake_provider", "fake_model", now_str, "deterministic", "test", "fake_provider", "fake_model", "deterministic"),
    )

    raw_conn.commit()
    raw_conn.close()

    # Step 2: Now open the v5 database with ContextOS Database class to trigger v5 -> v6 migration
    db = Database(db_path)
    await db.initialize()
    conn = db.connection()

    # Verify schema version is now 6
    async with conn.execute("SELECT MAX(version) FROM schema_version") as cursor:
        row = await cursor.fetchone()
        assert row[0] == 6

    # Verify new tables exist
    async with conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name IN ('connector_state', 'connector_items')"
    ) as cursor:
        rows = await cursor.fetchall()
        table_names = {row[0] for row in rows}
        assert "connector_state" in table_names
        assert "connector_items" in table_names

    # Verify all Phase 10 old data survives completely intact
    async with conn.execute("SELECT content FROM memories WHERE id=?", (mem_id,)) as cursor:
        row = await cursor.fetchone()
        assert row[0] == "Phase 10 existing memory fact"

    async with conn.execute("SELECT content FROM events WHERE id=?", (event_id,)) as cursor:
        row = await cursor.fetchone()
        assert row[0] == "Sanitized event content"

    async with conn.execute("SELECT relation_type FROM memory_relations WHERE id=?", (rel_id,)) as cursor:
        row = await cursor.fetchone()
        assert row[0] == "relates_to"

    async with conn.execute("SELECT label FROM graph_nodes WHERE id=?", (node_id,)) as cursor:
        row = await cursor.fetchone()
        assert row[0] == "Project Atlas"

    async with conn.execute("SELECT invocation_id FROM model_invocations WHERE invocation_id=?", (inv_id,)) as cursor:
        row = await cursor.fetchone()
        assert row[0] == inv_id

    await db.close()

    # Step 3: Reopen v6 DB to prove idempotency and usability
    db_reopened = Database(db_path)
    await db_reopened.initialize()
    async with db_reopened.connection().execute("SELECT MAX(version) FROM schema_version") as cursor:
        row = await cursor.fetchone()
        assert row[0] == 6
    await db_reopened.close()


@pytest.mark.asyncio
async def test_retry_side_effects_no_duplicate_memory(full_sqlite_stack):
    """Attack 7: Verify retry does not duplicate memory records in SQLite."""
    manager = full_sqlite_stack["manager"]
    conn = full_sqlite_stack["conn"]

    item = ConnectorItem(
        external_id="retry_idem_1",
        source_type="fake",
        source_uri="fake://retry_1",
        content="I am working on Project Atlas idempotent retry.",
        revision="r1",
    )
    connector = FakeConnector("conn-retry-idem", [item])
    manager.register(connector)

    # Initial sync
    res1 = await manager.sync("conn-retry-idem")
    assert res1.accepted >= 1

    # Verify 1 memory exists in SQLite
    async with conn.execute(
        "SELECT COUNT(*) FROM memories WHERE content LIKE '%Project Atlas idempotent retry%'"
    ) as cursor:
        count1 = (await cursor.fetchone())[0]
        assert count1 == 1

    # Invalidate connector_items entry to simulate retry before sync result completion
    await conn.execute("DELETE FROM connector_items WHERE connector_id = 'conn-retry-idem'")
    await conn.commit()

    # Re-sync same item (simulating retry of same source item)
    res2 = await manager.sync("conn-retry-idem")
    assert res2.status == "success"

    # Verify no duplicate memory was created in SQLite!
    async with conn.execute(
        "SELECT COUNT(*) FROM memories WHERE content LIKE '%Project Atlas idempotent retry%'"
    ) as cursor:
        count2 = (await cursor.fetchone())[0]
        assert count2 == 1


@pytest.mark.asyncio
async def test_cancellation_during_scan_checkpoint(full_sqlite_stack):
    """Attack 8: Cancel sync while connector scan is blocked; verify state, DB usability, and resume."""
    manager = full_sqlite_stack["manager"]
    connector_repo = full_sqlite_stack["connector_repo"]
    conn = full_sqlite_stack["conn"]

    class ScanCancellingConnector(FakeConnector):
        def __init__(self, connector_id: str, items: list[ConnectorItem]):
            super().__init__(connector_id, items)
            self.cancel_next = True

        async def scan(self, cursor: str | None) -> tuple[list[ConnectorItem], str | None]:
            if self.cancel_next:
                raise asyncio.CancelledError()
            return self.items, "c_resumed"

    items = [
        ConnectorItem(
            external_id="scan_c1",
            source_type="fake",
            source_uri="fake://sc1",
            content="I am working on Project Atlas scan cancellation.",
            revision="r1",
        )
    ]
    connector = ScanCancellingConnector("conn-scan-cancel", items)
    manager.register(connector)

    # Trigger cancellation during scan
    with pytest.raises(asyncio.CancelledError):
        await manager.sync("conn-scan-cancel")

    # State reflects cancellation safely and cursor was not corrupted
    state = await connector_repo.state("conn-scan-cancel")
    assert state is not None
    assert state.status == "cancelled"
    assert state.error_code == "CANCELLED"
    assert state.cursor is None

    # SQLite DB is fully usable
    async with conn.execute("SELECT 1") as cursor:
        assert (await cursor.fetchone())[0] == 1

    # Resume sync on subsequent run
    connector.cancel_next = False
    res = await manager.sync("conn-scan-cancel")
    assert res.status == "success"
    assert res.accepted >= 1
    assert res.next_cursor == "c_resumed"


@pytest.mark.asyncio
async def test_persistence_ordering_recovery_between_temporal_and_state(full_sqlite_stack):
    """Attacks 9 & 10: Inject failure between temporal commit and connector_items/cursor update."""
    manager = full_sqlite_stack["manager"]
    connector_repo = full_sqlite_stack["connector_repo"]
    conn = full_sqlite_stack["conn"]

    items = [
        ConnectorItem(
            external_id="order_1",
            source_type="fake",
            source_uri="fake://order1",
            content="I am working on Project Atlas ordering test.",
            revision="r1",
        )
    ]
    connector = FakeConnector("conn-order", items)
    manager.register(connector)

    # Failure injection: fail during save_item after temporal acceptance has already committed memory
    orig_save_item = connector_repo.save_item
    injected = True

    async def failing_save_item(*args, **kwargs):
        nonlocal injected
        if injected:
            raise RuntimeError("transient persistence failure during save_item")
        return await orig_save_item(*args, **kwargs)

    connector_repo.save_item = failing_save_item

    # Sync fails at save_item boundary
    res1 = await manager.sync("conn-order")
    assert res1.status == "partial"
    assert res1.failed == 1

    # Memory was committed in temporal, but connector_items is empty and cursor not advanced
    mids_before = await connector_repo.get_item_memory_ids("conn-order", "order_1")
    assert len(mids_before) == 0
    state1 = await connector_repo.state("conn-order")
    assert state1.cursor is None

    # Recover on next sync: memory is matched as duplicate without duplicating row, item and cursor persist
    injected = False
    res2 = await manager.sync("conn-order")
    assert res2.status == "success"

    # Exactly 1 memory exists in SQLite
    async with conn.execute(
        "SELECT COUNT(*) FROM memories WHERE content LIKE '%Project Atlas ordering test%'"
    ) as cursor:
        assert (await cursor.fetchone())[0] == 1

    # Item is safely recorded with memory id and cursor is advanced
    mids_after = await connector_repo.get_item_memory_ids("conn-order", "order_1")
    assert len(mids_after) == 1
    state2 = await connector_repo.state("conn-order")
    assert state2.status == "success"
    assert state2.cursor == "1"


@pytest.mark.asyncio
async def test_non_connector_provenance_protected_from_connector_deletion(full_sqlite_stack):
    """Attack 26: Deleting connector item must NOT expire memory created by non-connector provenance."""
    manager = full_sqlite_stack["manager"]
    connector_repo = full_sqlite_stack["connector_repo"]
    memory_repo = full_sqlite_stack["memory_repo"]
    manager._retention_policy = RetentionPolicy.EXPIRE_ON_SOURCE_DELETE

    # 1. Ingest a memory via non-connector provenance (e.g. CLI or MCP)
    cli_ingest = await full_sqlite_stack["ingestion"].ingest(
        from_models_IngestRequest := full_sqlite_stack["ingestion"].__class__.__dict__["ingest"]
    ) if False else None  # type check placeholder

    from contextos.core.models import IngestRequest
    from contextos.core.enums import SourceRole
    res_cli = await full_sqlite_stack["ingestion"].ingest(
        IngestRequest(
            content="I am working on Project Atlas non-connector root.",
            source_type="cli_input",
            source_role=SourceRole.USER,
        )
    )
    cli_resolution = await full_sqlite_stack["temporal"].accept(res_cli.candidates[0])
    cli_memory_id = cli_resolution.memory.id

    # 2. Connect a connector item to this memory
    item = ConnectorItem(
        external_id="non_conn_doc",
        source_type="fake",
        source_uri="fake://doc",
        content="I am working on Project Atlas non-connector root.",
        revision="r1",
    )
    await connector_repo.save_item("conn-shared", item, "hash1", [cli_memory_id], deleted=False)

    # 3. Mark connector item deleted
    del_item = ConnectorItem(
        external_id="non_conn_doc",
        source_type="fake",
        source_uri="fake://doc",
        content="deleted",
        revision="r2",
        deleted=True,
    )
    connector = FakeConnector("conn-shared", [del_item])
    manager.register(connector)

    del_res = await manager.sync("conn-shared")
    assert del_res.deleted == 1

    # 4. Prove that because source_type is "cli_input" (not "connector:..."), memory is NOT expired!
    mem = await memory_repo.get(cli_memory_id)
    assert mem is not None
    assert mem.status == MemoryStatus.ACTIVE  # Protected from expiration!
