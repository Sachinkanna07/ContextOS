"""Phase 11 connectors unit test suite."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from contextos.connectors.fake import FakeConnector
from contextos.connectors.json_import import JsonImportConnector
from contextos.connectors.local_files import LocalFileConnector
from contextos.connectors.manager import ConnectorManager
from contextos.connectors.models import (
    ConnectorItem,
    ConnectorSyncResult,
    ConnectorSyncState,
    RetentionPolicy,
)
from contextos.core.enums import MemoryStatus, SecretDetectionMode
from contextos.core.exceptions import IngestionError, SecretDetectedError
from contextos.embedding.deterministic import DeterministicEmbedding
from contextos.services.extraction import RuleBasedMemoryExtractor
from contextos.services.ingestion import IngestionPipeline
from contextos.services.secret_scanner import PatternSecretScanner
from contextos.services.temporal import TemporalMemoryService
from contextos.services.token_counter import DeterministicWordTokenCounter
from contextos.storage.connector_repo import SqliteConnectorRepository
from contextos.storage.database import Database
from contextos.storage.event_repo import SqliteEventRepository
from contextos.storage.lexical.bm25 import BM25Index
from contextos.storage.memory_repo import SqliteMemoryRepository
from contextos.storage.vector.in_memory import InMemoryVectorStore


def test_connector_item_model_validation():
    item = ConnectorItem(
        external_id="doc_1",
        source_type="local_file",
        source_uri="file:///doc_1.txt",
        content="I am working on Project Atlas.",
        revision="rev1",
        title="Doc 1",
        metadata={"author": "Alice", "priority": 1},
    )
    assert item.external_id == "doc_1"
    assert item.content == "I am working on Project Atlas."
    assert item.metadata["priority"] == 1


def test_connector_item_rejects_control_chars():
    with pytest.raises(ValueError, match="control characters"):
        ConnectorItem(
            external_id="doc\x001",
            source_type="local_file",
            source_uri="file:///doc1.txt",
            content="valid content",
            revision="rev1",
        )


@pytest.mark.asyncio
async def test_fake_connector_scan():
    items = [
        ConnectorItem(
            external_id="1",
            source_type="fake",
            source_uri="fake://1",
            content="content 1",
            revision="r1",
        )
    ]
    connector = FakeConnector("fake-1", items)
    assert await connector.health() is True
    scanned, next_cur = await connector.scan(None)
    assert len(scanned) == 1
    assert next_cur == "1"


@pytest.mark.asyncio
async def test_local_file_connector_scan_allowed_files(tmp_path: Path):
    root = tmp_path / "allowed_root"
    root.mkdir()
    f1 = root / "note.txt"
    f1.write_text("I am learning Rust.", encoding="utf-8")
    f2 = root / "doc.md"
    f2.write_text("I prefer concise answers.", encoding="utf-8")

    connector = LocalFileConnector("local-1", [root])
    assert await connector.health() is True
    items, next_cursor = await connector.scan(None)
    assert len(items) == 2
    ext_ids = {item.external_id for item in items}
    assert "note.txt" in ext_ids
    assert "doc.md" in ext_ids


@pytest.mark.asyncio
async def test_local_file_connector_ignores_unsupported_extensions(tmp_path: Path):
    root = tmp_path / "root"
    root.mkdir()
    (root / "script.py").write_text("print('hello')", encoding="utf-8")
    (root / "valid.txt").write_text("I am working on Project Atlas.", encoding="utf-8")

    connector = LocalFileConnector("local-2", [root])
    items, _ = await connector.scan(None)
    assert len(items) == 1
    assert items[0].external_id == "valid.txt"


# --- JSON / JSONL Matrix Tests ---

@pytest.mark.asyncio
async def test_json_import_connector_valid_file(tmp_path: Path):
    json_path = tmp_path / "data.json"
    data = [
        {
            "id": "item1",
            "content": "I am working on Project Beta.",
            "title": "Title 1",
            "metadata": {"tags": "test"},
            "timestamp": "2026-09-01T12:00:00Z",
        }
    ]
    json_path.write_text(json.dumps(data), encoding="utf-8")

    connector = JsonImportConnector("json-1", json_path)
    assert await connector.health() is True
    items, _ = await connector.scan(None)
    assert len(items) == 1
    assert items[0].external_id == "item1"
    assert items[0].content == "I am working on Project Beta."
    assert items[0].created_at is not None


@pytest.mark.asyncio
async def test_jsonl_import_connector_valid(tmp_path: Path):
    jsonl_path = tmp_path / "data.jsonl"
    lines = [
        json.dumps({"id": "l1", "content": "Line one content"}),
        json.dumps({"id": "l2", "content": "Line two content"}),
    ]
    jsonl_path.write_text("\n".join(lines), encoding="utf-8")

    connector = JsonImportConnector("jsonl-1", jsonl_path)
    items, _ = await connector.scan(None)
    assert len(items) == 2
    assert [i.external_id for i in items] == ["l1", "l2"]


@pytest.mark.asyncio
async def test_json_import_connector_malformed_json(tmp_path: Path):
    json_path = tmp_path / "bad.json"
    json_path.write_text("{not valid json: 123", encoding="utf-8")

    connector = JsonImportConnector("json-bad", json_path)
    with pytest.raises(Exception):
        await connector.scan(None)


@pytest.mark.asyncio
async def test_json_import_connector_rejects_duplicate_external_ids(tmp_path: Path):
    json_path = tmp_path / "dup.json"
    data = [
        {"id": "same_id", "content": "First content"},
        {"id": "same_id", "content": "Second content"},
    ]
    json_path.write_text(json.dumps(data), encoding="utf-8")

    connector = JsonImportConnector("json-dup", json_path)
    with pytest.raises(ValueError, match="duplicate id"):
        await connector.scan(None)


@pytest.mark.asyncio
async def test_json_import_connector_missing_id_or_content(tmp_path: Path):
    json_path = tmp_path / "missing.json"
    data = [{"title": "No id or content"}]
    json_path.write_text(json.dumps(data), encoding="utf-8")

    connector = JsonImportConnector("json-miss", json_path)
    with pytest.raises(ValueError, match="missing or invalid id or content"):
        await connector.scan(None)



@pytest.mark.asyncio
async def test_json_import_connector_oversized_content(tmp_path: Path):
    json_path = tmp_path / "huge.json"
    data = [{"id": "huge_1", "content": "A" * 105_000}]
    json_path.write_text(json.dumps(data), encoding="utf-8")

    connector = JsonImportConnector("json-huge", json_path)
    with pytest.raises(ValueError, match="ITEM_TOO_LARGE"):
        await connector.scan(None)


@pytest.mark.asyncio
async def test_json_import_connector_nested_metadata_rejected(tmp_path: Path):
    json_path = tmp_path / "nested_meta.json"
    data = [
        {
            "id": "nest_1",
            "content": "Valid content",
            "metadata": {"nested_key": {"deep": "not allowed"}},
        }
    ]
    json_path.write_text(json.dumps(data), encoding="utf-8")

    connector = JsonImportConnector("json-nest", json_path)
    with pytest.raises(ValueError, match="nested structures rejected"):
        await connector.scan(None)


@pytest.mark.asyncio
async def test_json_import_connector_secret_metadata_rejected(tmp_path: Path):
    json_path = tmp_path / "sec_meta.json"
    data = [
        {
            "id": "sec_meta_1",
            "content": "Valid content",
            "metadata": {"api_key": "sk-proj-secretsecretsecret"},
        }
    ]
    json_path.write_text(json.dumps(data), encoding="utf-8")

    connector = JsonImportConnector("json-secmeta", json_path)
    with pytest.raises(ValueError, match="secret detected in metadata"):
        await connector.scan(None)


@pytest.mark.asyncio
async def test_json_import_connector_invalid_timestamps(tmp_path: Path):
    json_path = tmp_path / "bad_time.json"
    data = [{"id": "t1", "content": "Valid content", "timestamp": "not-a-timestamp"}]
    json_path.write_text(json.dumps(data), encoding="utf-8")

    connector = JsonImportConnector("json-time", json_path)
    with pytest.raises(ValueError, match="invalid timestamp"):
        await connector.scan(None)


@pytest.mark.asyncio
async def test_json_import_control_characters_rejected(tmp_path: Path):
    json_path = tmp_path / "ctrl.json"
    data = [{"id": "bad\x00id", "content": "Valid content"}]
    json_path.write_text(json.dumps(data), encoding="utf-8")

    connector = JsonImportConnector("json-ctrl", json_path)
    with pytest.raises(ValueError, match="control characters"):
        await connector.scan(None)


# --- Same Revision + Different Hash & Rollback ---

@pytest.mark.asyncio
async def test_same_revision_different_hash_and_rollback(tmp_path: Path):
    db_path = tmp_path / "rev_test.db"
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

    # 1. Initial sync with revision r1
    item_v1 = ConnectorItem(
        external_id="doc1",
        source_type="fake",
        source_uri="fake://doc1",
        content="I am working on Project Atlas.",
        revision="r1",
    )
    connector = FakeConnector("conn-rev", [item_v1])
    manager.register(connector)

    res1 = await manager.sync("conn-rev")
    assert res1.accepted == 1

    # 2. Same revision "r1", but different content!
    item_v2 = ConnectorItem(
        external_id="doc1",
        source_type="fake",
        source_uri="fake://doc1",
        content="I am working on Project Beta now.",
        revision="r1",  # Same revision, different hash!
    )
    connector.items = [item_v2]

    res2 = await manager.sync("conn-rev")
    assert res2.unchanged == 0  # NOT skipped!
    assert res2.accepted >= 1

    # 3. Revision rollback: revision changes back to r0
    item_rollback = ConnectorItem(
        external_id="doc1",
        source_type="fake",
        source_uri="fake://doc1",
        content="I am working on Project Alpha original.",
        revision="r0",
    )
    connector.items = [item_rollback]

    res3 = await manager.sync("conn-rev")
    assert res3.unchanged == 0  # Replay/rollback re-evaluated
    assert res3.accepted >= 1

    await db.close()


# --- Retry and Backoff Policy ---

@pytest.mark.asyncio
async def test_retry_transient_with_injected_backoff(tmp_path: Path):
    db_path = tmp_path / "retry_test.db"
    db = Database(db_path)
    await db.initialize()
    conn = db.connection()

    memory_repo = SqliteMemoryRepository(conn)
    event_repo = SqliteEventRepository(conn)
    connector_repo = SqliteConnectorRepository(conn)

    backoff_calls = []

    def mock_backoff(attempt: int) -> None:
        backoff_calls.append(attempt)

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
        max_retries=3,
        backoff_fn=mock_backoff,
    )

    attempts = 0

    class TransientFailingConnector(FakeConnector):
        async def scan(self, cursor: str | None):
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise ConnectionError("network blip")
            return self.items, "c_ok"

    items = [
        ConnectorItem(
            external_id="item_transient",
            source_type="fake",
            source_uri="fake://t",
            content="I am working on Project Atlas.",
            revision="r1",
        )
    ]
    connector = TransientFailingConnector("conn-transient", items)
    manager.register(connector)

    result = await manager.sync("conn-transient")
    assert result.status == "success"
    assert attempts == 3
    # Proves backoff was injected and called for attempts 1 and 2
    assert backoff_calls == [1, 2]

    await db.close()


@pytest.mark.asyncio
async def test_no_retry_for_permanent_errors(tmp_path: Path):
    db_path = tmp_path / "no_retry_test.db"
    db = Database(db_path)
    await db.initialize()
    conn = db.connection()

    memory_repo = SqliteMemoryRepository(conn)
    event_repo = SqliteEventRepository(conn)
    connector_repo = SqliteConnectorRepository(conn)

    backoff_calls = []

    def mock_backoff(attempt: int) -> None:
        backoff_calls.append(attempt)

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
        max_retries=3,
        backoff_fn=mock_backoff,
    )

    # Privacy rejection is permanent
    item_sec = ConnectorItem(
        external_id="item_sec_noretry",
        source_type="fake",
        source_uri="fake://sec",
        content="I prefer secret API key: sk-proj-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
        revision="r1",
    )
    connector = FakeConnector("conn-perm", [item_sec])
    manager.register(connector)

    res = await manager.sync("conn-perm")
    assert res.rejected == 1
    # Proves zero retry backoffs were called for permanent privacy error
    assert backoff_calls == []

    await db.close()


# --- Connector Disable and Re-enable ---

@pytest.mark.asyncio
async def test_connector_disable_and_reenable(tmp_path: Path):
    db_path = tmp_path / "disable_test.db"
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
            external_id="dis_1",
            source_type="fake",
            source_uri="fake://dis1",
            content="I am working on Project Atlas.",
            revision="r1",
        )
    ]
    connector = FakeConnector("conn-disable", items)
    manager.register(connector)

    # 1. Initial sync -> success, memory exists
    res1 = await manager.sync("conn-disable")
    assert res1.status == "success"
    assert res1.accepted >= 1
    mids = await connector_repo.get_item_memory_ids("conn-disable", "dis_1")
    assert len(mids) >= 1

    # 2. Disable connector
    await manager.set_enabled("conn-disable", False)

    # Sync while disabled -> returns status="disabled"
    res_dis = await manager.sync("conn-disable")
    assert res_dis.status == "disabled"

    # Memories must NOT be purged!
    mem = await memory_repo.get(mids[0])
    assert mem is not None
    assert mem.status == MemoryStatus.ACTIVE

    # 3. Re-enable connector
    await manager.set_enabled("conn-disable", True)

    # Sync resumes safely -> items unchanged
    res_re = await manager.sync("conn-disable")
    assert res_re.status == "success"
    assert res_re.unchanged == 1
    assert res_re.accepted == 0

    await db.close()
