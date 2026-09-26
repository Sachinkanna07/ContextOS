"""Phase 1 contract tests against real temporary SQLite files."""

from datetime import datetime, timezone
from uuid import uuid4

import pytest
from pydantic import ValidationError

from contextos.core.enums import MemoryStatus, MemoryType
from contextos.core.exceptions import (
    ConcurrencyError, DuplicateMemoryError, InvalidTransitionError, MigrationError,
)
from contextos.core.models import Memory, MemoryFilters, MemoryUpdate
from contextos.services.memory import CoreMemoryService
from contextos.storage.database import Database
from contextos.storage.memory_repo import SqliteMemoryRepository


def service(db: Database) -> CoreMemoryService:
    return CoreMemoryService(SqliteMemoryRepository(db.connection()))


@pytest.mark.asyncio
async def test_create_get_update_and_provenance(db):
    engine = service(db)
    event_id = uuid4()
    original = Memory(content="I prefer Python", status=MemoryStatus.ACTIVE,
                      source_type="document", source_uri="notes.txt",
                      provenance_event_id=event_id, confidence=0.8,
                      importance=0.6, token_count=4)
    assert await engine.create(original) == original
    assert await engine.get(original.id) == original
    updated = await engine.update(original.id, MemoryUpdate(content="I prefer Rust", tags=["code"]))
    assert updated.content == "I prefer Rust"
    assert updated.content_hash != original.content_hash
    assert updated.version == 2
    assert updated.source_type == "document"
    assert updated.source_uri == "notes.txt"
    assert updated.provenance_event_id == event_id
    assert updated.confidence == 0.8
    assert updated.importance == 0.6
    assert updated.token_count == 4
    assert updated.tags == ["code"]


@pytest.mark.asyncio
async def test_list_filters_and_independent_memories(db):
    engine = service(db)
    first = await engine.create(Memory(content="One", type=MemoryType.FACT,
                                        status=MemoryStatus.ACTIVE, tags=["team"]))
    second = await engine.create(Memory(content="Two", type=MemoryType.PROJECT,
                                         status=MemoryStatus.ACTIVE))
    await engine.update(first.id, MemoryUpdate(content="Changed"))
    assert (await engine.get(second.id)).content == "Two"
    assert [m.id for m in await engine.list(MemoryFilters(type=MemoryType.FACT))] == [first.id]
    assert [m.id for m in await engine.list(MemoryFilters(tags=["team"]))] == [first.id]
    assert len(await engine.list(MemoryFilters(status=MemoryStatus.ACTIVE))) == 2
    assert {m.id for m in await engine.list(MemoryFilters(created_before=datetime.now(timezone.utc)))} == {
        first.id, second.id
    }


@pytest.mark.asyncio
async def test_persistence_and_initialization_from_empty_directory(tmp_path):
    path = tmp_path / "nested" / "memory.db"
    assert not path.parent.exists()
    db = Database(path)
    await db.initialize()
    memory = await service(db).create(Memory(content="Survives restart", source_uri="file.txt"))
    await db.close()
    assert path.is_file()
    reopened = Database(path)
    await reopened.initialize()
    try:
        assert await service(reopened).get(memory.id) == memory
        cursor = await reopened.connection().execute("SELECT MAX(version) FROM schema_version")
        assert (await cursor.fetchone())[0] == 1
    finally:
        await reopened.close()


@pytest.mark.asyncio
async def test_duplicate_id_rolls_back(db):
    engine = service(db)
    original = await engine.create(Memory(content="Original"))
    with pytest.raises(DuplicateMemoryError):
        await engine.create(Memory(id=original.id, content="Collision"))
    assert (await engine.get(original.id)).content == "Original"
    assert len(await engine.list(MemoryFilters())) == 1


def test_invalid_input():
    with pytest.raises(ValidationError):
        Memory(content="   ")
    with pytest.raises(ValidationError):
        MemoryUpdate(content="\t")
    with pytest.raises(ValidationError):
        Memory(content="Valid", confidence=1.1)
    with pytest.raises(ValidationError):
        Memory(content="Valid", token_count=-1)
    with pytest.raises(ValidationError):
        Memory(content="Valid", created_at=datetime(2026, 1, 1))


@pytest.mark.asyncio
async def test_timestamps_are_utc_precise_and_update(db):
    engine = service(db)
    created = await engine.create(Memory(content="Clock"))
    updated = await engine.update(created.id, MemoryUpdate(content="Clock changed"))
    assert created.created_at.tzinfo == timezone.utc
    assert updated.created_at == created.created_at
    assert updated.updated_at >= created.updated_at
    assert updated.updated_at.microsecond > 0
    assert (await engine.get(created.id)).updated_at == updated.updated_at


@pytest.mark.asyncio
async def test_lifecycle_and_soft_delete(db):
    engine = service(db)
    memory = await engine.create(Memory(content="Lifecycle"))
    active = await engine.transition(memory.id, MemoryStatus.ACTIVE)
    assert active.status == MemoryStatus.ACTIVE
    contradicted = await engine.transition(memory.id, MemoryStatus.CONTRADICTED)
    assert contradicted.status == MemoryStatus.CONTRADICTED
    await engine.transition(memory.id, MemoryStatus.ACTIVE)
    expired = await engine.transition(memory.id, MemoryStatus.EXPIRED)
    assert (await engine.transition(memory.id, MemoryStatus.HISTORICAL)).status == MemoryStatus.HISTORICAL
    deleted = await engine.delete(memory.id)
    assert deleted.status == MemoryStatus.DELETED
    assert await engine.get(memory.id) == deleted
    assert [m.id for m in await engine.list(MemoryFilters(status=MemoryStatus.DELETED))] == [memory.id]
    with pytest.raises(InvalidTransitionError):
        await engine.transition(memory.id, MemoryStatus.ACTIVE)


@pytest.mark.asyncio
async def test_supersede_keeps_both_records_and_links(db):
    engine = service(db)
    old = await engine.create(Memory(content="Old", status=MemoryStatus.ACTIVE))
    new = Memory(content="New", status=MemoryStatus.ACTIVE)
    await engine.supersede(old.id, new)
    old_after = await engine.get(old.id)
    new_after = await engine.get(new.id)
    assert old_after.status == MemoryStatus.SUPERSEDED
    assert old_after.superseded_by == new.id
    assert new_after.supersedes == old.id
    assert len(await engine.list(MemoryFilters())) == 2


@pytest.mark.asyncio
async def test_invalid_transition_and_stale_version(db):
    repo = SqliteMemoryRepository(db.connection())
    memory = await repo.create(Memory(content="Candidate"))
    with pytest.raises(InvalidTransitionError):
        await repo.update_status(memory.id, MemoryStatus.EXPIRED, memory.version)
    assert (await repo.get(memory.id)).status == MemoryStatus.CANDIDATE
    await repo.update_status(memory.id, MemoryStatus.ACTIVE, memory.version)
    with pytest.raises(InvalidTransitionError):
        await repo.update_status(memory.id, MemoryStatus.SUPERSEDED, memory.version + 1)
    with pytest.raises(ConcurrencyError):
        await repo.update(memory.id, MemoryUpdate(content="Stale"), memory.version)


@pytest.mark.asyncio
async def test_supersede_transaction_rolls_back_on_second_write_failure(db):
    engine = service(db)
    old = await engine.create(Memory(content="Old", status=MemoryStatus.ACTIVE))
    successor = Memory(content="New", status=MemoryStatus.ACTIVE)
    await db.connection().execute("""
        CREATE TRIGGER reject_supersede BEFORE UPDATE OF status ON memories
        WHEN NEW.id = OLD.id AND NEW.status = 'superseded'
        BEGIN SELECT RAISE(ABORT, 'injected failure'); END
    """)
    await db.connection().commit()
    with pytest.raises(Exception, match="injected failure"):
        await engine.supersede(old.id, successor)
    assert await engine.get(successor.id) is None
    assert (await engine.get(old.id)).status == MemoryStatus.ACTIVE


@pytest.mark.asyncio
async def test_newer_database_version_rejected(tmp_path):
    path = tmp_path / "future.db"
    db = Database(path)
    await db.initialize()
    await db.connection().execute(
        "INSERT INTO schema_version(version, description) VALUES (999, 'future')"
    )
    await db.connection().commit()
    await db.close()
    with pytest.raises(MigrationError):
        await Database(path).initialize()
