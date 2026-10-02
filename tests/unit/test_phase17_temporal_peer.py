"""A same-property peer beyond the old 500-row scan must still be observed."""

import pytest

from contextos.core.enums import MemoryStatus, TemporalOutcome
from contextos.core.models import Memory, MemorySlot
from contextos.services.temporal import TemporalMemoryService
from contextos.storage.database import Database
from contextos.storage.memory_repo import SqliteMemoryRepository


@pytest.mark.asyncio
async def test_peer_lookup_is_complete_after_500_unrelated_temporal_rows(tmp_path):
    db = Database(tmp_path / "peer.db")
    await db.initialize()
    try:
        repo = SqliteMemoryRepository(db.connection())
        for index in range(501):
            await repo.create(
                Memory(
                    content=f"Unrelated project state {index}.",
                    status=MemoryStatus.ACTIVE,
                    slot=MemorySlot(
                        subject="user", property="project_status", scope=f"project_{index}"
                    ),
                )
            )
        peer = await repo.create(
            Memory(
                content="I use Docker for project Atlas.",
                status=MemoryStatus.ACTIVE,
                slot=MemorySlot(subject="user", property="tool_usage", scope="atlas"),
            )
        )
        candidate = Memory(
            content="I use Docker for project Boreal.",
            slot=MemorySlot(subject="user", property="tool_usage", scope="boreal"),
        )
        decision = await TemporalMemoryService(repo).decide(candidate)
        assert decision.outcome == TemporalOutcome.COEXIST
        assert decision.related_memory_id == peer.id
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_batched_candidate_hydration_preserves_exact_ids_and_missing_rows(tmp_path):
    db = Database(tmp_path / "batch.db")
    await db.initialize()
    try:
        repo = SqliteMemoryRepository(db.connection())
        created = []
        for index in range(405):
            created.append(
                await repo.create(
                    Memory(
                        content=f"Batch record {index}.",
                        status=MemoryStatus.ACTIVE,
                    )
                )
            )
        ids = {str(memory.id) for memory in created}
        ids.add("00000000-0000-0000-0000-000000000000")
        found = await repo.get_many(ids)
        assert set(found) == {str(memory.id) for memory in created}
        assert found[str(created[-1].id)].content == created[-1].content
    finally:
        await db.close()
