"""SQLite persistence for typed memory relationships."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from uuid import UUID

import aiosqlite

from contextos.core.enums import RelationType
from contextos.core.models import MemoryRelation


def _relation_from_row(row: aiosqlite.Row) -> MemoryRelation:
    data = dict(row)
    data["id"] = UUID(data["id"])
    data["source_memory_id"] = UUID(data["source_memory_id"])
    data["target_memory_id"] = UUID(data["target_memory_id"])
    data["relation_type"] = RelationType(data["relation_type"])
    data["metadata"] = json.loads(data["metadata"])
    created_at = datetime.fromisoformat(data["created_at"])
    data["created_at"] = (
        created_at.replace(tzinfo=timezone.utc)
        if created_at.tzinfo is None else created_at
    )
    return MemoryRelation.model_validate(data)


class SqliteRelationRepository:
    """Small relation repository sharing the main SQLite transaction domain."""

    def __init__(self, db: aiosqlite.Connection) -> None:
        self._db = db

    async def create(self, relation: MemoryRelation) -> MemoryRelation:
        await self._db.execute(
            "INSERT INTO memory_relations "
            "(id, source_memory_id, target_memory_id, relation_type, confidence, "
            "created_at, metadata) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                str(relation.id), str(relation.source_memory_id),
                str(relation.target_memory_id), relation.relation_type.value,
                relation.confidence,
                relation.created_at.astimezone(timezone.utc).isoformat(),
                json.dumps(relation.metadata),
            ),
        )
        await self._db.commit()
        return relation

    async def get_relations(
        self, memory_id: UUID, direction: str = "both"
    ) -> list[MemoryRelation]:
        if direction == "outgoing":
            clause = "source_memory_id = ?"
            params = (str(memory_id),)
        elif direction == "incoming":
            clause = "target_memory_id = ?"
            params = (str(memory_id),)
        elif direction == "both":
            clause = "source_memory_id = ? OR target_memory_id = ?"
            params = (str(memory_id), str(memory_id))
        else:
            raise ValueError("direction must be outgoing, incoming, or both")
        cursor = await self._db.execute(
            f"SELECT * FROM memory_relations WHERE {clause} "
            "ORDER BY created_at, id",
            params,
        )
        return [_relation_from_row(row) for row in await cursor.fetchall()]

    async def delete_for_memory(self, memory_id: UUID) -> int:
        cursor = await self._db.execute(
            "DELETE FROM memory_relations "
            "WHERE source_memory_id = ? OR target_memory_id = ?",
            (str(memory_id), str(memory_id)),
        )
        await self._db.commit()
        return cursor.rowcount
