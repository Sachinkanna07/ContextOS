"""SQLite-backed Memory repository.

Implements the MemoryRepository protocol for CRUD operations on memories.
Handles serialization between Pydantic Memory models and SQLite rows.
"""

from __future__ import annotations

import json
import logging
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from uuid import UUID

import aiosqlite

from contextos.core.enums import VALID_TRANSITIONS, MemoryStatus, MemoryType, PrivacyLevel
from contextos.core.exceptions import (
    ConcurrencyError, DuplicateMemoryError, InvalidTransitionError, MemoryNotFoundError,
)
from contextos.core.models import Memory, MemoryFilters, MemoryUpdate

logger = logging.getLogger(__name__)


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _row_to_memory(row: aiosqlite.Row) -> Memory:
    """Convert a SQLite row to a Memory model."""
    data = dict(row)

    # Parse JSON fields
    data["tags"] = json.loads(data.get("tags", "[]"))

    # Convert UUID strings
    for field in ("id", "provenance_event_id", "superseded_by", "supersedes"):
        if data.get(field) is not None:
            data[field] = UUID(data[field])

    # Convert enum strings
    data["status"] = MemoryStatus(data["status"])
    data["type"] = MemoryType(data["type"])
    data["privacy_level"] = PrivacyLevel(data["privacy_level"])

    # Parse datetime strings
    for field in ("created_at", "updated_at", "last_accessed_at", "expires_at"):
        val = data.get(field)
        if val is not None:
            parsed = datetime.fromisoformat(val)
            data[field] = parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed

    return Memory.model_validate(data)


def _memory_to_row(memory: Memory) -> dict:
    """Convert a Memory model to a dict suitable for SQLite insertion."""
    return {
        "id": str(memory.id),
        "content": memory.content,
        "content_hash": memory.content_hash,
        "type": memory.type.value,
        "source_type": memory.source_type,
        "source_uri": memory.source_uri,
        "provenance_event_id": str(memory.provenance_event_id) if memory.provenance_event_id else None,
        "status": memory.status.value,
        "confidence": memory.confidence,
        "importance": memory.importance,
        "privacy_level": memory.privacy_level.value,
        "token_count": memory.token_count,
        "embedding_id": memory.embedding_id,
        "superseded_by": str(memory.superseded_by) if memory.superseded_by else None,
        "supersedes": str(memory.supersedes) if memory.supersedes else None,
        "access_count": memory.access_count,
        "created_at": memory.created_at.astimezone(timezone.utc).isoformat(),
        "updated_at": memory.updated_at.astimezone(timezone.utc).isoformat(),
        "last_accessed_at": (
            memory.last_accessed_at.astimezone(timezone.utc).isoformat()
            if memory.last_accessed_at
            else None
        ),
        "expires_at": (
            memory.expires_at.astimezone(timezone.utc).isoformat()
            if memory.expires_at
            else None
        ),
        "version": memory.version,
        "tags": json.dumps(memory.tags),
    }


class SqliteMemoryRepository:
    """SQLite implementation of the MemoryRepository protocol."""

    def __init__(self, db: aiosqlite.Connection) -> None:
        self._db = db
        self._write_lock = asyncio.Lock()

    @asynccontextmanager
    async def _transaction(self):
        async with self._write_lock:
            await self._db.execute("BEGIN IMMEDIATE")
            try:
                yield
                await self._db.commit()
            except BaseException:
                await self._db.rollback()
                raise

    async def get(self, memory_id: UUID) -> Memory | None:
        cursor = await self._db.execute(
            "SELECT * FROM memories WHERE id = ?", (str(memory_id),)
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return _row_to_memory(row)

    async def list(self, filters: MemoryFilters) -> list[Memory]:
        query = "SELECT * FROM memories WHERE 1=1"
        params: list = []

        if filters.status is not None:
            query += " AND status = ?"
            params.append(filters.status.value)

        if filters.type is not None:
            query += " AND type = ?"
            params.append(filters.type.value)

        if filters.privacy_level is not None:
            query += " AND privacy_level = ?"
            params.append(filters.privacy_level.value)

        if filters.source_type is not None:
            query += " AND source_type = ?"
            params.append(filters.source_type)

        if filters.min_confidence is not None:
            query += " AND confidence >= ?"
            params.append(filters.min_confidence)

        if filters.min_importance is not None:
            query += " AND importance >= ?"
            params.append(filters.min_importance)

        if filters.created_after is not None:
            query += " AND julianday(created_at) >= julianday(?)"
            params.append(filters.created_after.astimezone(timezone.utc).isoformat())

        if filters.created_before is not None:
            query += " AND julianday(created_at) <= julianday(?)"
            params.append(filters.created_before.astimezone(timezone.utc).isoformat())

        # Tag filtering: use JSON contains (SQLite json_each)
        if filters.tags:
            for tag in filters.tags:
                query += " AND EXISTS (SELECT 1 FROM json_each(tags) WHERE value = ?)"
                params.append(tag)

        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([filters.limit, filters.offset])

        cursor = await self._db.execute(query, params)
        rows = await cursor.fetchall()
        return [_row_to_memory(row) for row in rows]

    async def create(self, memory: Memory) -> Memory:
        row = _memory_to_row(memory)
        columns = ", ".join(row.keys())
        placeholders = ", ".join("?" for _ in row)

        async with self._transaction():
            try:
                await self._db.execute(
                    f"INSERT INTO memories ({columns}) VALUES ({placeholders})",
                    list(row.values()),
                )
            except aiosqlite.IntegrityError as exc:
                if await self.get(memory.id) is not None:
                    raise DuplicateMemoryError(str(memory.id)) from exc
                raise
        logger.debug("Created memory %s", memory.id)
        return memory

    async def update(
        self, memory_id: UUID, update: MemoryUpdate, expected_version: int
    ) -> Memory:
        # Build update fields
        fields: dict = {}
        if update.content is not None:
            fields["content"] = update.content
            # Recompute content hash
            from contextos.core.models import _content_hash
            fields["content_hash"] = _content_hash(update.content)
        if update.type is not None:
            fields["type"] = update.type.value
        if update.confidence is not None:
            fields["confidence"] = update.confidence
        if update.importance is not None:
            fields["importance"] = update.importance
        if update.privacy_level is not None:
            fields["privacy_level"] = update.privacy_level.value
        if update.expires_at is not None:
            fields["expires_at"] = update.expires_at.astimezone(timezone.utc).isoformat()
        if update.tags is not None:
            fields["tags"] = json.dumps(update.tags)

        async with self._transaction():
            current = await self._require_version(memory_id, expected_version)
            if not fields:
                return current
            fields["updated_at"] = _utcnow_iso()
            fields["version"] = expected_version + 1
            set_clause = ", ".join(f"{k} = ?" for k in fields)
            values = list(fields.values()) + [str(memory_id), expected_version]
            await self._db.execute(
                f"UPDATE memories SET {set_clause} WHERE id = ? AND version = ?", values
            )
            updated = await self.get(memory_id)
            assert updated is not None
            return updated

    async def _require_version(self, memory_id: UUID, expected_version: int) -> Memory:
        current = await self.get(memory_id)
        if current is None:
            raise MemoryNotFoundError(str(memory_id))
        if current.version != expected_version:
            raise ConcurrencyError(str(memory_id), expected_version, current.version)
        return current

    async def update_status(
        self, memory_id: UUID, new_status: MemoryStatus, expected_version: int
    ) -> Memory:
        async with self._transaction():
            current = await self._require_version(memory_id, expected_version)
            # These states require a separate operation with additional data or
            # physical deletion; a status-only write would leave invalid history.
            if new_status in (MemoryStatus.SUPERSEDED, MemoryStatus.MERGED,
                              MemoryStatus.PURGED) or new_status not in VALID_TRANSITIONS[current.status]:
                raise InvalidTransitionError(str(memory_id), current.status.value, new_status.value)
            await self._db.execute(
                "UPDATE memories SET status = ?, updated_at = ?, version = version + 1 "
                "WHERE id = ? AND version = ?",
                (new_status.value, _utcnow_iso(), str(memory_id), expected_version),
            )
            updated = await self.get(memory_id)
            assert updated is not None
            return updated

    async def supersede(self, old_id: UUID, successor: Memory, expected_version: int) -> Memory:
        """Store successor and link both records atomically; retain the old record."""
        if old_id == successor.id:
            raise ValueError("A memory cannot supersede itself")
        async with self._transaction():
            old = await self._require_version(old_id, expected_version)
            if MemoryStatus.SUPERSEDED not in VALID_TRANSITIONS[old.status]:
                raise InvalidTransitionError(str(old_id), old.status.value, MemoryStatus.SUPERSEDED.value)
            if successor.status != MemoryStatus.ACTIVE or successor.supersedes not in (None, old_id):
                raise ValueError("Successor must be active and refer to the old memory")
            successor = successor.model_copy(update={"supersedes": old_id})
            row = _memory_to_row(successor)
            columns = ", ".join(row)
            placeholders = ", ".join("?" for _ in row)
            try:
                await self._db.execute(
                    f"INSERT INTO memories ({columns}) VALUES ({placeholders})", list(row.values())
                )
            except aiosqlite.IntegrityError as exc:
                if await self.get(successor.id) is not None:
                    raise DuplicateMemoryError(str(successor.id)) from exc
                raise
            await self._db.execute(
                "UPDATE memories SET status = ?, superseded_by = ?, updated_at = ?, "
                "version = version + 1 WHERE id = ? AND version = ?",
                (MemoryStatus.SUPERSEDED.value, str(successor.id), _utcnow_iso(),
                 str(old_id), expected_version),
            )
            return successor

    async def update_access(self, memory_id: UUID) -> None:
        now = _utcnow_iso()
        async with self._transaction():
            await self._db.execute(
                "UPDATE memories SET access_count = access_count + 1, "
                "last_accessed_at = ? WHERE id = ?",
                (now, str(memory_id)),
            )

    async def delete(self, memory_id: UUID) -> None:
        async with self._transaction():
            await self._db.execute("DELETE FROM memories WHERE id = ?", (str(memory_id),))
        logger.debug("Hard-deleted memory %s from database", memory_id)

    async def count(self, filters: MemoryFilters | None = None) -> int:
        if filters is None:
            cursor = await self._db.execute("SELECT COUNT(*) FROM memories")
        else:
            query = "SELECT COUNT(*) FROM memories WHERE 1=1"
            params: list = []

            if filters.status is not None:
                query += " AND status = ?"
                params.append(filters.status.value)
            if filters.type is not None:
                query += " AND type = ?"
                params.append(filters.type.value)

            cursor = await self._db.execute(query, params)

        row = await cursor.fetchone()
        return row[0] if row else 0

    async def get_by_hash(self, content_hash: str) -> Memory | None:
        cursor = await self._db.execute(
            "SELECT * FROM memories WHERE content_hash = ? AND status != ? LIMIT 1",
            (content_hash, MemoryStatus.DELETED.value),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return _row_to_memory(row)
