"""SQLite-backed Event repository.

Implements the EventRepository protocol. Events are append-only — the only
mutation is purge, which physically removes the row.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from uuid import UUID

import aiosqlite

from contextos.core.enums import EventType
from contextos.core.models import EventFilters, RawEvent

logger = logging.getLogger(__name__)


def _row_to_event(row: aiosqlite.Row) -> RawEvent:
    """Convert a SQLite row to a RawEvent model."""
    data = dict(row)

    data["id"] = UUID(data["id"])
    data["event_type"] = EventType(data["event_type"])

    # Parse JSON fields
    data["metadata"] = json.loads(data.get("metadata", "{}"))
    data["memory_ids"] = [UUID(mid) for mid in json.loads(data.get("memory_ids", "[]"))]

    pscan = data.get("privacy_scan_result")
    data["privacy_scan_result"] = json.loads(pscan) if pscan else None

    # Parse timestamp
    ts = data.get("timestamp")
    if ts is not None:
        data["timestamp"] = datetime.fromisoformat(ts).replace(tzinfo=timezone.utc)

    return RawEvent.model_validate(data)


def _event_to_row(event: RawEvent) -> dict:
    """Convert a RawEvent model to a dict suitable for SQLite insertion."""
    return {
        "id": str(event.id),
        "event_type": event.event_type.value,
        "timestamp": event.timestamp.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_type": event.source_type,
        "source_uri": event.source_uri,
        "content": event.content,
        "content_hash": event.content_hash,
        "metadata": json.dumps(event.metadata),
        "privacy_scan_result": (
            json.dumps(event.privacy_scan_result)
            if event.privacy_scan_result is not None
            else None
        ),
        "memory_ids": json.dumps([str(mid) for mid in event.memory_ids]),
    }


class SqliteEventRepository:
    """SQLite implementation of the EventRepository protocol."""

    def __init__(self, db: aiosqlite.Connection) -> None:
        self._db = db

    async def append(self, event: RawEvent) -> None:
        row = _event_to_row(event)
        columns = ", ".join(row.keys())
        placeholders = ", ".join("?" for _ in row)

        await self._db.execute(
            f"INSERT INTO events ({columns}) VALUES ({placeholders})",
            list(row.values()),
        )
        await self._db.commit()
        logger.debug("Appended event %s (type: %s)", event.id, event.event_type.value)

    async def get(self, event_id: UUID) -> RawEvent | None:
        cursor = await self._db.execute(
            "SELECT * FROM events WHERE id = ?", (str(event_id),)
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return _row_to_event(row)

    async def list(self, filters: EventFilters) -> list[RawEvent]:
        query = "SELECT * FROM events WHERE 1=1"
        params: list = []

        if filters.event_type is not None:
            query += " AND event_type = ?"
            params.append(filters.event_type.value)

        if filters.source_type is not None:
            query += " AND source_type = ?"
            params.append(filters.source_type)

        if filters.after is not None:
            query += " AND timestamp >= ?"
            params.append(filters.after.strftime("%Y-%m-%dT%H:%M:%SZ"))

        if filters.before is not None:
            query += " AND timestamp <= ?"
            params.append(filters.before.strftime("%Y-%m-%dT%H:%M:%SZ"))

        if filters.memory_id is not None:
            # Search for memory_id within the JSON array
            query += " AND EXISTS (SELECT 1 FROM json_each(memory_ids) WHERE value = ?)"
            params.append(str(filters.memory_id))

        query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.extend([filters.limit, filters.offset])

        cursor = await self._db.execute(query, params)
        rows = await cursor.fetchall()
        return [_row_to_event(row) for row in rows]

    async def count(self, filters: EventFilters | None = None) -> int:
        if filters is None:
            cursor = await self._db.execute("SELECT COUNT(*) FROM events")
        else:
            query = "SELECT COUNT(*) FROM events WHERE 1=1"
            params: list = []

            if filters.event_type is not None:
                query += " AND event_type = ?"
                params.append(filters.event_type.value)

            cursor = await self._db.execute(query, params)

        row = await cursor.fetchone()
        return row[0] if row else 0
