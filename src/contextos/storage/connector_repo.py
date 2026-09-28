"""Persistence for connector cursors and source identity, never source bodies."""
from __future__ import annotations
import json
from datetime import datetime, timezone
from uuid import UUID
import aiosqlite
from contextos.connectors.models import ConnectorItem, ConnectorSyncState

def _now() -> str: return datetime.now(timezone.utc).isoformat()

class SqliteConnectorRepository:
    def __init__(self, connection: aiosqlite.Connection) -> None: self._db = connection
    async def state(self, connector_id: str) -> ConnectorSyncState | None:
        row = await (await self._db.execute("SELECT * FROM connector_state WHERE connector_id=?", (connector_id,))).fetchone()
        if row is None: return None
        return ConnectorSyncState(connector_id=row["connector_id"], connector_type=row["connector_type"], cursor=row["cursor"], enabled=bool(row["enabled"]), status=row["status"], error_code=row["error_code"], last_success_at=row["last_success_at"], last_attempt_at=row["last_attempt_at"])
    async def save_state(self, state: ConnectorSyncState) -> None:
        await self._db.execute("INSERT INTO connector_state(connector_id,connector_type,cursor,last_success_at,last_attempt_at,enabled,status,error_code) VALUES(?,?,?,?,?,?,?,?) ON CONFLICT(connector_id) DO UPDATE SET cursor=excluded.cursor,last_success_at=excluded.last_success_at,last_attempt_at=excluded.last_attempt_at,enabled=excluded.enabled,status=excluded.status,error_code=excluded.error_code", (state.connector_id,state.connector_type,state.cursor,state.last_success_at.isoformat() if state.last_success_at else None,state.last_attempt_at.isoformat() if state.last_attempt_at else None,int(state.enabled),state.status,state.error_code)); await self._db.commit()
    async def item_is_current(self, connector_id: str, item: ConnectorItem, content_hash: str) -> bool:
        row = await (await self._db.execute("SELECT revision,content_hash,deleted FROM connector_items WHERE connector_id=? AND external_id=?", (connector_id,item.external_id))).fetchone()
        return row is not None and not row["deleted"] and row["revision"] == item.revision and row["content_hash"] == content_hash
    async def save_item(self, connector_id: str, item: ConnectorItem, content_hash: str, memory_ids: list[UUID], deleted: bool = False) -> None:
        is_deleted = int(deleted or item.deleted)
        await self._db.execute("INSERT INTO connector_items(connector_id,external_id,revision,content_hash,source_uri,memory_ids,last_seen_at,deleted) VALUES(?,?,?,?,?,?,?,?) ON CONFLICT(connector_id,external_id) DO UPDATE SET revision=excluded.revision,content_hash=excluded.content_hash,source_uri=excluded.source_uri,memory_ids=excluded.memory_ids,last_seen_at=excluded.last_seen_at,deleted=excluded.deleted", (connector_id,item.external_id,item.revision,content_hash,item.source_uri,json.dumps([str(value) for value in memory_ids]),_now(),is_deleted)); await self._db.commit()

    async def get_item_memory_ids(self, connector_id: str, external_id: str) -> list[UUID]:
        row = await (await self._db.execute("SELECT memory_ids FROM connector_items WHERE connector_id=? AND external_id=?", (connector_id, external_id))).fetchone()
        if row is None or not row["memory_ids"]:
            return []
        try:
            raw = json.loads(row["memory_ids"])
            return [UUID(val) for val in raw]
        except Exception:
            return []

    async def count_active_references(self, memory_id: UUID) -> int:
        target = str(memory_id)
        cursor = await self._db.execute("SELECT memory_ids FROM connector_items WHERE deleted = 0")
        rows = await cursor.fetchall()
        count = 0
        for row in rows:
            if not row["memory_ids"]:
                continue
            try:
                raw = json.loads(row["memory_ids"])
                if target in raw:
                    count += 1
            except Exception:
                pass
        return count

    async def list_states(self) -> list[ConnectorSyncState]:
        cursor = await self._db.execute("SELECT * FROM connector_state ORDER BY connector_id")
        rows = await cursor.fetchall()
        return [
            ConnectorSyncState(
                connector_id=row["connector_id"],
                connector_type=row["connector_type"],
                cursor=row["cursor"],
                enabled=bool(row["enabled"]),
                status=row["status"],
                error_code=row["error_code"],
                last_success_at=row["last_success_at"],
                last_attempt_at=row["last_attempt_at"],
            )
            for row in rows
        ]
