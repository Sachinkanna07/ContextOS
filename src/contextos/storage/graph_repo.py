"""SQLite persistence for the rebuildable ContextOS memory graph."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from datetime import datetime, timezone
from uuid import UUID

import aiosqlite

from contextos.core.enums import GraphNodeType, GraphRelationType
from contextos.core.models import GraphEdge, GraphEdgeSupport, GraphNode


def _dt(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed


class SqliteGraphRepository:
    """Persistent graph projection sharing the main SQLite database."""

    def __init__(self, db: aiosqlite.Connection) -> None:
        self._db = db

    async def replace_all(
        self,
        nodes: list[GraphNode],
        edges: list[GraphEdge],
    ) -> None:
        """Atomically replace the derived graph; retain the old graph on failure."""
        node_ids = {node.id for node in nodes}
        if len(node_ids) != len(nodes):
            raise ValueError("Duplicate graph node ID")
        for edge in edges:
            if edge.source_node_id not in node_ids or edge.target_node_id not in node_ids:
                raise ValueError("Graph edge references an unknown node")
            if not edge.supports:
                raise ValueError("Graph edges require at least one support")

        for attempt in range(5):
            try:
                await self._db.execute("BEGIN IMMEDIATE")
                break
            except Exception as exc:
                msg = str(exc).lower()
                if attempt < 4 and ("locked" in msg or "busy" in msg or "cannot start a transaction" in msg):
                    await asyncio.sleep(0.01 * (2 ** attempt))
                    continue
                raise
        try:
            await self._db.execute("DELETE FROM graph_edge_supports")
            await self._db.execute("DELETE FROM graph_edges")
            await self._db.execute("DELETE FROM graph_nodes")
            await self._db.executemany(
                "INSERT INTO graph_nodes "
                "(id, node_type, canonical_key, label, metadata, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        str(node.id), node.node_type.value, node.canonical_key, node.label,
                        json.dumps(node.metadata, sort_keys=True),
                        node.created_at.astimezone(timezone.utc).isoformat(),
                        node.updated_at.astimezone(timezone.utc).isoformat(),
                    )
                    for node in nodes
                ],
            )
            await self._db.executemany(
                "INSERT INTO graph_edges "
                "(id, source_node_id, target_node_id, relation_type, confidence, directed, "
                "scope_key, metadata, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        str(edge.id), str(edge.source_node_id), str(edge.target_node_id),
                        edge.relation_type.value, edge.confidence, int(edge.directed),
                        edge.scope_key or "", json.dumps(edge.metadata, sort_keys=True),
                        edge.created_at.astimezone(timezone.utc).isoformat(),
                        edge.updated_at.astimezone(timezone.utc).isoformat(),
                    )
                    for edge in edges
                ],
            )
            supports = [support for edge in edges for support in edge.supports]
            await self._db.executemany(
                "INSERT INTO graph_edge_supports "
                "(edge_id, memory_id, confidence, provenance_event_id, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                [
                    (
                        str(item.edge_id), str(item.memory_id), item.confidence,
                        str(item.provenance_event_id) if item.provenance_event_id else None,
                        item.created_at.astimezone(timezone.utc).isoformat(),
                    )
                    for item in supports
                ],
            )
            await self._db.execute(
                "UPDATE graph_projection_state SET dirty = 0 WHERE singleton = 1"
            )
            await self._db.commit()
        except Exception:
            await self._db.rollback()
            raise

    async def nodes(self) -> list[GraphNode]:
        cursor = await self._db.execute("SELECT * FROM graph_nodes ORDER BY node_type, canonical_key")
        return [self._node(row) for row in await cursor.fetchall()]

    async def find_nodes(self, canonical_keys: set[str]) -> list[GraphNode]:
        if not canonical_keys:
            return []
        placeholders = ", ".join("?" for _ in canonical_keys)
        cursor = await self._db.execute(
            f"SELECT * FROM graph_nodes WHERE canonical_key IN ({placeholders}) "
            "ORDER BY node_type, canonical_key",
            tuple(sorted(canonical_keys)),
        )
        return [self._node(row) for row in await cursor.fetchall()]

    async def get_node(self, node_id: UUID) -> GraphNode | None:
        cursor = await self._db.execute("SELECT * FROM graph_nodes WHERE id = ?", (str(node_id),))
        row = await cursor.fetchone()
        return self._node(row) if row else None

    async def edges_for_nodes(
        self, node_ids: set[UUID], *, limit: int | None = None
    ) -> list[GraphEdge]:
        if not node_ids or (limit is not None and limit <= 0):
            return []
        raw = tuple(str(value) for value in sorted(node_ids, key=str))
        placeholders = ", ".join("?" for _ in raw)
        sql = (
            f"SELECT * FROM graph_edges WHERE source_node_id IN ({placeholders}) "
            f"OR target_node_id IN ({placeholders}) ORDER BY id"
        )
        params: tuple[object, ...] = raw + raw
        if limit is not None:
            sql += " LIMIT ?"
            params += (limit,)
        cursor = await self._db.execute(sql, params)
        rows = await cursor.fetchall()
        if not rows:
            return []
        edge_ids = tuple(row["id"] for row in rows)
        support_placeholders = ", ".join("?" for _ in edge_ids)
        support_cursor = await self._db.execute(
            f"SELECT * FROM graph_edge_supports WHERE edge_id IN ({support_placeholders}) "
            "ORDER BY edge_id, memory_id",
            edge_ids,
        )
        grouped: dict[str, list[GraphEdgeSupport]] = {}
        for item in await support_cursor.fetchall():
            grouped.setdefault(item["edge_id"], []).append(self._support(item))
        return [self._edge_from_data(row, grouped.get(row["id"], [])) for row in rows]

    async def all_edges(self) -> list[GraphEdge]:
        cursor = await self._db.execute("SELECT * FROM graph_edges ORDER BY id")
        rows = await cursor.fetchall()
        support_cursor = await self._db.execute(
            "SELECT * FROM graph_edge_supports ORDER BY edge_id, memory_id"
        )
        grouped: dict[str, list[GraphEdgeSupport]] = {}
        for item in await support_cursor.fetchall():
            grouped.setdefault(item["edge_id"], []).append(self._support(item))
        return [self._edge_from_data(row, grouped.get(row["id"], [])) for row in rows]

    async def counts(self) -> tuple[int, int, int]:
        values = []
        for table in ("graph_nodes", "graph_edges", "graph_edge_supports"):
            cursor = await self._db.execute(f"SELECT COUNT(*) FROM {table}")
            values.append((await cursor.fetchone())[0])
        return values[0], values[1], values[2]

    async def source_is_dirty(self) -> bool:
        cursor = await self._db.execute(
            "SELECT dirty FROM graph_projection_state WHERE singleton = 1"
        )
        row = await cursor.fetchone()
        return bool(row[0]) if row else True

    @staticmethod
    def _node(row: aiosqlite.Row) -> GraphNode:
        data = dict(row)
        return GraphNode(
            id=UUID(data["id"]),
            node_type=GraphNodeType(data["node_type"]),
            canonical_key=data["canonical_key"],
            label=data["label"],
            metadata=json.loads(data["metadata"]),
            created_at=_dt(data["created_at"]),
            updated_at=_dt(data["updated_at"]),
        )

    async def _edge(self, row: aiosqlite.Row) -> GraphEdge:
        data = dict(row)
        cursor = await self._db.execute(
            "SELECT * FROM graph_edge_supports WHERE edge_id = ? ORDER BY memory_id",
            (data["id"],),
        )
        supports = [self._support(item) for item in await cursor.fetchall()]
        return self._edge_from_data(row, supports)

    @staticmethod
    def _support(item: aiosqlite.Row) -> GraphEdgeSupport:
        return GraphEdgeSupport(
            edge_id=UUID(item["edge_id"]), memory_id=UUID(item["memory_id"]),
            confidence=item["confidence"],
            provenance_event_id=(
                UUID(item["provenance_event_id"]) if item["provenance_event_id"] else None
            ),
            created_at=_dt(item["created_at"]),
        )

    @staticmethod
    def _edge_from_data(row: aiosqlite.Row, supports: list[GraphEdgeSupport]) -> GraphEdge:
        data = dict(row)
        return GraphEdge(
            id=UUID(data["id"]), source_node_id=UUID(data["source_node_id"]),
            target_node_id=UUID(data["target_node_id"]),
            relation_type=GraphRelationType(data["relation_type"]),
            confidence=data["confidence"], directed=bool(data["directed"]),
            scope_key=data["scope_key"] or None,
            metadata=json.loads(data["metadata"]), supports=supports,
            created_at=_dt(data["created_at"]), updated_at=_dt(data["updated_at"]),
        )
