"""SQLite database management for ContextOS.

Handles connection pooling, WAL mode, migrations, and schema setup.
All database access goes through the Database class, which provides
the connection to repositories.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

import aiosqlite

from contextos.core.exceptions import MigrationError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema SQL — Phase 1 initial schema
# ---------------------------------------------------------------------------

SCHEMA_VERSION = 2

SCHEMA_SQL = """
-- Schema version tracking
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    description TEXT NOT NULL
);

-- Memories: the core data object
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    type TEXT NOT NULL DEFAULT 'context',
    source_type TEXT NOT NULL DEFAULT 'cli_input',
    source_uri TEXT,
    provenance_event_id TEXT,
    status TEXT NOT NULL DEFAULT 'candidate',
    confidence REAL NOT NULL DEFAULT 0.8,
    importance REAL NOT NULL DEFAULT 0.5,
    privacy_level TEXT NOT NULL DEFAULT 'personal',
    token_count INTEGER NOT NULL DEFAULT 0,
    embedding_id TEXT,
    superseded_by TEXT,
    supersedes TEXT,
    access_count INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    last_accessed_at TEXT,
    expires_at TEXT,
    version INTEGER NOT NULL DEFAULT 1,
    tags TEXT NOT NULL DEFAULT '[]'
);

-- Indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_memories_status ON memories(status);
CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(type);
CREATE INDEX IF NOT EXISTS idx_memories_content_hash ON memories(content_hash);
CREATE INDEX IF NOT EXISTS idx_memories_created_at ON memories(created_at);
CREATE INDEX IF NOT EXISTS idx_memories_privacy_level ON memories(privacy_level);
CREATE INDEX IF NOT EXISTS idx_memories_source_type ON memories(source_type);
CREATE INDEX IF NOT EXISTS idx_memories_provenance ON memories(provenance_event_id);

-- Memory relations (many-to-many)
CREATE TABLE IF NOT EXISTS memory_relations (
    id TEXT PRIMARY KEY,
    source_memory_id TEXT NOT NULL,
    target_memory_id TEXT NOT NULL,
    relation_type TEXT NOT NULL,
    confidence REAL NOT NULL DEFAULT 1.0,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    metadata TEXT NOT NULL DEFAULT '{}',
    FOREIGN KEY (source_memory_id) REFERENCES memories(id) ON DELETE CASCADE,
    FOREIGN KEY (target_memory_id) REFERENCES memories(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_relations_source ON memory_relations(source_memory_id);
CREATE INDEX IF NOT EXISTS idx_relations_target ON memory_relations(target_memory_id);
CREATE INDEX IF NOT EXISTS idx_relations_type ON memory_relations(relation_type);

-- Raw events: append-only audit log
CREATE TABLE IF NOT EXISTS events (
    id TEXT PRIMARY KEY,
    event_type TEXT NOT NULL,
    timestamp TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    source_type TEXT NOT NULL DEFAULT 'system',
    source_uri TEXT,
    content TEXT,
    content_hash TEXT,
    metadata TEXT NOT NULL DEFAULT '{}',
    privacy_scan_result TEXT,
    memory_ids TEXT NOT NULL DEFAULT '[]'
);

CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);
CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp);
CREATE INDEX IF NOT EXISTS idx_events_source_type ON events(source_type);
CREATE INDEX IF NOT EXISTS idx_events_content_hash ON events(content_hash);

-- Pipeline traces
CREATE TABLE IF NOT EXISTS traces (
    id TEXT PRIMARY KEY,
    trace_type TEXT NOT NULL,
    timestamp TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    query TEXT,
    stages TEXT NOT NULL DEFAULT '[]',
    total_latency_ms REAL NOT NULL DEFAULT 0.0,
    total_input_tokens INTEGER,
    total_output_tokens INTEGER,
    metadata TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_traces_timestamp ON traces(timestamp);
CREATE INDEX IF NOT EXISTS idx_traces_type ON traces(trace_type);
"""

MIGRATION_2_SQL = """
ALTER TABLE memories ADD COLUMN observed_at TEXT;
ALTER TABLE memories ADD COLUMN valid_from TEXT;
ALTER TABLE memories ADD COLUMN valid_to TEXT;
ALTER TABLE memories ADD COLUMN temporal_precision TEXT NOT NULL DEFAULT 'unknown';
ALTER TABLE memories ADD COLUMN temporal_status TEXT NOT NULL DEFAULT 'unspecified';
ALTER TABLE memories ADD COLUMN temporal_expression TEXT;
ALTER TABLE memories ADD COLUMN slot_json TEXT;
ALTER TABLE memories ADD COLUMN slot_key TEXT;
ALTER TABLE memories ADD COLUMN uncertain INTEGER NOT NULL DEFAULT 0;
ALTER TABLE memories ADD COLUMN negated INTEGER NOT NULL DEFAULT 0;
ALTER TABLE memories ADD COLUMN resolution_reason TEXT;
ALTER TABLE memories ADD COLUMN resolution_confidence REAL;
UPDATE memories SET observed_at = created_at WHERE observed_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_memories_slot_key ON memories(slot_key);
CREATE INDEX IF NOT EXISTS idx_memories_temporal_status ON memories(temporal_status);
CREATE INDEX IF NOT EXISTS idx_memories_validity ON memories(valid_from, valid_to);
CREATE UNIQUE INDEX IF NOT EXISTS idx_relations_unique
ON memory_relations(source_memory_id, target_memory_id, relation_type);
"""


class Database:
    """Manages the SQLite database connection and schema.

    Usage:
        db = Database(data_dir / "contextos.db")
        await db.initialize()
        # ... use db.connection() to get aiosqlite connection
        await db.close()
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._connection: aiosqlite.Connection | None = None

    @property
    def path(self) -> Path:
        return self._db_path

    async def initialize(self) -> None:
        """Open the database, set pragmas, and run schema migrations."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        if self._connection is not None:
            return
        self._connection = await aiosqlite.connect(
            str(self._db_path),
            detect_types=sqlite3.PARSE_DECLTYPES,
        )

        # Enable WAL mode for concurrent read access and crash resilience
        await self._connection.execute("PRAGMA journal_mode=WAL")

        # Enable foreign keys
        await self._connection.execute("PRAGMA foreign_keys=ON")

        # Reasonable busy timeout for concurrent access
        await self._connection.execute("PRAGMA busy_timeout=5000")

        # Row factory for dict-like access
        self._connection.row_factory = aiosqlite.Row

        try:
            await self._apply_schema()
        except Exception:
            await self.close()
            raise
        logger.info("Database initialized at %s", self._db_path)

    async def _apply_schema(self) -> None:
        """Apply the initial schema and ordered incremental migrations."""
        assert self._connection is not None

        # Check if schema_version table exists
        cursor = await self._connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_version'"
        )
        table_exists = await cursor.fetchone()

        if not table_exists:
            try:
                await self._connection.executescript(
                    "BEGIN IMMEDIATE;\n" + SCHEMA_SQL
                    + "\nINSERT INTO schema_version (version, description) "
                    "VALUES (1, 'Initial schema');\nCOMMIT;"
                )
            except Exception as exc:
                await self._connection.rollback()
                raise MigrationError("Failed to initialize database schema") from exc
            current_version = 1
        else:
            cursor = await self._connection.execute(
                "SELECT MAX(version) FROM schema_version"
            )
            row = await cursor.fetchone()
            current_version = row[0] if row and row[0] is not None else 0

            if current_version > SCHEMA_VERSION:
                raise MigrationError(
                    f"Database schema version {current_version} is newer than supported {SCHEMA_VERSION}"
                )
        if current_version < 2:
            try:
                await self._connection.executescript(
                    "BEGIN IMMEDIATE;\n" + MIGRATION_2_SQL
                    + "\nINSERT INTO schema_version (version, description) "
                    "VALUES (2, 'Temporal memory and resolution metadata');\nCOMMIT;"
                )
            except Exception as exc:
                await self._connection.rollback()
                raise MigrationError("Failed to apply schema migration 2") from exc
            current_version = 2

        if current_version != SCHEMA_VERSION:
            raise MigrationError(
                f"Database schema version {current_version} is unsupported"
            )
        logger.info("Applied schema version %d", SCHEMA_VERSION)

    def connection(self) -> aiosqlite.Connection:
        """Get the database connection. Raises if not initialized."""
        if self._connection is None:
            raise RuntimeError(
                "Database not initialized. Call await db.initialize() first."
            )
        return self._connection

    async def close(self) -> None:
        """Close the database connection."""
        if self._connection is not None:
            await self._connection.close()
            self._connection = None
            logger.info("Database connection closed")

    async def integrity_check(self) -> tuple[bool, str]:
        """Run SQLite integrity check. Returns (ok, message)."""
        conn = self.connection()
        cursor = await conn.execute("PRAGMA integrity_check")
        row = await cursor.fetchone()
        result = row[0] if row else "unknown"
        return result == "ok", result

    async def get_size_bytes(self) -> int:
        """Get database file size in bytes."""
        if self._db_path.exists():
            return self._db_path.stat().st_size
        return 0
