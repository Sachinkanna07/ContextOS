"""Connector orchestration that preserves ContextOS ingestion invariants."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from datetime import datetime, timezone
from typing import Any

from contextos.connectors.models import (
    ConnectorSyncResult,
    ConnectorSyncState,
    RetentionPolicy,
)
from contextos.connectors.protocols import Connector
from contextos.core.enums import MemoryStatus, SourceRole
from contextos.core.exceptions import IngestionError, SecretDetectedError
from contextos.core.models import IngestRequest

logger = logging.getLogger(__name__)


import sqlite3

def is_transient_error(exc: BaseException) -> bool:
    """Classify whether an error during scan or ingestion is transient and eligible for retry."""
    if isinstance(exc, (ConnectionError, TimeoutError, OSError)):
        return True
    if isinstance(exc, sqlite3.OperationalError):
        msg = str(exc).lower()
        if "locked" in msg or "busy" in msg or "cannot start a transaction" in msg:
            return True
    if isinstance(exc, RuntimeError) and "transient" in str(exc).lower():
        return True
    if getattr(exc, "is_transient", False):
        return True
    return False


class ConnectorManager:
    """Orchestrates credential-free connector syncs while preserving ingestion invariants."""

    def __init__(
        self,
        *,
        state_repo: Any,
        ingestion: Any,
        temporal: Any,
        retention_policy: RetentionPolicy = RetentionPolicy.KEEP_DERIVED_MEMORY,
        memory_repo: Any = None,
        max_retries: int = 3,
        backoff_fn: Any = None,
    ) -> None:
        self._state_repo = state_repo
        self._ingestion = ingestion
        self._temporal = temporal
        self._retention_policy = retention_policy
        self._memory_repo = memory_repo
        self._max_retries = max_retries
        self._backoff_fn = backoff_fn
        self._connectors: dict[str, Connector] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    def register(self, connector: Connector) -> None:
        if connector.connector_id in self._connectors:
            raise ValueError(f"duplicate connector id: {connector.connector_id}")
        self._connectors[connector.connector_id] = connector
        self._locks[connector.connector_id] = asyncio.Lock()

    def unregister(self, connector_id: str) -> None:
        """Safely release connector and lock to prevent unbounded growth."""
        if connector_id in self._connectors:
            del self._connectors[connector_id]
        if connector_id in self._locks:
            del self._locks[connector_id]

    def list_connectors(self) -> list[str]:
        return sorted(self._connectors)


    async def get_status(self, connector_id: str) -> ConnectorSyncState | None:
        if connector_id not in self._connectors:
            return None
        return await self._state_repo.state(connector_id)

    async def set_enabled(self, connector_id: str, enabled: bool) -> None:
        if connector_id not in self._connectors:
            raise KeyError(f"connector not found: {connector_id}")
        connector = self._connectors[connector_id]
        state = await self._state_repo.state(connector_id) or ConnectorSyncState(
            connector_id=connector_id, connector_type=connector.source_type
        )
        state.enabled = enabled
        await self._state_repo.save_state(state)

    async def sync(self, connector_id: str) -> ConnectorSyncResult:
        if connector_id not in self._connectors:
            raise KeyError(f"connector not found: {connector_id}")

        connector = self._connectors[connector_id]
        started = time.perf_counter()

        async with self._locks[connector_id]:
            state = await self._state_repo.state(connector_id) or ConnectorSyncState(
                connector_id=connector_id, connector_type=connector.source_type
            )
            if not state.enabled:
                return ConnectorSyncResult(
                    connector_id=connector_id,
                    status="disabled",
                    duration_ms=(time.perf_counter() - started) * 1000,
                )

            state.last_attempt_at = datetime.now(timezone.utc)
            state.status = "syncing"
            await self._state_repo.save_state(state)

            # --- Step 1: Scan with retry policy for transient errors ---
            items = None
            next_cursor = None
            scan_attempts = 0

            while scan_attempts < self._max_retries:
                scan_attempts += 1
                try:
                    items, next_cursor = await connector.scan(state.cursor)
                    break
                except asyncio.CancelledError:
                    state.status = "cancelled"
                    state.error_code = "CANCELLED"
                    await self._state_repo.save_state(state)
                    raise
                except (ValueError, TypeError):
                    state.status = "failed"
                    state.error_code = "VALIDATION_ERROR"
                    await self._state_repo.save_state(state)
                    return ConnectorSyncResult(
                        connector_id=connector_id,
                        status="failed",
                        failed=1,
                        error_code="VALIDATION_ERROR",
                        duration_ms=(time.perf_counter() - started) * 1000,
                    )
                except Exception as exc:
                    if not is_transient_error(exc) or scan_attempts >= self._max_retries:
                        state.status = "failed"
                        state.error_code = "SOURCE_FAILURE"
                        await self._state_repo.save_state(state)
                        return ConnectorSyncResult(
                            connector_id=connector_id,
                            status="failed",
                            failed=1,
                            error_code="SOURCE_FAILURE",
                            duration_ms=(time.perf_counter() - started) * 1000,
                        )
                    if self._backoff_fn is not None:
                        res = self._backoff_fn(scan_attempts)
                        if asyncio.iscoroutine(res):
                            await res
                    else:
                        await asyncio.sleep(0.01 * (2 ** (scan_attempts - 1)))

            assert items is not None
            result = ConnectorSyncResult(
                connector_id=connector_id,
                status="success",
                scanned=len(items),
                next_cursor=next_cursor,
            )

            # --- Step 2: Item loop ---
            for item in items:
                digest = hashlib.sha256(item.content.encode("utf-8")).hexdigest()

                # Unchanged skip check (before privacy / extraction / temporal!)
                if await self._state_repo.item_is_current(connector_id, item, digest):
                    result.unchanged += 1
                    continue

                # Deletion handling
                if item.deleted:
                    old_mids = await self._state_repo.get_item_memory_ids(connector_id, item.external_id)
                    await self._state_repo.save_item(connector_id, item, digest, [], deleted=True)
                    result.deleted += 1

                    if (
                        self._retention_policy == RetentionPolicy.EXPIRE_ON_SOURCE_DELETE
                        and self._memory_repo is not None
                    ):
                        for mid in old_mids:
                            # Multi-source provenance check: only expire if no other active item references it!
                            active_refs = await self._state_repo.count_active_references(mid)
                            if active_refs == 0:
                                try:
                                    existing = await self._memory_repo.get(mid)
                                    if (
                                        existing is not None
                                        and existing.status != MemoryStatus.EXPIRED
                                        and existing.source_type.startswith("connector:")
                                    ):
                                        await self._memory_repo.update_status(
                                            mid, MemoryStatus.EXPIRED, expected_version=existing.version
                                        )
                                except Exception:
                                    pass
                    continue

                # Process new / changed item with retries for transient errors
                item_success = False
                item_attempts = 0

                while item_attempts < self._max_retries:
                    item_attempts += 1
                    try:
                        ingest_res = await self._ingestion.ingest(
                            IngestRequest(
                                content=item.content,
                                source_type=f"connector:{connector.source_type}",
                                source_uri=item.source_uri,
                                source_role=SourceRole.USER,
                            )
                        )
                        ids = []
                        for candidate in ingest_res.candidates:
                            resolution = await self._temporal.accept(
                                candidate, provenance_event_id=ingest_res.event_id
                            )
                            ids.append(resolution.memory.id)

                        await self._state_repo.save_item(
                            connector_id, item, digest, ids, deleted=False
                        )
                        result.accepted += len(ids)
                        if ids:
                            result.updated += 1
                        item_success = True
                        break

                    except asyncio.CancelledError:
                        state.status = "cancelled"
                        state.error_code = "CANCELLED"
                        await self._state_repo.save_state(state)
                        raise

                    except SecretDetectedError:
                        # Privacy rejection is permanent — do NOT retry!
                        await self._state_repo.save_item(
                            connector_id, item, digest, [], deleted=False
                        )
                        result.rejected += 1
                        item_success = True
                        break

                    except (ValueError, TypeError, IngestionError):
                        # Permanent validation error / item too large — do NOT retry!
                        result.rejected += 1
                        item_success = True
                        break

                    except Exception as exc:
                        if not is_transient_error(exc) or item_attempts >= self._max_retries:
                            # Non-transient errors or exhausted retries fail without retrying further
                            break
                        if self._backoff_fn is not None:
                            res = self._backoff_fn(item_attempts)
                            if asyncio.iscoroutine(res):
                                await res
                        else:
                            await asyncio.sleep(0.01 * (2 ** (item_attempts - 1)))



                if not item_success:
                    result.failed += 1
                    result.status = "partial"
                    result.error_code = "ITEM_FAILURE"
                    break  # Stop processing further items to preserve cursor safety!

            # --- Step 3: Finalize state and cursor ---
            if result.failed == 0:
                state.cursor = next_cursor
                state.last_success_at = datetime.now(timezone.utc)
                state.status = "success"
                state.error_code = None
            else:
                state.status = "partial" if (result.accepted > 0 or result.unchanged > 0 or result.deleted > 0) else "failed"
                state.error_code = result.error_code or "ITEM_FAILURE"

            await self._state_repo.save_state(state)
            result.duration_ms = (time.perf_counter() - started) * 1000
            return result
