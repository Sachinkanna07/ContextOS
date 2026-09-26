"""System status and diagnostics API routes."""

from __future__ import annotations

import os
import time

from fastapi import APIRouter

from contextos import __version__
from contextos.api.server import get_service
from contextos.core.models import MemoryFilters, SystemStatus, TokenStats
from contextos.core.enums import MemoryStatus

router = APIRouter(tags=["system"])

_start_time = time.time()


@router.get("/status", response_model=SystemStatus)
async def status() -> SystemStatus:
    """Get system health and status."""
    memory_service = get_service("memory")
    database = get_service("database")
    embedding_service = get_service("embedding")
    vector_store = get_service("vector_store")
    bm25_index = get_service("bm25_index")

    total = await memory_service.list(MemoryFilters(limit=1))
    active_filters = MemoryFilters(status=MemoryStatus.ACTIVE, limit=1)

    # Get counts
    from contextos.core.protocols import MemoryRepository
    memory_repo: MemoryRepository = get_service("memory_repo")
    total_count = await memory_repo.count()
    active_count = await memory_repo.count(MemoryFilters(status=MemoryStatus.ACTIVE))

    from contextos.core.protocols import EventRepository
    event_repo: EventRepository = get_service("event_repo")
    event_count = await event_repo.count()

    vector_count = await vector_store.count()
    bm25_count = await bm25_index.count()

    db_size = await database.get_size_bytes()

    return SystemStatus(
        daemon_running=True,
        pid=os.getpid(),
        uptime_seconds=time.time() - _start_time,
        total_memories=total_count,
        active_memories=active_count,
        total_events=event_count,
        embedding_model=embedding_service.model_name,
        embedding_model_loaded=True,
        vector_index_size=vector_count,
        bm25_index_size=bm25_count,
        database_size_bytes=db_size,
        data_directory=str(database.path.parent),
    )


@router.get("/stats", response_model=TokenStats)
async def stats() -> TokenStats:
    """Get token savings and performance statistics."""
    memory_repo = get_service("memory_repo")

    total_count = await memory_repo.count()
    active_count = await memory_repo.count(MemoryFilters(status=MemoryStatus.ACTIVE))

    # Calculate total tokens stored
    active_memories = await memory_repo.list(
        MemoryFilters(status=MemoryStatus.ACTIVE, limit=500)
    )
    total_tokens = sum(m.token_count for m in active_memories)
    avg_tokens = total_tokens / len(active_memories) if active_memories else 0

    return TokenStats(
        total_tokens_stored=total_tokens,
        tokens_per_memory=avg_tokens,
        # Compilation stats are accumulated over time — Phase 1 tracks per-request
        total_compilations=0,
        total_tokens_compiled=0,
        total_tokens_saved=0,
        average_compression_ratio=0.0,
    )


@router.post("/doctor")
async def doctor() -> dict:
    """Run diagnostic checks."""
    results: dict = {
        "version": __version__,
        "checks": {},
    }

    # Database integrity
    database = get_service("database")
    ok, msg = await database.integrity_check()
    results["checks"]["database_integrity"] = {"ok": ok, "message": msg}

    # Embedding model
    try:
        embedding = get_service("embedding")
        dim = embedding.dimension
        results["checks"]["embedding_model"] = {
            "ok": True,
            "model": embedding.model_name,
            "dimension": dim,
        }
    except Exception as e:
        results["checks"]["embedding_model"] = {"ok": False, "error": str(e)}

    # Index consistency
    memory_repo = get_service("memory_repo")
    vector_store = get_service("vector_store")
    bm25_index = get_service("bm25_index")

    memory_count = await memory_repo.count(MemoryFilters(status=MemoryStatus.ACTIVE))
    vector_count = await vector_store.count()
    bm25_count = await bm25_index.count()

    results["checks"]["index_consistency"] = {
        "ok": True,
        "active_memories": memory_count,
        "vector_index": vector_count,
        "bm25_index": bm25_count,
        "warnings": [],
    }

    if vector_count != memory_count:
        results["checks"]["index_consistency"]["warnings"].append(
            f"Vector index ({vector_count}) != active memories ({memory_count})"
        )
    if bm25_count != memory_count:
        results["checks"]["index_consistency"]["warnings"].append(
            f"BM25 index ({bm25_count}) != active memories ({memory_count})"
        )

    if results["checks"]["index_consistency"]["warnings"]:
        results["checks"]["index_consistency"]["ok"] = False

    results["overall"] = all(
        c.get("ok", False) for c in results["checks"].values()
    )

    return results
