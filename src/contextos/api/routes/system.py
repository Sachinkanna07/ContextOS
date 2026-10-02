"""System status and diagnostics API routes."""

from __future__ import annotations

import os
import time
import asyncio

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
    database = get_service("database")
    embedding_service = get_service("embedding")
    vector_store = get_service("vector_store")
    bm25_index = get_service("bm25_index")

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
        data_directory="[LOCAL]",
    )


@router.get("/stats", response_model=TokenStats)
async def stats() -> TokenStats:
    """Report stored counts and observed model compilations with provenance."""
    conn = get_service("database").connection()
    cursor = await conn.execute(
        "SELECT COALESCE(SUM(token_count), 0), COUNT(*) FROM memories WHERE status = ?",
        (MemoryStatus.ACTIVE.value,),
    )
    total_tokens, memory_count = await cursor.fetchone()
    bases = await get_service("telemetry_query").context_measurement_bases()
    basis = bases[0] if len(bases) == 1 and bases[0]["source"] != "unknown" else None
    if basis is not None:
        cursor = await conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(compiled_context_tokens), 0), "
            "COALESCE(SUM(context_tokens_avoided), 0), "
            "COALESCE(SUM(candidate_context_tokens), 0) "
            "FROM model_invocations WHERE status = 'success'"
        )
        runs, compiled, avoided, candidates = await cursor.fetchone()
    else:
        runs = compiled = avoided = candidates = None
    from contextos.api.routes.desktop import _public_label

    return TokenStats(
        total_tokens_stored=total_tokens,
        tokens_per_memory=total_tokens / memory_count if memory_count else 0,
        # Standalone compilations are not persisted; these are successful model runs.
        total_compilations=runs,
        total_tokens_compiled=compiled,
        total_tokens_saved=avoided,
        average_compression_ratio=avoided / candidates if candidates else None,
        measurement_basis=f"{basis['source']}:{_public_label(basis['tokenizer'])}" if basis else None,
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

    from contextos.storage.database import SCHEMA_VERSION
    conn = database.connection()
    cursor = await conn.execute("SELECT MAX(version) FROM schema_version")
    version_row = await cursor.fetchone()
    schema_version = version_row[0] if version_row else None
    results["checks"]["schema"] = {
        "ok": schema_version == SCHEMA_VERSION,
        "version": schema_version, "expected": SCHEMA_VERSION,
    }
    cursor = await conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name IN "
        "('memories', 'memory_relations', 'model_invocations', 'connector_state', 'graph_nodes', 'graph_edges')"
    )
    tables = {row[0] for row in await cursor.fetchall()}
    results["checks"]["tables"] = {"ok": len(tables) == 6, "present": sorted(tables)}
    cursor = await conn.execute("PRAGMA index_list('memories')")
    indexes = {row[1] for row in await cursor.fetchall()}
    required_indexes = {"idx_memories_status", "idx_memories_slot_key"}
    results["checks"]["indexes"] = {
        "ok": required_indexes <= indexes,
        "required_present": sorted(indexes & required_indexes),
    }
    graph_repo = get_service("graph_repo")
    graph_nodes, graph_edges, _ = await graph_repo.counts()
    graph_dirty = await graph_repo.source_is_dirty()
    results["checks"]["graph_projection"] = {
        "ok": True, "dirty": graph_dirty, "nodes": graph_nodes, "edges": graph_edges,
        "suggestion": "Graph rebuilds on the next graph retrieval" if graph_dirty else None,
    }
    results["checks"]["telemetry"] = {
        "ok": True, "recorded_invocations": await get_service("telemetry_repo").count(),
    }
    connector_ids = get_service("connectors").list_connectors()
    connector_states = await get_service("connector_repo").list_states()
    results["checks"]["connectors"] = {
        "ok": True, "configured": len(connector_ids),
        "failed": sum(state.status == "failed" for state in connector_states),
    }
    settings = get_service("settings")
    results["checks"]["local_configuration"] = {
        "ok": settings.daemon.host in {"127.0.0.1", "localhost", "::1"}
              and os.access(database.path.parent, os.R_OK | os.W_OK),
        "data_directory_accessible": os.access(database.path.parent, os.R_OK | os.W_OK),
    }
    scan = get_service("secret_scanner").scan("Authorization: Bearer privatefixture1234567890")
    results["checks"]["privacy_scanner"] = {"ok": bool(scan.matches)}
    providers = get_service("providers")

    async def available(provider):
        try:
            return bool(await asyncio.wait_for(provider.health(), timeout=1.0))
        except Exception:
            return False

    healths = await asyncio.gather(*(available(provider) for provider in providers.values()))
    results["checks"]["providers"] = {
        "ok": any(healths), "configured": len(providers), "available": sum(healths),
        "optional_unavailable": len(providers) - sum(healths),
    }

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
        results["checks"]["embedding_model"] = {"ok": False, "error": e.__class__.__name__}

    # Index consistency
    memory_repo = get_service("memory_repo")
    vector_store = get_service("vector_store")
    bm25_index = get_service("bm25_index")

    from contextos.services.retrieval_index import INDEXED_STATUSES
    indexed_count = 0
    for memory_status in INDEXED_STATUSES:
        indexed_count += await memory_repo.count(MemoryFilters(status=memory_status))
    vector_count = await vector_store.count()
    bm25_count = await bm25_index.count()

    results["checks"]["index_consistency"] = {
        "ok": True,
        "indexable_memories": indexed_count,
        "vector_index": vector_count,
        "bm25_index": bm25_count,
        "warnings": [],
    }

    if vector_count != indexed_count:
        results["checks"]["index_consistency"]["warnings"].append(
            f"Vector index ({vector_count}) != indexable memories ({indexed_count})"
        )
    if bm25_count != indexed_count:
        results["checks"]["index_consistency"]["warnings"].append(
            f"BM25 index ({bm25_count}) != indexable memories ({indexed_count})"
        )

    if results["checks"]["index_consistency"]["warnings"]:
        results["checks"]["index_consistency"]["ok"] = False

    results["overall"] = all(
        c.get("ok", False) for c in results["checks"].values()
    )

    return results
