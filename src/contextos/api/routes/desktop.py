"""Bounded, privacy-conscious terminal product endpoints."""

from __future__ import annotations

import asyncio
import re
from datetime import UTC, datetime, timedelta
from typing import Any, Literal
from uuid import UUID  # noqa: TC003 -- FastAPI resolves UUID route parameters at runtime.

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from contextos.api.server import get_service
from contextos.core.enums import MemoryStatus, SourceRole
from contextos.core.models import IngestRequest, MemoryFilters

router = APIRouter(tags=["terminal"])


def _public_label(value: object) -> str:
    """Keep identifiers useful while refusing credential, URI and path shapes."""
    from contextos.services.explainability import safe_text
    clean = safe_text(value, 160)
    if re.search(
        r"(?i)(?:[a-z]:[\\/]|\\\\|[a-z][a-z0-9+.-]*://|"
        r"\b(?:api[_-]?key|password|secret|bearer|authorization|cookie|credential)\b|"
        r"sk-[a-z0-9_-]{8,}|(?:^|\s)/(?:[^/\s]+/)+)", clean,
    ):
        return "[REDACTED]"
    return clean


def _measurement_label(source: str | None) -> str:
    if source in {"provider_reported", "tokenizer_counted"}:
        return "MEASURED"
    if source == "approximated":
        return "APPROXIMATED"
    return "UNKNOWN"


class RememberRequest(BaseModel):
    text: str = Field(min_length=1, max_length=10_000, repr=False)


@router.get("/dashboard")
async def dashboard(
    model: str | None = Query(default=None, min_length=1, max_length=200, pattern=r".*\S.*"),
    provider: str | None = Query(default=None, min_length=1, max_length=200, pattern=r".*\S.*"),
    period: Literal["all", "today", "week"] = "all",
) -> dict[str, Any]:
    repo = get_service("memory_repo")
    telemetry = get_service("telemetry_query")
    connectors = get_service("connectors")
    states = {
        state.connector_id: state
        for state in await get_service("connector_repo").list_states()
    }
    tracked_cursor = await get_service("database").connection().execute(
        "SELECT connector_id, COUNT(*) FROM connector_items WHERE deleted = 0 GROUP BY connector_id"
    )
    tracked_items = {row[0]: row[1] for row in await tracked_cursor.fetchall()}
    now = datetime.now(UTC)
    start = (
        datetime(now.year, now.month, now.day, tzinfo=UTC) if period == "today"
        else now - timedelta(days=7) if period == "week" else None
    )
    recent = await telemetry.list_recent(limit=8, model_id=model, provider_id=provider, start=start)
    summary = await telemetry.summary_range(start=start, provider_id=provider,
                                            model_id=model, success_only=True)
    breakdown = await telemetry.provider_model_breakdown(start, provider, model)
    summary_data = summary.model_dump(mode="json")
    for group in ("by_provider", "by_model"):
        summary_data[group] = {
            _public_label(key): value for key, value in summary_data[group].items()
        }
    for row in breakdown:
        row["provider"] = _public_label(row["provider"])
        row["model"] = _public_label(row["model"])
        row["context_tokenizer"] = _public_label(row["context_tokenizer"])
    inventory = await get_service("model_discovery").list_models(get_service("providers"))
    discovered = [
        {"provider": _public_label(item.provider_id), "model": _public_label(item.model_id),
         "local": item.local, "enabled": item.enabled,
         "simulated": item.provider_id == "fake"}
        for item in inventory
    ][:50]
    mcp = get_service("settings").mcp
    return {
        "memories": {
            "active": await repo.count(MemoryFilters(status=MemoryStatus.ACTIVE)),
            "historical": await repo.count(MemoryFilters(status=MemoryStatus.HISTORICAL)),
            "expired": await repo.count(MemoryFilters(status=MemoryStatus.EXPIRED)),
        },
        "temporal": {
            "superseded": await repo.count(MemoryFilters(status=MemoryStatus.SUPERSEDED)),
            "contradicted": await repo.count(MemoryFilters(status=MemoryStatus.CONTRADICTED)),
        },
        "graph": await graph_stats(),
        "mcp": {"enabled": mcp.enabled, "transport": mcp.transport if mcp.enabled else None,
                "read": mcp.allow_read if mcp.enabled else False,
                "write": mcp.allow_write if mcp.enabled else False},
        "connectors": [
            {"id": _public_label(cid), "status": states[cid].status if cid in states else "idle",
             "enabled": states[cid].enabled if cid in states else True,
             "error_code": states[cid].error_code if cid in states else None,
             "type": _public_label(states[cid].connector_type) if cid in states else "unknown",
             "tracked_items": tracked_items.get(cid, 0),
             "last_success_at": states[cid].last_success_at if cid in states else None,
             "last_attempt_at": states[cid].last_attempt_at if cid in states else None,
             "cursor_present": bool(states[cid].cursor) if cid in states else False}
            for cid in connectors.list_connectors()
        ],
        "models": [
            item for item in discovered
            if (model is None or item["model"] == _public_label(model))
            and (provider is None or item["provider"] == _public_label(provider))
        ],
        "summary": summary_data,
        "provider_models": breakdown,
        "period": period,
        "context_measurement_bases": [
            {"source": row["source"], "tokenizer": _public_label(row["tokenizer"])}
            for row in await telemetry.context_measurement_bases(model, provider, start)
        ],
        "recent": [
            {
             "id": str(row.invocation_id), "timestamp": row.timestamp.isoformat(),
             "provider": _public_label(row.provider_id), "model": _public_label(row.model_id),
             "status": row.status,
             "preflight_input_tokens": row.preflight_input_tokens,
             "provider_input_tokens": row.provider_input_tokens,
             "provider_output_tokens": row.provider_output_tokens,
             "provider_measurement_source": row.token_measurement_source.value,
             "provider_measurement_label": _measurement_label(row.token_measurement_source.value),
             "context_measurement_source": (row.context_token_measurement_source.value
                                            if row.context_token_measurement_source else "unknown"),
             "context_measurement_label": _measurement_label(
                 row.context_token_measurement_source.value
                 if row.context_token_measurement_source else None
             ),
             "context_tokenizer": _public_label(row.context_tokenizer),
             "candidate_context_tokens": row.candidate_context_tokens,
             "compiled_context_tokens": row.compiled_context_tokens,
             "context_tokens_avoided": row.context_tokens_avoided,
             "lexical_candidate_count": row.lexical_candidate_count,
             "dense_candidate_count": row.dense_candidate_count,
             "selected_memory_count": row.selected_memory_count,
             "graph_expanded_count": row.graph_expanded_count,
             "retrieval_ms": row.retrieval_ms, "compilation_ms": row.compilation_ms,
             "provider_ms": row.provider_latency_ms, "error_code": _public_label(row.error_code)}
            for row in recent
        ],
    }


@router.get("/graph/stats")
async def graph_stats() -> dict[str, Any]:
    """Read projection counts without rebuilding or exposing node labels."""
    repo = get_service("graph_repo")
    nodes, edges, supports = await repo.counts()
    conn = get_service("database").connection()
    node_cursor = await conn.execute(
        "SELECT node_type, COUNT(*) FROM graph_nodes GROUP BY node_type"
    )
    edge_cursor = await conn.execute(
        "SELECT relation_type, COUNT(*) FROM graph_edges GROUP BY relation_type"
    )
    return {
        "nodes": nodes, "edges": edges, "supports": supports,
        "dirty": await repo.source_is_dirty(),
        "average_total_degree": round(2 * edges / nodes, 3) if nodes else None,
        "node_types": {row[0]: row[1] for row in await node_cursor.fetchall()},
        "edge_types": {row[0]: row[1] for row in await edge_cursor.fetchall()},
    }


async def _graph_paths(query: str, seed_id: UUID | None = None) -> dict[str, Any]:
    expansion = await get_service("graph").expand(
        query_text=query, seed_memory_ids=[seed_id] if seed_id else [],
        max_hops=3, max_nodes=50, max_edges=100,
    )
    candidates = sorted(expansion.candidate_scores,
                        key=lambda item: (-expansion.candidate_scores[item], str(item)))[:10]
    return {
        "candidate_count": len(expansion.candidate_scores),
        "candidates": [{
            "memory_id": str(memory_id),
            "graph_score": expansion.candidate_scores[memory_id],
            "paths": [
                path.model_dump(mode="json")
                for path in expansion.candidate_paths.get(memory_id, [])[:5]
            ],
        } for memory_id in candidates],
        "truncated": len(expansion.candidate_scores) > 10,
    }


@router.get("/graph/search")
async def graph_search(entity: str = Query(min_length=1, max_length=128)) -> dict[str, Any]:
    return await _graph_paths(entity)


@router.get("/graph/show/{memory_id}")
async def graph_show(memory_id: UUID) -> dict[str, Any]:
    return await _graph_paths("", memory_id)


@router.get("/temporal/current")
async def temporal_current(limit: int = Query(default=25, ge=1, le=50)) -> dict[str, Any]:
    return await _temporal_rows("active", limit)


@router.get("/temporal/conflicts")
async def temporal_conflicts(limit: int = Query(default=25, ge=1, le=50)) -> dict[str, Any]:
    return await _temporal_rows("contradicted", limit)


async def _temporal_rows(status: str, limit: int) -> dict[str, Any]:
    conn = get_service("database").connection()
    cursor = await conn.execute(
        "SELECT id, status, temporal_status, observed_at, valid_from, superseded_by "
        "FROM memories WHERE status = ? ORDER BY observed_at DESC, id DESC LIMIT ?",
        (status, limit),
    )
    rows = await cursor.fetchall()
    return {"status": status, "memories": [
        {"memory_id": row[0], "lifecycle": row[1], "temporal_status": row[2],
         "observed_at": row[3], "effective_at": row[4], "superseded_by": row[5]}
        for row in rows
    ]}


@router.get("/temporal/history/{memory_id}")
async def temporal_history(memory_id: UUID, include_content: bool = False) -> dict[str, Any]:
    memory = await get_service("memory_repo").get(memory_id)
    if memory is None:
        raise HTTPException(status_code=404, detail="Memory not found")
    if memory.slot is None:
        return {"memory_id": str(memory_id), "history": [], "relations": []}
    conn = get_service("database").connection()
    cursor = await conn.execute(
        "SELECT id, status, temporal_status, observed_at, valid_from, superseded_by, content "
        "FROM memories WHERE slot_key = ? "
        "ORDER BY COALESCE(valid_from, observed_at, created_at) DESC, id DESC LIMIT 50",
        (memory.slot.key,),
    )
    from contextos.services.explainability import safe_text
    history = []
    for row in await cursor.fetchall():
        item = {"memory_id": row[0], "lifecycle": row[1], "temporal_status": row[2],
                "observed_at": row[3], "effective_at": row[4], "superseded_by": row[5]}
        if include_content:
            item["content"] = safe_text(row[6], 1000)
        history.append(item)
    evidence = await get_service("explainability").temporal_resolver.resolve_memory_evidence(memory)
    return {"memory_id": str(memory_id), "history": history,
            "relations": evidence["relations"][:50]}


@router.get("/connectors")
async def list_connectors() -> list[dict[str, Any]]:
    manager = get_service("connectors")
    states = {item.connector_id: item for item in await get_service("connector_repo").list_states()}
    return [{"id": _public_label(cid), "registered": True,
             "status": states[cid].status if cid in states else "idle",
             "enabled": states[cid].enabled if cid in states else True,
             "error_code": states[cid].error_code if cid in states else None,
             "last_success_at": states[cid].last_success_at if cid in states else None,
             "last_attempt_at": states[cid].last_attempt_at if cid in states else None}
            for cid in manager.list_connectors()]


@router.post("/connectors/{connector_id}/sync")
async def sync_connector(connector_id: str) -> dict[str, Any]:
    manager = get_service("connectors")
    if connector_id not in manager.list_connectors():
        raise HTTPException(status_code=404, detail="Connector is not registered")
    result = await manager.sync(connector_id)
    return dict(result.model_dump(mode="json"))


@router.post("/remember")
async def remember(request: RememberRequest) -> dict[str, Any]:
    result = await get_service("ingestion").ingest(
        IngestRequest(content=request.text, source_type="cli", source_role=SourceRole.USER)
    )
    accepted: list[str] = []
    for candidate in result.candidates:
        try:
            resolved = await get_service("temporal").accept(
                candidate, provenance_event_id=result.event_id
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            raise HTTPException(status_code=500, detail={
                "error": "PARTIAL_WRITE" if accepted else "MEMORY_WRITE_FAILED",
                "accepted_ids": accepted,
            }) from None
        accepted.append(str(resolved.memory.id))
    return {"accepted_ids": accepted, "count": len(accepted),
            "secrets_detected": result.secrets_detected}
