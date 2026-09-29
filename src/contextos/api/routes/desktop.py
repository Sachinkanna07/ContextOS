"""Bounded, privacy-conscious terminal product endpoints."""

from __future__ import annotations

import asyncio
import time

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from contextos.api.server import get_service
from contextos.core.enums import MemoryStatus, SourceRole
from contextos.core.models import IngestRequest, MemoryFilters

router = APIRouter(tags=["terminal"])
_model_cache: tuple[float, object, list[dict]] | None = None


class RememberRequest(BaseModel):
    text: str = Field(min_length=1, max_length=10_000, repr=False)


@router.get("/dashboard")
async def dashboard(model: str | None = Query(default=None, max_length=200)) -> dict:
    repo = get_service("memory_repo")
    telemetry = get_service("telemetry_query")
    connectors = get_service("connectors")
    states = {state.connector_id: state for state in await get_service("connector_repo").list_states()}
    recent = await telemetry.list_recent(limit=8, model_id=model)
    summary = await telemetry.summary_range(model_id=model, success_only=True)
    global _model_cache
    providers = get_service("providers")
    if _model_cache is None or _model_cache[1] is not providers or time.monotonic() >= _model_cache[0]:
        async def discover(provider):
            try:
                return await asyncio.wait_for(provider.list_models(), timeout=1.0)
            except Exception:
                return []
        inventories = await asyncio.gather(*(discover(p) for p in providers.values()))
        discovered = [
            {"provider": item.provider_id, "model": item.model_id,
             "local": item.local, "enabled": item.enabled,
             "simulated": item.provider_id == "fake"}
            for inventory in inventories for item in inventory
        ][:50]
        _model_cache = (time.monotonic() + 10.0, providers, discovered)
    discovered = _model_cache[2]
    return {
        "memories": {
            "active": await repo.count(MemoryFilters(status=MemoryStatus.ACTIVE)),
            "historical": await repo.count(MemoryFilters(status=MemoryStatus.HISTORICAL)),
            "expired": await repo.count(MemoryFilters(status=MemoryStatus.EXPIRED)),
        },
        "connectors": [
            {"id": cid, "status": states[cid].status if cid in states else "idle",
             "enabled": states[cid].enabled if cid in states else True,
             "error_code": states[cid].error_code if cid in states else None}
            for cid in connectors.list_connectors()
        ],
        "models": [item for item in discovered if model is None or item["model"] == model],
        "summary": summary.model_dump(mode="json"),
        "context_measurement_bases": await telemetry.context_measurement_bases(model),
        "recent": [
            {"id": str(row.invocation_id), "timestamp": row.timestamp.isoformat(),
             "provider": row.provider_id, "model": row.model_id, "status": row.status,
             "preflight_input_tokens": row.preflight_input_tokens,
             "provider_input_tokens": row.provider_input_tokens,
             "provider_output_tokens": row.provider_output_tokens,
             "provider_measurement_source": row.token_measurement_source.value,
             "context_measurement_source": (row.context_token_measurement_source.value
                                            if row.context_token_measurement_source else "unknown"),
             "context_tokenizer": row.context_tokenizer,
             "candidate_context_tokens": row.candidate_context_tokens,
             "compiled_context_tokens": row.compiled_context_tokens,
             "context_tokens_avoided": row.context_tokens_avoided,
             "graph_expanded_count": row.graph_expanded_count,
             "retrieval_ms": row.retrieval_ms, "compilation_ms": row.compilation_ms}
            for row in recent
        ],
    }


@router.get("/connectors")
async def list_connectors() -> list[dict]:
    manager = get_service("connectors")
    states = {item.connector_id: item for item in await get_service("connector_repo").list_states()}
    return [{"id": cid, "registered": True,
             "status": states[cid].status if cid in states else "idle",
             "enabled": states[cid].enabled if cid in states else True,
             "error_code": states[cid].error_code if cid in states else None,
             "last_success_at": states[cid].last_success_at if cid in states else None}
            for cid in manager.list_connectors()]


@router.post("/connectors/{connector_id}/sync")
async def sync_connector(connector_id: str) -> dict:
    manager = get_service("connectors")
    if connector_id not in manager.list_connectors():
        raise HTTPException(status_code=404, detail="Connector is not registered")
    result = await manager.sync(connector_id)
    return result.model_dump(mode="json")


@router.post("/remember")
async def remember(request: RememberRequest) -> dict:
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
