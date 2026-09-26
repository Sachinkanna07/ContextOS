"""Memory management API routes."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Query

from contextos.api.server import get_service
from contextos.core.enums import MemoryStatus, MemoryType, PrivacyLevel
from contextos.core.models import Memory, MemoryFilters, MemoryUpdate

router = APIRouter(tags=["memories"])


@router.get("/memories", response_model=list[Memory])
async def list_memories(
    status: MemoryStatus | None = None,
    type: MemoryType | None = None,
    privacy_level: PrivacyLevel | None = None,
    source_type: str | None = None,
    min_confidence: float | None = None,
    min_importance: float | None = None,
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> list[Memory]:
    """List memories with optional filters."""
    memory_service = get_service("memory")
    filters = MemoryFilters(
        status=status,
        type=type,
        privacy_level=privacy_level,
        source_type=source_type,
        min_confidence=min_confidence,
        min_importance=min_importance,
        limit=limit,
        offset=offset,
    )
    return await memory_service.list(filters)


@router.get("/memories/{memory_id}", response_model=Memory)
async def get_memory(memory_id: UUID) -> Memory:
    """Get a specific memory by ID."""
    memory_service = get_service("memory")
    memory = await memory_service.get(memory_id)
    if memory is None:
        from contextos.core.exceptions import MemoryNotFoundError
        raise MemoryNotFoundError(str(memory_id))
    return memory


@router.patch("/memories/{memory_id}", response_model=Memory)
async def update_memory(memory_id: UUID, update: MemoryUpdate) -> Memory:
    """Update a memory's content or metadata."""
    memory_service = get_service("memory")
    return await memory_service.update(memory_id, update)


@router.post("/memories/{memory_id}/transition", response_model=Memory)
async def transition_memory(
    memory_id: UUID,
    new_status: MemoryStatus,
    reason: str = "",
) -> Memory:
    """Transition a memory to a new lifecycle state."""
    memory_service = get_service("memory")
    return await memory_service.transition(memory_id, new_status, reason)


@router.delete("/memories/{memory_id}")
async def delete_memory(memory_id: UUID) -> dict:
    """Soft-delete a memory."""
    memory_service = get_service("memory")
    await memory_service.delete(memory_id)
    return {"status": "deleted", "memory_id": str(memory_id)}


@router.delete("/memories/{memory_id}/purge")
async def purge_memory(memory_id: UUID) -> dict:
    """Hard-delete a memory (irreversible)."""
    memory_service = get_service("memory")
    await memory_service.purge(memory_id)
    return {"status": "purged", "memory_id": str(memory_id)}
