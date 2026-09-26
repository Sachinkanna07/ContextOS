"""Memory management service for ContextOS.

Business logic layer for memory CRUD, lifecycle transitions, and search.
Enforces state machine rules from core.enums.VALID_TRANSITIONS.
"""

from __future__ import annotations

import logging
from uuid import UUID

from contextos.core.enums import VALID_TRANSITIONS, EventType, MemoryStatus
from contextos.core.exceptions import (
    InvalidTransitionError,
    MemoryNotFoundError,
)
from contextos.core.models import (
    Memory,
    MemoryFilters,
    MemoryUpdate,
    RawEvent,
    ScoredMemory,
)
from contextos.core.protocols import (
    EmbeddingService,
    EventRepository,
    LexicalIndex,
    MemoryRepository,
    VectorStore,
)

logger = logging.getLogger(__name__)


class CoreMemoryService:
    """Phase 1 memory CRUD and lifecycle without retrieval or indexing dependencies."""

    def __init__(self, repository: MemoryRepository) -> None:
        self._repository = repository

    async def create(self, memory: Memory) -> Memory:
        return await self._repository.create(memory)

    async def get(self, memory_id: UUID) -> Memory | None:
        return await self._repository.get(memory_id)

    async def list(self, filters: MemoryFilters) -> list[Memory]:
        return await self._repository.list(filters)

    async def update(self, memory_id: UUID, update: MemoryUpdate) -> Memory:
        current = await self._repository.get(memory_id)
        if current is None:
            raise MemoryNotFoundError(str(memory_id))
        return await self._repository.update(memory_id, update, current.version)

    async def transition(self, memory_id: UUID, status: MemoryStatus) -> Memory:
        current = await self._repository.get(memory_id)
        if current is None:
            raise MemoryNotFoundError(str(memory_id))
        return await self._repository.update_status(memory_id, status, current.version)

    async def supersede(self, old_id: UUID, successor: Memory) -> Memory:
        current = await self._repository.get(old_id)
        if current is None:
            raise MemoryNotFoundError(str(old_id))
        return await self._repository.supersede(old_id, successor, current.version)

    async def delete(self, memory_id: UUID) -> Memory:
        return await self.transition(memory_id, MemoryStatus.DELETED)


class MemoryManager:
    """Business logic for memory management.

    Implements the MemoryService protocol.
    """

    def __init__(
        self,
        *,
        memory_repo: MemoryRepository,
        event_repo: EventRepository,
        vector_store: VectorStore,
        lexical_index: LexicalIndex,
        embedding_service: EmbeddingService,
    ) -> None:
        self._memory_repo = memory_repo
        self._event_repo = event_repo
        self._vector_store = vector_store
        self._lexical_index = lexical_index
        self._embedding_service = embedding_service

    async def get(self, memory_id: UUID) -> Memory | None:
        return await self._memory_repo.get(memory_id)

    async def list(self, filters: MemoryFilters) -> list[Memory]:
        return await self._memory_repo.list(filters)

    async def search(self, query: str, limit: int = 50) -> list[ScoredMemory]:
        """Quick search across memories using vector similarity."""
        try:
            query_vec = await self._embedding_service.embed_query(query)
            results = await self._vector_store.search(vector=query_vec, top_k=limit)

            scored: list[ScoredMemory] = []
            for r in results:
                memory = await self._memory_repo.get(UUID(r.id))
                if memory and memory.status == MemoryStatus.ACTIVE:
                    scored.append(ScoredMemory(
                        memory=memory,
                        final_score=r.score,
                        vector_score=r.score,
                    ))
                    await self._memory_repo.update_access(memory.id)

            return scored
        except Exception:
            logger.warning("Vector search failed in memory manager", exc_info=True)
            return []

    async def update(self, memory_id: UUID, update: MemoryUpdate) -> Memory:
        memory = await self._memory_repo.get(memory_id)
        if memory is None:
            raise MemoryNotFoundError(str(memory_id))

        # Record the update event (before/after snapshot)
        before_content = memory.content

        updated = await self._memory_repo.update(
            memory_id, update, expected_version=memory.version
        )

        # If content changed, re-embed and re-index
        if update.content is not None and update.content != before_content:
            try:
                embeddings = await self._embedding_service.embed([updated.content])
                await self._vector_store.delete([str(memory_id)])
                await self._vector_store.add(
                    ids=[str(memory_id)],
                    vectors=embeddings,
                    metadata=[{"type": updated.type.value, "status": updated.status.value}],
                )
            except Exception:
                logger.warning("Failed to re-embed memory %s", memory_id, exc_info=True)

            try:
                await self._lexical_index.delete(str(memory_id))
                await self._lexical_index.index(
                    doc_id=str(memory_id),
                    text=updated.content,
                    metadata={"type": updated.type.value},
                )
            except Exception:
                logger.warning("Failed to re-index memory %s", memory_id, exc_info=True)

        # Record event
        await self._event_repo.append(RawEvent(
            event_type=EventType.MEMORY_UPDATED,
            source_type="system",
            metadata={
                "memory_id": str(memory_id),
                "before_content": before_content,
                "after_content": updated.content,
                "fields_changed": [
                    k for k, v in update.model_dump(exclude_none=True).items()
                ],
            },
            memory_ids=[memory_id],
        ))

        return updated

    async def transition(
        self, memory_id: UUID, new_status: MemoryStatus, reason: str = ""
    ) -> Memory:
        """Transition a memory to a new lifecycle state."""
        memory = await self._memory_repo.get(memory_id)
        if memory is None:
            raise MemoryNotFoundError(str(memory_id))

        # Validate transition
        allowed = VALID_TRANSITIONS.get(memory.status, set())
        if new_status not in allowed:
            raise InvalidTransitionError(
                str(memory_id), memory.status.value, new_status.value
            )

        updated = await self._memory_repo.update_status(
            memory_id, new_status, expected_version=memory.version
        )

        # Record transition event
        await self._event_repo.append(RawEvent(
            event_type=EventType.MEMORY_TRANSITION,
            source_type="system",
            metadata={
                "memory_id": str(memory_id),
                "from_status": memory.status.value,
                "to_status": new_status.value,
                "reason": reason,
            },
            memory_ids=[memory_id],
        ))

        logger.info(
            "Memory %s transitioned: %s → %s (reason: %s)",
            memory_id, memory.status.value, new_status.value, reason,
        )

        return updated

    async def delete(self, memory_id: UUID) -> None:
        """Soft delete: transition to DELETED, remove from indices."""
        memory = await self._memory_repo.get(memory_id)
        if memory is None:
            raise MemoryNotFoundError(str(memory_id))

        # Transition to DELETED (validates allowed transitions)
        if memory.status != MemoryStatus.DELETED:
            # Find the path to DELETED
            allowed = VALID_TRANSITIONS.get(memory.status, set())
            if MemoryStatus.DELETED not in allowed:
                raise InvalidTransitionError(
                    str(memory_id), memory.status.value, MemoryStatus.DELETED.value
                )
            await self._memory_repo.update_status(
                memory_id, MemoryStatus.DELETED, expected_version=memory.version
            )

        # Remove from indices
        try:
            await self._vector_store.delete([str(memory_id)])
        except Exception:
            logger.warning("Failed to remove memory %s from vector store", memory_id)
        try:
            await self._lexical_index.delete(str(memory_id))
        except Exception:
            logger.warning("Failed to remove memory %s from BM25 index", memory_id)

        # Record event
        await self._event_repo.append(RawEvent(
            event_type=EventType.MEMORY_DELETED,
            source_type="system",
            metadata={"memory_id": str(memory_id), "type": "soft_delete"},
            memory_ids=[memory_id],
        ))

    async def purge(self, memory_id: UUID) -> None:
        """Hard delete: destroy from all stores."""
        memory = await self._memory_repo.get(memory_id)
        if memory is None:
            raise MemoryNotFoundError(str(memory_id))

        # Remove from indices
        try:
            await self._vector_store.delete([str(memory_id)])
        except Exception:
            logger.warning("Failed to purge memory %s from vector store", memory_id)
        try:
            await self._lexical_index.delete(str(memory_id))
        except Exception:
            logger.warning("Failed to purge memory %s from BM25 index", memory_id)

        # Remove from database
        await self._memory_repo.delete(memory_id)

        # Record purge event (no content — it's destroyed)
        await self._event_repo.append(RawEvent(
            event_type=EventType.MEMORY_PURGED,
            source_type="system",
            metadata={"memory_id": str(memory_id), "type": "hard_delete"},
        ))

        logger.info("Memory %s purged from all stores", memory_id)

    async def history(self, topic: str) -> list[Memory]:
        """Get temporal evolution of memories related to a topic.

        Returns memories (including historical/superseded) related to the topic,
        ordered by creation time.
        """
        # Search across all statuses
        scored = await self.search(topic, limit=100)

        # Also include non-active memories that match
        all_memories = [sm.memory for sm in scored]

        # Sort by creation time ascending (oldest first)
        all_memories.sort(key=lambda m: m.created_at)

        return all_memories
