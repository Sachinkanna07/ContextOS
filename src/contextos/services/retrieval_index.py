"""Explicit synchronization of ephemeral retrieval indexes with SQLite."""

from __future__ import annotations

import asyncio

from contextos.core.enums import MemoryStatus
from contextos.core.models import Memory, MemoryFilters
from contextos.core.protocols import EmbeddingService, LexicalIndex, MemoryRepository, VectorStore


INDEXED_STATUSES = (
    MemoryStatus.ACTIVE,
    MemoryStatus.HISTORICAL,
    MemoryStatus.SUPERSEDED,
    MemoryStatus.CONTRADICTED,
    MemoryStatus.EXPIRED,
)


class RetrievalIndexSynchronizer:
    """Rebuild both indexes when the persisted memory fingerprint changes."""

    def __init__(
        self,
        *,
        memory_repo: MemoryRepository,
        lexical_index: LexicalIndex,
        vector_store: VectorStore,
        embedding_service: EmbeddingService,
    ) -> None:
        self._memory_repo = memory_repo
        self._lexical_index = lexical_index
        self._vector_store = vector_store
        self._embedding_service = embedding_service
        self._fingerprint: tuple[tuple[str, int, str, str], ...] | None = None
        self._lock = asyncio.Lock()

    async def ensure_current(self, *, force: bool = False) -> bool:
        """Synchronize indexes; return whether a rebuild occurred."""
        async with self._lock:
            memories = await self._load_indexable_memories()
            fingerprint = tuple(
                sorted(
                    (str(memory.id), memory.version, memory.content_hash, memory.status.value)
                    for memory in memories
                )
            )
            if not force and fingerprint == self._fingerprint:
                return False
            await self._rebuild(memories)
            self._fingerprint = fingerprint
            return True

    async def _load_indexable_memories(self) -> list[Memory]:
        memories: list[Memory] = []
        for status in INDEXED_STATUSES:
            offset = 0
            while True:
                page = await self._memory_repo.list(
                    MemoryFilters(status=status, limit=500, offset=offset)
                )
                memories.extend(page)
                if len(page) < 500:
                    break
                offset += len(page)
        return memories

    async def _rebuild(self, memories: list[Memory]) -> None:
        documents = {str(memory.id): memory.content for memory in memories}
        ids = [str(memory.id) for memory in memories]
        vectors = (
            await self._embedding_service.embed([memory.content for memory in memories])
            if memories
            else []
        )
        metadata = [
            {
                "status": memory.status.value,
                "type": memory.type.value,
                "source_type": memory.source_type,
            }
            for memory in memories
        ]
        # Generate and validate the more failure-prone dense corpus before
        # replacing either live index.
        await self._vector_store.rebuild(ids, vectors, metadata)
        await self._lexical_index.rebuild(documents)
