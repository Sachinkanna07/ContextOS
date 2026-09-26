"""Hybrid retrieval engine for ContextOS.

Implements multi-strategy retrieval with Reciprocal Rank Fusion (RRF).
Phase 1: Vector + BM25, fused with RRF. No reranker.

The retrieval engine is the core intelligence of the system — it determines
which memories are relevant to a given query.
"""

from __future__ import annotations

import logging
import time
from uuid import UUID

from contextos.core.enums import MemoryStatus
from contextos.core.models import (
    Memory,
    MemoryFilters,
    RetrievalConfig,
    RetrievalResult,
    RetrievalTrace,
    ScoredMemory,
    StageTrace,
    VectorResult,
    LexicalResult,
)
from contextos.core.protocols import (
    EmbeddingService,
    LexicalIndex,
    MemoryRepository,
    VectorStore,
)

logger = logging.getLogger(__name__)

# Default retrieval config
DEFAULT_CONFIG = RetrievalConfig()


class HybridRetrievalEngine:
    """Multi-strategy retrieval with RRF fusion.

    Implements the RetrievalService protocol.
    """

    def __init__(
        self,
        *,
        memory_repo: MemoryRepository,
        vector_store: VectorStore,
        lexical_index: LexicalIndex,
        embedding_service: EmbeddingService,
    ) -> None:
        self._memory_repo = memory_repo
        self._vector_store = vector_store
        self._lexical_index = lexical_index
        self._embedding_service = embedding_service

    async def retrieve(
        self, query: str, config: RetrievalConfig | None = None
    ) -> RetrievalResult:
        """Execute hybrid retrieval: vector + BM25 → RRF fusion → dedup."""
        cfg = config or DEFAULT_CONFIG
        trace_stages: list[StageTrace] = []
        t0 = time.perf_counter()

        # --- Stage 1: Vector Search ---
        t_vec = time.perf_counter()
        vector_results = await self._vector_search(query, cfg.vector_top_k)
        vec_latency = (time.perf_counter() - t_vec) * 1000
        trace_stages.append(StageTrace(
            stage_name="vector_search",
            input_count=1,
            output_count=len(vector_results),
            latency_ms=vec_latency,
            metadata={"top_k": cfg.vector_top_k},
        ))

        # --- Stage 2: BM25 Search ---
        t_bm25 = time.perf_counter()
        bm25_results = await self._bm25_search(query, cfg.bm25_top_k)
        bm25_latency = (time.perf_counter() - t_bm25) * 1000
        trace_stages.append(StageTrace(
            stage_name="bm25_search",
            input_count=1,
            output_count=len(bm25_results),
            latency_ms=bm25_latency,
            metadata={"top_k": cfg.bm25_top_k},
        ))

        # --- Stage 3: RRF Fusion ---
        t_fuse = time.perf_counter()
        fused = self._rrf_fuse(vector_results, bm25_results, cfg.rrf_k)
        fuse_latency = (time.perf_counter() - t_fuse) * 1000
        trace_stages.append(StageTrace(
            stage_name="rrf_fusion",
            input_count=len(vector_results) + len(bm25_results),
            output_count=len(fused),
            latency_ms=fuse_latency,
            metadata={"rrf_k": cfg.rrf_k},
        ))

        # --- Stage 4: Resolve memories from IDs ---
        t_resolve = time.perf_counter()
        scored_memories = await self._resolve_memories(fused, vector_results, bm25_results, cfg)
        resolve_latency = (time.perf_counter() - t_resolve) * 1000
        trace_stages.append(StageTrace(
            stage_name="memory_resolution",
            input_count=len(fused),
            output_count=len(scored_memories),
            latency_ms=resolve_latency,
        ))

        # --- Stage 5: Deduplication ---
        t_dedup = time.perf_counter()
        deduped = self._deduplicate(scored_memories)
        dedup_latency = (time.perf_counter() - t_dedup) * 1000
        trace_stages.append(StageTrace(
            stage_name="deduplication",
            input_count=len(scored_memories),
            output_count=len(deduped),
            latency_ms=dedup_latency,
            metadata={"removed": len(scored_memories) - len(deduped)},
        ))

        # Apply max_results limit
        final = deduped[: cfg.max_results]

        # Filter by min_score
        if cfg.min_score > 0:
            final = [sm for sm in final if sm.final_score >= cfg.min_score]

        total_latency = (time.perf_counter() - t0) * 1000

        # Build strategy_results for tracing
        strategy_results: dict[str, list[ScoredMemory]] = {}
        vec_scored = [sm for sm in scored_memories if sm.vector_score is not None]
        bm25_scored = [sm for sm in scored_memories if sm.bm25_score is not None]
        if vec_scored:
            strategy_results["vector"] = vec_scored
        if bm25_scored:
            strategy_results["bm25"] = bm25_scored

        return RetrievalResult(
            query=query,
            memories=final,
            strategy_results=strategy_results,
            trace=RetrievalTrace(
                stages=trace_stages,
                total_latency_ms=total_latency,
                total_candidates=len(fused),
                total_results=len(final),
            ),
        )

    # --- Internal Methods ---

    async def _vector_search(
        self, query: str, top_k: int
    ) -> list[tuple[str, float]]:
        """Run vector similarity search. Returns list of (memory_id, score)."""
        try:
            query_embedding = await self._embedding_service.embed_query(query)
            results: list[VectorResult] = await self._vector_store.search(
                vector=query_embedding, top_k=top_k
            )
            return [(r.id, r.score) for r in results]
        except Exception:
            logger.warning("Vector search failed, returning empty results", exc_info=True)
            return []

    async def _bm25_search(
        self, query: str, top_k: int
    ) -> list[tuple[str, float]]:
        """Run BM25 lexical search. Returns list of (memory_id, score)."""
        try:
            results: list[LexicalResult] = await self._lexical_index.search(
                query=query, top_k=top_k
            )
            return [(r.id, r.score) for r in results]
        except Exception:
            logger.warning("BM25 search failed, returning empty results", exc_info=True)
            return []

    @staticmethod
    def _rrf_fuse(
        vector_results: list[tuple[str, float]],
        bm25_results: list[tuple[str, float]],
        k: int = 60,
    ) -> list[tuple[str, float]]:
        """Fuse results using Reciprocal Rank Fusion.

        RRF_score(d) = Σ 1 / (k + rank_i(d))

        where rank_i(d) is the 1-based rank of document d in strategy i.
        Documents not present in a strategy's results are treated as having
        infinite rank (contributing 0 to the sum).
        """
        scores: dict[str, float] = {}

        for rank, (doc_id, _score) in enumerate(vector_results, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)

        for rank, (doc_id, _score) in enumerate(bm25_results, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)

        # Sort by RRF score descending
        fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return fused

    async def _resolve_memories(
        self,
        fused: list[tuple[str, float]],
        vector_results: list[tuple[str, float]],
        bm25_results: list[tuple[str, float]],
        config: RetrievalConfig,
    ) -> list[ScoredMemory]:
        """Resolve memory IDs to full Memory objects with scores."""
        # Build score lookup maps
        vec_scores = {doc_id: score for doc_id, score in vector_results}
        bm25_scores = {doc_id: score for doc_id, score in bm25_results}

        scored_memories: list[ScoredMemory] = []

        for rank, (doc_id, rrf_score) in enumerate(fused, start=1):
            try:
                memory = await self._memory_repo.get(UUID(doc_id))
            except (ValueError, Exception):
                logger.warning("Failed to resolve memory ID: %s", doc_id)
                continue

            if memory is None:
                continue

            # Filter by status
            if not self._should_include(memory, config):
                continue

            scored_memories.append(ScoredMemory(
                memory=memory,
                final_score=rrf_score,
                vector_score=vec_scores.get(doc_id),
                bm25_score=bm25_scores.get(doc_id),
                rrf_rank=rank,
            ))

            # Update access tracking
            await self._memory_repo.update_access(memory.id)

        return scored_memories

    @staticmethod
    def _should_include(memory: Memory, config: RetrievalConfig) -> bool:
        """Check if a memory should be included based on its status and config."""
        if memory.status == MemoryStatus.ACTIVE:
            return True
        if memory.status == MemoryStatus.SUPERSEDED and config.include_superseded:
            return True
        if memory.status == MemoryStatus.CONTRADICTED and config.include_contradicted:
            return True
        if memory.status == MemoryStatus.EXPIRED and config.include_expired:
            return True
        return False

    @staticmethod
    def _deduplicate(memories: list[ScoredMemory]) -> list[ScoredMemory]:
        """Remove exact duplicates (by content_hash) from results.

        Keeps the higher-ranked (earlier in list) memory.
        Also removes superseded memories when their successor is present.
        """
        seen_hashes: set[str] = set()
        seen_ids: set[UUID] = set()
        result: list[ScoredMemory] = []

        # First pass: collect all IDs present
        present_ids = {sm.memory.id for sm in memories}

        for sm in memories:
            # Skip if we've seen this content hash
            if sm.memory.content_hash in seen_hashes:
                continue

            # Skip if this memory is superseded and its successor is in results
            if (
                sm.memory.status == MemoryStatus.SUPERSEDED
                and sm.memory.superseded_by is not None
                and sm.memory.superseded_by in present_ids
            ):
                continue

            seen_hashes.add(sm.memory.content_hash)
            seen_ids.add(sm.memory.id)
            result.append(sm)

        return result
