"""Side-effect-free lexical, dense, and hybrid memory retrieval."""

from __future__ import annotations

import time
from collections.abc import Sequence
from uuid import UUID

from contextos.core.enums import (
    CandidateTemporalStatus,
    MemoryStatus,
    RetrievalMode,
    TemporalScope,
)
from contextos.core.models import (
    LexicalResult,
    Memory,
    RetrievalConfig,
    RetrievalQuery,
    RetrievalResult,
    RetrievalTrace,
    ScoredMemory,
    StageTrace,
    VectorResult,
)
from contextos.core.protocols import EmbeddingService, LexicalIndex, MemoryRepository, VectorStore
from contextos.services.retrieval_index import RetrievalIndexSynchronizer


class HybridRetrievalEngine:
    """Retrieve with BM25, cosine similarity, or Reciprocal Rank Fusion.

    RRF combines ranks rather than adding incomparable BM25 and cosine scores.
    A bounded metadata factor can increase the relevance score by at most 5%.
    """

    def __init__(
        self,
        *,
        memory_repo: MemoryRepository,
        vector_store: VectorStore,
        lexical_index: LexicalIndex,
        embedding_service: EmbeddingService,
        index_synchronizer: RetrievalIndexSynchronizer | None = None,
    ) -> None:
        self._memory_repo = memory_repo
        self._vector_store = vector_store
        self._lexical_index = lexical_index
        self._embedding_service = embedding_service
        self._index_synchronizer = index_synchronizer

    async def retrieve(
        self,
        query: str | RetrievalQuery,
        config: RetrievalConfig | None = None,
    ) -> RetrievalResult:
        request = self._coerce_query(query, config)
        started = time.perf_counter()
        stages: list[StageTrace] = []

        if self._index_synchronizer is not None:
            sync_started = time.perf_counter()
            rebuilt = await self._index_synchronizer.ensure_current()
            stages.append(self._stage(
                "index_sync", 0, 0, sync_started, {"rebuilt": rebuilt}
            ))

        corpus_size = max(
            await self._lexical_index.count(),
            await self._vector_store.count(),
        )
        candidate_limit = max(corpus_size, request.k)
        lexical: list[LexicalResult] = []
        dense: list[VectorResult] = []

        if request.mode in {
            RetrievalMode.LEXICAL,
            RetrievalMode.HYBRID,
            RetrievalMode.GRAPH,
            RetrievalMode.HYBRID_GRAPH,
        }:
            stage_started = time.perf_counter()
            lexical = await self._lexical_index.search(request.text, candidate_limit)
            stages.append(self._stage(
                "lexical_search", 1, len(lexical), stage_started, {"top_k": candidate_limit}
            ))

        if request.mode in {
            RetrievalMode.DENSE,
            RetrievalMode.HYBRID,
            RetrievalMode.GRAPH,
            RetrievalMode.HYBRID_GRAPH,
        }:
            stage_started = time.perf_counter()
            vector = await self._embedding_service.embed_query(request.text)
            dense = await self._vector_store.search(vector, candidate_limit)
            stages.append(self._stage(
                "dense_search", 1, len(dense), stage_started, {"top_k": candidate_limit}
            ))

        resolution_started = time.perf_counter()
        ids = {item.id for item in lexical} | {item.id for item in dense}
        memories = await self._eligible_memories(ids, request)
        lexical = [item for item in lexical if item.id in memories]
        dense = [item for item in dense if item.id in memories]
        stages.append(self._stage(
            "eligibility_filter",
            len(ids),
            len(memories),
            resolution_started,
            {"temporal_scope": request.temporal_scope.value},
        ))

        rank_started = time.perf_counter()
        scored = self._rank(request, memories, lexical, dense, rrf_k=(config.rrf_k if config else 60))
        minimum = config.min_score if config else 0.0
        pre_limit_scored = [item for item in scored if item.final_score >= minimum]
        pre_limit_ids = [str(item.memory.id) for item in pre_limit_scored]
        scored = pre_limit_scored[:request.k]
        for rank, item in enumerate(scored, 1):
            item.rank = rank
        stages.append(self._stage(
            "fusion_rerank",
            len(set(item.id for item in lexical) | set(item.id for item in dense)),
            len(scored),
            rank_started,
            {"mode": request.mode.value, "method": "rrf", "metadata_cap": 0.05},
        ))

        total_latency = (time.perf_counter() - started) * 1000
        trace = RetrievalTrace(
            stages=stages if request.include_trace else [],
            total_latency_ms=total_latency,
            total_candidates=len(ids),
            total_results=len(scored),
            lexical_candidate_ids=[str(item.id) for item in lexical[:100]],
            dense_candidate_ids=[str(item.id) for item in dense[:100]],
            pre_limit_candidate_ids=pre_limit_ids[:200],
            channel_candidates_truncated=len(lexical) > 100 or len(dense) > 100,
            pre_limit_candidates_truncated=len(pre_limit_ids) > 200,
        )
        strategy_results = {
            name: [item for item in scored if name in item.retrieval_sources]
            for name in ("lexical", "dense")
            if any(name in item.retrieval_sources for item in scored)
        }
        return RetrievalResult(
            query=request.text,
            memories=scored,
            strategy_results=strategy_results,
            trace=trace,
        )

    @staticmethod
    def _coerce_query(
        query: str | RetrievalQuery, config: RetrievalConfig | None
    ) -> RetrievalQuery:
        if isinstance(query, RetrievalQuery):
            return query
        if config is None:
            return RetrievalQuery(text=query)
        statuses = {MemoryStatus.ACTIVE}
        if config.include_superseded:
            statuses.add(MemoryStatus.SUPERSEDED)
        if config.include_contradicted:
            statuses.add(MemoryStatus.CONTRADICTED)
        if config.include_expired:
            statuses.add(MemoryStatus.EXPIRED)
        return RetrievalQuery(text=query, k=config.max_results, allowed_statuses=statuses)

    async def _eligible_memories(
        self, ids: set[str], request: RetrievalQuery
    ) -> dict[str, Memory]:
        eligible: dict[str, Memory] = {}
        for raw_id in sorted(ids):
            try:
                memory = await self._memory_repo.get(UUID(raw_id))
            except ValueError:
                continue
            if memory is not None and self._eligible(memory, request):
                eligible[raw_id] = memory
        return eligible

    @staticmethod
    def _eligible(memory: Memory, query: RetrievalQuery) -> bool:
        if memory.status in {MemoryStatus.DELETED, MemoryStatus.PURGED, MemoryStatus.MERGED}:
            return False
        if query.allowed_statuses is not None:
            if memory.status not in query.allowed_statuses:
                return False
        else:
            statuses = {
                TemporalScope.CURRENT: {MemoryStatus.ACTIVE},
                TemporalScope.HISTORICAL: {
                    MemoryStatus.HISTORICAL,
                    MemoryStatus.SUPERSEDED,
                },
                TemporalScope.ALL: {
                    MemoryStatus.ACTIVE,
                    MemoryStatus.HISTORICAL,
                    MemoryStatus.SUPERSEDED,
                    MemoryStatus.CONTRADICTED,
                    MemoryStatus.EXPIRED,
                },
            }[query.temporal_scope]
            if memory.status not in statuses:
                return False
        if query.allowed_statuses is None:
            if (
                query.temporal_scope == TemporalScope.CURRENT
                and memory.temporal_status in {
                    CandidateTemporalStatus.FUTURE,
                    CandidateTemporalStatus.HISTORICAL,
                }
            ):
                return False
            if (
                query.temporal_scope == TemporalScope.HISTORICAL
                and memory.temporal_status == CandidateTemporalStatus.FUTURE
            ):
                return False
        if query.allowed_memory_types and memory.type not in query.allowed_memory_types:
            return False
        if query.source_types and memory.source_type not in query.source_types:
            return False
        if query.tags and not query.tags.issubset(memory.tags):
            return False
        if query.created_after and memory.created_at < query.created_after:
            return False
        if query.created_before and memory.created_at > query.created_before:
            return False
        if query.min_confidence is not None and memory.confidence < query.min_confidence:
            return False
        return not (
            query.min_importance is not None and memory.importance < query.min_importance
        )

    @staticmethod
    def _rank(
        query: RetrievalQuery,
        memories: dict[str, Memory],
        lexical: Sequence[LexicalResult],
        dense: Sequence[VectorResult],
        *,
        rrf_k: int,
    ) -> list[ScoredMemory]:
        lexical_ranks = {item.id: rank for rank, item in enumerate(lexical, 1)}
        dense_ranks = {item.id: rank for rank, item in enumerate(dense, 1)}
        lexical_scores = {item.id: item.score for item in lexical}
        dense_scores = {item.id: item.score for item in dense}
        identifiers = set(lexical_ranks) | set(dense_ranks)
        results: list[ScoredMemory] = []
        for identifier in identifiers:
            sources: list[str] = []
            base_score = 0.0
            if identifier in lexical_ranks:
                sources.append("lexical")
                base_score += 1.0 / (rrf_k + lexical_ranks[identifier])
            if identifier in dense_ranks:
                sources.append("dense")
                base_score += 1.0 / (rrf_k + dense_ranks[identifier])
            memory = memories[identifier]
            adjustment = (
                base_score * 0.05 * ((memory.confidence + memory.importance) / 2.0)
                if query.apply_metadata_rerank
                else 0.0
            )
            results.append(ScoredMemory(
                memory=memory,
                final_score=base_score + adjustment,
                vector_score=dense_scores.get(identifier),
                bm25_score=lexical_scores.get(identifier),
                lexical_rank=lexical_ranks.get(identifier),
                dense_rank=dense_ranks.get(identifier),
                metadata_adjustment=adjustment,
                retrieval_sources=sources,
            ))
        results.sort(key=lambda item: (-item.final_score, str(item.memory.id)))
        return results

    @staticmethod
    def _stage(
        name: str,
        input_count: int,
        output_count: int,
        started: float,
        metadata: dict[str, object] | None = None,
    ) -> StageTrace:
        return StageTrace(
            stage_name=name,
            input_count=input_count,
            output_count=output_count,
            latency_ms=(time.perf_counter() - started) * 1000,
            metadata=metadata or {},
        )
