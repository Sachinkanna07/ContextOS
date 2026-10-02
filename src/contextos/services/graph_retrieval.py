"""Graph-only and rank-fused graph-augmented memory retrieval."""

from __future__ import annotations

import time

from contextos.core.enums import RetrievalMode
from contextos.core.models import (
    RetrievalConfig,
    RetrievalQuery,
    RetrievalResult,
    RetrievalTrace,
    ScoredMemory,
    StageTrace,
)
from contextos.core.protocols import MemoryRepository
from contextos.services.graph import MemoryGraphService
from contextos.services.retrieval import HybridRetrievalEngine


class GraphAugmentedRetrievalEngine:
    """Add graph candidates using rank fusion, never raw-score addition."""

    def __init__(
        self,
        *,
        base_engine: HybridRetrievalEngine,
        graph_service: MemoryGraphService,
        memory_repo: MemoryRepository,
        rrf_k: int = 60,
    ) -> None:
        self._base = base_engine
        self._graph = graph_service
        self._memory_repo = memory_repo
        self._rrf_k = rrf_k

    async def retrieve(
        self,
        query: str | RetrievalQuery,
        config: RetrievalConfig | None = None,
    ) -> RetrievalResult:
        request = HybridRetrievalEngine._coerce_query(query, config)
        if request.mode not in {RetrievalMode.GRAPH, RetrievalMode.HYBRID_GRAPH}:
            return await self._base.retrieve(request, config)

        started = time.perf_counter()
        base_result = RetrievalResult(query=request.text)
        if request.mode == RetrievalMode.HYBRID_GRAPH:
            base_request = request.model_copy(update={
                "mode": RetrievalMode.HYBRID,
                "k": min(200, max(request.k * 3, 20)),
            })
            base_result = await self._base.retrieve(base_request, config)

        graph_started = time.perf_counter()
        expansion = await self._graph.expand(
            query_text=request.text,
            seed_memory_ids=[item.memory.id for item in base_result.memories[:10]],
            max_hops=request.graph_max_hops,
            min_confidence=request.graph_min_confidence,
            max_nodes=request.graph_max_nodes,
            max_edges=request.graph_max_edges,
        )
        graph_ranked = sorted(
            expansion.candidate_scores,
            key=lambda item: (-expansion.candidate_scores[item], str(item)),
        )
        graph_ranks = {memory_id: rank for rank, memory_id in enumerate(graph_ranked, 1)}
        base_ranks = {item.memory.id: rank for rank, item in enumerate(base_result.memories, 1)}
        base_by_id = {item.memory.id: item for item in base_result.memories}

        identifiers = set(graph_ranks)
        if request.mode == RetrievalMode.HYBRID_GRAPH:
            identifiers.update(base_ranks)
        fused: list[ScoredMemory] = []
        for memory_id in identifiers:
            memory = (
                base_by_id[memory_id].memory
                if memory_id in base_by_id
                else await self._memory_repo.get(memory_id)
            )
            if memory is None or not HybridRetrievalEngine._eligible(memory, request):
                continue
            score = 0.0
            sources: list[str] = []
            if memory_id in base_ranks:
                score += 1.0 / (self._rrf_k + base_ranks[memory_id])
                sources.extend(base_by_id[memory_id].retrieval_sources)
            if memory_id in graph_ranks:
                score += 1.0 / (self._rrf_k + graph_ranks[memory_id])
                sources.append("graph")
            original = base_by_id.get(memory_id)
            fused.append(ScoredMemory(
                memory=memory,
                final_score=score,
                vector_score=original.vector_score if original else None,
                bm25_score=original.bm25_score if original else None,
                lexical_rank=original.lexical_rank if original else None,
                dense_rank=original.dense_rank if original else None,
                rrf_rank=base_ranks.get(memory_id, 0),
                metadata_adjustment=original.metadata_adjustment if original else 0.0,
                retrieval_sources=list(dict.fromkeys(sources)),
                graph_score=expansion.candidate_scores.get(memory_id),
                graph_rank=graph_ranks.get(memory_id),
                graph_paths=expansion.candidate_paths.get(memory_id, []),
            ))
        fused.sort(key=lambda item: (-item.final_score, str(item.memory.id)))
        pre_limit_ids = [str(item.memory.id) for item in fused]
        fused = fused[:request.k]
        for rank, item in enumerate(fused, 1):
            item.rank = rank

        graph_stage = StageTrace(
            stage_name="graph_expansion",
            input_count=len(expansion.seed_node_ids),
            output_count=len(graph_ranked),
            latency_ms=(time.perf_counter() - graph_started) * 1000,
            metadata={
                "seed_node_ids": [str(value) for value in expansion.seed_node_ids],
                "visited_node_ids": [str(value) for value in expansion.visited_node_ids],
                "traversed_edge_ids": [str(value) for value in expansion.traversed_edge_ids],
                "max_hops": request.graph_max_hops,
                "min_confidence": request.graph_min_confidence,
                "fusion": "reciprocal_rank",
            },
        )
        stages = [*base_result.trace.stages, graph_stage] if request.include_trace else []
        trace = RetrievalTrace(
            stages=stages,
            total_latency_ms=(time.perf_counter() - started) * 1000,
            total_candidates=len(identifiers),
            total_results=len(fused),
            lexical_candidate_ids=base_result.trace.lexical_candidate_ids,
            dense_candidate_ids=base_result.trace.dense_candidate_ids,
            pre_limit_candidate_ids=pre_limit_ids[:200],
            channel_candidates_truncated=base_result.trace.channel_candidates_truncated,
            pre_limit_candidates_truncated=len(pre_limit_ids) > 200,
        )
        strategies = dict(base_result.strategy_results)
        strategies["graph"] = [item for item in fused if "graph" in item.retrieval_sources]
        return RetrievalResult(
            query=request.text, memories=fused, strategy_results=strategies, trace=trace,
        )
