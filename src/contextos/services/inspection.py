"""Bounded RAG inspection built from one Phase 13 explanation execution."""

from __future__ import annotations

import time
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from contextos.core.enums import RetrievalMode
from contextos.core.models import RetrievalQuery
from contextos.services.explainability import (
    ExplainabilityService,
    ExplanationRequest,
    ExplanationTrace,
)
from contextos.services.token_counter import get_token_counter_for_model


class InspectionRequest(ExplanationRequest):
    graph: bool = False
    compare: bool = False
    target_model: str | None = Field(default=None, max_length=200)


class RAGInspection(BaseModel):
    inspection_id: str
    explanation_trace_id: str
    query: dict[str, Any]
    configuration: dict[str, Any]
    stages: list[dict[str, Any]]
    candidates: list[dict[str, Any]]
    context_diff: dict[str, Any]
    final_context: dict[str, Any]
    requested_memory: dict[str, Any] | None = None
    comparison: dict[str, Any] | None = None
    provider_dispatch: dict[str, Any]
    timing: dict[str, float]
    content: str | None = None


class RAGInspector:
    def __init__(self, services: dict[str, Any]) -> None:
        self._services = services
        self._explain: ExplainabilityService = services["explainability"]

    async def inspect(self, request: InspectionRequest) -> RAGInspection:
        started = time.perf_counter()
        trace, retrieved, selection, compiled = await self._explain.execute(request)
        counter = (
            get_token_counter_for_model(request.target_model)
            if request.target_model
            else self._explain.token_counter
        )
        if counter is None:
            raise RuntimeError("Inspection token counter is unavailable")
        tokenizer_name = counter.encoding_name
        if request.target_model and tokenizer_name.startswith("profile-"):
            tokenizer_name = (
                "qwen-profile" if "qwen" in request.target_model.lower() else "claude-profile"
            )
        candidate_tokens = sum(counter.count(row.memory.content) for row in retrieved.memories)
        optimized_tokens = sum(
            counter.count(row.memory.content) for row in selection.selected_memories
        )
        compiled_tokens = counter.count(compiled.context_text)
        avoided = max(0, candidate_tokens - compiled_tokens)
        stages = [
            {
                **stage,
                "removed_count": max(0, stage["input_count"] - stage["output_count"]),
            }
            for stage in trace.stages[:20]
        ]
        duplicate_count = sum(fact.reason.value == "duplicate" for fact in compiled.excluded_facts)
        compiled_provenance = sum(bool(fact.provenance_event_ids) for fact in compiled.facts)
        comparison = await self._compare(request, trace) if request.compare else None
        return RAGInspection(
            inspection_id=str(uuid4()),
            explanation_trace_id=trace.trace_id,
            query={"characters": len(request.query), "content_redacted": True},
            configuration={
                "mode": trace.strategy,
                "graph": trace.strategy
                in {RetrievalMode.GRAPH.value, RetrievalMode.HYBRID_GRAPH.value},
                "budget": request.budget,
                "limit": request.limit,
                "temporal_scope": request.temporal_scope.value,
                "target_model_requested": request.target_model is not None,
            },
            stages=stages,
            candidates=trace.candidates,
            context_diff={
                "candidate_tokens": candidate_tokens,
                "optimized_tokens": optimized_tokens,
                "compiled_tokens": compiled_tokens,
                "tokens_removed": avoided,
                "net_token_change": compiled_tokens - candidate_tokens,
                "reduction_ratio": avoided / candidate_tokens if candidate_tokens else 0.0,
                "selected_memories": len(selection.selected_memories),
                "rejected_memories": max(
                    0, len(retrieved.memories) - len(selection.selected_memories)
                ),
                "facts_emitted": len(compiled.facts),
                "facts_excluded": len(compiled.excluded_facts),
                "duplicate_facts_excluded": duplicate_count,
                "provenance_coverage": (
                    compiled_provenance / len(compiled.facts) if compiled.facts else None
                ),
                "token_measurement_source": counter.measurement_source.value,
                "tokenizer": tokenizer_name,
                "token_basis": "target_model_recount"
                if request.target_model
                else "pipeline_counter_recount",
                "text_exposed": request.include_content,
            },
            final_context=trace.final_context,
            requested_memory=trace.requested_memory,
            comparison=comparison,
            provider_dispatch=trace.provider_dispatch,
            timing={
                "pipeline_ms": trace.measured_pipeline_ms,
                "inspection_ms": (time.perf_counter() - started) * 1000,
            },
            content=trace.content,
        )

    async def _compare(self, request: InspectionRequest, trace: ExplanationTrace) -> dict[str, Any]:
        """Explicit extra retrievals; no optimizer/compiler replay or quality claims."""
        modes = (
            RetrievalMode.LEXICAL,
            RetrievalMode.DENSE,
            RetrievalMode.HYBRID,
            RetrievalMode.HYBRID_GRAPH,
        )
        primary_ids = [row["memory_id"] for row in trace.candidates]
        rank_primary = {memory_id: rank for rank, memory_id in enumerate(primary_ids, 1)}
        rows: list[dict[str, Any]] = []
        for mode in modes:
            started = time.perf_counter()
            result = await self._services["retrieval"].retrieve(
                RetrievalQuery(
                    text=request.query,
                    mode=mode,
                    k=request.limit,
                    temporal_scope=request.temporal_scope,
                )
            )
            ids = [str(item.memory.id) for item in result.memories]
            rows.append(
                {
                    "mode": mode.value,
                    "candidate_ids": ids,
                    "candidate_count": len(ids),
                    "overlap_with_inspection": len(set(ids) & set(primary_ids)),
                    "rank_movement": [
                        {
                            "memory_id": memory_id,
                            "inspection_rank": rank_primary[memory_id],
                            "comparison_rank": rank,
                        }
                        for rank, memory_id in enumerate(ids, 1)
                        if memory_id in rank_primary
                    ],
                    "latency_ms": (time.perf_counter() - started) * 1000,
                }
            )
        return {
            "runs": rows,
            "contextos_final_memory_ids": trace.final_context["memories_contributing"],
            "ground_truth_metrics": None,
            "quality_status": "NOT_AVAILABLE: no ground truth supplied",
        }
