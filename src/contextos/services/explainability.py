"""Deterministic, ephemeral explanations for one existing pipeline execution."""

from __future__ import annotations

import re
import time
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from contextos.core.enums import MemoryStatus, RetrievalMode, TemporalScope
from contextos.core.models import (
    CompilationConfig,
    ContextBudget,
    Memory,
    ProviderDispatchEvidence,
    RetrievalQuery,
    RetrievalResult,
)
from contextos.services.retrieval import HybridRetrievalEngine


class ExplanationRequest(BaseModel):
    query: str = Field(min_length=1, max_length=10_000, repr=False)
    mode: RetrievalMode = RetrievalMode.HYBRID
    budget: int = Field(default=1000, ge=1, le=8_000)
    limit: int = Field(default=25, ge=1, le=100)
    graph: bool = True
    include_content: bool = False
    target_memory_id: UUID | None = None
    temporal_scope: TemporalScope = TemporalScope.CURRENT


class ExplanationTrace(BaseModel):
    trace_id: str
    query_id: str
    strategy: str
    stages: list[dict]
    candidates: list[dict]
    selected: list[str]
    excluded: list[dict]
    final_context: dict
    content: str | None = None
    evidence_scope: str = "retrieved candidates only"
    requested_memory: dict | None = None
    provider_dispatch: dict[str, Any] = Field(
        default_factory=lambda: {
            "state": "NOT_ATTEMPTED",
            "status_message": "prepared by ContextOS; no provider dispatch attempted",
        }
    )
    execution_ms: float = 0.0
    measured_pipeline_ms: float = 0.0
    explanation_overhead_ms: float = 0.0


_UNSAFE = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x1b\x07]*(?:\x07|\x1b\\)|.)")
_BIDI = {chr(value) for value in (*range(0x202A, 0x202F), *range(0x2066, 0x206A))}
_SAFE_SOURCE_TYPES = {
    "cli_input", "cli", "mcp", "api", "file", "local_file", "json_import",
    "fake", "system", "benchmark", "manual",
}


def safe_text(value: object, limit: int = 160) -> str:
    text = _UNSAFE.sub("", str(value or ""))
    return "".join(char for char in text if char.isprintable() and char not in _BIDI)[:limit]


def safe_source_type(value: str) -> str:
    if value in _SAFE_SOURCE_TYPES:
        return value
    if value.startswith("connector:") and value.removeprefix("connector:") in {"local_file", "json_import", "fake"}:
        return value
    return "unknown"


class TemporalEvidenceResolver:
    """Resolve structured, non-inferred temporal evidence for memories and relations."""

    def __init__(self, memory_repo: Any, relation_repo: Any = None) -> None:
        self.memory_repo = memory_repo
        self.relation_repo = relation_repo

    async def resolve_memory_evidence(
        self,
        memory: Memory,
        retrieval_request: RetrievalQuery | None = None,
    ) -> dict[str, Any]:
        eligible = (
            HybridRetrievalEngine._eligible(memory, retrieval_request)
            if retrieval_request is not None
            else (memory.status == MemoryStatus.ACTIVE)
        )
        reason_code = self._reason_code(memory, eligible)

        relations_evidence: list[dict[str, Any]] = []
        if self.relation_repo is not None:
            raw_relations = await self.relation_repo.get_relations(memory.id, direction="both")
            for rel in raw_relations[:50]:
                related_id = (
                    rel.target_memory_id
                    if rel.source_memory_id == memory.id
                    else rel.source_memory_id
                )
                related_mem = await self.memory_repo.get(related_id) if self.memory_repo else None
                target_mem = await self.memory_repo.get(rel.target_memory_id) if self.memory_repo else None
                related_state = (
                    "missing" if related_mem is None else
                    "deleted" if related_mem.status in {MemoryStatus.DELETED, MemoryStatus.PURGED} else
                    "present"
                )
                relations_evidence.append({
                    "relation_id": str(rel.id),
                    "relation_type": rel.relation_type.value,
                    "source_memory_id": str(rel.source_memory_id),
                    "target_memory_id": str(rel.target_memory_id),
                    "related_memory_id": str(related_id),
                    "confidence": rel.confidence,
                    "created_at": rel.created_at.isoformat(),
                    "related_memory_state": related_state,
                    "target_deleted": target_mem is None or target_mem.status in {
                        MemoryStatus.DELETED, MemoryStatus.PURGED,
                    },
                    "acceptance_rationale": None,
                })

        return {
            "status": memory.temporal_status.value,
            "lifecycle": memory.status.value,
            "temporal_status": memory.temporal_status.value,
            "eligible": eligible,
            "reason_code": reason_code,
            "reason": "eligible_by_retrieval_policy" if eligible else "temporal_policy_ineligible",
            "replacement_memory_id": str(memory.superseded_by) if memory.superseded_by else None,
            "observed_at": memory.observed_at.isoformat() if memory.observed_at else None,
            "valid_from": memory.valid_from.isoformat() if memory.valid_from else None,
            "valid_to": memory.valid_to.isoformat() if memory.valid_to else None,
            "relations": relations_evidence,
            "acceptance_rationale": None,
        }

    @staticmethod
    def _reason_code(memory: Memory, eligible: bool) -> str:
        if memory.status == MemoryStatus.SUPERSEDED:
            return "REPLACED_BY_CURRENT_STATE"
        if memory.status == MemoryStatus.HISTORICAL:
            return "HISTORICAL_RECORD"
        if memory.status == MemoryStatus.CONTRADICTED:
            return "CONTRADICTED_STATE"
        if memory.status == MemoryStatus.EXPIRED:
            return "LIFECYCLE_EXPIRED"
        if memory.status in {MemoryStatus.DELETED, MemoryStatus.PURGED}:
            return "LIFECYCLE_DELETED"
        if memory.status == MemoryStatus.ACTIVE:
            return "CURRENT_STATE" if eligible else "ACTIVE_INELIGIBLE_FOR_QUERY"
        return "UNKNOWN_STATE"


class ExplainabilityService:
    """Collect explanations from the same retrieval/optimization/compile results."""

    def __init__(self, services: dict) -> None:
        self.services = services
        self.token_counter = services.get("token_counter") or getattr(
            services.get("optimizer"), "_token_counter", None
        )
        self.temporal_resolver = TemporalEvidenceResolver(
            services.get("memory_repo"), services.get("relation_repo")
        )

    async def explain(self, request: ExplanationRequest) -> ExplanationTrace:
        started = time.perf_counter()
        mode = request.mode
        if not request.graph and mode in {RetrievalMode.GRAPH, RetrievalMode.HYBRID_GRAPH}:
            mode = RetrievalMode.HYBRID
        if request.graph and mode == RetrievalMode.HYBRID:
            mode = RetrievalMode.HYBRID_GRAPH
        retrieval_request = RetrievalQuery(
            text=request.query,
            mode=mode,
            k=request.limit,
            include_trace=True,
            temporal_scope=request.temporal_scope,
        )
        retrieved = await self.services["retrieval"].retrieve(retrieval_request)
        optimizer_started = time.perf_counter()
        selection = self.services["optimizer"].optimize(
            request.query, retrieved.memories, ContextBudget(max_tokens=request.budget)
        )
        optimizer_ms = (time.perf_counter() - optimizer_started) * 1000
        compilation_started = time.perf_counter()
        compiled = await self.services["compilation"].compile(
            request.query, selection, CompilationConfig(budget=request.budget)
        )
        compilation_ms = (time.perf_counter() - compilation_started) * 1000
        pipeline_ms = retrieved.trace.total_latency_ms + optimizer_ms + compilation_ms

        return await self.build_trace(
            request=request,
            retrieval_request=retrieval_request,
            retrieved=retrieved,
            selection=selection,
            compiled=compiled,
            dispatch_evidence=None,
            started_at=started,
            pipeline_ms=pipeline_ms,
        )

    async def build_trace(
        self,
        request: ExplanationRequest,
        retrieved: RetrievalResult,
        selection: Any,
        compiled: Any,
        retrieval_request: RetrievalQuery | None = None,
        dispatch_evidence: ProviderDispatchEvidence | None = None,
        started_at: float | None = None,
        pipeline_ms: float | None = None,
    ) -> ExplanationTrace:
        trace_id = str(uuid4())
        if retrieval_request is None:
            retrieval_request = RetrievalQuery(
                text=request.query,
                mode=request.mode,
                k=request.limit,
                include_trace=True,
                temporal_scope=request.temporal_scope,
            )

        decisions = {item.memory_id: item for item in selection.trace.decisions}
        source_content = {str(item.memory.id): item.memory.content for item in retrieved.memories}
        compiled_memory_ids = {str(value) for value in compiled.included_memory_ids}
        facts_by_memory: dict[str, list[dict]] = {}
        for fact in compiled.facts[:200]:
            for memory_id in fact.source_memory_ids:
                facts_by_memory.setdefault(str(memory_id), []).append({
                    "fact_id": fact.fact_id,
                    "action": (
                        "MERGED" if len(fact.source_memory_ids) > 1 else
                        "RESCUED_FROM_OVERSIZED_MEMORY" if fact.input_kind.value == "oversized_rescue" else
                        "RAW_INCLUDED" if fact.text == source_content.get(str(memory_id)) else "UNKNOWN"
                    ),
                    "token_cost": fact.token_cost,
                    "source_memory_ids": [str(value) for value in fact.source_memory_ids[:20]],
                    "provenance_event_ids": [str(value) for value in fact.provenance_event_ids[:20]],
                    "provenance_preserved": bool(fact.source_memory_ids),
                })
        for fact in compiled.excluded_facts[:200]:
            for memory_id in fact.source_memory_ids:
                facts_by_memory.setdefault(str(memory_id), []).append({
                    "fact_id": fact.fact_id,
                    "action": "DEDUPLICATED" if fact.reason.value == "duplicate" else "DROPPED",
                    "reason": fact.reason.value,
                    "token_cost": fact.token_cost,
                    "source_memory_ids": [str(value) for value in fact.source_memory_ids[:20]],
                    "provenance_event_ids": [],
                    "provenance_preserved": bool(fact.source_memory_ids),
                })

        candidate_rows: list[dict] = []
        excluded: list[dict] = []
        mode = retrieval_request.mode

        for item in retrieved.memories[:request.limit]:
            memory = item.memory
            key = str(memory.id)
            decision = decisions.get(memory.id)
            selected = bool(decision and decision.selected)

            graph_paths_data = []
            for path in item.graph_paths[:5]:
                p_nodes = [
                    {
                        "node_id": str(n.node_id),
                        "node_type": n.node_type.value,
                        "label": safe_text(n.label, 120) if n.label else None,
                        "project_scope": safe_text(n.project_scope, 64) if n.project_scope else None,
                    }
                    for n in getattr(path, "path_nodes", [])[:4]
                ]
                p_edges = [
                    {
                        "edge_type": e.edge_type.value,
                        "confidence": e.confidence,
                        "supporting_memory_ids": [str(x) for x in e.supporting_memory_ids[:20]],
                        "project_scope": safe_text(e.project_scope, 64) if e.project_scope else None,
                    }
                    for e in getattr(path, "path_edges", [])[:3]
                ]
                graph_paths_data.append({
                    "seed_node_ids": [str(value) for value in path.seed_node_ids[:20]],
                    "node_ids": [str(value) for value in path.node_ids[:4]],
                    "node_types": [value.value for value in path.node_types[:4]],
                    "hop_count": path.hop_count,
                    "relation_types": [value.value for value in path.edge_types[:3]],
                    "supporting_memory_ids": [str(value) for value in path.source_memory_ids[:20]],
                    "score": path.graph_contribution,
                    "path_nodes": p_nodes,
                    "path_edges": p_edges,
                    "scope_match": getattr(path, "scope_match", None),
                })

            temporal_data = await self.temporal_resolver.resolve_memory_evidence(memory, retrieval_request)

            row = {
                "memory_id": key,
                "rank": item.rank,
                "status": memory.status.value,
                "confidence": memory.confidence,
                "importance": memory.importance,
                "token_cost": decision.token_cost if decision else None,
                "content_tokens": decision.content_tokens if decision else None,
                "retrieval": {
                    "origin": (
                        "graph_expanded"
                        if "graph" in item.retrieval_sources and not ("lexical" in item.retrieval_sources or "dense" in item.retrieval_sources)
                        else "direct"
                    ),
                    "mode": mode.value,
                    "lexical_rank": item.lexical_rank,
                    "lexical_bm25_score": item.bm25_score,
                    "dense_rank": item.dense_rank,
                    "dense_score": item.vector_score,
                    "lexical_rrf_contribution": (
                        ((1.0 / (60 + item.lexical_rank)) if item.lexical_rank else 0.0)
                        if mode not in {RetrievalMode.GRAPH, RetrievalMode.HYBRID_GRAPH} else None
                    ),
                    "dense_rrf_contribution": (
                        ((1.0 / (60 + item.dense_rank)) if item.dense_rank else 0.0)
                        if mode not in {RetrievalMode.GRAPH, RetrievalMode.HYBRID_GRAPH} else None
                    ),
                    "base_rank": item.rrf_rank if mode in {RetrievalMode.GRAPH, RetrievalMode.HYBRID_GRAPH} else None,
                    "base_rrf_contribution": (1.0 / (60 + item.rrf_rank)) if item.rrf_rank else None,
                    "graph_rrf_contribution": (1.0 / (60 + item.graph_rank)) if item.graph_rank else 0.0,
                    "graph_score": item.graph_score,
                    "graph_rank": item.graph_rank,
                    "fused_score": item.final_score,
                    "metadata_adjustment": item.metadata_adjustment,
                    "sources": list(item.retrieval_sources),
                },
                "graph": graph_paths_data,
                "temporal": temporal_data,
                "optimizer": {
                    "selected": decision.selected if decision else False,
                    "reason": (
                        decision.exclusion_reason.value
                        if decision and decision.exclusion_reason
                        else ("selected" if selected else "not_available")
                    ),
                    "utility": decision.marginal_utility if decision else None,
                    "redundancy": decision.redundancy if decision else None,
                    "redundant_with": str(decision.redundant_with) if decision and decision.redundant_with else None,
                    "budget": selection.trace.available_tokens,
                    "decision": decision.model_dump(mode="json") if decision else None,
                },
                "compiler": facts_by_memory.get(key, []),
                "compiler_transformations": [fact["action"] for fact in facts_by_memory.get(key, [])],
                "selected": selected,
                "compiler_included": key in compiled_memory_ids,
                "provenance": {
                    "source_type": safe_source_type(memory.source_type),
                    "event_id": str(memory.provenance_event_id) if memory.provenance_event_id else None,
                },
            }
            if request.include_content:
                row["content"] = safe_text(memory.content, 1000)
            candidate_rows.append(row)
            if not selected:
                excluded.append({"memory_id": key, "reason": row["optimizer"]["reason"]})

        selected_ids = [str(item.memory.id) for item in selection.selected_memories[:request.limit]]

        requested_memory = None
        if request.target_memory_id is not None:
            requested_id = str(request.target_memory_id)
            matched = next((row for row in candidate_rows if row["memory_id"] == requested_id), None)
            if matched is not None:
                if matched["selected"]:
                    if requested_id in compiled_memory_ids:
                        requested_memory = {
                            "memory_id": requested_id,
                            "status": "retrieved_and_selected",
                            "reason_code": "SELECTED",
                            "reason": "selected_and_compiled",
                        }
                    else:
                        requested_memory = {
                            "memory_id": requested_id,
                            "status": "optimizer_selected_not_compiled",
                            "reason_code": "COMPILER_EXCLUDED",
                            "reason": "not_in_compiled_context",
                        }
                else:
                    requested_memory = {
                        "memory_id": requested_id,
                        "status": "retrieved_but_excluded",
                        "reason_code": "RETRIEVED_BUT_OPTIMIZER_EXCLUDED",
                        "reason": matched["optimizer"]["reason"],
                    }
            elif requested_id in getattr(retrieved.trace, "pre_limit_candidate_ids", []):
                requested_memory = {
                    "memory_id": requested_id,
                    "status": "retrieval_excluded",
                    "reason_code": "RESULT_LIMIT",
                    "reason": "result_limit_exceeded",
                }
            elif (
                getattr(retrieved.trace, "pre_limit_candidates_truncated", False)
                or getattr(retrieved.trace, "channel_candidates_truncated", False)
            ):
                requested_memory = {
                    "memory_id": requested_id,
                    "status": "not_available",
                    "reason_code": "NOT_AVAILABLE",
                    "reason": "candidate_evidence_truncated",
                }
            else:
                repo = self.services.get("memory_repo")
                memory = await repo.get(request.target_memory_id) if repo is not None else None
                if memory is None:
                    requested_memory = {
                        "memory_id": requested_id,
                        "status": "not_available",
                        "reason_code": "NOT_AVAILABLE",
                        "reason": "not_retrieved_or_not_observed",
                    }
                elif not HybridRetrievalEngine._eligible(memory, retrieval_request):
                    requested_memory = {
                        "memory_id": requested_id,
                        "status": "not_retrieved",
                        "reason_code": "TEMPORALLY_INELIGIBLE",
                        "reason": "temporal_policy_ineligible",
                        "lifecycle": memory.status.value,
                        "temporal_status": memory.temporal_status.value,
                    }
                else:
                    lexical_idx = self.services.get("lexical_index") or self.services.get("bm25_index")
                    vector_st = self.services.get("vector_store")
                    in_lexical = bool(
                        lexical_idx and (
                            lexical_idx.contains(requested_id)
                            if hasattr(lexical_idx, "contains")
                            else (requested_id in getattr(lexical_idx, "_documents", {}))
                        )
                    )
                    in_vector = bool(
                        vector_st and (
                            vector_st.contains(requested_id)
                            if hasattr(vector_st, "contains")
                            else (requested_id in getattr(vector_st, "_ids", []))
                        )
                    )
                    if (
                        not in_lexical and not in_vector
                        and retrieval_request.mode not in {
                            RetrievalMode.GRAPH, RetrievalMode.HYBRID_GRAPH,
                        }
                    ):
                        requested_memory = {
                            "memory_id": requested_id,
                            "status": "not_retrieved",
                            "reason_code": "INDEX_NOT_PRESENT",
                            "reason": "absent_from_retrieval_index",
                        }
                    elif retrieval_request.mode in {
                        RetrievalMode.GRAPH, RetrievalMode.HYBRID_GRAPH,
                    } and not in_lexical and not in_vector:
                        requested_memory = {
                            "memory_id": requested_id,
                            "status": "not_available",
                            "reason_code": "NOT_AVAILABLE",
                            "reason": "graph_channel_membership_not_proven",
                        }
                    else:
                        requested_memory = {
                            "memory_id": requested_id,
                            "status": "not_retrieved",
                            "reason_code": "CHANNEL_NOT_RETRIEVED",
                            "reason": "channel_not_retrieved",
                        }

        provider_dispatch_data: dict[str, Any]
        if dispatch_evidence is not None:
            provider_dispatch_data = dispatch_evidence.model_dump(mode="json")
        else:
            provider_dispatch_data = {
                "state": "NOT_ATTEMPTED",
                "status_message": "prepared by ContextOS; no provider dispatch attempted",
            }

        measured_pipeline = pipeline_ms if pipeline_ms is not None else (
            retrieved.trace.total_latency_ms + selection.trace.latency_ms + getattr(compiled, "total_tokens", 0)
        )
        execution_ms = (time.perf_counter() - started_at) * 1000 if started_at else 0.0

        return ExplanationTrace(
            trace_id=trace_id,
            query_id=trace_id,
            strategy=mode.value,
            stages=[
                {"name": safe_text(stage.stage_name, 64), "input_count": stage.input_count,
                 "output_count": stage.output_count, "latency_ms": stage.latency_ms}
                for stage in retrieved.trace.stages[:20]
            ] + [
                {"name": "optimizer", "input_count": selection.trace.candidate_count,
                 "output_count": selection.trace.selected_count,
                 "latency_ms": selection.trace.latency_ms,
                 "tokens_used": selection.trace.tokens_used,
                 "token_budget": selection.trace.available_tokens},
                *[{"name": safe_text(stage.stage_name, 64), "input_count": stage.input_count,
                   "output_count": stage.output_count, "latency_ms": stage.latency_ms,
                   "input_tokens": stage.input_tokens, "output_tokens": stage.output_tokens}
                  for stage in compiled.trace.stages[:15]],
            ][:20],
            candidates=candidate_rows,
            selected=selected_ids,
            excluded=excluded[:request.limit],
            final_context={
                "candidate_count": retrieved.trace.total_candidates,
                "candidate_tokens": sum(row["content_tokens"] or 0 for row in candidate_rows),
                "optimized_tokens": selection.total_tokens,
                "compiled_tokens": compiled.total_tokens,
                "token_budget": compiled.budget,
                "token_measurement_source": (
                    self.token_counter.measurement_source.value if self.token_counter else "unknown"
                ),
                "tokenizer": self.token_counter.encoding_name if self.token_counter else "unknown",
                "facts_emitted": len(compiled.facts),
                "memories_contributing": [str(value) for value in compiled.included_memory_ids[:request.limit]],
                "graph_contribution_count": sum(row["retrieval"]["origin"] == "graph_expanded" for row in candidate_rows),
                "temporal_filter_count": next(
                    (stage.input_count - stage.output_count
                     for stage in retrieved.trace.stages
                     if stage.stage_name == "eligibility_filter"),
                    0,
                ),
            },
            content=compiled.context_text[:20_000] if request.include_content else None,
            evidence_scope="retrieved candidates only",
            requested_memory=requested_memory,
            provider_dispatch=provider_dispatch_data,
            execution_ms=execution_ms,
            measured_pipeline_ms=measured_pipeline,
            explanation_overhead_ms=max(0.0, execution_ms - measured_pipeline),
        )
