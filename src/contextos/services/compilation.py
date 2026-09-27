"""Deterministic query-aware compilation of selected memory facts."""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Iterable
from dataclasses import dataclass
from uuid import UUID

from contextos.core.enums import (
    CandidateTemporalStatus,
    CompilationStrategy,
    CompilerInputKind,
    CompressionLevel,
    FactExclusionReason,
    MemoryStatus,
    PrivacyLevel,
)
from contextos.core.models import (
    CompilationConfig,
    CompilationTrace,
    CompiledContext,
    ContextFact,
    ExcludedContextFact,
    ScoredMemory,
    SelectionResult,
    StageTrace,
)
from contextos.core.protocols import TokenCounter
from contextos.services.optimization import (
    information_tokens,
    redundancy_similarity,
)


_NEGATION = re.compile(
    r"\b(?:don't|doesn't|do not|does not|did not|never|no longer|not anymore|stopped)\b",
    re.IGNORECASE,
)
_UNCERTAINTY = re.compile(
    r"\b(?:might|may|maybe|perhaps|possibly|could|uncertain|not sure)\b",
    re.IGNORECASE,
)
_HISTORICAL = re.compile(
    r"\b(?:previously|before|formerly|historically|used to|during a previous|stopped)\b",
    re.IGNORECASE,
)
_CURRENT = re.compile(r"\b(?:currently|now|today|still)\b", re.IGNORECASE)
_FUTURE = re.compile(r"\b(?:will|plan to|intends? to|going to|might learn)\b", re.IGNORECASE)
_CAUSAL = re.compile(
    r"\b(?:because|due to|caused by|as a result|therefore|so that)\b",
    re.IGNORECASE,
)
_WHY_QUERY = re.compile(r"\b(?:why|reason|cause|because)\b", re.IGNORECASE)
_CLAUSE_BOUNDARY = re.compile(r"(?<=[.!?])\s+|;\s+|,\s+(?=and\b)")


@dataclass(frozen=True)
class _CompilerInput:
    scored: ScoredMemory
    kind: CompilerInputKind


def _compiler_tokens(text: str) -> frozenset[str]:
    """Compiler-local lexical normalization without changing Phase 5."""
    return frozenset(
        "local" if token == "locally" else token
        for token in information_tokens(text)
    )


def fact_is_supported(fact_text: str, source_text: str) -> bool:
    """Return whether fact tokens occur in source order without additions."""
    pattern = r"[a-z0-9]+(?:[+#._-][a-z0-9]+)*"
    fact_tokens = re.findall(pattern, fact_text.casefold())
    source_tokens = iter(re.findall(pattern, source_text.casefold()))
    return all(any(source == token for source in source_tokens) for token in fact_tokens)


class QueryAwareContextCompiler:
    """Extract, merge, order, and serialize source-supported facts."""

    def __init__(self, *, token_counter: TokenCounter) -> None:
        self._token_counter = token_counter

    async def compile(
        self,
        query: str,
        memories: list[ScoredMemory] | SelectionResult,
        config: CompilationConfig | None = None,
    ) -> CompiledContext:
        cfg = config or CompilationConfig()
        started = time.perf_counter()
        stages: list[StageTrace] = []
        inputs = self._compiler_inputs(memories)
        input_tokens = sum(
            self._token_counter.count(item.scored.memory.content) for item in inputs
        )
        normal_input_count = sum(
            item.kind == CompilerInputKind.NORMAL_SELECTED for item in inputs
        )
        rescue_input_count = len(inputs) - normal_input_count

        ir_started = time.perf_counter()
        facts, excluded = self._build_ir(query, inputs, cfg)
        ir_fact_count = len(facts)
        stages.append(StageTrace(
            stage_name="fact_ir",
            input_count=len(inputs),
            output_count=len(facts),
            latency_ms=(time.perf_counter() - ir_started) * 1000,
            input_tokens=input_tokens,
            metadata={
                "normal_selected": normal_input_count,
                "oversized_rescue": rescue_input_count,
            },
        ))

        dedup_started = time.perf_counter()
        if cfg.strategy != CompilationStrategy.RAW_CONCAT:
            facts, duplicate_exclusions = self._deduplicate(facts)
            excluded.extend(duplicate_exclusions)
        stages.append(StageTrace(
            stage_name="fact_deduplication",
            input_count=len(facts) + len(duplicate_exclusions)
            if cfg.strategy != CompilationStrategy.RAW_CONCAT
            else len(facts),
            output_count=len(facts),
            latency_ms=(time.perf_counter() - dedup_started) * 1000,
        ))

        serialization_started = time.perf_counter()
        included: list[ContextFact] = []
        for fact in facts:
            proposed = self._serialize([*included, fact], cfg.format)
            proposed_tokens = self._token_counter.count(proposed)
            if proposed_tokens <= cfg.budget:
                included.append(fact)
            else:
                excluded.append(ExcludedContextFact(
                    fact_id=fact.fact_id,
                    source_memory_ids=fact.source_memory_ids,
                    input_kind=fact.input_kind,
                    reason=FactExclusionReason.BUDGET,
                    token_cost=fact.token_cost,
                ))
        context_text = self._serialize(included, cfg.format)
        output_tokens = self._token_counter.count(context_text)
        stages.append(StageTrace(
            stage_name="serialization",
            input_count=len(facts),
            output_count=len(included),
            latency_ms=(time.perf_counter() - serialization_started) * 1000,
            input_tokens=sum(fact.token_cost for fact in facts),
            output_tokens=output_tokens,
            metadata={"format": cfg.format},
        ))

        memory_ids = self._ordered_memory_ids(included)
        provenance_map = {
            fact.fact_id: list(fact.source_memory_ids) for fact in included
        }
        provenance_coverage = (
            sum(bool(fact.source_memory_ids) for fact in included) / len(included)
            if included
            else 1.0
        )
        source_texts = {
            item.scored.memory.id: item.scored.memory.content for item in inputs
        }
        unsupported_count = sum(
            not any(
                fact_is_supported(fact.text, source_texts.get(source_id, ""))
                for source_id in fact.source_memory_ids
            )
            for fact in included
        )
        unsupported_rate = unsupported_count / len(included) if included else 0.0
        total_latency = (time.perf_counter() - started) * 1000
        compression_ratio = output_tokens / input_tokens if input_tokens else 0.0
        utilization = output_tokens / cfg.budget if cfg.budget else 0.0

        return CompiledContext(
            query=query,
            context_text=context_text,
            total_tokens=output_tokens,
            budget=cfg.budget,
            memories_considered=len(inputs),
            memories_included=len(memory_ids),
            memories_excluded=len(inputs) - len(memory_ids),
            compression_ratio=compression_ratio,
            included_memory_ids=memory_ids,
            included_fact_ids=[fact.fact_id for fact in included],
            facts=included,
            excluded_facts=excluded,
            provenance_map=provenance_map,
            input_tokens=input_tokens,
            utilization=utilization,
            unsupported_fact_rate=unsupported_rate,
            provenance_coverage=provenance_coverage,
            strategy=cfg.strategy,
            compression_level=cfg.compression_level,
            trace=CompilationTrace(
                stages=stages,
                memories_considered=len(inputs),
                memories_included=len(memory_ids),
                memories_excluded=len(inputs) - len(memory_ids),
                normal_selected_inputs=normal_input_count,
                oversized_rescue_inputs=rescue_input_count,
                rescued_facts_included=sum(
                    fact.input_kind == CompilerInputKind.OVERSIZED_RESCUE
                    for fact in included
                ),
                facts_created=ir_fact_count,
                facts_included=len(included),
                facts_excluded=len(excluded),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                provenance_coverage=provenance_coverage,
                total_latency_ms=total_latency,
            ),
        )

    def _build_ir(
        self,
        query: str,
        memories: list[_CompilerInput],
        config: CompilationConfig,
    ) -> tuple[list[ContextFact], list[ExcludedContextFact]]:
        facts: list[ContextFact] = []
        excluded: list[ExcludedContextFact] = []
        query_terms = _compiler_tokens(query)
        for compiler_input in memories:
            scored = compiler_input.scored
            memory = scored.memory
            if memory.privacy_level == PrivacyLevel.RESTRICTED:
                fact_id = self._fact_id(memory.content, [memory.id])
                excluded.append(ExcludedContextFact(
                    fact_id=fact_id,
                    source_memory_ids=[memory.id],
                    input_kind=compiler_input.kind,
                    reason=FactExclusionReason.PRIVACY_RESTRICTED,
                    token_cost=self._token_counter.count(memory.content),
                ))
                continue

            if (
                compiler_input.kind == CompilerInputKind.NORMAL_SELECTED
                and (
                    config.strategy in {
                        CompilationStrategy.RAW_CONCAT,
                        CompilationStrategy.DEDUP_ONLY,
                    }
                    or config.compression_level == CompressionLevel.NONE
                )
            ):
                texts = [memory.content.strip()]
            else:
                clauses = self._split_clauses(memory.content)
                scored_clauses = [
                    (clause, self._query_relevance(clause, query_terms))
                    for clause in clauses
                ]
                texts = [
                    self._compress_clause(
                        clause,
                        query,
                        query_terms,
                        config.compression_level
                        if config.compression_level != CompressionLevel.NONE
                        else CompressionLevel.LIGHT,
                    )
                    for clause, relevance in scored_clauses
                    if relevance > 0.0
                ]
                texts = [text for text in texts if text]
                if not texts:
                    fact_id = self._fact_id(memory.content, [memory.id])
                    excluded.append(ExcludedContextFact(
                        fact_id=fact_id,
                        source_memory_ids=[memory.id],
                        input_kind=compiler_input.kind,
                        reason=FactExclusionReason.QUERY_IRRELEVANT,
                        token_cost=self._token_counter.count(memory.content),
                    ))
                    continue

            for text in texts:
                fact = self._make_fact(
                    text, scored, query_terms, compiler_input.kind
                )
                facts.append(fact)
        return facts, excluded

    @staticmethod
    def _split_clauses(text: str) -> list[str]:
        return [
            clause.strip()
            for clause in _CLAUSE_BOUNDARY.split(" ".join(text.split()))
            if clause.strip()
        ]

    def _compress_clause(
        self,
        clause: str,
        query: str,
        query_terms: frozenset[str],
        level: CompressionLevel,
    ) -> str:
        text = clause.strip()
        if not _WHY_QUERY.search(query) and not _NEGATION.search(text):
            match = re.search(r"\s+(?:mainly\s+)?for\s+(.+?)([.!?]?)$", text, re.IGNORECASE)
            if match:
                reason_terms = _compiler_tokens(match.group(1))
                if not (reason_terms & query_terms):
                    text = text[:match.start()].rstrip() + match.group(2)
        if (
            level == CompressionLevel.AGGRESSIVE
            and not (_NEGATION.search(text) or _UNCERTAINTY.search(text))
        ):
            text = re.sub(r"^(?:the\s+)?user\s+", "", text, flags=re.IGNORECASE)
            if text:
                text = text[0].upper() + text[1:]
        return text

    def _make_fact(
        self,
        text: str,
        scored: ScoredMemory,
        query_terms: frozenset[str],
        input_kind: CompilerInputKind,
    ) -> ContextFact:
        memory = scored.memory
        temporal = (
            memory.temporal_status
            if memory.temporal_status != CandidateTemporalStatus.UNSPECIFIED
            else self._temporal_status(text, memory.status)
        )
        event_ids = (
            [memory.provenance_event_id] if memory.provenance_event_id is not None else []
        )
        return ContextFact(
            fact_id=self._fact_id(text, [memory.id]),
            text=text,
            source_memory_ids=[memory.id],
            input_kind=input_kind,
            provenance_event_ids=event_ids,
            memory_type=memory.type,
            temporal_status=temporal,
            confidence=memory.confidence,
            importance=memory.importance,
            negated=bool(_NEGATION.search(text)),
            uncertain=bool(_UNCERTAINTY.search(text)),
            causal=bool(_CAUSAL.search(text)),
            query_relevance=self._query_relevance(text, query_terms),
            token_cost=self._token_counter.count(text),
        )

    @staticmethod
    def _query_relevance(text: str, query_terms: frozenset[str]) -> float:
        if not query_terms:
            return 1.0
        fact_terms = _compiler_tokens(text)
        return min(1.0, len(fact_terms & query_terms) / len(query_terms))

    @staticmethod
    def _temporal_status(
        text: str, memory_status: MemoryStatus
    ) -> CandidateTemporalStatus:
        if _HISTORICAL.search(text) or memory_status in {
            MemoryStatus.HISTORICAL,
            MemoryStatus.SUPERSEDED,
        }:
            return CandidateTemporalStatus.HISTORICAL
        if _FUTURE.search(text):
            return CandidateTemporalStatus.FUTURE
        if _CURRENT.search(text) or memory_status == MemoryStatus.ACTIVE:
            return CandidateTemporalStatus.CURRENT
        return CandidateTemporalStatus.UNSPECIFIED

    def _deduplicate(
        self, facts: list[ContextFact]
    ) -> tuple[list[ContextFact], list[ExcludedContextFact]]:
        kept: list[ContextFact] = []
        excluded: list[ExcludedContextFact] = []
        for fact in facts:
            match_index = next(
                (
                    index
                    for index, existing in enumerate(kept)
                    if self._merge_compatible(existing, fact)
                ),
                None,
            )
            if match_index is None:
                kept.append(fact)
                continue
            existing = kept[match_index]
            merged = self._merge_facts(existing, fact)
            kept[match_index] = merged
            excluded.append(ExcludedContextFact(
                fact_id=fact.fact_id,
                source_memory_ids=fact.source_memory_ids,
                input_kind=fact.input_kind,
                reason=FactExclusionReason.DUPLICATE,
                token_cost=fact.token_cost,
            ))
        return kept, excluded

    @staticmethod
    def _merge_compatible(left: ContextFact, right: ContextFact) -> bool:
        if (
            left.memory_type != right.memory_type
            or left.negated != right.negated
            or left.uncertain != right.uncertain
            or left.temporal_status != right.temporal_status
        ):
            return False
        similarity = redundancy_similarity(
            information_tokens(left.text),
            information_tokens(right.text),
        )
        return similarity >= 0.75

    def _merge_facts(self, left: ContextFact, right: ContextFact) -> ContextFact:
        representative = min(
            (left, right),
            key=lambda fact: (
                -self._modifier_evidence(fact),
                fact.token_cost,
                fact.text.casefold(),
                fact.fact_id,
            ),
        )
        source_ids = self._ordered_unique([*left.source_memory_ids, *right.source_memory_ids])
        event_ids = self._ordered_unique(
            [*left.provenance_event_ids, *right.provenance_event_ids]
        )
        return representative.model_copy(update={
            "fact_id": self._fact_id(representative.text, source_ids),
            "source_memory_ids": source_ids,
            "provenance_event_ids": event_ids,
            "confidence": max(left.confidence, right.confidence),
            "importance": max(left.importance, right.importance),
            "query_relevance": max(left.query_relevance, right.query_relevance),
            "input_kind": (
                CompilerInputKind.NORMAL_SELECTED
                if CompilerInputKind.NORMAL_SELECTED
                in {left.input_kind, right.input_kind}
                else CompilerInputKind.OVERSIZED_RESCUE
            ),
        })

    @staticmethod
    def _compiler_inputs(
        memories: list[ScoredMemory] | SelectionResult,
    ) -> list[_CompilerInput]:
        if isinstance(memories, SelectionResult):
            selected = [
                _CompilerInput(item, CompilerInputKind.NORMAL_SELECTED)
                for item in memories.selected_memories
            ]
            selected_ids = {item.scored.memory.id for item in selected}
            rescued = [
                _CompilerInput(item, CompilerInputKind.OVERSIZED_RESCUE)
                for item in memories.compiler_rescue_candidates
                if item.memory.id not in selected_ids
            ]
            return [*selected, *rescued]
        return [
            _CompilerInput(item, CompilerInputKind.NORMAL_SELECTED)
            for item in memories
        ]

    @staticmethod
    def _modifier_evidence(fact: ContextFact) -> int:
        """Prefer merged wording that explicitly carries critical modifiers."""
        score = 0
        if fact.negated and _NEGATION.search(fact.text):
            score += 1
        if fact.uncertain and _UNCERTAINTY.search(fact.text):
            score += 1
        if fact.causal and _CAUSAL.search(fact.text):
            score += 1
        if (
            fact.temporal_status == CandidateTemporalStatus.HISTORICAL
            and _HISTORICAL.search(fact.text)
        ):
            score += 1
        if (
            fact.temporal_status == CandidateTemporalStatus.CURRENT
            and _CURRENT.search(fact.text)
        ):
            score += 1
        if (
            fact.temporal_status == CandidateTemporalStatus.FUTURE
            and _FUTURE.search(fact.text)
        ):
            score += 1
        return score

    @staticmethod
    def _ordered_unique(values: Iterable[UUID]) -> list[UUID]:
        return list(dict.fromkeys(values))

    @staticmethod
    def _ordered_memory_ids(facts: list[ContextFact]) -> list[UUID]:
        return list(dict.fromkeys(
            source_id for fact in facts for source_id in fact.source_memory_ids
        ))

    @staticmethod
    def _fact_id(text: str, source_ids: list[UUID]) -> str:
        normalized = " ".join(text.casefold().split())
        payload = normalized + "|" + "|".join(str(value) for value in source_ids)
        return "cf_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _serialize(facts: list[ContextFact], output_format: str) -> str:
        if not facts:
            return ""
        if output_format == "json":
            return json.dumps(
                {"facts": [fact.text for fact in facts]},
                ensure_ascii=False,
                separators=(",", ":"),
            )
        return "USER CONTEXT\n" + "\n".join(f"- {fact.text}" for fact in facts)


# Compatibility name retained for API and external imports.
GreedyContextCompiler = QueryAwareContextCompiler
