"""Whole-memory token-aware selection after retrieval."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from uuid import UUID

from contextos.core.enums import (
    ExclusionReason,
    MemoryStatus,
    OptimizationStrategy,
)
from contextos.core.models import (
    CandidateDecision,
    ContextBudget,
    OptimizationTrace,
    ScoredMemory,
    SelectionResult,
)
from contextos.core.protocols import TokenCounter


_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "because", "for", "from",
    "has", "in", "is", "it", "of", "on", "or", "that", "the", "to", "user",
    "was", "with",
}
_CANONICAL = {
    "answers": "response",
    "answer": "response",
    "brief": "concise",
    "short": "concise",
    "responses": "response",
    "explanation": "response",
    "explanations": "response",
    "likes": "prefer",
    "prefers": "prefer",
    "uses": "use",
    "used": "use",
    "using": "use",
}
_VALID_STATUSES = {
    MemoryStatus.ACTIVE,
    MemoryStatus.HISTORICAL,
    MemoryStatus.SUPERSEDED,
    MemoryStatus.CONTRADICTED,
}
_LIFECYCLE_FACTOR = {
    MemoryStatus.ACTIVE: 1.0,
    MemoryStatus.HISTORICAL: 0.95,
    MemoryStatus.SUPERSEDED: 0.90,
    MemoryStatus.CONTRADICTED: 0.80,
}


def information_tokens(text: str) -> frozenset[str]:
    """Extract deterministic concept tokens for overlap and novelty."""
    values: set[str] = set()
    for raw in re.findall(r"[a-z0-9]+(?:[+#._-][a-z0-9]+)*", text.casefold()):
        if raw in _STOP_WORDS:
            continue
        values.add(_CANONICAL.get(raw, raw))
    return frozenset(values)


def jaccard_similarity(left: frozenset[str], right: frozenset[str]) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def redundancy_similarity(left: frozenset[str], right: frozenset[str]) -> float:
    """Combine Jaccard with containment to catch concise paraphrase subsets."""
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    containment = len(left & right) / min(len(left), len(right))
    return max(jaccard_similarity(left, right), containment)


@dataclass
class _Candidate:
    scored: ScoredMemory
    index: int
    content_tokens: int
    cost: int
    concepts: frozenset[str]
    normalized_relevance: float = 0.0
    importance_contribution: float = 0.0
    confidence_contribution: float = 0.0
    support_contribution: float = 0.0
    lifecycle_multiplier: float = 0.0
    base_utility: float = 0.0
    marginal_utility: float = 0.0
    redundancy: float = 0.0
    selected: bool = False
    exclusion_reason: ExclusionReason | None = None
    redundant_with: UUID | None = None


class MemoryContextOptimizer:
    """Select whole memories using bounded utility, novelty, and token cost."""

    redundancy_threshold = 0.60
    minimum_relative_relevance = 0.15

    def __init__(self, *, token_counter: TokenCounter) -> None:
        self._token_counter = token_counter

    def optimize(
        self,
        query: str,
        candidates: list[ScoredMemory],
        budget: ContextBudget,
        strategy: OptimizationStrategy = OptimizationStrategy.CONTEXTOS,
    ) -> SelectionResult:
        del query  # Phase 4 scores already encode query relevance.
        started = time.perf_counter()
        prepared = self._prepare(candidates, budget)
        eligible = [
            candidate for candidate in prepared
            if candidate.exclusion_reason is None
        ]

        if strategy == OptimizationStrategy.TOP_RANK_STOP:
            selected = self._top_rank(eligible, budget.available_tokens)
        elif strategy == OptimizationStrategy.TOP_RANK_SKIP:
            selected = self._top_rank_skip(eligible, budget.available_tokens)
        elif strategy == OptimizationStrategy.GREEDY:
            selected = self._greedy(eligible, budget.available_tokens)
        else:
            selected = self._contextos(eligible, budget.available_tokens)

        selected.sort(key=self._output_order)
        content_tokens = sum(candidate.content_tokens for candidate in selected)
        overhead_tokens = len(selected) * budget.overhead_per_memory
        total_tokens = content_tokens + overhead_tokens
        remaining = budget.available_tokens - total_tokens
        decisions = [self._decision(candidate, budget) for candidate in prepared]
        latency_ms = (time.perf_counter() - started) * 1000
        utilization = (
            total_tokens / budget.available_tokens
            if budget.available_tokens
            else 0.0
        )
        trace = OptimizationTrace(
            strategy=strategy,
            candidate_count=len(candidates),
            eligible_count=len(eligible),
            selected_count=len(selected),
            budget_tokens=budget.max_tokens,
            reserved_tokens=budget.reserved_tokens,
            available_tokens=budget.available_tokens,
            tokens_used=total_tokens,
            remaining_tokens=remaining,
            latency_ms=latency_ms,
            decisions=decisions,
        )
        return SelectionResult(
            strategy=strategy,
            selected_memories=[candidate.scored for candidate in selected],
            total_tokens=total_tokens,
            content_tokens=content_tokens,
            overhead_tokens=overhead_tokens,
            budget=budget,
            remaining_tokens=remaining,
            utilization=utilization,
            trace=trace,
        )

    def _prepare(
        self, candidates: list[ScoredMemory], budget: ContextBudget
    ) -> list[_Candidate]:
        prepared = [
            _Candidate(
                scored=scored,
                index=index,
                content_tokens=self._token_counter.count(scored.memory.content),
                cost=self._token_counter.count(scored.memory.content)
                + budget.overhead_per_memory,
                concepts=information_tokens(scored.memory.content),
            )
            for index, scored in enumerate(candidates)
        ]
        best_by_id: dict[UUID, _Candidate] = {}
        for candidate in prepared:
            existing = best_by_id.get(candidate.scored.memory.id)
            if existing is None or self._input_order(candidate) < self._input_order(existing):
                if existing is not None:
                    existing.exclusion_reason = ExclusionReason.DUPLICATE_ID
                best_by_id[candidate.scored.memory.id] = candidate
            else:
                candidate.exclusion_reason = ExclusionReason.DUPLICATE_ID

        eligible = [
            candidate
            for candidate in prepared
            if candidate.exclusion_reason is None
            and candidate.scored.memory.status in _VALID_STATUSES
        ]
        maximum_score = max(
            (candidate.scored.final_score for candidate in eligible),
            default=0.0,
        )
        for candidate in prepared:
            status = candidate.scored.memory.status
            if candidate.exclusion_reason is not None:
                continue
            if status not in _VALID_STATUSES:
                candidate.exclusion_reason = ExclusionReason.INVALID_LIFECYCLE
                continue
            candidate.normalized_relevance = (
                candidate.scored.final_score / maximum_score if maximum_score else 0.0
            )
            candidate.importance_contribution = 0.10 * candidate.scored.memory.importance
            candidate.confidence_contribution = 0.10 * candidate.scored.memory.confidence
            source_count = len(set(candidate.scored.retrieval_sources))
            candidate.support_contribution = 0.05 * min(source_count, 2) / 2
            candidate.lifecycle_multiplier = _LIFECYCLE_FACTOR[status]
            utility = (
                0.75 * candidate.normalized_relevance
                + candidate.importance_contribution
                + candidate.confidence_contribution
                + candidate.support_contribution
            )
            candidate.base_utility = min(1.0, utility) * candidate.lifecycle_multiplier
        return prepared

    @staticmethod
    def _input_order(candidate: _Candidate) -> tuple[float, float, str, int]:
        rank = candidate.scored.rank if candidate.scored.rank > 0 else float("inf")
        return (
            rank,
            -candidate.scored.final_score,
            str(candidate.scored.memory.id),
            candidate.index,
        )

    @staticmethod
    def _output_order(candidate: _Candidate) -> tuple[float, float, str]:
        rank = candidate.scored.rank if candidate.scored.rank > 0 else float("inf")
        return (rank, -candidate.scored.final_score, str(candidate.scored.memory.id))

    def _top_rank(
        self, candidates: list[_Candidate], available: int
    ) -> list[_Candidate]:
        selected: list[_Candidate] = []
        used = 0
        stopped = False
        for candidate in sorted(candidates, key=self._input_order):
            if stopped:
                candidate.exclusion_reason = self._budget_reason(candidate, available)
                continue
            if used + candidate.cost > available:
                candidate.exclusion_reason = self._budget_reason(candidate, available)
                stopped = True
                continue
            candidate.selected = True
            candidate.marginal_utility = candidate.base_utility
            selected.append(candidate)
            used += candidate.cost
        return selected

    def _greedy(
        self, candidates: list[_Candidate], available: int
    ) -> list[_Candidate]:
        selected: list[_Candidate] = []
        used = 0
        ranked = sorted(
            candidates,
            key=lambda candidate: (
                -(candidate.base_utility / max(candidate.cost, 1)),
                *self._input_order(candidate),
            ),
        )
        for candidate in ranked:
            candidate.marginal_utility = candidate.base_utility
            if used + candidate.cost <= available:
                candidate.selected = True
                selected.append(candidate)
                used += candidate.cost
            else:
                candidate.exclusion_reason = self._budget_reason(candidate, available)
        return selected

    def _top_rank_skip(
        self, candidates: list[_Candidate], available: int
    ) -> list[_Candidate]:
        """Select in retrieval order, skipping non-fitting candidates."""
        selected: list[_Candidate] = []
        used = 0
        for candidate in sorted(candidates, key=self._input_order):
            candidate.marginal_utility = candidate.base_utility
            if used + candidate.cost <= available:
                candidate.selected = True
                selected.append(candidate)
                used += candidate.cost
            else:
                candidate.exclusion_reason = self._budget_reason(candidate, available)
        return selected

    def _contextos(
        self, candidates: list[_Candidate], available: int
    ) -> list[_Candidate]:
        selected: list[_Candidate] = []
        remaining = list(candidates)
        used = 0
        covered: set[str] = set()
        while remaining:
            viable: list[tuple[float, _Candidate]] = []
            for candidate in list(remaining):
                if candidate.normalized_relevance < self.minimum_relative_relevance:
                    candidate.exclusion_reason = ExclusionReason.LOW_RELEVANCE
                    remaining.remove(candidate)
                    continue
                redundancy, redundant_with = self._maximum_redundancy(candidate, selected)
                candidate.redundancy = redundancy
                candidate.redundant_with = redundant_with
                if redundancy >= self.redundancy_threshold:
                    candidate.exclusion_reason = ExclusionReason.REDUNDANT
                    remaining.remove(candidate)
                    continue
                novelty = (
                    len(candidate.concepts - covered) / len(candidate.concepts)
                    if candidate.concepts
                    else 0.0
                )
                candidate.marginal_utility = (
                    candidate.base_utility
                    * (0.65 + 0.35 * novelty)
                    * (1.0 - 0.70 * redundancy)
                )
                density = candidate.marginal_utility / max(candidate.cost, 1)
                viable.append((density, candidate))
            if not viable:
                break
            viable.sort(
                key=lambda item: (
                    -item[0],
                    *self._input_order(item[1]),
                )
            )
            candidate = viable[0][1]
            remaining.remove(candidate)
            if used + candidate.cost > available:
                candidate.exclusion_reason = self._budget_reason(candidate, available)
                continue
            candidate.selected = True
            selected.append(candidate)
            used += candidate.cost
            covered.update(candidate.concepts)
        return selected

    @staticmethod
    def _maximum_redundancy(
        candidate: _Candidate, selected: list[_Candidate]
    ) -> tuple[float, UUID | None]:
        best = 0.0
        matched: UUID | None = None
        for existing in selected:
            similarity = redundancy_similarity(candidate.concepts, existing.concepts)
            if similarity > best:
                best = similarity
                matched = existing.scored.memory.id
        return best, matched

    @staticmethod
    def _budget_reason(candidate: _Candidate, available: int) -> ExclusionReason:
        return (
            ExclusionReason.OVERSIZED
            if candidate.cost > available
            else ExclusionReason.BUDGET_EXHAUSTED
        )

    @staticmethod
    def _decision(candidate: _Candidate, budget: ContextBudget) -> CandidateDecision:
        return CandidateDecision(
            memory_id=candidate.scored.memory.id,
            token_cost=candidate.cost,
            content_tokens=candidate.content_tokens,
            overhead_tokens=budget.overhead_per_memory,
            retrieval_score=candidate.scored.final_score,
            normalized_relevance=candidate.normalized_relevance,
            importance_contribution=candidate.importance_contribution,
            confidence_contribution=candidate.confidence_contribution,
            support_contribution=candidate.support_contribution,
            lifecycle_multiplier=candidate.lifecycle_multiplier,
            base_utility=candidate.base_utility,
            marginal_utility=candidate.marginal_utility,
            redundancy=candidate.redundancy,
            selected=candidate.selected,
            exclusion_reason=candidate.exclusion_reason,
            redundant_with=candidate.redundant_with,
        )
