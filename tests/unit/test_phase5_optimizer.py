"""Phase 5 token-aware selection acceptance tests."""

from __future__ import annotations

from uuid import UUID

import pytest

from contextos.benchmarks.optimization import (
    budget_utilization,
    coverage_efficiency,
    information_unit_coverage,
    redundancy_rate,
    synthetic_candidates,
    weighted_information_unit_coverage,
)
from contextos.core.enums import (
    ExclusionReason,
    MemoryStatus,
    OptimizationStrategy,
)
from contextos.core.models import ContextBudget, Memory, ScoredMemory
from contextos.services.optimization import MemoryContextOptimizer
from contextos.services.token_counter import DeterministicWordTokenCounter


def candidate(
    number: int,
    content: str,
    score: float = 1.0,
    *,
    rank: int = 0,
    importance: float = 0.5,
    confidence: float = 0.8,
    status: MemoryStatus = MemoryStatus.ACTIVE,
) -> ScoredMemory:
    return ScoredMemory(
        memory=Memory(
            id=UUID(f"30000000-0000-0000-0000-{number:012d}"),
            content=content,
            status=status,
            importance=importance,
            confidence=confidence,
        ),
        final_score=score,
        rank=rank,
        retrieval_sources=["lexical", "dense"],
    )


@pytest.fixture
def counter() -> DeterministicWordTokenCounter:
    return DeterministicWordTokenCounter()


@pytest.fixture
def optimizer(counter) -> MemoryContextOptimizer:
    return MemoryContextOptimizer(token_counter=counter)


@pytest.mark.parametrize("strategy", list(OptimizationStrategy))
@pytest.mark.parametrize("budget", [0, 3, 7, 20, 100])
def test_never_exceeds_budget(optimizer, strategy, budget):
    candidates = [
        candidate(1, "alpha beta gamma", 1.0),
        candidate(2, "delta epsilon zeta eta", 0.8),
    ]
    result = optimizer.optimize("query", candidates, ContextBudget(max_tokens=budget), strategy)
    assert result.total_tokens <= result.budget.available_tokens


def test_zero_budget(optimizer):
    result = optimizer.optimize(
        "query", [candidate(1, "alpha")], ContextBudget(max_tokens=0)
    )
    assert result.selected_memories == []
    assert result.remaining_tokens == 0


def test_empty_candidate_list(optimizer):
    result = optimizer.optimize("query", [], ContextBudget(max_tokens=20))
    assert result.total_tokens == 0
    assert result.trace.candidate_count == 0


def test_one_fitting_candidate(optimizer):
    item = candidate(1, "alpha beta")
    result = optimizer.optimize("query", [item], ContextBudget(max_tokens=4))
    assert [entry.memory.id for entry in result.selected_memories] == [item.memory.id]


def test_one_oversized_candidate(optimizer):
    result = optimizer.optimize(
        "query",
        [candidate(1, "alpha beta gamma")],
        ContextBudget(max_tokens=4, overhead_per_memory=2),
    )
    assert result.selected_memories == []
    assert result.trace.decisions[0].exclusion_reason == ExclusionReason.OVERSIZED


def test_exact_budget_boundary(optimizer):
    result = optimizer.optimize(
        "query",
        [candidate(1, "alpha beta gamma")],
        ContextBudget(max_tokens=5, overhead_per_memory=2),
    )
    assert result.total_tokens == 5
    assert result.remaining_tokens == 0


def test_top_rank_stops_at_first_nonfitting_item(optimizer):
    items = [
        candidate(1, "alpha beta", rank=1),
        candidate(2, "one two three four five six", rank=2),
        candidate(3, "tiny", rank=3),
    ]
    result = optimizer.optimize(
        "query", items, ContextBudget(max_tokens=8), OptimizationStrategy.TOP_RANK
    )
    assert [item.memory.id for item in result.selected_memories] == [items[0].memory.id]


def test_top_rank_skip_skips_oversized_high_ranked_candidate(optimizer):
    oversized = candidate(1, "one two three four five six seven", rank=1)
    fitting = candidate(2, "small fact", rank=2)
    result = optimizer.optimize(
        "query",
        [oversized, fitting],
        ContextBudget(max_tokens=5, overhead_per_memory=2),
        OptimizationStrategy.TOP_RANK_SKIP,
    )
    assert [item.memory.id for item in result.selected_memories] == [fitting.memory.id]
    assert result.trace.decisions[0].exclusion_reason == ExclusionReason.OVERSIZED


def test_top_rank_skip_continues_to_fitting_lower_rank(optimizer):
    items = [
        candidate(1, "first fact", rank=1),
        candidate(2, "second has too many words here", rank=2),
        candidate(3, "third", rank=3),
    ]
    result = optimizer.optimize(
        "query",
        items,
        ContextBudget(max_tokens=8, overhead_per_memory=2),
        OptimizationStrategy.TOP_RANK_SKIP,
    )
    assert [item.memory.id for item in result.selected_memories] == [
        items[0].memory.id,
        items[2].memory.id,
    ]
    assert result.total_tokens <= 8


def test_top_rank_skip_preserves_original_ranking_order(optimizer):
    items = [
        candidate(3, "rank three", rank=3),
        candidate(1, "rank one", rank=1),
        candidate(2, "rank two", rank=2),
    ]
    result = optimizer.optimize(
        "query",
        items,
        ContextBudget(max_tokens=20),
        OptimizationStrategy.TOP_RANK_SKIP,
    )
    assert [item.memory.id for item in result.selected_memories] == [
        items[1].memory.id,
        items[2].memory.id,
        items[0].memory.id,
    ]


def test_top_rank_skip_does_not_rerank_by_relevance_or_utility(optimizer):
    high_rank_low_score = candidate(
        1,
        "first compact fact",
        0.1,
        rank=1,
        importance=0.0,
        confidence=0.0,
    )
    low_rank_high_score = candidate(
        2,
        "second compact fact",
        1.0,
        rank=2,
        importance=1.0,
        confidence=1.0,
    )
    result = optimizer.optimize(
        "query",
        [high_rank_low_score, low_rank_high_score],
        ContextBudget(max_tokens=5, overhead_per_memory=2),
        OptimizationStrategy.TOP_RANK_SKIP,
    )
    assert [item.memory.id for item in result.selected_memories] == [
        high_rank_low_score.memory.id
    ]


def test_greedy_utility_per_token_baseline(optimizer):
    items = [
        candidate(1, "long relevant fact with several extra words", 1.0, rank=1),
        candidate(2, "compact fact", 0.8, rank=2),
    ]
    result = optimizer.optimize(
        "query", items, ContextBudget(max_tokens=4), OptimizationStrategy.GREEDY
    )
    assert result.selected_memories[0].memory.id == items[1].memory.id


def test_optimized_selection_is_deterministic(optimizer):
    items = [
        candidate(1, "runtime ollama", 0.9),
        candidate(2, "hardware memory limit", 0.8),
    ]
    first = optimizer.optimize("query", items, ContextBudget(max_tokens=20))
    second = optimizer.optimize("query", items, ContextBudget(max_tokens=20))
    assert [item.memory.id for item in first.selected_memories] == [
        item.memory.id for item in second.selected_memories
    ]


def test_short_relevant_beats_long_equally_relevant(optimizer):
    short = candidate(1, "qwen failure memory", 1.0)
    long = candidate(2, "qwen failure memory " + "memory " * 30, 1.0)
    result = optimizer.optimize("query", [long, short], ContextBudget(max_tokens=10))
    assert [item.memory.id for item in result.selected_memories] == [short.memory.id]


def test_irrelevant_tiny_memory_not_selected_because_cheap(optimizer):
    relevant = candidate(1, "hardware memory limitation", 1.0)
    tiny = candidate(2, "biryani", 0.01)
    result = optimizer.optimize("query", [relevant, tiny], ContextBudget(max_tokens=20))
    assert [item.memory.id for item in result.selected_memories] == [relevant.memory.id]
    assert result.trace.decisions[1].exclusion_reason == ExclusionReason.LOW_RELEVANCE


def test_redundant_candidates_penalized(optimizer):
    first = candidate(1, "User prefers concise answers.", 1.0)
    duplicate = candidate(2, "User likes short responses.", 0.9)
    result = optimizer.optimize("query", [first, duplicate], ContextBudget(max_tokens=50))
    assert len(result.selected_memories) == 1
    rejected = next(decision for decision in result.trace.decisions if not decision.selected)
    assert rejected.exclusion_reason == ExclusionReason.REDUNDANT
    assert rejected.redundant_with is not None


def test_complementary_candidates_preserved(optimizer):
    items = [
        candidate(1, "Ollama local runtime", 1.0),
        candidate(2, "Machine memory limitation", 0.9),
        candidate(3, "Qwen 9B successful", 0.8),
    ]
    result = optimizer.optimize("query", items, ContextBudget(max_tokens=30))
    assert len(result.selected_memories) == 3


def test_importance_contribution_is_bounded(optimizer):
    items = [
        candidate(1, "alpha", 1.0, importance=0.0),
        candidate(2, "beta", 1.0, importance=1.0),
    ]
    result = optimizer.optimize("query", items, ContextBudget(max_tokens=20))
    contributions = [decision.importance_contribution for decision in result.trace.decisions]
    assert max(contributions) == 0.1
    assert min(contributions) == 0.0


def test_confidence_contribution_is_bounded(optimizer):
    items = [
        candidate(1, "alpha", 1.0, confidence=0.0),
        candidate(2, "beta", 1.0, confidence=1.0),
    ]
    result = optimizer.optimize("query", items, ContextBudget(max_tokens=20))
    contributions = [decision.confidence_contribution for decision in result.trace.decisions]
    assert max(contributions) == 0.1
    assert min(contributions) == 0.0


@pytest.mark.parametrize(
    "status",
    [MemoryStatus.CANDIDATE, MemoryStatus.EXPIRED, MemoryStatus.DELETED, MemoryStatus.MERGED],
)
def test_invalid_lifecycle_is_excluded(optimizer, status):
    result = optimizer.optimize(
        "query", [candidate(1, "alpha", status=status)], ContextBudget(max_tokens=20)
    )
    assert result.selected_memories == []
    assert result.trace.decisions[0].exclusion_reason == ExclusionReason.INVALID_LIFECYCLE


def test_active_memory_wins_current_historical_conflict(optimizer):
    historical = candidate(
        1,
        "primary language python",
        1.0,
        status=MemoryStatus.SUPERSEDED,
    )
    active = candidate(2, "primary language C++17", 1.0)
    result = optimizer.optimize(
        "current primary language",
        [historical, active],
        ContextBudget(max_tokens=6, overhead_per_memory=2),
    )
    assert [item.memory.id for item in result.selected_memories] == [active.memory.id]


def test_token_count_is_deterministic_and_current(counter, optimizer):
    text = "C++17, Ollama: local inference."
    assert counter.count(text) == counter.count(text) == 7
    item = candidate(1, text)
    item.memory.token_count = 999
    result = optimizer.optimize("query", [item], ContextBudget(max_tokens=20))
    assert result.trace.decisions[0].content_tokens == 7


def test_token_overhead_and_reserved_budget_accounted(optimizer):
    result = optimizer.optimize(
        "query",
        [candidate(1, "alpha beta")],
        ContextBudget(max_tokens=10, reserved_tokens=3, overhead_per_memory=4),
    )
    assert result.content_tokens == 2
    assert result.overhead_tokens == 4
    assert result.total_tokens == 6
    assert result.remaining_tokens == 1


def test_remaining_budget_correct(optimizer):
    result = optimizer.optimize(
        "query",
        [candidate(1, "alpha beta")],
        ContextBudget(max_tokens=10, overhead_per_memory=2),
    )
    assert result.remaining_tokens == 6


def test_trace_counts_and_no_raw_content(optimizer):
    result = optimizer.optimize(
        "query",
        [candidate(1, "alpha"), candidate(2, "beta")],
        ContextBudget(max_tokens=20),
    )
    assert result.trace.candidate_count == 2
    assert result.trace.eligible_count == 2
    assert result.trace.selected_count == 2
    assert "alpha" not in result.trace.model_dump_json()


def test_selection_does_not_mutate_retrieval_objects(optimizer):
    items = [candidate(1, "alpha"), candidate(2, "beta")]
    before = [item.model_dump() for item in items]
    optimizer.optimize("query", items, ContextBudget(max_tokens=20))
    assert [item.model_dump() for item in items] == before


def test_budget_larger_than_corpus(optimizer):
    items = [candidate(1, "alpha unique"), candidate(2, "beta distinct")]
    result = optimizer.optimize("query", items, ContextBudget(max_tokens=1_000))
    assert len(result.selected_memories) == 2
    assert result.remaining_tokens == 1_000 - result.total_tokens


def test_ties_use_stable_memory_id_order(optimizer):
    first = candidate(2, "beta", 1.0)
    second = candidate(1, "alpha", 1.0)
    result = optimizer.optimize(
        "query", [first, second], ContextBudget(max_tokens=20)
    )
    assert [item.memory.id for item in result.selected_memories] == [
        second.memory.id,
        first.memory.id,
    ]


def test_duplicate_memory_ids_are_deduplicated(optimizer):
    first = candidate(1, "alpha", 1.0)
    duplicate = first.model_copy(deep=True)
    duplicate.final_score = 0.5
    result = optimizer.optimize(
        "query", [first, duplicate], ContextBudget(max_tokens=20)
    )
    assert len(result.selected_memories) == 1
    assert any(
        decision.exclusion_reason == ExclusionReason.DUPLICATE_ID
        for decision in result.trace.decisions
    )


def test_information_unit_coverage_metric():
    annotations = {"a": {"runtime"}, "b": {"hardware", "failure"}}
    assert information_unit_coverage(
        ["a", "b"], annotations, {"runtime", "hardware", "failure", "success"}
    ) == 0.75
    assert information_unit_coverage([], annotations, set()) == 0.0


def test_weighted_coverage_metric():
    annotations = {"a": {"critical"}, "b": {"minor"}}
    assert weighted_information_unit_coverage(
        ["a"], annotations, {"critical": 3.0, "minor": 1.0}
    ) == 0.75


def test_redundancy_metric():
    items = [
        candidate(1, "concise response"),
        candidate(2, "short answer"),
    ]
    assert redundancy_rate(items) == 1.0
    assert redundancy_rate(items[:1]) == 0.0


def test_utilization_and_efficiency_metrics():
    assert budget_utilization(40, 100) == 0.4
    assert budget_utilization(0, 0) == 0.0
    assert coverage_efficiency(0.75, 30) == 0.025
    assert coverage_efficiency(0.0, 0) == 0.0


def test_hand_calculated_optimization_example(optimizer):
    items = [
        candidate(1, "alpha beta", 1.0),
        candidate(2, "gamma delta", 0.5),
        candidate(3, "epsilon zeta eta theta", 0.9),
    ]
    result = optimizer.optimize(
        "query",
        items,
        ContextBudget(max_tokens=8, overhead_per_memory=2),
        OptimizationStrategy.GREEDY,
    )
    assert [item.memory.id for item in result.selected_memories] == [
        items[0].memory.id,
        items[1].memory.id,
    ]
    assert result.total_tokens == 8


def test_adversarial_many_small_irrelevant_memories(optimizer):
    relevant = candidate(1, "machine memory model constraint", 1.0)
    distractors = [
        candidate(index + 2, f"snack{index}", 0.001)
        for index in range(100)
    ]
    result = optimizer.optimize(
        "query", [*distractors, relevant], ContextBudget(max_tokens=30)
    )
    assert [item.memory.id for item in result.selected_memories] == [relevant.memory.id]


def test_realistic_scale_is_deterministic_and_budget_safe(optimizer):
    items = synthetic_candidates(1_000)
    budget = ContextBudget(max_tokens=500)
    first = optimizer.optimize("machine constraints", items, budget)
    second = optimizer.optimize("machine constraints", items, budget)
    assert first.total_tokens <= 500
    assert [item.memory.id for item in first.selected_memories] == [
        item.memory.id for item in second.selected_memories
    ]
