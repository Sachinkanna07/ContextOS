"""Offline Phase 5 token-selection evaluation and scale smoke test."""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass
from uuid import UUID

from contextos.core.enums import MemoryStatus, MemoryType, OptimizationStrategy
from contextos.core.models import ContextBudget, Memory, ScoredMemory, SelectionResult
from contextos.services.optimization import (
    MemoryContextOptimizer,
    information_tokens,
    redundancy_similarity,
)
from contextos.services.token_counter import DeterministicWordTokenCounter


SMOKE_QUERY = "Which local model should I use given my machine and previous attempts?"
INFORMATION_UNITS = {"runtime", "failed_large", "successful_small", "hardware_limit"}
UNIT_WEIGHTS = {
    "runtime": 1.0,
    "failed_large": 1.2,
    "successful_small": 1.0,
    "hardware_limit": 1.2,
}
STRATEGY_LABELS = {
    OptimizationStrategy.TOP_RANK_STOP: "top_rank_stop",
    OptimizationStrategy.TOP_RANK_SKIP: "top_rank_skip",
    OptimizationStrategy.GREEDY: "greedy",
    OptimizationStrategy.CONTEXTOS: "contextos",
}


@dataclass(frozen=True)
class OptimizationMetrics:
    tokens: int
    utilization: float
    coverage: float
    weighted_coverage: float
    relevance_retained: float
    redundancy: float
    efficiency: float


def information_unit_coverage(
    selected_ids: list[str],
    annotations: dict[str, set[str]],
    required_units: set[str],
) -> float:
    if not required_units:
        return 0.0
    covered = set().union(*(annotations.get(identifier, set()) for identifier in selected_ids))
    return len(covered & required_units) / len(required_units)


def weighted_information_unit_coverage(
    selected_ids: list[str],
    annotations: dict[str, set[str]],
    unit_weights: dict[str, float],
) -> float:
    total = sum(unit_weights.values())
    if total <= 0:
        return 0.0
    covered = set().union(*(annotations.get(identifier, set()) for identifier in selected_ids))
    return sum(weight for unit, weight in unit_weights.items() if unit in covered) / total


def retained_relevance(
    selected: list[ScoredMemory], candidates: list[ScoredMemory]
) -> float:
    total = sum(item.final_score for item in candidates)
    return sum(item.final_score for item in selected) / total if total else 0.0


def redundancy_rate(selected: list[ScoredMemory]) -> float:
    if len(selected) < 2:
        return 0.0
    similarities: list[float] = []
    for index, left in enumerate(selected):
        for right in selected[index + 1:]:
            similarities.append(redundancy_similarity(
                information_tokens(left.memory.content),
                information_tokens(right.memory.content),
            ))
    return sum(similarities) / len(similarities)


def budget_utilization(tokens: int, available_tokens: int) -> float:
    return tokens / available_tokens if available_tokens > 0 else 0.0


def coverage_efficiency(coverage: float, tokens: int) -> float:
    return coverage / tokens if tokens > 0 else 0.0


def _padded(base: str, target: int, repeated_token: str) -> str:
    counter = DeterministicWordTokenCounter()
    current = counter.count(base)
    if current > target:
        raise ValueError("Base content already exceeds target")
    return base + (" " + repeated_token) * (target - current)


def smoke_candidates() -> tuple[list[ScoredMemory], dict[str, set[str]]]:
    specs = [
        (1, _padded("User uses Ollama.", 12, "Ollama"), 0.95, {"runtime"}),
        (
            2,
            _padded(
                "Qwen 30B failed due to insufficient available memory.",
                20,
                "memory",
            ),
            0.93,
            {"failed_large"},
        ),
        (
            3,
            _padded("User successfully runs Qwen 9B locally.", 18, "Qwen"),
            0.90,
            {"successful_small"},
        ),
        (
            4,
            _padded(
                "Historically Qwen 30B failed due to insufficient available memory.",
                180,
                "memory",
            ),
            0.92,
            {"failed_large"},
        ),
        (
            5,
            _padded("User prefers concise responses.", 10, "responses"),
            0.10,
            set(),
        ),
        (
            6,
            _padded("User uses Ollama for local model inference.", 16, "Ollama"),
            0.88,
            {"runtime"},
        ),
        (
            7,
            _padded(
                "User's machine has limited memory for very large models.",
                17,
                "memory",
            ),
            0.87,
            {"hardware_limit"},
        ),
    ]
    candidates: list[ScoredMemory] = []
    annotations: dict[str, set[str]] = {}
    for rank, (number, content, score, units) in enumerate(specs, 1):
        identifier = UUID(f"10000000-0000-0000-0000-{number:012d}")
        status = MemoryStatus.HISTORICAL if number == 4 else MemoryStatus.ACTIVE
        scored = ScoredMemory(
            memory=Memory(
                id=identifier,
                content=content,
                status=status,
                type=MemoryType.FACT,
                importance=0.8,
                confidence=0.9,
            ),
            final_score=score,
            rank=rank,
            retrieval_sources=["lexical", "dense"],
        )
        candidates.append(scored)
        annotations[str(identifier)] = units
    return candidates, annotations


def measure(
    result: SelectionResult,
    candidates: list[ScoredMemory],
    annotations: dict[str, set[str]],
) -> OptimizationMetrics:
    identifiers = [str(item.memory.id) for item in result.selected_memories]
    coverage = information_unit_coverage(identifiers, annotations, INFORMATION_UNITS)
    return OptimizationMetrics(
        tokens=result.total_tokens,
        utilization=result.utilization,
        coverage=coverage,
        weighted_coverage=weighted_information_unit_coverage(
            identifiers, annotations, UNIT_WEIGHTS
        ),
        relevance_retained=retained_relevance(result.selected_memories, candidates),
        redundancy=redundancy_rate(result.selected_memories),
        efficiency=coverage_efficiency(coverage, result.total_tokens),
    )


def synthetic_candidates(count: int = 1_000) -> list[ScoredMemory]:
    return [
        ScoredMemory(
            memory=Memory(
                id=UUID(f"20000000-0000-0000-0000-{index:012d}"),
                content=(
                    f"Memory topic{index} records constraint group{index % 37} "
                    f"tool{index % 83} outcome{index % 19}."
                ),
                status=MemoryStatus.ACTIVE,
                importance=(index % 10) / 10,
                confidence=0.8,
            ),
            final_score=1.0 / (1 + index % 100),
            rank=index + 1,
            retrieval_sources=["dense"] if index % 2 else ["lexical", "dense"],
        )
        for index in range(count)
    ]


def main() -> None:
    counter = DeterministicWordTokenCounter()
    optimizer = MemoryContextOptimizer(token_counter=counter)
    candidates, annotations = smoke_candidates()
    aggregate: dict[OptimizationStrategy, list[OptimizationMetrics]] = {
        strategy: [] for strategy in OptimizationStrategy
    }
    print("Smoke scenario")
    print(
        "Budget  Strategy       IDs                 Tokens  Util   Coverage  "
        "Weighted  Relevance  Redundancy  Efficiency  Exclusions"
    )
    for budget_size in (40, 60, 100, 250):
        for strategy in OptimizationStrategy:
            result = optimizer.optimize(
                SMOKE_QUERY,
                candidates,
                ContextBudget(max_tokens=budget_size),
                strategy,
            )
            metrics = measure(result, candidates, annotations)
            aggregate[strategy].append(metrics)
            ids = ",".join(str(item.memory.id)[-2:] for item in result.selected_memories) or "-"
            exclusions = Counter(
                decision.exclusion_reason.value
                for decision in result.trace.decisions
                if decision.exclusion_reason is not None
            )
            exclusion_text = ",".join(
                f"{reason}:{count}" for reason, count in sorted(exclusions.items())
            ) or "-"
            print(
                f"{budget_size:>6}  {STRATEGY_LABELS[strategy]:<14} {ids:<19} "
                f"{metrics.tokens:>6}  {metrics.utilization:>5.3f}  "
                f"{metrics.coverage:>8.3f}  {metrics.weighted_coverage:>8.3f}  "
                f"{metrics.relevance_retained:>9.3f}  "
                f"{metrics.redundancy:>10.3f}  {metrics.efficiency:>10.4f}  "
                f"{exclusion_text}"
            )

    print("\nAverage evaluation metrics across budgets")
    print(
        "Strategy    Tokens  Utilization  Coverage  Weighted  Relevance  "
        "Redundancy  Efficiency"
    )
    for strategy, rows in aggregate.items():
        count = len(rows)
        print(
            f"{STRATEGY_LABELS[strategy]:<13} "
            f"{sum(row.tokens for row in rows) / count:>6.1f}  "
            f"{sum(row.utilization for row in rows) / count:>11.3f}  "
            f"{sum(row.coverage for row in rows) / count:>8.3f}  "
            f"{sum(row.weighted_coverage for row in rows) / count:>8.3f}  "
            f"{sum(row.relevance_retained for row in rows) / count:>9.3f}  "
            f"{sum(row.redundancy for row in rows) / count:>10.3f}  "
            f"{sum(row.efficiency for row in rows) / count:>10.4f}"
        )

    scale = synthetic_candidates()
    started = time.perf_counter()
    first = optimizer.optimize(
        "Find machine tools and constraints",
        scale,
        ContextBudget(max_tokens=500),
    )
    elapsed_ms = (time.perf_counter() - started) * 1000
    second = optimizer.optimize(
        "Find machine tools and constraints",
        scale,
        ContextBudget(max_tokens=500),
    )
    deterministic = [
        item.memory.id for item in first.selected_memories
    ] == [item.memory.id for item in second.selected_memories]
    print("\nSynthetic scale")
    print(
        f"candidates=1000 selected={len(first.selected_memories)} "
        f"tokens={first.total_tokens}/500 latency_ms={elapsed_ms:.3f} "
        f"deterministic={deterministic}"
    )


if __name__ == "__main__":
    main()
