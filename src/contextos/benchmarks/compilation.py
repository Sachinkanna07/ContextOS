"""Deterministic Phase 6 compiler evaluation and scale benchmark."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from uuid import UUID

from contextos.core.enums import (
    CompilationStrategy,
    CompressionLevel,
    MemoryStatus,
    MemoryType,
)
from contextos.core.models import CompilationConfig, CompiledContext, Memory, ScoredMemory
from contextos.services.compilation import QueryAwareContextCompiler, fact_is_supported
from contextos.services.optimization import information_tokens, redundancy_similarity
from contextos.services.token_counter import DeterministicWordTokenCounter


SMOKE_QUERY = "Which local model should I use given my machine and previous attempts?"
UNIT_TERMS = {
    "runtime": {"ollama"},
    "failed_large": {"qwen", "30b", "failed", "memory"},
    "successful_small": {"qwen", "9b", "successfully"},
}
UNIT_WEIGHTS = {"runtime": 1.0, "failed_large": 1.3, "successful_small": 1.0}


@dataclass(frozen=True)
class CompilationMetrics:
    input_tokens: int
    output_tokens: int
    compression_ratio: float
    information_recall: float
    weighted_preservation: float
    unsupported_fact_rate: float
    redundancy: float
    budget_violations: int
    provenance_coverage: float


def smoke_memories() -> list[ScoredMemory]:
    contents = [
        "User currently uses Ollama for local model inference.",
        (
            "During a previous attempt the user tried Qwen 30B locally, but model "
            "loading failed because the machine did not have enough available memory."
        ),
        "User successfully runs Qwen 9B locally.",
        "User prefers concise technical responses.",
        (
            "Historical debugging notes discussed terminal colors and repeated setup steps. "
            "The logs included unrelated package installation details and console output. "
            "During a previous attempt the user tried Qwen 30B locally, but model "
            "loading failed because the machine did not have enough available memory. "
            "Additional historical notes repeated the same failure without new evidence. "
            "The session ended after reviewing unrelated shell configuration."
        ),
    ]
    return [
        ScoredMemory(
            memory=Memory(
                id=UUID(f"40000000-0000-0000-0000-{index:012d}"),
                content=content,
                status=MemoryStatus.HISTORICAL if index == 5 else MemoryStatus.ACTIVE,
                type=MemoryType.PREFERENCE if index == 4 else MemoryType.FACT,
                confidence=0.9,
                importance=0.8,
            ),
            final_score=1.0 - index * 0.05,
            rank=index,
            retrieval_sources=["lexical", "dense"],
        )
        for index, content in enumerate(contents, 1)
    ]


def information_unit_recall(context: CompiledContext) -> float:
    emitted = information_tokens(" ".join(fact.text for fact in context.facts))
    covered = sum(terms <= emitted for terms in UNIT_TERMS.values())
    return covered / len(UNIT_TERMS)


def weighted_preservation(context: CompiledContext) -> float:
    emitted = information_tokens(" ".join(fact.text for fact in context.facts))
    total = sum(UNIT_WEIGHTS.values())
    preserved = sum(
        weight for unit, weight in UNIT_WEIGHTS.items() if UNIT_TERMS[unit] <= emitted
    )
    return preserved / total


def unsupported_fact_rate(
    context: CompiledContext, source_memories: list[ScoredMemory]
) -> float:
    if not context.facts:
        return 0.0
    sources = {item.memory.id: item.memory.content for item in source_memories}
    unsupported = 0
    for fact in context.facts:
        supported = any(
            fact_is_supported(fact.text, sources.get(source_id, ""))
            for source_id in fact.source_memory_ids
        )
        unsupported += not supported
    return unsupported / len(context.facts)


def fact_redundancy(context: CompiledContext) -> float:
    if len(context.facts) < 2:
        return 0.0
    redundant_pairs = 0
    pair_count = 0
    for index, left in enumerate(context.facts):
        for right in context.facts[index + 1:]:
            pair_count += 1
            similarity = redundancy_similarity(
                information_tokens(left.text),
                information_tokens(right.text),
            )
            redundant_pairs += similarity >= 0.75
    return redundant_pairs / pair_count


def provenance_coverage(context: CompiledContext) -> float:
    if not context.facts:
        return 1.0
    return sum(
        bool(context.provenance_map.get(fact.fact_id)) for fact in context.facts
    ) / len(context.facts)


def budget_violation_rate(contexts: list[CompiledContext]) -> float:
    if not contexts:
        return 0.0
    return sum(context.total_tokens > context.budget for context in contexts) / len(contexts)


def measure(
    context: CompiledContext, source_memories: list[ScoredMemory]
) -> CompilationMetrics:
    return CompilationMetrics(
        input_tokens=context.input_tokens,
        output_tokens=context.total_tokens,
        compression_ratio=context.compression_ratio,
        information_recall=information_unit_recall(context),
        weighted_preservation=weighted_preservation(context),
        unsupported_fact_rate=unsupported_fact_rate(context, source_memories),
        redundancy=fact_redundancy(context),
        budget_violations=int(context.total_tokens > context.budget),
        provenance_coverage=provenance_coverage(context),
    )


def synthetic_memories(count: int = 1_000) -> list[ScoredMemory]:
    return [
        ScoredMemory(
            memory=Memory(
                id=UUID(f"50000000-0000-0000-0000-{index:012d}"),
                content=(
                    f"Project topic{index} currently has constraint group{index % 41}. "
                    f"Unrelated note category{index % 73} is archived."
                ),
                status=MemoryStatus.ACTIVE,
                type=MemoryType.PROJECT,
            ),
            final_score=1.0 / (index + 1),
            rank=index + 1,
        )
        for index in range(count)
    ]


async def main() -> None:
    compiler = QueryAwareContextCompiler(
        token_counter=DeterministicWordTokenCounter()
    )
    memories = smoke_memories()
    strategies = (
        CompilationStrategy.RAW_CONCAT,
        CompilationStrategy.DEDUP_ONLY,
        CompilationStrategy.CONTEXTOS_COMPILER,
    )
    print(
        "Budget  Strategy             In  Out  Ratio  Recall  Weighted  "
        "Unsupported  Redundancy  Violations  Provenance"
    )
    results: dict[tuple[int, CompilationStrategy], CompiledContext] = {}
    for budget in (40, 60, 100, 200):
        for strategy in strategies:
            context = await compiler.compile(
                SMOKE_QUERY,
                memories,
                CompilationConfig(
                    budget=budget,
                    strategy=strategy,
                    compression_level=CompressionLevel.LIGHT,
                ),
            )
            results[(budget, strategy)] = context
            metrics = measure(context, memories)
            print(
                f"{budget:>6}  {strategy.value:<20} "
                f"{metrics.input_tokens:>3}  {metrics.output_tokens:>3}  "
                f"{metrics.compression_ratio:>5.3f}  "
                f"{metrics.information_recall:>6.3f}  "
                f"{metrics.weighted_preservation:>8.3f}  "
                f"{metrics.unsupported_fact_rate:>11.3f}  "
                f"{metrics.redundancy:>10.3f}  "
                f"{metrics.budget_violations:>10}  "
                f"{metrics.provenance_coverage:>10.3f}"
            )

    print("\nExact compiled contexts")
    for budget in (40, 60, 100, 200):
        print(f"\nBudget {budget}")
        for strategy in strategies:
            text = results[(budget, strategy)].context_text or "<empty>"
            print(f"[{strategy.value}]\n{text}")

    scale = synthetic_memories()
    started = time.perf_counter()
    first = await compiler.compile(
        "What project constraints are current?",
        scale,
        CompilationConfig(budget=500),
    )
    latency_ms = (time.perf_counter() - started) * 1000
    second = await compiler.compile(
        "What project constraints are current?",
        scale,
        CompilationConfig(budget=500),
    )
    print("\nSynthetic scale")
    print(
        f"inputs=1000 facts={len(first.facts)} tokens={first.total_tokens}/500 "
        f"latency_ms={latency_ms:.3f} "
        f"deterministic={first.context_text == second.context_text}"
    )


if __name__ == "__main__":
    asyncio.run(main())
