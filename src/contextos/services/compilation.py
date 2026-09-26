"""Context compiler for ContextOS.

Implements budget-constrained context assembly from retrieved memories.
Phase 1: Greedy knapsack approach — maximizes value_density (score / token_cost).

This is the module that directly impacts token savings — the core value
proposition of ContextOS.
"""

from __future__ import annotations

import logging
import time

from contextos.core.enums import PrivacyLevel
from contextos.core.models import (
    CompiledContext,
    CompilationConfig,
    CompilationTrace,
    ScoredMemory,
    StageTrace,
)
from contextos.core.protocols import TokenCounter

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = CompilationConfig()


class GreedyContextCompiler:
    """Phase 1 context compiler using greedy knapsack strategy.

    Strategy:
    1. Compute value_density = (retrieval_score * importance * confidence) / token_count
    2. Sort by value_density descending.
    3. Greedily add memories until budget is exhausted.
    4. Format selected memories into a context string.

    Implements the CompilationService protocol.
    """

    def __init__(self, *, token_counter: TokenCounter) -> None:
        self._token_counter = token_counter

    async def compile(
        self,
        query: str,
        memories: list[ScoredMemory],
        config: CompilationConfig | None = None,
    ) -> CompiledContext:
        """Compile retrieved memories into budget-constrained context."""
        cfg = config or DEFAULT_CONFIG
        trace_stages: list[StageTrace] = []
        t0 = time.perf_counter()

        # --- Stage 1: Privacy Filter ---
        t_filter = time.perf_counter()
        filtered = self._privacy_filter(memories)
        filter_latency = (time.perf_counter() - t_filter) * 1000
        trace_stages.append(StageTrace(
            stage_name="privacy_filter",
            input_count=len(memories),
            output_count=len(filtered),
            latency_ms=filter_latency,
            metadata={"removed": len(memories) - len(filtered)},
        ))

        # --- Stage 2: Value Density Ranking ---
        t_rank = time.perf_counter()
        ranked = self._rank_by_value_density(filtered)
        rank_latency = (time.perf_counter() - t_rank) * 1000

        value_densities = {
            str(sm.memory.id): vd for sm, vd in ranked
        }

        trace_stages.append(StageTrace(
            stage_name="value_density_ranking",
            input_count=len(filtered),
            output_count=len(ranked),
            latency_ms=rank_latency,
            metadata={"top_5_densities": dict(list(value_densities.items())[:5])},
        ))

        # --- Stage 3: Budget-Constrained Selection ---
        t_select = time.perf_counter()
        selected, total_candidate_tokens = self._select_within_budget(ranked, cfg.budget)
        select_latency = (time.perf_counter() - t_select) * 1000

        trace_stages.append(StageTrace(
            stage_name="budget_selection",
            input_count=len(ranked),
            output_count=len(selected),
            latency_ms=select_latency,
            input_tokens=total_candidate_tokens,
            metadata={"budget": cfg.budget},
        ))

        # --- Stage 4: Format Output ---
        t_format = time.perf_counter()
        context_text = self._format_context(selected, cfg)
        compiled_tokens = self._token_counter.count(context_text)
        format_latency = (time.perf_counter() - t_format) * 1000

        trace_stages.append(StageTrace(
            stage_name="formatting",
            input_count=len(selected),
            output_count=1,
            latency_ms=format_latency,
            output_tokens=compiled_tokens,
        ))

        total_latency = (time.perf_counter() - t0) * 1000

        # Compute compression ratio
        compression_ratio = (
            compiled_tokens / total_candidate_tokens
            if total_candidate_tokens > 0
            else 0.0
        )

        return CompiledContext(
            query=query,
            context_text=context_text,
            total_tokens=compiled_tokens,
            budget=cfg.budget,
            memories_considered=len(memories),
            memories_included=len(selected),
            memories_excluded=len(memories) - len(selected),
            compression_ratio=compression_ratio,
            included_memory_ids=[sm.memory.id for sm in selected],
            trace=CompilationTrace(
                stages=trace_stages,
                memories_considered=len(memories),
                memories_included=len(selected),
                memories_excluded=len(memories) - len(selected),
                value_densities=value_densities,
                total_latency_ms=total_latency,
            ),
        )

    # --- Internal Methods ---

    @staticmethod
    def _privacy_filter(memories: list[ScoredMemory]) -> list[ScoredMemory]:
        """Remove RESTRICTED memories from compilation candidates.

        RESTRICTED memories must never be sent to external LLMs.
        """
        return [
            sm for sm in memories
            if sm.memory.privacy_level != PrivacyLevel.RESTRICTED
        ]

    @staticmethod
    def _rank_by_value_density(
        memories: list[ScoredMemory],
    ) -> list[tuple[ScoredMemory, float]]:
        """Compute value_density and sort descending.

        value = retrieval_score * importance * confidence
        cost = token_count (minimum 1 to avoid division by zero)
        value_density = value / cost
        """
        ranked: list[tuple[ScoredMemory, float]] = []

        for sm in memories:
            value = sm.final_score * sm.memory.importance * sm.memory.confidence
            cost = max(1, sm.memory.token_count)
            density = value / cost
            ranked.append((sm, density))

        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked

    def _select_within_budget(
        self,
        ranked: list[tuple[ScoredMemory, float]],
        budget: int,
    ) -> tuple[list[ScoredMemory], int]:
        """Greedily select memories that fit within the token budget.

        Returns (selected_memories, total_candidate_tokens).
        """
        selected: list[ScoredMemory] = []
        tokens_used = 0
        total_candidate_tokens = sum(sm.memory.token_count for sm, _ in ranked)

        # Reserve tokens for formatting overhead (numbering, newlines, labels)
        # Estimate: ~10 tokens per memory for formatting
        formatting_overhead_per_memory = 10

        for sm, _density in ranked:
            memory_cost = sm.memory.token_count + formatting_overhead_per_memory

            if tokens_used + memory_cost <= budget:
                selected.append(sm)
                tokens_used += memory_cost

        return selected, total_candidate_tokens

    @staticmethod
    def _format_context(
        memories: list[ScoredMemory],
        config: CompilationConfig,
    ) -> str:
        """Format selected memories into a context string."""
        if not memories:
            return ""

        if config.format == "json":
            import json

            items = []
            for sm in memories:
                item: dict = {"content": sm.memory.content, "type": sm.memory.type.value}
                if config.include_sources and sm.memory.source_type:
                    item["source"] = sm.memory.source_type
                if config.include_confidence:
                    item["confidence"] = sm.memory.confidence
                items.append(item)
            return json.dumps(items, indent=2)

        # Text format
        lines: list[str] = []
        lines.append("## User Context")
        lines.append("")

        for i, sm in enumerate(memories, 1):
            mem = sm.memory
            line = f"{i}. {mem.content}"

            annotations: list[str] = []
            if config.include_sources and mem.source_type:
                annotations.append(f"source: {mem.source_type}")
            if config.include_confidence:
                annotations.append(f"confidence: {mem.confidence:.1%}")

            if annotations:
                line += f"  ({', '.join(annotations)})"

            lines.append(line)

        return "\n".join(lines)
