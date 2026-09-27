"""Phase 6 query-aware compiler acceptance tests."""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest

from contextos.benchmarks.compilation import (
    budget_violation_rate,
    fact_redundancy,
    information_unit_recall,
    provenance_coverage,
    smoke_memories,
    synthetic_memories,
    unsupported_fact_rate,
    weighted_preservation,
)
from contextos.core.enums import (
    CandidateTemporalStatus,
    CompilationStrategy,
    CompressionLevel,
    FactExclusionReason,
    MemoryStatus,
    MemoryType,
    PrivacyLevel,
)
from contextos.core.models import CompilationConfig, Memory, ScoredMemory
from contextos.services.compilation import QueryAwareContextCompiler, fact_is_supported
from contextos.services.token_counter import DeterministicWordTokenCounter


def scored(
    number: int,
    content: str,
    *,
    memory_type: MemoryType = MemoryType.CONTEXT,
    status: MemoryStatus = MemoryStatus.ACTIVE,
    privacy: PrivacyLevel = PrivacyLevel.PERSONAL,
) -> ScoredMemory:
    return ScoredMemory(
        memory=Memory(
            id=UUID(f"60000000-0000-0000-0000-{number:012d}"),
            content=content,
            type=memory_type,
            status=status,
            privacy_level=privacy,
        ),
        final_score=1.0,
        rank=number,
    )


@pytest.fixture
def compiler() -> QueryAwareContextCompiler:
    return QueryAwareContextCompiler(
        token_counter=DeterministicWordTokenCounter()
    )


async def compile_with(
    compiler: QueryAwareContextCompiler,
    query: str,
    memories: list[ScoredMemory],
    *,
    budget: int = 1_000,
    strategy: CompilationStrategy = CompilationStrategy.CONTEXTOS_COMPILER,
    level: CompressionLevel = CompressionLevel.LIGHT,
    output_format: str = "text",
):
    return await compiler.compile(
        query,
        memories,
        CompilationConfig(
            budget=budget,
            strategy=strategy,
            compression_level=level,
            format=output_format,
        ),
    )


@pytest.mark.asyncio
async def test_raw_concat_baseline(compiler):
    items = [scored(1, "Alpha fact."), scored(2, "Beta fact.")]
    result = await compile_with(
        compiler, "facts", items, strategy=CompilationStrategy.RAW_CONCAT
    )
    assert [fact.text for fact in result.facts] == ["Alpha fact.", "Beta fact."]
    assert result.strategy == CompilationStrategy.RAW_CONCAT


@pytest.mark.asyncio
async def test_dedup_baseline_merges_sources_without_rewriting(compiler):
    items = [
        scored(1, "User prefers concise answers.", memory_type=MemoryType.PREFERENCE),
        scored(2, "User likes short responses.", memory_type=MemoryType.PREFERENCE),
    ]
    result = await compile_with(
        compiler, "response preference", items,
        strategy=CompilationStrategy.DEDUP_ONLY,
    )
    assert len(result.facts) == 1
    assert len(result.facts[0].source_memory_ids) == 2
    assert result.facts[0].text in {item.memory.content for item in items}


@pytest.mark.asyncio
async def test_deterministic_compiler_removes_irrelevant_clause(compiler):
    item = scored(1, "User uses Ollama for inference, and user enjoys biryani.")
    result = await compile_with(compiler, "Which inference runtime is used?", [item])
    assert "Ollama" in result.context_text
    assert "biryani" not in result.context_text


@pytest.mark.asyncio
async def test_empty_input(compiler):
    result = await compile_with(compiler, "anything", [])
    assert result.context_text == ""
    assert result.total_tokens == 0
    assert result.provenance_coverage == 1.0


@pytest.mark.asyncio
async def test_zero_budget(compiler):
    result = await compile_with(compiler, "alpha", [scored(1, "Alpha fact.")], budget=0)
    assert result.context_text == ""
    assert result.total_tokens == 0
    assert result.excluded_facts[0].reason == FactExclusionReason.BUDGET


@pytest.mark.asyncio
async def test_one_memory(compiler):
    item = scored(1, "User uses Ollama.")
    result = await compile_with(compiler, "Ollama runtime", [item])
    assert result.included_memory_ids == [item.memory.id]
    assert len(result.facts) == 1


@pytest.mark.asyncio
async def test_exact_duplicate_memories(compiler):
    items = [scored(1, "User uses Ollama."), scored(2, "User uses Ollama.")]
    result = await compile_with(compiler, "Ollama", items)
    assert len(result.facts) == 1
    assert result.facts[0].source_memory_ids == [
        items[0].memory.id,
        items[1].memory.id,
    ]


@pytest.mark.asyncio
async def test_near_duplicate_memories(compiler):
    items = [
        scored(1, "User prefers short answers.", memory_type=MemoryType.PREFERENCE),
        scored(2, "User prefers concise responses.", memory_type=MemoryType.PREFERENCE),
    ]
    result = await compile_with(compiler, "response preference", items)
    assert len(result.facts) == 1


@pytest.mark.asyncio
async def test_complementary_memories_remain_distinct(compiler):
    items = [
        scored(1, "User uses Ollama."),
        scored(2, "Qwen 9B runs successfully."),
    ]
    result = await compile_with(compiler, "Ollama Qwen", items)
    assert len(result.facts) == 2


@pytest.mark.asyncio
async def test_negation_is_preserved(compiler):
    result = await compile_with(
        compiler, "Does the user use Docker?", [scored(1, "I don't use Docker anymore.")]
    )
    assert "don't" in result.context_text
    assert result.facts[0].negated


@pytest.mark.asyncio
async def test_temporal_semantics_are_preserved(compiler):
    result = await compile_with(
        compiler, "Python before", [scored(1, "I used Python before.")]
    )
    assert "before" in result.context_text
    assert result.facts[0].temporal_status == CandidateTemporalStatus.HISTORICAL


@pytest.mark.asyncio
async def test_uncertainty_is_preserved(compiler):
    result = await compile_with(
        compiler, "Rust learning", [scored(1, "I might learn Rust.")]
    )
    assert "might" in result.context_text
    assert result.facts[0].uncertain


@pytest.mark.asyncio
async def test_causal_relation_is_preserved(compiler):
    result = await compile_with(
        compiler,
        "Why did Qwen fail?",
        [scored(1, "Qwen 30B failed due to insufficient memory.")],
    )
    assert "due to insufficient memory" in result.context_text
    assert result.facts[0].causal


@pytest.mark.asyncio
async def test_preference_and_knowledge_are_not_merged(compiler):
    items = [
        scored(1, "User prefers C++.", memory_type=MemoryType.PREFERENCE),
        scored(2, "User knows C++.", memory_type=MemoryType.SKILL),
    ]
    result = await compile_with(compiler, "C++", items)
    assert len(result.facts) == 2


@pytest.mark.asyncio
async def test_current_and_stopped_use_are_not_flattened(compiler):
    items = [
        scored(1, "User uses Ollama."),
        scored(2, "User stopped using Ollama.", status=MemoryStatus.HISTORICAL),
    ]
    result = await compile_with(compiler, "Ollama use", items)
    assert len(result.facts) == 2
    assert "stopped" in result.context_text


@pytest.mark.asyncio
async def test_oversized_memory_yields_relevant_fact(compiler):
    content = (
        "Unrelated console output was recorded. " * 30
        + "Qwen 30B failed due to insufficient memory. "
        + "Unrelated package details followed. " * 30
    )
    item = scored(1, content)
    result = await compile_with(compiler, "Why did Qwen fail?", [item], budget=20)
    assert result.total_tokens <= 20
    assert "Qwen 30B failed due to insufficient memory." in result.context_text
    assert "console output" not in result.context_text


@pytest.mark.asyncio
async def test_query_specific_compression(compiler):
    item = scored(1, "User is learning C++17 mainly for systems interviews.")
    language = await compile_with(compiler, "What language am I learning?", [item])
    reason = await compile_with(compiler, "Why am I learning C++17?", [item])
    assert "systems interviews" not in language.context_text
    assert "systems interviews" in reason.context_text


@pytest.mark.asyncio
async def test_exact_budget_and_one_below(compiler):
    item = scored(1, "User uses Ollama.")
    unconstrained = await compile_with(compiler, "Ollama", [item])
    exact = await compile_with(
        compiler, "Ollama", [item], budget=unconstrained.total_tokens
    )
    below = await compile_with(
        compiler, "Ollama", [item], budget=unconstrained.total_tokens - 1
    )
    assert exact.total_tokens == exact.budget
    assert below.total_tokens <= below.budget
    assert below.facts == []


@pytest.mark.asyncio
async def test_formatting_overhead_is_counted(compiler):
    result = await compile_with(
        compiler, "alpha", [scored(1, "Alpha fact.")]
    )
    assert result.total_tokens > sum(fact.token_cost for fact in result.facts)
    counter = DeterministicWordTokenCounter()
    assert result.total_tokens == counter.count(result.context_text)


@pytest.mark.asyncio
async def test_provenance_one_to_one(compiler):
    item = scored(1, "User uses Ollama.")
    result = await compile_with(compiler, "Ollama", [item])
    fact = result.facts[0]
    assert result.provenance_map[fact.fact_id] == [item.memory.id]


@pytest.mark.asyncio
async def test_provenance_many_to_one_and_source_order(compiler):
    items = [scored(1, "User uses Ollama."), scored(2, "User uses Ollama.")]
    result = await compile_with(compiler, "Ollama", items)
    fact = result.facts[0]
    assert result.provenance_map[fact.fact_id] == [
        items[0].memory.id,
        items[1].memory.id,
    ]


@pytest.mark.asyncio
async def test_excluded_fact_retains_reason_and_provenance(compiler):
    item = scored(1, "User uses Ollama.")
    result = await compile_with(compiler, "Ollama", [item], budget=1)
    assert result.excluded_facts[0].reason == FactExclusionReason.BUDGET
    assert result.excluded_facts[0].source_memory_ids == [item.memory.id]


@pytest.mark.asyncio
async def test_no_emitted_fact_without_provenance_or_support(compiler):
    items = [scored(1, "User uses Ollama."), scored(2, "Qwen 9B runs.")]
    result = await compile_with(compiler, "Ollama Qwen", items)
    sources = {item.memory.id: item.memory.content for item in items}
    assert all(fact.source_memory_ids for fact in result.facts)
    assert all(
        any(fact_is_supported(fact.text, sources[source_id])
            for source_id in fact.source_memory_ids)
        for fact in result.facts
    )
    assert result.unsupported_fact_rate == 0.0


@pytest.mark.asyncio
async def test_output_is_deterministic(compiler):
    items = [scored(1, "User uses Ollama."), scored(2, "Qwen 9B runs.")]
    first = await compile_with(compiler, "Ollama Qwen", items)
    second = await compile_with(compiler, "Ollama Qwen", items)
    assert first.context_text == second.context_text
    assert first.provenance_map == second.provenance_map


@pytest.mark.asyncio
async def test_retrieval_objects_are_not_mutated(compiler):
    items = [scored(1, "User uses Ollama.")]
    before = [item.model_dump() for item in items]
    await compile_with(compiler, "Ollama", items)
    assert [item.model_dump() for item in items] == before


@pytest.mark.asyncio
async def test_privacy_metadata_is_omitted(compiler):
    event_id = uuid4()
    item = scored(1, "User uses Ollama.")
    item.memory.source_uri = "private://secret-fingerprint"
    item.memory.provenance_event_id = event_id
    result = await compile_with(compiler, "Ollama", [item], output_format="json")
    assert "private://" not in result.context_text
    assert str(event_id) not in result.context_text
    assert result.facts[0].provenance_event_ids == [event_id]


@pytest.mark.asyncio
async def test_restricted_memory_is_excluded(compiler):
    item = scored(
        1,
        "Restricted local secret.",
        privacy=PrivacyLevel.RESTRICTED,
    )
    result = await compile_with(compiler, "secret", [item])
    assert result.facts == []
    assert result.excluded_facts[0].reason == FactExclusionReason.PRIVACY_RESTRICTED


@pytest.mark.asyncio
async def test_machine_readable_output(compiler):
    result = await compile_with(
        compiler, "Ollama", [scored(1, "User uses Ollama.")], output_format="json"
    )
    dumped = result.model_dump(mode="json")
    assert dumped["facts"][0]["source_memory_ids"]
    assert dumped["provenance_map"]
    assert '"facts"' in result.context_text


@pytest.mark.asyncio
async def test_many_tiny_facts_never_exceed_budget(compiler):
    items = [scored(index, f"Topic fact{index}.") for index in range(1, 101)]
    result = await compile_with(compiler, "topic facts", items, budget=30)
    assert result.total_tokens <= 30


@pytest.mark.asyncio
async def test_compression_ratio_metric(compiler):
    result = await compile_with(
        compiler,
        "Ollama",
        [scored(1, "User uses Ollama. Unrelated sentence.")],
    )
    assert result.compression_ratio == pytest.approx(
        result.total_tokens / result.input_tokens
    )


@pytest.mark.asyncio
async def test_information_unit_and_weighted_metrics(compiler):
    memories = smoke_memories()
    result = await compile_with(compiler, "local model machine previous", memories)
    assert information_unit_recall(result) == 1.0
    assert weighted_preservation(result) == 1.0


@pytest.mark.asyncio
async def test_unsupported_fact_metric(compiler):
    memories = smoke_memories()
    result = await compile_with(compiler, "local model machine previous", memories)
    assert unsupported_fact_rate(result, memories) == 0.0


@pytest.mark.asyncio
async def test_provenance_coverage_metric(compiler):
    result = await compile_with(
        compiler, "Ollama", [scored(1, "User uses Ollama.")]
    )
    assert provenance_coverage(result) == 1.0


@pytest.mark.asyncio
async def test_redundancy_metric(compiler):
    duplicate = await compile_with(
        compiler,
        "Ollama",
        [scored(1, "User uses Ollama."), scored(2, "User uses Ollama.")],
        strategy=CompilationStrategy.RAW_CONCAT,
    )
    deduped = await compile_with(
        compiler,
        "Ollama",
        [scored(1, "User uses Ollama."), scored(2, "User uses Ollama.")],
        strategy=CompilationStrategy.DEDUP_ONLY,
    )
    assert fact_redundancy(duplicate) == 1.0
    assert fact_redundancy(deduped) == 0.0


@pytest.mark.asyncio
async def test_budget_violation_metric(compiler):
    result = await compile_with(
        compiler, "Ollama", [scored(1, "User uses Ollama.")], budget=1
    )
    assert budget_violation_rate([result]) == 0.0
    assert budget_violation_rate([]) == 0.0


@pytest.mark.asyncio
async def test_scale_1000_is_deterministic_and_budget_safe(compiler):
    items = synthetic_memories(1_000)
    first = await compile_with(
        compiler, "current project constraints", items, budget=500
    )
    second = await compile_with(
        compiler, "current project constraints", items, budget=500
    )
    assert first.total_tokens <= 500
    assert first.context_text == second.context_text
