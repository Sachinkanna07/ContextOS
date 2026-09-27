"""Phase 5 to Phase 6 oversized-candidate rescue contract tests."""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest

from contextos.api.routes import retrieval as retrieval_route
from contextos.core.enums import (
    CompilationStrategy,
    CompilerInputKind,
    ExclusionReason,
    MemoryStatus,
    MemoryType,
    PrivacyLevel,
)
from contextos.core.models import (
    CompilationConfig,
    ContextBudget,
    Memory,
    RetrievalResult,
    ScoredMemory,
)
from contextos.services.compilation import QueryAwareContextCompiler
from contextos.services.optimization import MemoryContextOptimizer
from contextos.services.token_counter import DeterministicWordTokenCounter


def item(
    number: int,
    content: str,
    score: float,
    *,
    status: MemoryStatus = MemoryStatus.ACTIVE,
    privacy: PrivacyLevel = PrivacyLevel.PERSONAL,
    event_id: UUID | None = None,
) -> ScoredMemory:
    return ScoredMemory(
        memory=Memory(
            id=UUID(f"70000000-0000-0000-0000-{number:012d}"),
            content=content,
            type=MemoryType.FACT,
            status=status,
            privacy_level=privacy,
            provenance_event_id=event_id,
            confidence=0.9,
            importance=0.8,
        ),
        final_score=score,
        rank=number,
        retrieval_sources=["lexical", "dense"],
    )


def long_memory(fact: str, marker: str = "archive") -> str:
    padding = " ".join(f"{marker}{index}" for index in range(320)) + "."
    return f"{fact} {padding}"


@pytest.fixture
def pipeline():
    counter = DeterministicWordTokenCounter()
    return (
        MemoryContextOptimizer(token_counter=counter),
        QueryAwareContextCompiler(token_counter=counter),
    )


@pytest.mark.asyncio
async def test_relevant_oversized_candidate_reaches_compiler_as_compact_fact(pipeline):
    optimizer, compiler = pipeline
    normal = item(1, "Ollama is the local model runner.", 0.85)
    oversized = item(
        2,
        long_memory(
            "Qwen30B failed on the local machine because VRAM was insufficient."
        ),
        1.0,
    )

    selection = optimizer.optimize(
        "local model machine attempts", [normal, oversized], ContextBudget(max_tokens=12)
    )
    result = await compiler.compile(
        "Why did Qwen30B fail on the local machine?",
        selection,
        CompilationConfig(budget=40),
    )

    assert [value.memory.id for value in selection.compiler_rescue_candidates] == [
        oversized.memory.id
    ]
    rescued = [
        fact for fact in result.facts
        if fact.input_kind == CompilerInputKind.OVERSIZED_RESCUE
    ]
    assert [fact.text for fact in rescued] == [
        "Qwen30B failed on the local machine because VRAM was insufficient."
    ]
    assert "archive319" not in result.context_text
    assert result.total_tokens <= result.budget


@pytest.mark.asyncio
async def test_rescue_is_extractive_even_for_raw_concat_baseline(pipeline):
    optimizer, compiler = pipeline
    oversized = item(
        1,
        long_memory("Qwen30B failed locally because VRAM was insufficient."),
        1.0,
    )
    selection = optimizer.optimize(
        "Qwen30B local failure", [oversized], ContextBudget(max_tokens=10)
    )
    result = await compiler.compile(
        "Qwen30B local failure",
        selection,
        CompilationConfig(budget=30, strategy=CompilationStrategy.RAW_CONCAT),
    )

    assert [fact.text for fact in result.facts] == [
        "Qwen30B failed locally because VRAM was insufficient."
    ]
    assert "archive319" not in result.context_text
    assert result.total_tokens <= result.budget


@pytest.mark.asyncio
async def test_rescue_preserves_provenance_and_has_structured_trace(pipeline):
    optimizer, compiler = pipeline
    event_id = uuid4()
    oversized = item(
        1,
        long_memory("Qwen30B failed on this machine due to insufficient VRAM."),
        1.0,
        event_id=event_id,
    )
    selection = optimizer.optimize(
        "Qwen30B machine failure", [oversized], ContextBudget(max_tokens=10)
    )
    result = await compiler.compile(
        "Qwen30B machine failure", selection, CompilationConfig(budget=30)
    )

    assert result.trace.normal_selected_inputs == 0
    assert result.trace.oversized_rescue_inputs == 1
    assert result.trace.rescued_facts_included == 1
    assert result.facts[0].source_memory_ids == [oversized.memory.id]
    assert result.facts[0].provenance_event_ids == [event_id]
    assert result.provenance_map[result.facts[0].fact_id] == [oversized.memory.id]


def test_rescue_rejects_irrelevant_invalid_private_and_redundant_candidates(pipeline):
    optimizer, _ = pipeline
    selected = item(1, "The local runner is Ollama.", 1.0)
    irrelevant = item(2, long_memory("A garden uses drip irrigation."), 0.01)
    invalid = item(
        3,
        long_memory("Qwen30B failed locally."),
        0.9,
        status=MemoryStatus.EXPIRED,
    )
    private = item(
        4,
        long_memory("Qwen30B failed locally."),
        0.9,
        privacy=PrivacyLevel.RESTRICTED,
    )
    budget_exhausted = item(
        6, "Qwen14B needed a smaller quantization on this local machine.", 0.85
    )
    redundant = item(
        5,
        long_memory("The local runner is Ollama.", marker="The local runner is Ollama"),
        0.95,
    )

    selection = optimizer.optimize(
        "local model",
        [selected, irrelevant, invalid, private, redundant, budget_exhausted],
        ContextBudget(max_tokens=12),
    )
    rescue_ids = {value.memory.id for value in selection.compiler_rescue_candidates}
    reasons = {
        decision.memory_id: decision.exclusion_reason
        for decision in selection.trace.decisions
    }

    assert irrelevant.memory.id not in rescue_ids
    assert invalid.memory.id not in rescue_ids
    assert private.memory.id not in rescue_ids
    assert redundant.memory.id not in rescue_ids
    assert budget_exhausted.memory.id not in rescue_ids
    assert reasons[irrelevant.memory.id] == ExclusionReason.LOW_RELEVANCE
    assert reasons[invalid.memory.id] == ExclusionReason.INVALID_LIFECYCLE
    assert reasons[private.memory.id] == ExclusionReason.OVERSIZED
    assert reasons[redundant.memory.id] == ExclusionReason.REDUNDANT
    assert reasons[budget_exhausted.memory.id] == ExclusionReason.BUDGET_EXHAUSTED


def test_phase5_whole_memory_selection_and_accounting_are_unchanged(pipeline):
    optimizer, _ = pipeline
    short = item(1, "Ollama runs locally.", 0.8)
    oversized = item(2, long_memory("Qwen30B failed locally."), 1.0)
    result = optimizer.optimize(
        "local model", [short, oversized], ContextBudget(max_tokens=8)
    )

    assert result.selected_memories == [short]
    assert result.total_tokens == result.content_tokens + 2
    assert result.content_tokens == DeterministicWordTokenCounter().count(
        short.memory.content
    )
    assert result.overhead_tokens == 2
    assert all(value.memory.id != oversized.memory.id for value in result.selected_memories)
    assert result.trace.decisions[1].exclusion_reason == ExclusionReason.OVERSIZED


@pytest.mark.asyncio
async def test_phase4_scored_memory_list_contract_remains_normal_selected(pipeline):
    _, compiler = pipeline
    phase4_output = [item(1, "Ollama runs local models.", 1.0)]
    result = await compiler.compile("local models", phase4_output)

    assert result.trace.normal_selected_inputs == 1
    assert result.trace.oversized_rescue_inputs == 0
    assert result.facts[0].input_kind == CompilerInputKind.NORMAL_SELECTED


@pytest.mark.asyncio
async def test_normal_selected_and_oversized_rescue_coexist(pipeline):
    optimizer, compiler = pipeline
    normal = item(1, "Qwen9B works on the local machine.", 0.9)
    oversized = item(
        2,
        long_memory("Qwen30B failed on the local machine due to limited VRAM."),
        1.0,
    )
    selection = optimizer.optimize(
        "local model machine attempts", [normal, oversized], ContextBudget(max_tokens=10)
    )
    result = await compiler.compile(
        "local model machine attempts", selection, CompilationConfig(budget=45)
    )

    assert {fact.input_kind for fact in result.facts} == {
        CompilerInputKind.NORMAL_SELECTED,
        CompilerInputKind.OVERSIZED_RESCUE,
    }
    assert set(result.included_memory_ids) == {normal.memory.id, oversized.memory.id}


@pytest.mark.asyncio
async def test_compile_route_passes_typed_selection_contract(monkeypatch, pipeline):
    optimizer, compiler = pipeline
    normal = item(1, "Qwen9B works locally.", 0.9)
    oversized = item(2, long_memory("Qwen30B failed locally."), 1.0)

    class RetrievalStub:
        async def retrieve(self, query, config):
            return RetrievalResult(query=query, memories=[normal, oversized])

    services = {
        "retrieval": RetrievalStub(),
        "optimizer": optimizer,
        "compilation": compiler,
    }
    monkeypatch.setattr(retrieval_route, "get_service", services.__getitem__)
    result = await retrieval_route.compile_context(retrieval_route.CompileRequest(
        query="local model attempts",
        config=CompilationConfig(budget=12),
    ))

    assert result.trace.normal_selected_inputs == 1
    assert result.trace.oversized_rescue_inputs == 1
    assert any(
        fact.input_kind == CompilerInputKind.OVERSIZED_RESCUE
        for fact in result.facts
    )

