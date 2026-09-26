"""Phase 2 deterministic candidate extraction contract."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from contextos.core.enums import (
    CandidateAction,
    CandidateTemporalStatus,
    MemoryType,
    SourceRole,
)
from contextos.core.models import CandidateMemory, IngestRequest, ScanResult
from contextos.services.extraction import MAX_CANDIDATES, RuleBasedMemoryExtractor
from contextos.services.ingestion import IngestionPipeline


@pytest.fixture
def extractor() -> RuleBasedMemoryExtractor:
    return RuleBasedMemoryExtractor()


@pytest.mark.asyncio
async def test_single_preference(extractor):
    result = await extractor.extract("I prefer concise technical answers.")
    assert len(result) == 1
    assert result[0].memory_type == MemoryType.PREFERENCE
    assert result[0].content == "User prefers concise technical answers"


@pytest.mark.asyncio
async def test_single_goal(extractor):
    result = await extractor.extract("I want to learn Kubernetes.")
    assert result[0].memory_type == MemoryType.GOAL


@pytest.mark.asyncio
async def test_multi_sentence_atomicity(extractor):
    result = await extractor.extract("I use Linux. I prefer dark mode. I'm building ContextOS.")
    assert len(result) == 3


@pytest.mark.asyncio
async def test_multiple_facts_in_one_sentence(extractor):
    result = await extractor.extract("I use Ollama and I prefer short answers.")
    assert [item.content for item in result] == [
        "User uses Ollama",
        "User prefers concise answers",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text", ["okay", "thanks", "nice", "the weather looks good", "lol", "continue", "yes"]
)
async def test_filler_is_ignored(extractor, text):
    assert await extractor.extract(text) == []


@pytest.mark.asyncio
async def test_negation_survives(extractor):
    result = await extractor.extract("I don't use Docker anymore.")
    assert result[0].content == "User does not use Docker anymore"
    assert result[0].metadata["negated"] is True
    assert result[0].action_hint == CandidateAction.SUPERSEDE


@pytest.mark.asyncio
async def test_historical_fact(extractor):
    result = await extractor.extract("I used to focus on Python.")
    assert result[0].temporal_status == CandidateTemporalStatus.HISTORICAL


@pytest.mark.asyncio
async def test_current_fact(extractor):
    result = await extractor.extract("I am learning C++.")
    assert result[0].temporal_status == CandidateTemporalStatus.CURRENT
    assert result[0].memory_type == MemoryType.SKILL


@pytest.mark.asyncio
async def test_future_intent(extractor):
    result = await extractor.extract("I plan to learn Rust next month.")
    assert result[0].temporal_status == CandidateTemporalStatus.FUTURE
    assert result[0].memory_type == MemoryType.GOAL


@pytest.mark.asyncio
async def test_uncertain_future_intent(extractor):
    result = await extractor.extract("I might learn Go.")
    assert result[0].temporal_status == CandidateTemporalStatus.FUTURE
    assert result[0].confidence < 0.7


@pytest.mark.asyncio
async def test_switch_produces_prior_and_current_candidates(extractor):
    result = await extractor.extract("I'm switching from Python to C++.")
    assert len(result) == 2
    assert result[0].temporal_status == CandidateTemporalStatus.HISTORICAL
    assert result[0].action_hint == CandidateAction.SUPERSEDE
    assert result[1].temporal_status == CandidateTemporalStatus.CURRENT
    assert "Python" in result[0].content
    assert "C++" in result[1].content


@pytest.mark.asyncio
async def test_confidence_differs_by_certainty(extractor):
    certain = await extractor.extract("I definitely use Linux.")
    uncertain = await extractor.extract("I think I might switch to Linux.")
    assert certain[0].confidence > uncertain[0].confidence


@pytest.mark.asyncio
async def test_importance_differs_from_temporary_remark(extractor):
    preference = await extractor.extract("I prefer concise answers.")
    temporary = await extractor.extract("I am busy today.")
    assert preference[0].importance > temporary[0].importance


@pytest.mark.asyncio
async def test_provenance_and_evidence_preserved(extractor):
    result = await extractor.extract(
        "I prefer dark mode.", source_type="conversation", source_uri="chat://42"
    )
    candidate = result[0]
    assert candidate.source_type == "conversation"
    assert candidate.source_uri == "chat://42"
    assert candidate.source_role == SourceRole.USER
    assert candidate.evidence == "I prefer dark mode."
    assert candidate.evidence_start == 0
    assert candidate.evidence_end == len(candidate.evidence)


@pytest.mark.asyncio
async def test_duplicate_candidate_suppression(extractor):
    result = await extractor.extract("I use Linux. I use Linux.")
    assert len(result) == 1


@pytest.mark.asyncio
async def test_punctuation_and_casing_robustness(extractor):
    lower = await extractor.extract("i prefer dark mode!!!")
    title = await extractor.extract("I PREFER dark mode.")
    assert lower[0].memory_type == title[0].memory_type == MemoryType.PREFERENCE
    assert lower[0].content.casefold() == title[0].content.casefold()


@pytest.mark.asyncio
@pytest.mark.parametrize("text", ["", " \t\n "])
async def test_empty_and_whitespace_input(extractor, text):
    assert await extractor.extract(text) == []


@pytest.mark.asyncio
async def test_very_long_input_is_bounded(extractor):
    text = " ".join(f"I prefer option {index}." for index in range(10_000))
    result = await extractor.extract(text)
    assert len(result) == MAX_CANDIDATES
    assert all(len(item.content) <= 10_000 for item in result)
    assert all(item.metadata["input_truncated"] for item in result)


@pytest.mark.asyncio
async def test_repeated_fact_with_case_difference_is_suppressed(extractor):
    result = await extractor.extract("I use Linux. i use linux!")
    assert len(result) == 1


@pytest.mark.asyncio
async def test_assistant_claim_is_ignored_unless_confirmed(extractor):
    ignored = await extractor.extract("I prefer Rust.", source_role=SourceRole.ASSISTANT)
    confirmed = await extractor.extract(
        "The user prefers Rust.",
        source_role=SourceRole.ASSISTANT,
        confirmed_user_information=True,
    )
    assert ignored == []
    assert len(confirmed) == 1
    assert confirmed[0].metadata["confirmed_user_information"] is True


@pytest.mark.asyncio
async def test_output_is_deterministic(extractor):
    text = "I used to use Python, but now I prefer Rust."
    first = await extractor.extract(text, source_uri="chat://stable")
    second = await extractor.extract(text, source_uri="chat://stable")
    assert [item.model_dump() for item in first] == [item.model_dump() for item in second]


@pytest.mark.asyncio
async def test_required_smoke_semantics(extractor):
    text = (
        "I used to focus mainly on Python, but now I'm preparing for C++ systems interviews.\n"
        "I also prefer concise technical answers.\n"
        "Maybe I'll learn Rust later."
    )
    result = await extractor.extract(text)
    assert len(result) == 4
    assert [item.memory_type for item in result] == [
        MemoryType.GOAL,
        MemoryType.GOAL,
        MemoryType.PREFERENCE,
        MemoryType.GOAL,
    ]
    assert [item.temporal_status for item in result] == [
        CandidateTemporalStatus.HISTORICAL,
        CandidateTemporalStatus.CURRENT,
        CandidateTemporalStatus.CURRENT,
        CandidateTemporalStatus.FUTURE,
    ]
    assert result[3].confidence < result[1].confidence


def test_candidate_model_validation():
    with pytest.raises(ValidationError):
        CandidateMemory(
            content=" ", evidence="I prefer Rust", memory_type=MemoryType.PREFERENCE
        )
    with pytest.raises(ValidationError):
        CandidateMemory(
            content="User prefers Rust",
            evidence="I prefer Rust",
            memory_type=MemoryType.PREFERENCE,
            evidence_start=5,
        )


@pytest.mark.asyncio
async def test_ingestion_returns_candidates_without_long_term_writes():
    class Scanner:
        def scan(self, text):
            return ScanResult(scanned_length=len(text))

    class Events:
        def __init__(self):
            self.items = []

        async def append(self, event):
            self.items.append(event)

    memory_repo = AsyncMock()
    embedding = AsyncMock()
    vector = AsyncMock()
    lexical = AsyncMock()
    token_counter = AsyncMock()
    events = Events()
    pipeline = IngestionPipeline(
        secret_scanner=Scanner(),
        memory_extractor=RuleBasedMemoryExtractor(),
        memory_repo=memory_repo,
        event_repo=events,
        embedding_service=embedding,
        vector_store=vector,
        lexical_index=lexical,
        token_counter=token_counter,
    )
    result = await pipeline.ingest(IngestRequest(content="I prefer concise answers."))
    assert len(result.candidates) == 1
    assert len(events.items) == 1
    memory_repo.create.assert_not_awaited()
    embedding.embed.assert_not_awaited()
    vector.add.assert_not_awaited()
    lexical.index.assert_not_awaited()
