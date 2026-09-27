"""Phase 7 temporal resolution acceptance and integration tests."""

from __future__ import annotations

import socket
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

import pytest
import pytest_asyncio

from contextos.benchmarks.temporal import (
    classification_accuracy,
    coexistence_accuracy,
    current_state_accuracy,
    evaluation_cases,
    false_supersession_rate,
    historical_state_recall,
    outcome_precision,
    outcome_recall,
    run_evaluation,
    timeline_consistency_violations,
)
from contextos.core.enums import (
    CandidateTemporalStatus,
    MemoryStatus,
    MemoryType,
    RelationType,
    RetrievalMode,
    TemporalOutcome,
    TemporalPrecision,
    TemporalScope,
)
from contextos.core.exceptions import InvalidTransitionError
from contextos.core.models import (
    CompilationConfig,
    Memory,
    MemorySlot,
    RetrievalConfig,
    RetrievalQuery,
    ScoredMemory,
)
from contextos.embedding.deterministic import DeterministicEmbedding
from contextos.services.compilation import QueryAwareContextCompiler
from contextos.services.retrieval import HybridRetrievalEngine
from contextos.services.retrieval_index import RetrievalIndexSynchronizer
from contextos.services.temporal import TemporalMemoryService, TemporalSlotAnalyzer
from contextos.services.token_counter import DeterministicWordTokenCounter
from contextos.storage.database import Database, SCHEMA_SQL, SCHEMA_VERSION
from contextos.storage.lexical.bm25 import BM25Index
from contextos.storage.memory_repo import SqliteMemoryRepository
from contextos.storage.relation_repo import SqliteRelationRepository
from contextos.storage.vector.in_memory import InMemoryVectorStore


UTC = timezone.utc


def moment(year: int, month: int = 1, day: int = 1) -> datetime:
    return datetime(year, month, day, 12, tzinfo=UTC)


def candidate(
    number: int,
    content: str,
    *,
    observed_at: datetime,
    valid_from: datetime | None = None,
    confidence: float = 0.95,
    temporal_status: CandidateTemporalStatus = CandidateTemporalStatus.UNSPECIFIED,
    provenance: int | None = None,
) -> Memory:
    return Memory(
        id=UUID(f"90000000-0000-0000-0000-{number:012d}"),
        content=content,
        type=MemoryType.FACT,
        status=MemoryStatus.CANDIDATE,
        observed_at=observed_at,
        valid_from=valid_from,
        confidence=confidence,
        provenance_event_id=(
            UUID(f"91000000-0000-0000-0000-{provenance:012d}")
            if provenance is not None else None
        ),
        temporal_status=temporal_status,
    )


@pytest_asyncio.fixture
async def temporal_stack(tmp_path: Path):
    database = Database(tmp_path / "temporal.db")
    await database.initialize()
    repository = SqliteMemoryRepository(database.connection())
    relations = SqliteRelationRepository(database.connection())
    service = TemporalMemoryService(repository)
    yield database, repository, relations, service
    await database.close()


@pytest.mark.asyncio
async def test_explicit_supersession_current_history_provenance_and_trace(temporal_stack):
    _, _, _, service = temporal_stack
    old = candidate(
        1, "User primarily uses Python for interview preparation.",
        observed_at=moment(2025), provenance=1,
    )
    new = candidate(
        2, "User now primarily uses C++17 for systems interviews.",
        observed_at=moment(2026), provenance=2,
    )
    await service.resolve(old)
    result = await service.resolve(new)

    assert result.decision.outcome == TemporalOutcome.SUPERSEDE
    assert result.decision.evidence == ["same_slot", "explicit_change_cue"]
    assert [change.to_status for change in result.decision.changes] == [
        MemoryStatus.SUPERSEDED, MemoryStatus.ACTIVE,
    ]
    slot = result.decision.slot
    current = await service.get_current_state(slot)
    history = await service.get_history(slot)
    assert [memory.id for memory in current] == [new.id]
    assert [memory.id for memory in history] == [old.id, new.id]
    assert history[0].superseded_by == new.id
    assert history[1].supersedes == old.id
    assert history[0].provenance_event_id == old.provenance_event_id
    assert history[1].provenance_event_id == new.provenance_event_id


@pytest.mark.asyncio
async def test_compatible_language_scopes_coexist(temporal_stack):
    _, _, _, service = temporal_stack
    ml = candidate(1, "User uses Python for machine learning.", observed_at=moment(2026))
    systems = candidate(2, "User uses C++ for systems programming.", observed_at=moment(2026, 2))
    first = await service.resolve(ml)
    second = await service.resolve(systems)

    assert first.memory.status == MemoryStatus.ACTIVE
    assert second.decision.outcome == TemporalOutcome.COEXIST
    assert second.memory.status == MemoryStatus.ACTIVE
    assert first.decision.slot.key != second.decision.slot.key
    assert (await service.get_current_state(first.decision.slot))[0].id == ml.id
    assert (await service.get_current_state(second.decision.slot))[0].id == systems.id


@pytest.mark.asyncio
async def test_future_and_uncertain_claims_do_not_supersede_current(temporal_stack):
    _, repository, _, service = temporal_stack
    current = candidate(1, "User currently uses Python.", observed_at=moment(2026))
    future = candidate(2, "User might learn Rust next year.", observed_at=moment(2026, 2))
    await service.resolve(current)
    result = await service.resolve(future)

    assert result.decision.outcome == TemporalOutcome.COEXIST
    assert result.memory.temporal_status == CandidateTemporalStatus.FUTURE
    assert result.memory.uncertain
    assert [memory.id for memory in await service.get_current_state(result.decision.slot)] == [
        current.id
    ]
    assert [memory.id for memory in await service.get_future()] == [future.id]
    assert (await repository.get(current.id)).status == MemoryStatus.ACTIVE


@pytest.mark.asyncio
async def test_explicit_correction_preserves_both_records_and_relation(temporal_stack):
    _, repository, relations, service = temporal_stack
    old = candidate(1, "My machine has 16 GB RAM.", observed_at=moment(2026), provenance=1)
    corrected = candidate(
        2, "Correction: my machine actually has 32 GB RAM.",
        observed_at=moment(2026, 2), provenance=2,
    )
    await service.resolve(old)
    result = await service.resolve(corrected)

    assert result.decision.outcome == TemporalOutcome.CORRECT
    assert (await repository.get(old.id)).status == MemoryStatus.SUPERSEDED
    assert (await repository.get(corrected.id)).status == MemoryStatus.ACTIVE
    links = await relations.get_relations(corrected.id, "outgoing")
    assert [(link.relation_type, link.target_memory_id) for link in links] == [
        (RelationType.CORRECTS, old.id)
    ]


@pytest.mark.asyncio
async def test_negated_transition_supersedes_general_usage(temporal_stack):
    _, repository, _, service = temporal_stack
    old = candidate(1, "I use Docker.", observed_at=moment(2025))
    stopped = candidate(2, "I no longer use Docker.", observed_at=moment(2026))
    await service.resolve(old)
    result = await service.resolve(stopped)
    assert result.decision.outcome == TemporalOutcome.SUPERSEDE
    assert result.memory.negated
    assert (await repository.get(old.id)).status == MemoryStatus.SUPERSEDED


@pytest.mark.asyncio
async def test_scoped_negation_coexists_with_general_usage(temporal_stack):
    _, repository, _, service = temporal_stack
    general = candidate(1, "I use Docker generally.", observed_at=moment(2026))
    scoped = candidate(2, "I don't use Docker for Project X.", observed_at=moment(2026, 2))
    await service.resolve(general)
    result = await service.resolve(scoped)
    assert result.decision.outcome == TemporalOutcome.COEXIST
    assert result.memory.negated
    assert (await repository.get(general.id)).status == MemoryStatus.ACTIVE
    assert result.memory.status == MemoryStatus.ACTIVE


@pytest.mark.asyncio
async def test_unresolved_contradiction_is_symmetric_and_not_current(temporal_stack):
    _, repository, relations, service = temporal_stack
    attended = candidate(1, "I attended Event X.", observed_at=moment(2026))
    denied = candidate(2, "I did not attend Event X.", observed_at=moment(2026))
    first = await service.resolve(attended)
    result = await service.resolve(denied)

    assert result.decision.outcome == TemporalOutcome.CONTRADICT
    assert (await repository.get(attended.id)).status == MemoryStatus.CONTRADICTED
    assert result.memory.status == MemoryStatus.CONTRADICTED
    assert await service.get_current_state(first.decision.slot) == []
    outgoing = await relations.get_relations(attended.id, "outgoing")
    incoming = await relations.get_relations(attended.id, "incoming")
    assert outgoing[0].target_memory_id == denied.id
    assert incoming[0].source_memory_id == denied.id


@pytest.mark.asyncio
async def test_default_retrieval_config_excludes_unresolved_contradictions(temporal_stack):
    _, repository, _, service = temporal_stack
    await service.resolve(candidate(1, "I attended Event X.", observed_at=moment(2026)))
    await service.resolve(candidate(2, "I did not attend Event X.", observed_at=moment(2026)))
    embedding = DeterministicEmbedding(64)
    lexical = BM25Index()
    vector = InMemoryVectorStore(64)
    sync = RetrievalIndexSynchronizer(
        memory_repo=repository, lexical_index=lexical, vector_store=vector,
        embedding_service=embedding,
    )
    engine = HybridRetrievalEngine(
        memory_repo=repository, lexical_index=lexical, vector_store=vector,
        embedding_service=embedding, index_synchronizer=sync,
    )
    result = await engine.retrieve("Event X attendance", RetrievalConfig())
    assert result.memories == []


@pytest.mark.asyncio
async def test_duplicate_and_semantic_no_change_do_not_create_transition(temporal_stack):
    _, repository, _, service = temporal_stack
    first = candidate(1, "User uses Python for ML.", observed_at=moment(2026))
    exact = candidate(2, "User uses Python for ML.", observed_at=moment(2026, 2))
    paraphrase = candidate(3, "User currently uses Python for machine learning.", observed_at=moment(2026, 3))
    await service.resolve(first)
    duplicate = await service.resolve(exact)
    unchanged = await service.resolve(paraphrase)

    assert duplicate.decision.outcome == TemporalOutcome.DUPLICATE
    assert unchanged.decision.outcome == TemporalOutcome.NO_CHANGE
    assert await repository.get(exact.id) is None
    assert await repository.get(paraphrase.id) is None
    assert await repository.count() == 1


@pytest.mark.asyncio
async def test_three_step_chain_is_deterministic_and_acyclic(temporal_stack):
    _, repository, _, service = temporal_stack
    python = candidate(1, "User uses Python for systems interviews.", observed_at=moment(2024))
    cpp = candidate(2, "User now uses C++ for systems interviews.", observed_at=moment(2025))
    rust = candidate(3, "User now uses Rust for systems interviews.", observed_at=moment(2026))
    await service.resolve(python)
    await service.resolve(cpp)
    result = await service.resolve(rust)

    history = await service.get_history(result.decision.slot)
    assert [memory.id for memory in history] == [python.id, cpp.id, rust.id]
    assert (await repository.get(python.id)).superseded_by == cpp.id
    middle = await repository.get(cpp.id)
    assert middle.supersedes == python.id and middle.superseded_by == rust.id
    assert (await repository.get(rust.id)).supersedes == cpp.id
    assert (await service.get_previous(rust.id)).id == cpp.id
    visited: set[UUID] = set()
    current = await repository.get(rust.id)
    while current and current.supersedes:
        assert current.id not in visited
        visited.add(current.id)
        current = await repository.get(current.supersedes)


@pytest.mark.asyncio
async def test_supersession_transaction_rolls_back_everything(temporal_stack):
    database, repository, _, service = temporal_stack
    old = candidate(1, "User uses Python for systems interviews.", observed_at=moment(2025))
    new = candidate(2, "User now uses C++ for systems interviews.", observed_at=moment(2026))
    await service.resolve(old)
    await database.connection().execute(
        "CREATE TRIGGER reject_temporal_relation BEFORE INSERT ON memory_relations "
        "BEGIN SELECT RAISE(FAIL, 'injected temporal failure'); END"
    )
    await database.connection().commit()

    with pytest.raises(sqlite3.IntegrityError):
        await service.resolve(new)
    persisted_old = await repository.get(old.id)
    assert persisted_old.status == MemoryStatus.ACTIVE
    assert persisted_old.superseded_by is None
    assert await repository.get(new.id) is None


@pytest.mark.asyncio
async def test_restart_persists_timeline_and_relations(tmp_path: Path):
    path = tmp_path / "restart-temporal.db"
    database = Database(path)
    await database.initialize()
    repository = SqliteMemoryRepository(database.connection())
    service = TemporalMemoryService(repository)
    old = candidate(1, "User uses Python for systems interviews.", observed_at=moment(2025))
    new = candidate(2, "User now uses C++ for systems interviews.", observed_at=moment(2026))
    await service.resolve(old)
    resolved = await service.resolve(new)
    await database.close()

    reopened = Database(path)
    await reopened.initialize()
    try:
        repository = SqliteMemoryRepository(reopened.connection())
        service = TemporalMemoryService(repository)
        history = await service.get_history(resolved.decision.slot)
        relations = await SqliteRelationRepository(reopened.connection()).get_relations(new.id)
        assert [memory.id for memory in history] == [old.id, new.id]
        assert relations[0].relation_type == RelationType.SUPERSEDES
    finally:
        await reopened.close()


@pytest.mark.asyncio
async def test_out_of_order_effective_time_does_not_replace_current(temporal_stack):
    _, repository, _, service = temporal_stack
    current = candidate(
        1, "User currently uses C++ for systems interviews.",
        observed_at=moment(2026, 6), valid_from=moment(2026, 1),
    )
    imported = candidate(
        2, "User used Python for systems interviews in 2025.",
        observed_at=moment(2026, 9), valid_from=moment(2025, 1),
    )
    await service.resolve(current)
    result = await service.resolve(imported)
    assert result.decision.outcome == TemporalOutcome.ADD_NEW
    assert result.memory.temporal_status == CandidateTemporalStatus.HISTORICAL
    assert (await repository.get(current.id)).status == MemoryStatus.ACTIVE
    assert result.memory.observed_at > result.memory.valid_from


def test_temporal_precision_exact_date_and_vague_language():
    analyzer = TemporalSlotAnalyzer()
    dated = analyzer.prepare(Memory(content="User used Python on 2025-04-03."))
    yearly = analyzer.prepare(Memory(content="User used Python in 2025."))
    vague = analyzer.prepare(Memory(content="User recently used Python."))
    assert dated.temporal_precision == TemporalPrecision.DATE
    assert dated.valid_from == datetime(2025, 4, 3, tzinfo=UTC)
    assert yearly.temporal_precision == TemporalPrecision.YEAR
    assert yearly.valid_from is None
    assert vague.temporal_precision == TemporalPrecision.RELATIVE
    assert vague.valid_from is None


@pytest.mark.asyncio
async def test_low_confidence_uncertain_statement_is_conservative(temporal_stack):
    _, repository, _, service = temporal_stack
    current = candidate(1, "User uses Python.", observed_at=moment(2026))
    uncertain = candidate(
        2, "I think I prefer C++ now.", observed_at=moment(2026, 2), confidence=0.55
    )
    await service.resolve(current)
    result = await service.resolve(uncertain)
    assert result.decision.outcome == TemporalOutcome.COEXIST
    assert (await repository.get(current.id)).status == MemoryStatus.ACTIVE


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [MemoryStatus.DELETED, MemoryStatus.PURGED])
async def test_terminal_candidate_cannot_be_reactivated(temporal_stack, status):
    _, _, _, service = temporal_stack
    terminal = candidate(1, "User uses Python.", observed_at=moment(2026)).model_copy(
        update={"status": status}
    )
    with pytest.raises(InvalidTransitionError):
        await service.resolve(terminal)


@pytest.mark.asyncio
async def test_current_and_historical_retrieval_integration(temporal_stack):
    _, repository, _, service = temporal_stack
    old = candidate(1, "User uses Python for systems interviews.", observed_at=moment(2025))
    new = candidate(2, "User now uses C++17 for systems interviews.", observed_at=moment(2026))
    await service.resolve(old)
    await service.resolve(new)
    embedding = DeterministicEmbedding(64)
    lexical = BM25Index()
    vector = InMemoryVectorStore(64)
    sync = RetrievalIndexSynchronizer(
        memory_repo=repository, lexical_index=lexical, vector_store=vector,
        embedding_service=embedding,
    )
    engine = HybridRetrievalEngine(
        memory_repo=repository, lexical_index=lexical, vector_store=vector,
        embedding_service=embedding, index_synchronizer=sync,
    )
    current = await engine.retrieve(RetrievalQuery(
        text="systems interview language", mode=RetrievalMode.LEXICAL,
        temporal_scope=TemporalScope.CURRENT,
    ))
    historical = await engine.retrieve(RetrievalQuery(
        text="systems interview Python", mode=RetrievalMode.LEXICAL,
        temporal_scope=TemporalScope.HISTORICAL,
    ))
    assert [memory.memory.id for memory in current.memories] == [new.id]
    assert [memory.memory.id for memory in historical.memories] == [old.id]


@pytest.mark.asyncio
async def test_compiler_uses_persisted_temporal_status(temporal_stack):
    _, _, _, service = temporal_stack
    historical = candidate(
        1, "User used Python for systems interviews.",
        observed_at=moment(2026), temporal_status=CandidateTemporalStatus.HISTORICAL,
    )
    result = await service.resolve(historical)
    compiler = QueryAwareContextCompiler(token_counter=DeterministicWordTokenCounter())
    compiled = await compiler.compile(
        "previous systems interview language",
        [ScoredMemory(memory=result.memory, final_score=1.0)],
        CompilationConfig(budget=50),
    )
    assert compiled.facts[0].temporal_status == CandidateTemporalStatus.HISTORICAL


@pytest.mark.asyncio
async def test_trace_is_deterministic_and_contains_no_raw_memory(temporal_stack):
    _, _, _, service = temporal_stack
    old = candidate(1, "User uses Python for systems interviews.", observed_at=moment(2025))
    new = candidate(2, "User now uses C++ for systems interviews.", observed_at=moment(2026))
    await service.resolve(old)
    decision = await service.decide(TemporalSlotAnalyzer().prepare(new))
    dump = decision.model_dump(mode="json")
    assert dump["outcome"] == "supersede"
    assert dump["evidence"] == ["same_slot", "explicit_change_cue"]
    assert "Python" not in str(dump) and "C++" not in str(dump)


@pytest.mark.asyncio
async def test_phase7_resolution_has_no_network_dependency(temporal_stack, monkeypatch):
    _, _, _, service = temporal_stack
    def blocked(*args, **kwargs):
        raise AssertionError("network access attempted")
    monkeypatch.setattr(socket.socket, "connect", blocked)
    result = await service.resolve(candidate(
        1, "User uses Python for ML.", observed_at=moment(2026)
    ))
    assert result.memory.status == MemoryStatus.ACTIVE


@pytest.mark.asyncio
async def test_migration_from_phase6_schema_preserves_memory(tmp_path: Path):
    path = tmp_path / "phase6.db"
    connection = sqlite3.connect(path)
    connection.executescript(SCHEMA_SQL)
    connection.execute(
        "INSERT INTO schema_version(version, description) VALUES (1, 'Initial schema')"
    )
    connection.execute(
        "INSERT INTO memories "
        "(id, content, content_hash, status, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (
            "92000000-0000-0000-0000-000000000001", "Preserved Phase 6 memory",
            "legacyhash", "active", moment(2026).isoformat(), moment(2026).isoformat(),
        ),
    )
    connection.commit()
    connection.close()

    database = Database(path)
    await database.initialize()
    try:
        cursor = await database.connection().execute("SELECT MAX(version) FROM schema_version")
        assert (await cursor.fetchone())[0] == SCHEMA_VERSION
        memory = await SqliteMemoryRepository(database.connection()).get(
            UUID("92000000-0000-0000-0000-000000000001")
        )
        assert memory.content == "Preserved Phase 6 memory"
        assert memory.observed_at == moment(2026)
        assert memory.temporal_precision == TemporalPrecision.UNKNOWN
    finally:
        await database.close()


def test_evaluation_dataset_contains_34_memories_across_required_behaviors():
    cases = evaluation_cases()
    assert len(cases) * 2 == 34
    assert {case.expected for case in cases} >= {
        TemporalOutcome.SUPERSEDE,
        TemporalOutcome.CORRECT,
        TemporalOutcome.COEXIST,
        TemporalOutcome.CONTRADICT,
        TemporalOutcome.DUPLICATE,
        TemporalOutcome.NO_CHANGE,
        TemporalOutcome.ADD_NEW,
    }


def test_contradiction_precision_and_recall_metrics():
    expected = [TemporalOutcome.CONTRADICT, TemporalOutcome.COEXIST]
    predicted = [TemporalOutcome.CONTRADICT, TemporalOutcome.CONTRADICT]
    assert outcome_precision(expected, predicted, TemporalOutcome.CONTRADICT) == 0.5
    assert outcome_recall(expected, predicted, TemporalOutcome.CONTRADICT) == 1.0


def test_supersession_precision_recall_and_false_supersession_metrics():
    expected = [TemporalOutcome.SUPERSEDE, TemporalOutcome.COEXIST]
    predicted = [TemporalOutcome.SUPERSEDE, TemporalOutcome.SUPERSEDE]
    assert outcome_precision(expected, predicted, TemporalOutcome.SUPERSEDE) == 0.5
    assert outcome_recall(expected, predicted, TemporalOutcome.SUPERSEDE) == 1.0
    assert false_supersession_rate(expected, predicted) == 1.0


def test_coexistence_current_history_and_consistency_metrics():
    expected = [
        TemporalOutcome.COEXIST,
        TemporalOutcome.CORRECT,
        TemporalOutcome.ADD_NEW,
    ]
    predicted = [
        TemporalOutcome.COEXIST,
        TemporalOutcome.SUPERSEDE,
        TemporalOutcome.ADD_NEW,
    ]
    assert coexistence_accuracy(expected, predicted) == 1.0
    assert current_state_accuracy(expected, predicted) == 1.0
    assert historical_state_recall(expected, predicted) == 1.0
    assert classification_accuracy(expected, predicted) == pytest.approx(2 / 3)
    assert timeline_consistency_violations(expected, predicted) == 1


@pytest.mark.asyncio
async def test_contextos_evaluation_metrics_are_deterministic():
    first = await run_evaluation()
    second = await run_evaluation()
    assert first == second
    contextos = first["CONTEXTOS_TEMPORAL"]
    assert contextos["contradiction_precision"] == 1.0
    assert contextos["contradiction_recall"] == 1.0
    assert contextos["supersession_precision"] == 1.0
    assert contextos["supersession_recall"] == 1.0
    assert contextos["coexistence_accuracy"] == 1.0
    assert contextos["false_supersession_rate"] == 0.0
    assert contextos["current_state_accuracy"] == 1.0
    assert contextos["historical_state_recall"] == 1.0
    assert contextos["timeline_consistency_violations"] == 0.0
