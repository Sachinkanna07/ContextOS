"""Deterministic Phase 7 temporal evaluation, smoke, and scale harnesses."""

from __future__ import annotations

import asyncio
import json
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from contextos.core.enums import (
    CandidateTemporalStatus,
    MemoryStatus,
    MemoryType,
    RetrievalMode,
    TemporalOutcome,
    TemporalScope,
)
from contextos.core.models import Memory, MemorySlot, RetrievalQuery
from contextos.embedding.deterministic import DeterministicEmbedding
from contextos.services.retrieval import HybridRetrievalEngine
from contextos.services.retrieval_index import RetrievalIndexSynchronizer
from contextos.services.temporal import TemporalMemoryService
from contextos.storage.database import Database
from contextos.storage.lexical.bm25 import BM25Index
from contextos.storage.memory_repo import SqliteMemoryRepository
from contextos.storage.vector.in_memory import InMemoryVectorStore


UTC = timezone.utc


@dataclass(frozen=True)
class TemporalEvaluationCase:
    name: str
    previous: str
    candidate: str
    expected: TemporalOutcome
    previous_observed: datetime
    candidate_observed: datetime
    previous_valid: datetime | None = None
    candidate_valid: datetime | None = None


def evaluation_cases() -> list[TemporalEvaluationCase]:
    """Seventeen balanced cases, totaling 34 synthetic memory records."""
    old = datetime(2025, 1, 1, tzinfo=UTC)
    new = datetime(2026, 1, 1, tzinfo=UTC)
    late = datetime(2026, 9, 1, tzinfo=UTC)
    return [
        TemporalEvaluationCase("language_transition", "User uses Python for systems interviews.", "User now uses C++17 for systems interviews.", TemporalOutcome.SUPERSEDE, old, new),
        TemporalEvaluationCase("ram_correction", "My machine has 16 GB RAM.", "Correction: my machine actually has 32 GB RAM.", TemporalOutcome.CORRECT, old, new),
        TemporalEvaluationCase("tool_negation", "I use Docker.", "I no longer use Docker.", TemporalOutcome.SUPERSEDE, old, new),
        TemporalEvaluationCase("response_change", "User prefers concise answers.", "User now prefers detailed answers.", TemporalOutcome.SUPERSEDE, old, new),
        TemporalEvaluationCase("local_model_change", "User uses Ollama as a local model.", "User now uses Qwen9B as a local model.", TemporalOutcome.SUPERSEDE, old, new),
        TemporalEvaluationCase("project_change", "Project Atlas is active.", "Project Atlas is now paused.", TemporalOutcome.SUPERSEDE, old, new),
        TemporalEvaluationCase("skill_progression", "User is a Rust beginner.", "User is now proficient in Rust.", TemporalOutcome.SUPERSEDE, old, new),
        TemporalEvaluationCase("language_scopes", "User uses Python for machine learning.", "User uses C++ for systems programming.", TemporalOutcome.COEXIST, old, new),
        TemporalEvaluationCase("web_scope", "User uses Python for machine learning.", "User uses JavaScript for web development.", TemporalOutcome.COEXIST, old, new),
        TemporalEvaluationCase("future_plan", "User currently uses Python.", "User might learn Rust next year.", TemporalOutcome.COEXIST, old, new),
        TemporalEvaluationCase("uncertain_change", "User uses Python.", "I think I prefer C++ now.", TemporalOutcome.COEXIST, old, new),
        TemporalEvaluationCase("scoped_negation", "I use Docker generally.", "I don't use Docker for Project X.", TemporalOutcome.COEXIST, old, new),
        TemporalEvaluationCase("attendance_conflict", "I attended Event X.", "I did not attend Event X.", TemporalOutcome.CONTRADICT, old, new),
        TemporalEvaluationCase("preference_conflict", "User prefers concise answers.", "User prefers detailed answers.", TemporalOutcome.CONTRADICT, old, new),
        TemporalEvaluationCase("exact_duplicate", "User uses Python for ML.", "User uses Python for ML.", TemporalOutcome.DUPLICATE, old, new),
        TemporalEvaluationCase("semantic_no_change", "User uses Python for ML.", "User currently uses Python for machine learning.", TemporalOutcome.NO_CHANGE, old, new),
        TemporalEvaluationCase("late_historical_import", "User currently uses C++ for systems interviews.", "User used Python for systems interviews in 2025.", TemporalOutcome.ADD_NEW, new, late, new, old),
    ]


def classification_accuracy(expected: list[TemporalOutcome], predicted: list[TemporalOutcome]) -> float:
    return sum(left == right for left, right in zip(expected, predicted, strict=True)) / len(expected)


def outcome_precision(
    expected: list[TemporalOutcome], predicted: list[TemporalOutcome], outcome: TemporalOutcome
) -> float:
    chosen = [index for index, value in enumerate(predicted) if value == outcome]
    if not chosen:
        return 0.0 if outcome in expected else 1.0
    return sum(expected[index] == outcome for index in chosen) / len(chosen)


def outcome_recall(
    expected: list[TemporalOutcome], predicted: list[TemporalOutcome], outcome: TemporalOutcome
) -> float:
    actual = [index for index, value in enumerate(expected) if value == outcome]
    return sum(predicted[index] == outcome for index in actual) / len(actual) if actual else 1.0


def false_supersession_rate(
    expected: list[TemporalOutcome], predicted: list[TemporalOutcome]
) -> float:
    wrong = [
        index for index, value in enumerate(predicted)
        if value in {TemporalOutcome.SUPERSEDE, TemporalOutcome.CORRECT}
        and expected[index] not in {TemporalOutcome.SUPERSEDE, TemporalOutcome.CORRECT}
    ]
    negatives = sum(
        value not in {TemporalOutcome.SUPERSEDE, TemporalOutcome.CORRECT}
        for value in expected
    )
    return len(wrong) / negatives if negatives else 0.0


def coexistence_accuracy(expected: list[TemporalOutcome], predicted: list[TemporalOutcome]) -> float:
    indices = [index for index, value in enumerate(expected) if value == TemporalOutcome.COEXIST]
    return sum(predicted[index] == TemporalOutcome.COEXIST for index in indices) / len(indices)


def current_state_accuracy(
    expected: list[TemporalOutcome], predicted: list[TemporalOutcome]
) -> float:
    def family(value: TemporalOutcome) -> str:
        if value in {TemporalOutcome.SUPERSEDE, TemporalOutcome.CORRECT}:
            return "new_current"
        if value in {TemporalOutcome.DUPLICATE, TemporalOutcome.NO_CHANGE}:
            return "old_current"
        return value.value
    return sum(
        family(left) == family(right)
        for left, right in zip(expected, predicted, strict=True)
    ) / len(expected)


def historical_state_recall(
    expected: list[TemporalOutcome], predicted: list[TemporalOutcome]
) -> float:
    indices = [
        index for index, value in enumerate(expected)
        if value in {TemporalOutcome.SUPERSEDE, TemporalOutcome.CORRECT, TemporalOutcome.ADD_NEW}
    ]
    return sum(
        predicted[index] in {TemporalOutcome.SUPERSEDE, TemporalOutcome.CORRECT}
        if expected[index] in {TemporalOutcome.SUPERSEDE, TemporalOutcome.CORRECT}
        else predicted[index] == TemporalOutcome.ADD_NEW
        for index in indices
    ) / len(indices)


def timeline_consistency_violations(
    expected: list[TemporalOutcome], predicted: list[TemporalOutcome]
) -> int:
    """Count decisions that would assign the wrong current/history relationship."""
    return sum(
        left != right
        for left, right in zip(expected, predicted, strict=True)
    )


def naive_latest_write(cases: list[TemporalEvaluationCase]) -> list[TemporalOutcome]:
    return [
        TemporalOutcome.DUPLICATE if case.previous == case.candidate
        else TemporalOutcome.SUPERSEDE
        for case in cases
    ]


def timestamp_only(cases: list[TemporalEvaluationCase]) -> list[TemporalOutcome]:
    return [
        TemporalOutcome.DUPLICATE if case.previous == case.candidate
        else TemporalOutcome.SUPERSEDE
        if case.candidate_observed >= case.previous_observed
        else TemporalOutcome.COEXIST
        for case in cases
    ]


async def _contextos_predictions(
    cases: list[TemporalEvaluationCase], directory: Path
) -> list[TemporalOutcome]:
    predicted: list[TemporalOutcome] = []
    for index, case in enumerate(cases):
        database = Database(directory / f"case-{index}.db")
        await database.initialize()
        try:
            service = TemporalMemoryService(SqliteMemoryRepository(database.connection()))
            previous = Memory(
                id=uuid5(NAMESPACE_URL, f"{case.name}:previous"),
                content=case.previous,
                type=MemoryType.FACT,
                status=MemoryStatus.CANDIDATE,
                observed_at=case.previous_observed,
                valid_from=case.previous_valid,
            )
            candidate = Memory(
                id=uuid5(NAMESPACE_URL, f"{case.name}:candidate"),
                content=case.candidate,
                type=MemoryType.FACT,
                status=MemoryStatus.CANDIDATE,
                observed_at=case.candidate_observed,
                valid_from=case.candidate_valid,
            )
            await service.resolve(previous)
            predicted.append((await service.resolve(candidate)).decision.outcome)
        finally:
            await database.close()
    return predicted


def metric_summary(
    expected: list[TemporalOutcome], predicted: list[TemporalOutcome]
) -> dict[str, float]:
    return {
        "relation_classification_accuracy": classification_accuracy(expected, predicted),
        "supersession_precision": outcome_precision(expected, predicted, TemporalOutcome.SUPERSEDE),
        "supersession_recall": outcome_recall(expected, predicted, TemporalOutcome.SUPERSEDE),
        "contradiction_precision": outcome_precision(expected, predicted, TemporalOutcome.CONTRADICT),
        "contradiction_recall": outcome_recall(expected, predicted, TemporalOutcome.CONTRADICT),
        "coexistence_accuracy": coexistence_accuracy(expected, predicted),
        "false_supersession_rate": false_supersession_rate(expected, predicted),
        "current_state_accuracy": current_state_accuracy(expected, predicted),
        "historical_state_recall": historical_state_recall(expected, predicted),
        "timeline_consistency_violations": float(
            timeline_consistency_violations(expected, predicted)
        ),
    }


async def run_evaluation() -> dict[str, dict[str, float]]:
    cases = evaluation_cases()
    expected = [case.expected for case in cases]
    with tempfile.TemporaryDirectory(prefix="contextos-temporal-eval-") as raw:
        contextos = await _contextos_predictions(cases, Path(raw))
    return {
        "NAIVE_LATEST_WRITE": metric_summary(expected, naive_latest_write(cases)),
        "TIMESTAMP_ONLY": metric_summary(expected, timestamp_only(cases)),
        "CONTEXTOS_TEMPORAL": metric_summary(expected, contextos),
    }


async def run_smoke() -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="contextos-temporal-smoke-") as raw:
        path = Path(raw) / "timeline.db"
        database = Database(path)
        await database.initialize()
        try:
            repository = SqliteMemoryRepository(database.connection())
            service = TemporalMemoryService(repository)
            statements = [
                "User primarily uses Python for systems interview preparation.",
                "User switched from Python and now primarily uses C++17 for systems interviews.",
                "User might learn Rust later.",
                "User uses Python for machine learning projects.",
            ]
            resolutions = []
            for index, content in enumerate(statements):
                resolutions.append(await service.resolve(Memory(
                    id=uuid5(NAMESPACE_URL, f"smoke:{index}"),
                    content=content,
                    type=MemoryType.FACT,
                    status=MemoryStatus.CANDIDATE,
                    observed_at=datetime(2025 + min(index, 1), index + 1, 1, tzinfo=UTC),
                )))

            systems_slot = resolutions[1].decision.slot
            ml_slot = resolutions[3].decision.slot
            embedding = DeterministicEmbedding(64)
            lexical = BM25Index()
            vector = InMemoryVectorStore(64)
            sync = RetrievalIndexSynchronizer(
                memory_repo=repository, lexical_index=lexical, vector_store=vector,
                embedding_service=embedding,
            )
            retrieval = HybridRetrievalEngine(
                memory_repo=repository, lexical_index=lexical, vector_store=vector,
                embedding_service=embedding, index_synchronizer=sync,
            )
            queries = {
                "systems_now": RetrievalQuery(
                    text="language systems interviews now", mode=RetrievalMode.LEXICAL
                ),
                "before_cpp": RetrievalQuery(
                    text="Python systems interviews before", mode=RetrievalMode.LEXICAL,
                    temporal_scope=TemporalScope.HISTORICAL,
                ),
                "ml": RetrievalQuery(
                    text="language machine learning", mode=RetrievalMode.LEXICAL
                ),
                "future": RetrievalQuery(
                    text="learn later Rust", mode=RetrievalMode.LEXICAL,
                    temporal_scope=TemporalScope.ALL,
                ),
            }
            query_results = {}
            for name, query in queries.items():
                result = await retrieval.retrieve(query)
                query_results[name] = [item.memory.content for item in result.memories]

            return {
                "database_on_disk": path.exists(),
                "current_state": {
                    "systems_interviews": [
                        memory.content for memory in await service.get_current_state(systems_slot)
                    ],
                    "machine_learning": [
                        memory.content for memory in await service.get_current_state(ml_slot)
                    ],
                },
                "history": {
                    "systems_interviews": [
                        memory.content for memory in await service.get_history(systems_slot)
                    ]
                },
                "future": [memory.content for memory in await service.get_future()],
                "queries": query_results,
                "traces": [result.decision.model_dump(mode="json") for result in resolutions],
            }
        finally:
            await database.close()


async def run_scale(count: int = 1_000) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="contextos-temporal-scale-") as raw:
        path = Path(raw) / "scale.db"
        database = Database(path)
        await database.initialize()
        repository = SqliteMemoryRepository(database.connection())
        service = TemporalMemoryService(repository)
        started = time.perf_counter()
        for index in range(count):
            await service.resolve(Memory(
                id=uuid5(NAMESPACE_URL, f"scale:{index}"),
                content=f"Synthetic value {index} for benchmark slot {index}.",
                type=MemoryType.FACT,
                status=MemoryStatus.CANDIDATE,
                observed_at=datetime(2026, 1, 1, tzinfo=UTC),
                slot=MemorySlot(
                    subject="synthetic", property="benchmark_value", scope=f"slot_{index}"
                ),
            ))
        resolution_ms = (time.perf_counter() - started) * 1000
        lookup_started = time.perf_counter()
        for index in range(100):
            await service.get_current_state(
                MemorySlot(
                    subject="synthetic", property="benchmark_value", scope=f"slot_{index}"
                )
            )
        lookup_ms = (time.perf_counter() - lookup_started) * 1000
        await database.close()

        restart_started = time.perf_counter()
        reopened = Database(path)
        await reopened.initialize()
        try:
            reopened_repo = SqliteMemoryRepository(reopened.connection())
            persisted = await reopened_repo.count()
            restart_ms = (time.perf_counter() - restart_started) * 1000
        finally:
            await reopened.close()
        return {
            "memories": count,
            "resolution_total_ms": round(resolution_ms, 3),
            "resolution_average_ms": round(resolution_ms / count, 6),
            "timeline_lookups": 100,
            "timeline_lookup_total_ms": round(lookup_ms, 3),
            "timeline_lookup_average_ms": round(lookup_ms / 100, 6),
            "restart_and_count_ms": round(restart_ms, 3),
            "persisted_after_restart": persisted,
            "claim": "synthetic timing only; not a production scalability claim",
        }


async def _main() -> None:
    print(json.dumps({
        "evaluation": await run_evaluation(),
        "smoke": await run_smoke(),
        "scale": await run_scale(),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    asyncio.run(_main())
