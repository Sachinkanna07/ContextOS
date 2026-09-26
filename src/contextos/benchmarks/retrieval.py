"""Deterministic, offline Phase 4 retrieval evaluation.

Run with: python -m contextos.benchmarks.retrieval
"""

from __future__ import annotations

import asyncio
import math
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from contextos.core.enums import MemoryStatus, MemoryType, RetrievalMode, TemporalScope
from contextos.core.models import Memory, RetrievalQuery
from contextos.embedding.deterministic import DeterministicEmbedding
from contextos.services.retrieval import HybridRetrievalEngine
from contextos.services.retrieval_index import RetrievalIndexSynchronizer
from contextos.storage.database import Database
from contextos.storage.lexical.bm25 import BM25Index
from contextos.storage.memory_repo import SqliteMemoryRepository
from contextos.storage.vector.in_memory import InMemoryVectorStore


@dataclass(frozen=True)
class EvaluationCase:
    query_id: str
    text: str
    qrels: dict[str, int]
    temporal_scope: TemporalScope = TemporalScope.CURRENT
    memory_types: set[MemoryType] | None = None


@dataclass
class EvaluationSummary:
    recall: float
    precision: float
    hit_rate: float
    mrr: float
    ndcg: float
    average_latency_ms: float
    stage_latency_ms: dict[str, float] = field(default_factory=dict)
    rankings: dict[str, list[str]] = field(default_factory=dict)


def recall_at_k(retrieved: list[str], qrels: dict[str, int], k: int) -> float:
    relevant = {identifier for identifier, grade in qrels.items() if grade > 0}
    if not relevant:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def precision_at_k(retrieved: list[str], qrels: dict[str, int], k: int) -> float:
    if k <= 0:
        return 0.0
    relevant = {identifier for identifier, grade in qrels.items() if grade > 0}
    return len(set(retrieved[:k]) & relevant) / k


def hit_rate_at_k(retrieved: list[str], qrels: dict[str, int], k: int) -> float:
    return float(any(qrels.get(identifier, 0) > 0 for identifier in retrieved[:k]))


def reciprocal_rank(retrieved: list[str], qrels: dict[str, int]) -> float:
    for rank, identifier in enumerate(retrieved, 1):
        if qrels.get(identifier, 0) > 0:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list[str], qrels: dict[str, int], k: int) -> float:
    def dcg(grades: list[int]) -> float:
        return sum((2**grade - 1) / math.log2(rank + 1) for rank, grade in enumerate(grades, 1))

    actual = dcg([qrels.get(identifier, 0) for identifier in retrieved[:k]])
    ideal = dcg(sorted(qrels.values(), reverse=True)[:k])
    return actual / ideal if ideal else 0.0


def evaluation_memories() -> list[Memory]:
    specs = [
        ("m01", "User currently uses Ollama for local model inference.",
         MemoryStatus.ACTIVE, MemoryType.FACT),
        ("m02", "User previously focused primarily on Python.",
         MemoryStatus.HISTORICAL, MemoryType.SKILL),
        ("m03", "User currently focuses on C++17 for systems interviews.",
         MemoryStatus.ACTIVE, MemoryType.GOAL),
        ("m04", "User prefers concise technical responses.",
         MemoryStatus.ACTIVE, MemoryType.PREFERENCE),
        ("m05", "User enjoys biryani.", MemoryStatus.ACTIVE, MemoryType.PREFERENCE),
        (
            "m06",
            "Qwen 30B, the much larger model, failed locally because available "
            "memory was insufficient.",
            MemoryStatus.ACTIVE,
            MemoryType.FACT,
        ),
        ("m07", "User has successfully run Qwen 9B locally.",
         MemoryStatus.ACTIVE, MemoryType.FACT),
        ("m08", "The Atlas project uses PostgreSQL for durable storage.",
         MemoryStatus.ACTIVE, MemoryType.PROJECT),
        ("m09", "User tests Python services with pytest.",
         MemoryStatus.ACTIVE, MemoryType.PROCEDURE),
        ("m10", "User likes detailed travel stories.",
         MemoryStatus.ACTIVE, MemoryType.PREFERENCE),
        ("m11", "A deleted note claimed the runtime was Docker.",
         MemoryStatus.DELETED, MemoryType.FACT),
        ("m12", "An expired goal was to study Java.",
         MemoryStatus.EXPIRED, MemoryType.GOAL),
        ("m13", "User builds local search tools with SQLite.",
         MemoryStatus.ACTIVE, MemoryType.PROJECT),
        ("m14", "User previously used llama.cpp for inference.",
         MemoryStatus.SUPERSEDED, MemoryType.FACT),
    ]
    from uuid import UUID

    return [
        Memory(
            id=UUID(f"00000000-0000-0000-0000-{index:012d}"),
            content=content,
            status=status,
            type=memory_type,
            source_type="benchmark",
            confidence=0.9,
            importance=0.7,
            tags=[short_id],
        )
        for index, (short_id, content, status, memory_type) in enumerate(specs, 1)
    ]


def evaluation_cases() -> list[EvaluationCase]:
    def uid(number: int) -> str:
        return f"00000000-0000-0000-0000-{number:012d}"

    return [
        EvaluationCase(
            "q1", "Which local AI runtime does the user use?", {uid(1): 2, uid(7): 1}
        ),
        EvaluationCase("q2", "What programming language is the user focusing on now?", {uid(3): 2}),
        EvaluationCase(
            "q3", "What language did the user focus on previously?", {uid(2): 2},
            TemporalScope.HISTORICAL,
        ),
        EvaluationCase("q4", "What response style does the user prefer?", {uid(4): 2}),
        EvaluationCase(
            "q5",
            "What happened when the user tried a much larger Qwen model?",
            {uid(6): 2},
        ),
        EvaluationCase("q6", "Which database backs the Atlas project?", {uid(8): 2}),
        EvaluationCase(
            "q7", "What active career learning objective involves systems?", {uid(3): 2},
            memory_types={MemoryType.GOAL},
        ),
    ]


async def build_engine(path: Path) -> tuple[Database, HybridRetrievalEngine]:
    database = Database(path)
    await database.initialize()
    repository = SqliteMemoryRepository(database.connection())
    for memory in evaluation_memories():
        await repository.create(memory)
    embedding = DeterministicEmbedding()
    lexical = BM25Index()
    vector = InMemoryVectorStore(embedding.dimension)
    synchronizer = RetrievalIndexSynchronizer(
        memory_repo=repository,
        lexical_index=lexical,
        vector_store=vector,
        embedding_service=embedding,
    )
    engine = HybridRetrievalEngine(
        memory_repo=repository,
        lexical_index=lexical,
        vector_store=vector,
        embedding_service=embedding,
        index_synchronizer=synchronizer,
    )
    return database, engine


async def evaluate(
    engine: HybridRetrievalEngine, mode: RetrievalMode, *, k: int = 5
) -> EvaluationSummary:
    totals = {"recall": 0.0, "precision": 0.0, "hit": 0.0, "mrr": 0.0, "ndcg": 0.0}
    latencies: list[float] = []
    stage_totals: dict[str, float] = {}
    stage_counts: dict[str, int] = {}
    rankings: dict[str, list[str]] = {}
    cases = evaluation_cases()
    for case in cases:
        started = time.perf_counter()
        result = await engine.retrieve(RetrievalQuery(
            text=case.text,
            k=k,
            mode=mode,
            temporal_scope=case.temporal_scope,
            allowed_memory_types=case.memory_types,
        ))
        latencies.append((time.perf_counter() - started) * 1000)
        for stage in result.trace.stages:
            stage_totals[stage.stage_name] = (
                stage_totals.get(stage.stage_name, 0.0) + stage.latency_ms
            )
            stage_counts[stage.stage_name] = stage_counts.get(stage.stage_name, 0) + 1
        identifiers = [str(item.memory.id) for item in result.memories]
        rankings[case.query_id] = [item.memory.content for item in result.memories]
        totals["recall"] += recall_at_k(identifiers, case.qrels, k)
        totals["precision"] += precision_at_k(identifiers, case.qrels, k)
        totals["hit"] += hit_rate_at_k(identifiers, case.qrels, k)
        totals["mrr"] += reciprocal_rank(identifiers, case.qrels)
        totals["ndcg"] += ndcg_at_k(identifiers, case.qrels, k)
    count = len(cases)
    return EvaluationSummary(
        recall=totals["recall"] / count,
        precision=totals["precision"] / count,
        hit_rate=totals["hit"] / count,
        mrr=totals["mrr"] / count,
        ndcg=totals["ndcg"] / count,
        average_latency_ms=sum(latencies) / count,
        stage_latency_ms={
            name: total / stage_counts[name] for name, total in stage_totals.items()
        },
        rankings=rankings,
    )


async def _main() -> None:
    with tempfile.TemporaryDirectory(prefix="contextos-retrieval-") as directory:
        database, engine = await build_engine(Path(directory) / "benchmark.db")
        try:
            print("Mode       Recall@5  Precision@5  HitRate@5  MRR     NDCG@5  Avg ms")
            summaries: dict[RetrievalMode, EvaluationSummary] = {}
            for mode in RetrievalMode:
                summaries[mode] = await evaluate(engine, mode)
                item = summaries[mode]
                print(
                    f"{mode.value:<10} {item.recall:>8.3f}  {item.precision:>11.3f}  "
                    f"{item.hit_rate:>9.3f}  {item.mrr:>6.3f}  {item.ndcg:>6.3f}  "
                    f"{item.average_latency_ms:>6.3f}"
                )
            print("\nAverage stage latency (ms):")
            print("Mode       Lexical   Dense   Fusion   Total")
            for mode, item in summaries.items():
                print(
                    f"{mode.value:<10} "
                    f"{item.stage_latency_ms.get('lexical_search', 0.0):>7.3f} "
                    f"{item.stage_latency_ms.get('dense_search', 0.0):>7.3f} "
                    f"{item.stage_latency_ms.get('fusion_rerank', 0.0):>7.3f} "
                    f"{item.average_latency_ms:>7.3f}"
                )
            print("\nSmoke rankings (top 3):")
            for case in evaluation_cases()[:5]:
                print(f"\n{case.query_id}: {case.text}")
                for mode in RetrievalMode:
                    ranking = summaries[mode].rankings[case.query_id][:3]
                    print(f"  {mode.value:<8} " + " | ".join(ranking))
        finally:
            await database.close()


if __name__ == "__main__":
    asyncio.run(_main())
