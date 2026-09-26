"""Phase 4 retrieval, synchronization, and evaluation acceptance tests."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import pytest_asyncio
from pydantic import ValidationError

from contextos.benchmarks.retrieval import (
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from contextos.core.enums import MemoryStatus, MemoryType, RetrievalMode, TemporalScope
from contextos.core.models import Memory, MemoryUpdate, RetrievalQuery
from contextos.embedding.deterministic import DeterministicEmbedding
from contextos.services.retrieval import HybridRetrievalEngine
from contextos.services.retrieval_index import RetrievalIndexSynchronizer
from contextos.storage.database import Database
from contextos.storage.lexical.bm25 import BM25Index
from contextos.storage.memory_repo import SqliteMemoryRepository
from contextos.storage.vector.in_memory import InMemoryVectorStore


@pytest_asyncio.fixture
async def retrieval_stack(tmp_path: Path):
    database = Database(tmp_path / "retrieval.db")
    await database.initialize()
    repository = SqliteMemoryRepository(database.connection())
    memories = [
        Memory(
            content="User currently uses Ollama for local model inference.",
            status=MemoryStatus.ACTIVE,
            type=MemoryType.FACT,
            source_type="cli",
            confidence=0.9,
            importance=0.8,
        ),
        Memory(
            content="User previously focused primarily on Python.",
            status=MemoryStatus.HISTORICAL,
            type=MemoryType.SKILL,
            source_type="import",
        ),
        Memory(
            content="User currently focuses on C++17 for systems interviews.",
            status=MemoryStatus.ACTIVE,
            type=MemoryType.GOAL,
            source_type="cli",
            tags=["career"],
        ),
        Memory(
            content="User prefers concise technical responses.",
            status=MemoryStatus.ACTIVE,
            type=MemoryType.PREFERENCE,
            source_type="cli",
        ),
        Memory(
            content="A deleted Docker runtime note.",
            status=MemoryStatus.DELETED,
            type=MemoryType.FACT,
        ),
        Memory(
            content="An expired Java learning goal.",
            status=MemoryStatus.EXPIRED,
            type=MemoryType.GOAL,
        ),
        Memory(
            content="User previously used llama.cpp for inference.",
            status=MemoryStatus.SUPERSEDED,
            type=MemoryType.FACT,
        ),
        Memory(
            content="A gardening calendar tracks tomato watering.",
            status=MemoryStatus.ACTIVE,
            type=MemoryType.PROJECT,
        ),
    ]
    for memory in memories:
        await repository.create(memory)
    embedding = DeterministicEmbedding(64)
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
    yield database, repository, memories, embedding, lexical, vector, synchronizer, engine
    await database.close()


@pytest.mark.asyncio
async def test_bm25_exact_retrieval_and_normalization():
    index = BM25Index()
    await index.index("right", "C++17 handles systems interviews.")
    await index.index("wrong", "Travel plans and garden tools.")
    results = await index.search("C++17, SYSTEMS!")
    assert [item.id for item in results] == ["right"]


@pytest.mark.asyncio
async def test_bm25_ranking_and_top_k():
    index = BM25Index()
    await index.rebuild({"a": "atlas atlas postgres", "b": "atlas notes", "c": "garden"})
    results = await index.search("atlas postgres", top_k=1)
    assert [item.id for item in results] == ["a"]


@pytest.mark.asyncio
async def test_bm25_empty_query_corpus_and_rebuild():
    index = BM25Index()
    assert await index.search("anything") == []
    await index.rebuild({"a": "first unique"})
    assert await index.search("") == []
    await index.rebuild({"b": "second unique"})
    assert await index.search("first") == []
    assert [item.id for item in await index.search("second")] == ["b"]


@pytest.mark.asyncio
async def test_cosine_similarity_and_duplicate_vector_update():
    store = InMemoryVectorStore(2)
    await store.add(["a", "b"], [[1, 0], [0, 1]], [{}, {}])
    assert [item.id for item in await store.search([0.9, 0.1])] == ["a", "b"]
    await store.add(["a"], [[0, 1]], [{}])
    assert (await store.search([0, 1]))[0].id == "a"
    assert await store.count() == 2


@pytest.mark.asyncio
async def test_vector_zero_and_dimension_validation():
    store = InMemoryVectorStore(2)
    await store.add(["a"], [[1, 0]], [{}])
    assert await store.search([0, 0]) == []
    with pytest.raises(ValueError, match="dimension"):
        await store.search([1, 0, 0])
    with pytest.raises(ValueError, match="dimension"):
        await store.add(["b"], [[1, 0, 0]], [{}])


@pytest.mark.asyncio
async def test_dense_semantic_retrieval(retrieval_stack):
    *_, engine = retrieval_stack
    result = await engine.retrieve(RetrievalQuery(
        text="Which local AI runtime handles inference?", mode=RetrievalMode.DENSE, k=3
    ))
    assert "Ollama" in result.memories[0].memory.content


@pytest.mark.asyncio
async def test_modes_and_hybrid_duplicate_fusion(retrieval_stack):
    *_, engine = retrieval_stack
    for mode in RetrievalMode:
        result = await engine.retrieve(RetrievalQuery(text="Ollama inference", mode=mode, k=3))
        assert result.memories
        assert len({item.memory.id for item in result.memories}) == len(result.memories)
        if mode == RetrievalMode.HYBRID:
            assert result.memories[0].retrieval_sources == ["lexical", "dense"]


@pytest.mark.asyncio
async def test_top_k_and_deterministic_order(retrieval_stack):
    *_, engine = retrieval_stack
    query = RetrievalQuery(text="user local tools", k=2)
    first = await engine.retrieve(query)
    second = await engine.retrieve(query)
    assert len(first.memories) <= 2
    assert [item.memory.id for item in first.memories] == [
        item.memory.id for item in second.memories
    ]


def test_empty_query_is_invalid():
    with pytest.raises(ValidationError):
        RetrievalQuery(text="  ")


@pytest.mark.asyncio
async def test_deleted_and_expired_excluded(retrieval_stack):
    *_, engine = retrieval_stack
    deleted = await engine.retrieve(RetrievalQuery(text="Docker runtime", k=10))
    expired = await engine.retrieve(RetrievalQuery(text="Java learning", k=10))
    assert all(item.memory.status != MemoryStatus.DELETED for item in deleted.memories)
    assert all(item.memory.status != MemoryStatus.EXPIRED for item in expired.memories)


@pytest.mark.asyncio
async def test_expired_status_can_be_requested_explicitly(retrieval_stack):
    *_, engine = retrieval_stack
    result = await engine.retrieve(RetrievalQuery(
        text="Java learning",
        allowed_statuses={MemoryStatus.EXPIRED},
        k=10,
    ))
    assert result.memories[0].memory.status == MemoryStatus.EXPIRED


@pytest.mark.asyncio
async def test_superseded_default_and_historical_behavior(retrieval_stack):
    *_, engine = retrieval_stack
    current = await engine.retrieve(RetrievalQuery(text="llama.cpp inference", k=10))
    historical = await engine.retrieve(RetrievalQuery(
        text="llama.cpp inference", temporal_scope=TemporalScope.HISTORICAL, k=10
    ))
    assert all(item.memory.status != MemoryStatus.SUPERSEDED for item in current.memories)
    assert historical.memories[0].memory.status == MemoryStatus.SUPERSEDED


@pytest.mark.asyncio
async def test_current_memory_preference(retrieval_stack):
    *_, engine = retrieval_stack
    current = await engine.retrieve(RetrievalQuery(text="language focus now", k=5))
    history = await engine.retrieve(RetrievalQuery(
        text="language focus previously", temporal_scope=TemporalScope.HISTORICAL, k=5
    ))
    assert "C++17" in current.memories[0].memory.content
    assert "Python" in history.memories[0].memory.content


@pytest.mark.asyncio
async def test_memory_type_source_and_tag_filters(retrieval_stack):
    *_, engine = retrieval_stack
    result = await engine.retrieve(RetrievalQuery(
        text="systems career",
        allowed_memory_types={MemoryType.GOAL},
        source_types={"cli"},
        tags={"career"},
        k=10,
    ))
    assert len(result.memories) == 1
    assert result.memories[0].memory.type == MemoryType.GOAL


@pytest.mark.asyncio
async def test_retrieval_does_not_mutate_access_metadata(retrieval_stack):
    _, repository, memories, *_, engine = retrieval_stack
    before = await repository.get(memories[0].id)
    await engine.retrieve(RetrievalQuery(text="Ollama", k=1))
    after = await repository.get(memories[0].id)
    assert before and after
    assert (after.access_count, after.last_accessed_at, after.version) == (
        before.access_count, before.last_accessed_at, before.version
    )


@pytest.mark.asyncio
async def test_trace_accuracy_and_optional_trace(retrieval_stack):
    *_, engine = retrieval_stack
    result = await engine.retrieve(RetrievalQuery(text="Ollama", k=2))
    assert result.trace.total_results == len(result.memories)
    assert result.trace.total_candidates >= len(result.memories)
    assert {stage.stage_name for stage in result.trace.stages} >= {
        "index_sync", "lexical_search", "dense_search", "eligibility_filter", "fusion_rerank"
    }
    hidden = await engine.retrieve(RetrievalQuery(text="Ollama", include_trace=False))
    assert hidden.trace.stages == []


@pytest.mark.asyncio
async def test_created_memory_index_consistency(retrieval_stack):
    _, repository, *_middle, engine = retrieval_stack
    created = await repository.create(Memory(
        content="ZephyrDB is the selected analytics database.", status=MemoryStatus.ACTIVE
    ))
    result = await engine.retrieve(RetrievalQuery(text="ZephyrDB", mode=RetrievalMode.LEXICAL))
    assert result.memories[0].memory.id == created.id


@pytest.mark.asyncio
async def test_updated_memory_index_consistency(retrieval_stack):
    _, repository, memories, *_middle, engine = retrieval_stack
    target = memories[-1]
    await engine.retrieve(RetrievalQuery(text="gardening"))
    await repository.update(
        target.id,
        MemoryUpdate(content="NebulaDB stores astronomy records."),
        target.version,
    )
    assert not (await engine.retrieve(RetrievalQuery(text="gardening"))).memories
    updated = await engine.retrieve(RetrievalQuery(text="NebulaDB"))
    assert updated.memories[0].memory.id == target.id


@pytest.mark.asyncio
async def test_expired_memory_index_consistency(retrieval_stack):
    _, repository, memories, *_middle, engine = retrieval_stack
    target = memories[0]
    await repository.update_status(target.id, MemoryStatus.EXPIRED, target.version)
    result = await engine.retrieve(RetrievalQuery(text="Ollama", k=10))
    assert all(item.memory.id != target.id for item in result.memories)


@pytest.mark.asyncio
async def test_deleted_memory_index_consistency(retrieval_stack):
    _, repository, memories, *_middle, engine = retrieval_stack
    target = memories[-1]
    await engine.retrieve(RetrievalQuery(text="gardening"))
    await repository.delete(target.id)
    result = await engine.retrieve(RetrievalQuery(text="gardening", k=10))
    assert all(item.memory.id != target.id for item in result.memories)


@pytest.mark.asyncio
async def test_supersede_index_consistency(retrieval_stack):
    _, repository, memories, *_middle, engine = retrieval_stack
    old = memories[0]
    successor = Memory(
        content="User currently uses LocalAI for local model inference.",
        status=MemoryStatus.ACTIVE,
    )
    await repository.supersede(old.id, successor, old.version)
    current = await engine.retrieve(RetrievalQuery(text="local model inference", k=10))
    historical = await engine.retrieve(RetrievalQuery(
        text="Ollama inference", temporal_scope=TemporalScope.HISTORICAL, k=10
    ))
    assert current.memories[0].memory.id == successor.id
    assert any(item.memory.id == old.id for item in historical.memories)


@pytest.mark.asyncio
async def test_restart_rebuild_from_disk(tmp_path: Path):
    path = tmp_path / "restart.db"
    database = Database(path)
    await database.initialize()
    repository = SqliteMemoryRepository(database.connection())
    memory = await repository.create(
        Memory(
            content="QuasarCLI is the deployment tool.",
            status=MemoryStatus.ACTIVE,
        )
    )
    await database.close()

    reopened = Database(path)
    await reopened.initialize()
    repository = SqliteMemoryRepository(reopened.connection())
    embedding = DeterministicEmbedding(32)
    lexical = BM25Index()
    vector = InMemoryVectorStore(32)
    sync = RetrievalIndexSynchronizer(
        memory_repo=repository, lexical_index=lexical, vector_store=vector,
        embedding_service=embedding,
    )
    engine = HybridRetrievalEngine(
        memory_repo=repository, lexical_index=lexical, vector_store=vector,
        embedding_service=embedding, index_synchronizer=sync,
    )
    result = await engine.retrieve(RetrievalQuery(text="QuasarCLI deployment"))
    assert result.memories[0].memory.id == memory.id
    await reopened.close()


@pytest.mark.parametrize(
    ("retrieved", "qrels", "k", "expected"),
    [
        (["a", "x"], {"a": 1, "b": 1}, 2, 0.5),
        ([], {"a": 1}, 5, 0.0),
        (["a"], {}, 5, 0.0),
        (["a", "b"], {"a": 1, "b": 1}, 10, 1.0),
    ],
)
def test_recall_at_k(retrieved, qrels, k, expected):
    assert recall_at_k(retrieved, qrels, k) == expected


def test_precision_and_hit_rate_math():
    assert precision_at_k(["a", "x", "b"], {"a": 1, "b": 1}, 3) == pytest.approx(2 / 3)
    assert precision_at_k([], {"a": 1}, 5) == 0.0
    assert hit_rate_at_k(["x", "a"], {"a": 1}, 1) == 0.0
    assert hit_rate_at_k(["x", "a"], {"a": 1}, 2) == 1.0


def test_mrr_math_with_multiple_relevant_documents():
    assert reciprocal_rank(["x", "b", "a"], {"a": 1, "b": 2}) == 0.5
    assert reciprocal_rank([], {"a": 1}) == 0.0


def test_ndcg_graded_relevance_math_and_large_k():
    actual = 3 / math.log2(2) + 7 / math.log2(3)
    ideal = 7 / math.log2(2) + 3 / math.log2(3)
    assert ndcg_at_k(["b", "a"], {"a": 3, "b": 2}, 10) == pytest.approx(actual / ideal)
    assert ndcg_at_k([], {"a": 3}, 10) == 0.0
    assert ndcg_at_k(["x"], {}, 10) == 0.0


@pytest.mark.asyncio
async def test_adversarial_distractor_does_not_beat_exact_identifier(retrieval_stack):
    *_, engine = retrieval_stack
    result = await engine.retrieve(RetrievalQuery(
        text="Ollama", mode=RetrievalMode.HYBRID, k=5
    ))
    assert result.memories[0].memory.content.startswith("User currently uses Ollama")
