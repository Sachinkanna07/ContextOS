"""Deterministic acceptance tests for Phase 8 memory graph retrieval."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from uuid import uuid4

import pytest
import pytest_asyncio

from contextos.core.enums import (
    GraphNodeType,
    GraphRelationType,
    MemoryStatus,
    MemoryType,
    RetrievalMode,
    TemporalScope,
)
from contextos.core.models import (
    GraphEdge,
    GraphEdgeSupport,
    GraphNode,
    Memory,
    MemoryRelation,
    MemoryUpdate,
    RetrievalQuery,
)
from contextos.embedding.deterministic import DeterministicEmbedding
from contextos.services.graph import (
    DeterministicEntityExtractor,
    MemoryGraphService,
    stable_edge_id,
    stable_node_id,
)
from contextos.services.graph_retrieval import GraphAugmentedRetrievalEngine
from contextos.services.retrieval import HybridRetrievalEngine
from contextos.services.retrieval_index import RetrievalIndexSynchronizer
from contextos.storage.database import MIGRATION_2_SQL, SCHEMA_SQL, SCHEMA_VERSION, Database
from contextos.storage.graph_repo import SqliteGraphRepository
from contextos.storage.lexical.bm25 import BM25Index
from contextos.storage.memory_repo import SqliteMemoryRepository
from contextos.storage.relation_repo import SqliteRelationRepository
from contextos.storage.vector.in_memory import InMemoryVectorStore


@pytest_asyncio.fixture
async def graph_stack(tmp_path: Path):
    database = Database(tmp_path / "graph.db")
    await database.initialize()
    memory_repo = SqliteMemoryRepository(database.connection())
    relation_repo = SqliteRelationRepository(database.connection())
    graph_repo = SqliteGraphRepository(database.connection())
    service = MemoryGraphService(
        memory_repo=memory_repo, relation_repo=relation_repo, graph_repo=graph_repo,
    )
    yield database, memory_repo, relation_repo, graph_repo, service
    await database.close()


async def _add(repo: SqliteMemoryRepository, content: str, **updates) -> Memory:
    memory = Memory(
        content=content,
        type=updates.pop("type", MemoryType.PROJECT),
        status=updates.pop("status", MemoryStatus.ACTIVE),
        **updates,
    )
    return await repo.create(memory)


def _retrieval_engine(memory_repo, service, dimension: int = 16):
    embedding = DeterministicEmbedding(dimension)
    lexical = BM25Index()
    vector = InMemoryVectorStore(dimension)
    base = HybridRetrievalEngine(
        memory_repo=memory_repo, lexical_index=lexical, vector_store=vector,
        embedding_service=embedding,
        index_synchronizer=RetrievalIndexSynchronizer(
            memory_repo=memory_repo, lexical_index=lexical, vector_store=vector,
            embedding_service=embedding,
        ),
    )
    return GraphAugmentedRetrievalEngine(
        base_engine=base, graph_service=service, memory_repo=memory_repo,
    )


def test_explicit_relations_and_stable_canonical_ids():
    extractor = DeterministicEntityExtractor()
    relation = extractor.relations("Project Atlas uses Ollama")[0]
    assert relation.source.key == "atlas"
    assert relation.target.key == "ollama"
    assert relation.relation_type == GraphRelationType.USES
    assert relation.scope_key == "atlas"
    assert stable_node_id(GraphNodeType.PROJECT, "atlas") == stable_node_id(
        GraphNodeType.PROJECT, "atlas"
    )


@pytest.mark.parametrize(
    "text",
    [
        "Atlas does not use Docker",
        "Atlas never uses Docker",
        "Atlas no longer uses Docker",
    ],
)
def test_negation_prevents_structural_edges(text: str):
    assert DeterministicEntityExtractor().relations(text) == []


def test_cooccurrence_does_not_invent_relation():
    extractor = DeterministicEntityExtractor()
    assert extractor.relations("Atlas, Ollama, and Qwen30B were discussed") == []


def test_compound_explicit_claims_do_not_cross_or_form_phrase_edges():
    relations = DeterministicEntityExtractor().relations(
        "Project A uses Python and Project B uses Rust."
    )
    assert {(item.source.key, item.target.key) for item in relations} == {
        ("a", "python"), ("b", "rust"),
    }


def test_rejected_consideration_and_embedded_dot_name_are_safe():
    extractor = DeterministicEntityExtractor()
    assert extractor.relations("Project Atlas considered Docker but rejected it.") == []
    relation = extractor.relations("Project Atlas used llama.cpp.")[0]
    assert relation.target.key == "llama.cpp"


def test_temporal_replacement_sentence_keeps_current_explicit_relation():
    relations = DeterministicEntityExtractor().relations(
        "Project Atlas used Docker previously but now uses Podman."
    )
    assert [(item.relation_type, item.target.key) for item in relations] == [
        (GraphRelationType.USES, "podman")
    ]


@pytest.mark.asyncio
async def test_graph_rebuild_persists_nodes_edges_and_provenance(graph_stack):
    database, memory_repo, _, graph_repo, service = graph_stack
    event_id = uuid4()
    memory = await _add(
        memory_repo, "Atlas uses Ollama", provenance_event_id=event_id,
    )
    counts = await service.rebuild()
    assert counts[0] >= 3
    assert counts[1] >= 3
    uses = [edge for edge in await graph_repo.all_edges() if edge.relation_type == GraphRelationType.USES]
    assert len(uses) == 1
    assert uses[0].supports[0].memory_id == memory.id
    assert uses[0].supports[0].provenance_event_id == event_id

    db_path = database.path
    await database.close()
    reopened = Database(db_path)
    await reopened.initialize()
    persisted = await SqliteGraphRepository(reopened.connection()).all_edges()
    assert any(edge.relation_type == GraphRelationType.USES for edge in persisted)
    await reopened.close()


@pytest.mark.asyncio
async def test_edge_dedup_keeps_multiple_supports(graph_stack):
    _, memory_repo, _, graph_repo, service = graph_stack
    first = await _add(memory_repo, "Atlas uses Ollama")
    second = await _add(memory_repo, "Project Atlas uses Ollama", source_type="import")
    await service.rebuild()
    uses = [edge for edge in await graph_repo.all_edges() if edge.relation_type == GraphRelationType.USES]
    assert len(uses) == 1
    assert {item.memory_id for item in uses[0].supports} == {first.id, second.id}


@pytest.mark.asyncio
async def test_node_dedupe_and_directed_edge(graph_stack):
    _, memory_repo, _, graph_repo, service = graph_stack
    await _add(memory_repo, "Atlas uses Ollama")
    await _add(memory_repo, "Nova uses Ollama")
    await service.rebuild()
    ollama = [node for node in await graph_repo.nodes() if node.canonical_key == "ollama"]
    uses = [edge for edge in await graph_repo.all_edges() if edge.relation_type == GraphRelationType.USES]
    assert len(ollama) == 1
    assert len(uses) == 2
    assert all(edge.directed for edge in uses)
    assert {edge.scope_key for edge in uses} == {"atlas", "nova"}


@pytest.mark.asyncio
async def test_unsupported_edge_is_rejected(graph_stack):
    _, _, _, graph_repo, _ = graph_stack
    source = GraphNode(
        id=stable_node_id(GraphNodeType.PROJECT, "source"),
        node_type=GraphNodeType.PROJECT, canonical_key="source", label="source",
    )
    target = GraphNode(
        id=stable_node_id(GraphNodeType.TOOL, "target"),
        node_type=GraphNodeType.TOOL, canonical_key="target", label="target",
    )
    with pytest.raises(ValueError, match="support"):
        await graph_repo.replace_all([source, target], [GraphEdge(
            id=stable_edge_id(source.id, target.id, GraphRelationType.USES, None),
            source_node_id=source.id, target_node_id=target.id,
            relation_type=GraphRelationType.USES,
        )])


@pytest.mark.asyncio
async def test_phase7_relation_is_reused_with_symmetric_semantics(graph_stack):
    _, memory_repo, relation_repo, graph_repo, service = graph_stack
    left = await _add(memory_repo, "Atlas deployment profile")
    right = await _add(memory_repo, "Atlas staging profile")
    relation = MemoryRelation(
        source_memory_id=left.id, target_memory_id=right.id,
        relation_type="coexists_with", confidence=0.87,
    )
    await relation_repo.create(relation)
    await service.rebuild()
    reused = [
        edge for edge in await graph_repo.all_edges()
        if edge.relation_type == GraphRelationType.COEXISTS_WITH
    ]
    assert len(reused) == 1
    assert reused[0].directed is False
    assert reused[0].confidence == pytest.approx(0.87)


@pytest.mark.asyncio
async def test_deleted_support_is_removed_and_empty_edge_disappears(graph_stack):
    _, memory_repo, _, graph_repo, service = graph_stack
    memory = await _add(memory_repo, "Atlas uses Ollama")
    await service.rebuild()
    await memory_repo.update_status(memory.id, MemoryStatus.DELETED, memory.version)
    await service.ensure_current()
    assert not any(edge.relation_type == GraphRelationType.USES for edge in await graph_repo.all_edges())


@pytest.mark.asyncio
async def test_one_deleted_support_keeps_edge_with_remaining_support(graph_stack):
    _, memory_repo, _, graph_repo, service = graph_stack
    first = await _add(memory_repo, "Atlas uses Ollama")
    second = await _add(memory_repo, "Project Atlas uses Ollama", source_type="import")
    await service.rebuild()
    await memory_repo.update_status(first.id, MemoryStatus.DELETED, first.version)
    await service.ensure_current()
    uses = [edge for edge in await graph_repo.all_edges() if edge.relation_type == GraphRelationType.USES]
    assert len(uses) == 1
    assert [support.memory_id for support in uses[0].supports] == [second.id]


@pytest.mark.asyncio
async def test_purged_memory_is_absent_after_rebuild(graph_stack):
    _, memory_repo, _, graph_repo, service = graph_stack
    memory = await _add(memory_repo, "Atlas uses Ollama")
    await service.rebuild()
    await memory_repo.delete(memory.id)
    await service.ensure_current()
    assert all(support.memory_id != memory.id for edge in await graph_repo.all_edges() for support in edge.supports)


@pytest.mark.asyncio
async def test_bounded_cycle_safe_traversal(graph_stack):
    _, memory_repo, _, _, service = graph_stack
    await _add(memory_repo, "Atlas uses Ollama")
    await _add(memory_repo, "Ollama runs Qwen9B")
    await service.rebuild()
    result = await service.expand(query_text="Project Atlas", max_hops=2, max_nodes=4, max_edges=5)
    assert len(result.visited_node_ids) <= 4
    assert len(result.traversed_edge_ids) <= 5
    assert all(path.hop_count <= 2 for paths in result.candidate_paths.values() for path in paths)


@pytest.mark.asyncio
async def test_explicit_dependency_cycle_terminates(graph_stack):
    _, memory_repo, _, _, service = graph_stack
    await _add(memory_repo, "Docker depends on Podman")
    await _add(memory_repo, "Podman depends on Ollama")
    await _add(memory_repo, "Ollama depends on Docker")
    result = await service.expand(
        query_text="Docker", max_hops=3, max_nodes=10, max_edges=10,
    )
    assert len(result.visited_node_ids) <= 6
    assert len(result.traversed_edge_ids) <= 10
    assert all(path.hop_count <= 3 for paths in result.candidate_paths.values() for path in paths)


@pytest.mark.asyncio
async def test_hop_confidence_relation_and_node_bounds_are_enforced(graph_stack):
    _, memory_repo, _, _, service = graph_stack
    first = await _add(memory_repo, "Atlas uses Ollama")
    second = await _add(memory_repo, "Ollama runs Qwen 9B locally")
    await service.rebuild()
    one = await service.expand(query_text="Atlas", max_hops=1)
    two = await service.expand(query_text="Atlas", max_hops=2)
    assert first.id in one.candidate_scores
    assert second.id not in one.candidate_scores
    assert second.id in two.candidate_scores
    assert not (await service.expand(query_text="Atlas", min_confidence=0.99)).candidate_scores
    assert not (await service.expand(
        query_text="Atlas", relation_types={GraphRelationType.RUNS}
    )).candidate_scores
    bounded = await service.expand(query_text="Atlas", max_nodes=2, max_edges=2)
    assert len(bounded.visited_node_ids) <= 2
    assert len(bounded.traversed_edge_ids) <= 2


@pytest.mark.asyncio
async def test_traversal_is_deterministic(graph_stack):
    _, memory_repo, _, _, service = graph_stack
    await _add(memory_repo, "Atlas uses Ollama")
    await _add(memory_repo, "Ollama runs Qwen 9B locally")
    first = await service.expand(query_text="Atlas")
    second = await service.expand(query_text="Atlas")
    assert first.model_dump(exclude={"candidate_paths": {}}) == second.model_dump(exclude={"candidate_paths": {}})
    assert first.candidate_paths == second.candidate_paths


@pytest.mark.asyncio
async def test_hub_scope_prevents_cross_project_contamination(graph_stack):
    _, memory_repo, _, _, service = graph_stack
    a = await _add(memory_repo, "Project A uses Ollama")
    await _add(memory_repo, "Project B uses Ollama")
    failure = await _add(memory_repo, "Project B works on Qwen30B")
    result = await service.expand(query_text="Project A", max_hops=3)
    assert a.id in result.candidate_scores
    assert failure.id not in result.candidate_scores


@pytest.mark.asyncio
async def test_graph_only_and_hybrid_graph_use_rank_fusion(graph_stack):
    _, memory_repo, _, _, service = graph_stack
    atlas = await _add(memory_repo, "Atlas uses Ollama")
    runs = await _add(memory_repo, "Ollama runs Qwen9B")
    embedding = DeterministicEmbedding(32)
    lexical = BM25Index()
    vector = InMemoryVectorStore(32)
    sync = RetrievalIndexSynchronizer(
        memory_repo=memory_repo, lexical_index=lexical, vector_store=vector,
        embedding_service=embedding,
    )
    base = HybridRetrievalEngine(
        memory_repo=memory_repo, lexical_index=lexical, vector_store=vector,
        embedding_service=embedding, index_synchronizer=sync,
    )
    engine = GraphAugmentedRetrievalEngine(
        base_engine=base, graph_service=service, memory_repo=memory_repo,
    )
    graph = await engine.retrieve(RetrievalQuery(
        text="Project Atlas", mode=RetrievalMode.GRAPH, k=5, graph_max_hops=2,
    ))
    assert {item.memory.id for item in graph.memories} >= {atlas.id, runs.id}
    assert all("graph" in item.retrieval_sources for item in graph.memories)
    assert all(item.final_score <= 1 / 61 for item in graph.memories)

    hybrid = await engine.retrieve(RetrievalQuery(
        text="Atlas local inference model", mode=RetrievalMode.HYBRID_GRAPH, k=5,
    ))
    assert runs.id in {item.memory.id for item in hybrid.memories}
    assert hybrid.trace.stages[-1].metadata["fusion"] == "reciprocal_rank"


@pytest.mark.asyncio
async def test_expired_memory_is_excluded_current_and_included_all(graph_stack):
    _, memory_repo, _, _, service = graph_stack
    expired = await _add(memory_repo, "Atlas uses LegacyRuntime", status=MemoryStatus.EXPIRED)
    engine = _retrieval_engine(memory_repo, service)
    current = await engine.retrieve(RetrievalQuery(text="Atlas", mode=RetrievalMode.GRAPH))
    all_time = await engine.retrieve(RetrievalQuery(
        text="Atlas", mode=RetrievalMode.GRAPH, temporal_scope=TemporalScope.ALL,
    ))
    assert expired.id not in {item.memory.id for item in current.memories}
    assert expired.id in {item.memory.id for item in all_time.memories}


@pytest.mark.asyncio
async def test_graph_trace_contains_ids_and_no_memory_content(graph_stack):
    _, memory_repo, _, _, service = graph_stack
    memory = await _add(memory_repo, "Atlas uses Ollama")
    await service.rebuild()
    result = await service.expand(query_text="Atlas")
    path = result.candidate_paths[memory.id][0]
    serialized = path.model_dump_json()
    assert "Atlas uses Ollama" not in serialized
    assert path.seed_node_ids and path.edge_types and path.node_types


@pytest.mark.asyncio
async def test_current_vs_historical_lifecycle_filter(graph_stack):
    _, memory_repo, _, _, service = graph_stack
    old = await _add(memory_repo, "Atlas uses llama.cpp", status=MemoryStatus.SUPERSEDED)
    current = await _add(memory_repo, "Atlas uses Ollama")
    embedding = DeterministicEmbedding(16)
    lexical = BM25Index()
    vector = InMemoryVectorStore(16)
    base = HybridRetrievalEngine(
        memory_repo=memory_repo, lexical_index=lexical, vector_store=vector,
        embedding_service=embedding,
        index_synchronizer=RetrievalIndexSynchronizer(
            memory_repo=memory_repo, lexical_index=lexical, vector_store=vector,
            embedding_service=embedding,
        ),
    )
    engine = GraphAugmentedRetrievalEngine(base_engine=base, graph_service=service, memory_repo=memory_repo)
    now = await engine.retrieve(RetrievalQuery(text="Atlas", mode=RetrievalMode.GRAPH))
    history = await engine.retrieve(RetrievalQuery(
        text="Atlas", mode=RetrievalMode.GRAPH, temporal_scope=TemporalScope.HISTORICAL,
    ))
    assert current.id in {item.memory.id for item in now.memories}
    assert old.id not in {item.memory.id for item in now.memories}
    assert old.id in {item.memory.id for item in history.memories}


@pytest.mark.asyncio
async def test_update_rebuild_removes_stale_relation(graph_stack):
    _, memory_repo, _, graph_repo, service = graph_stack
    memory = await _add(memory_repo, "Atlas uses Ollama")
    await service.rebuild()
    await memory_repo.update(
        memory.id, MemoryUpdate(content="Nova uses vLLM"), memory.version,
    )
    await service.ensure_current()
    edges = await graph_repo.all_edges()
    assert not any(edge.relation_type == GraphRelationType.USES and edge.scope_key == "atlas" for edge in edges)
    assert any(edge.relation_type == GraphRelationType.USES and edge.scope_key == "nova" for edge in edges)


@pytest.mark.asyncio
async def test_failed_atomic_rebuild_keeps_memory_and_previous_graph(graph_stack):
    _, memory_repo, _, graph_repo, service = graph_stack
    memory = await _add(memory_repo, "Atlas uses Ollama")
    await service.rebuild()
    before = await graph_repo.counts()
    source = stable_node_id(GraphNodeType.PROJECT, "bad-source")
    target = stable_node_id(GraphNodeType.TOOL, "bad-target")
    edge_id = stable_edge_id(source, target, GraphRelationType.USES, None)
    nodes = [
        GraphNode(id=source, node_type=GraphNodeType.PROJECT, canonical_key="bad-source", label="bad-source"),
        GraphNode(id=target, node_type=GraphNodeType.TOOL, canonical_key="bad-target", label="bad-target"),
    ]
    edge = GraphEdge(
        id=edge_id, source_node_id=source, target_node_id=target,
        relation_type=GraphRelationType.USES,
        supports=[GraphEdgeSupport(edge_id=edge_id, memory_id=uuid4())],
    )
    with pytest.raises(Exception):
        await graph_repo.replace_all(nodes, [edge])
    assert await graph_repo.counts() == before
    assert await memory_repo.get(memory.id) is not None


@pytest.mark.asyncio
async def test_phase7_to_phase8_incremental_migration(tmp_path: Path):
    path = tmp_path / "phase7.db"
    memory_id = uuid4()
    connection = sqlite3.connect(path)
    connection.executescript(SCHEMA_SQL)
    connection.execute(
        "INSERT INTO schema_version(version, description) VALUES (1, 'Initial schema')"
    )
    connection.executescript(MIGRATION_2_SQL)
    connection.execute(
        "INSERT INTO schema_version(version, description) "
        "VALUES (2, 'Temporal memory and resolution metadata')"
    )
    connection.execute(
        "INSERT INTO memories(id, content, content_hash, status) VALUES (?, ?, ?, ?)",
        (str(memory_id), "Atlas uses Ollama", "phase7-hash", MemoryStatus.ACTIVE.value),
    )
    connection.commit()
    connection.close()

    reopened = Database(path)
    await reopened.initialize()
    cursor = await reopened.connection().execute("SELECT MAX(version) FROM schema_version")
    assert (await cursor.fetchone())[0] == SCHEMA_VERSION
    assert await SqliteGraphRepository(reopened.connection()).counts() == (0, 0, 0)
    assert (await SqliteMemoryRepository(reopened.connection()).get(memory_id)).content == "Atlas uses Ollama"
    await reopened.close()


@pytest.mark.asyncio
async def test_persisted_dirty_state_avoids_full_graph_rescan(graph_stack):
    _, memory_repo, _, graph_repo, service = graph_stack
    await _add(memory_repo, "Atlas uses Ollama")
    assert await graph_repo.source_is_dirty()
    assert await service.ensure_current() is True
    assert not await graph_repo.source_is_dirty()
    assert await service.ensure_current() is False
    await _add(memory_repo, "Ollama runs Qwen 9B locally")
    assert await graph_repo.source_is_dirty()
    assert await service.ensure_current() is True
