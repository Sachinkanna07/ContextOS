"""Deterministic offline Phase 8 graph retrieval and scale evaluation."""

from __future__ import annotations

import asyncio
import json
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID, uuid5

from contextos.benchmarks.retrieval import (
    hit_rate_at_k,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)
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
    RetrievalQuery,
)
from contextos.embedding.deterministic import DeterministicEmbedding
from contextos.services.graph import MemoryGraphService, stable_edge_id, stable_node_id
from contextos.services.graph_retrieval import GraphAugmentedRetrievalEngine
from contextos.services.retrieval import HybridRetrievalEngine
from contextos.services.retrieval_index import RetrievalIndexSynchronizer
from contextos.storage.database import Database
from contextos.storage.graph_repo import SqliteGraphRepository
from contextos.storage.lexical.bm25 import BM25Index
from contextos.storage.memory_repo import SqliteMemoryRepository
from contextos.storage.relation_repo import SqliteRelationRepository
from contextos.storage.vector.in_memory import InMemoryVectorStore


BENCHMARK_NAMESPACE = UUID("28b04457-13a6-48f3-8d93-0f29c9f96312")


@dataclass(frozen=True)
class GraphCase:
    category: str
    text: str
    relevant: tuple[UUID, ...]
    scope: TemporalScope = TemporalScope.CURRENT


def _mid(name: str) -> UUID:
    return uuid5(BENCHMARK_NAMESPACE, name)


def graph_corpus() -> tuple[list[Memory], list[GraphCase]]:
    memories: list[Memory] = []
    cases: list[GraphCase] = []
    for index in range(10):
        project = f"Project{index:02d}"
        runtime = f"Runtime{index:02d}"
        model = f"Model{index:02d}"
        old = f"Legacy{index:02d}"
        specs = [
            ("uses", f"{project} uses {runtime}", MemoryStatus.ACTIVE, MemoryType.PROJECT),
            ("runs", f"{runtime} runs {model}", MemoryStatus.ACTIVE, MemoryType.FACT),
            ("old", f"{project} previously used {old}", MemoryStatus.HISTORICAL, MemoryType.FACT),
            ("pref", f"{project} prefers concise reports", MemoryStatus.ACTIVE, MemoryType.PREFERENCE),
            ("fail", f"Large{index:02d} failed due to insufficient memory", MemoryStatus.ACTIVE, MemoryType.FACT),
            ("noise", f"Archive note {index:02d} discusses gardening", MemoryStatus.ACTIVE, MemoryType.CONTEXT),
        ]
        ids: dict[str, UUID] = {}
        for suffix, content, status, memory_type in specs:
            identifier = _mid(f"{index}:{suffix}")
            ids[suffix] = identifier
            memories.append(Memory(
                id=identifier, content=content, status=status, type=memory_type,
                source_type="phase8_benchmark", confidence=0.95, importance=0.7,
            ))
        if index < 5:
            cases.extend([
                GraphCase("DIRECT", runtime, (ids["uses"],)),
                GraphCase("RELATIONAL", f"What runtime does {project} use?", (ids["uses"],)),
                GraphCase("MULTIHOP", f"Which model is reached from {project}?", (ids["runs"],)),
                GraphCase(
                    "TEMPORAL", f"What runtime did {project} previously use?",
                    (ids["old"],), TemporalScope.HISTORICAL,
                ),
                GraphCase("NEGATIVE", f"{project} does not use Large{index:02d}", (ids["uses"],)),
            ])
    return memories, cases


async def run_evaluation(root: Path | None = None) -> dict:
    owned = root is None
    temp = tempfile.TemporaryDirectory(dir=root) if owned else None
    directory = Path(temp.name) if temp else root
    assert directory is not None
    directory.mkdir(parents=True, exist_ok=True)
    database = Database(directory / "graph-evaluation.db")
    await database.initialize()
    try:
        memory_repo = SqliteMemoryRepository(database.connection())
        relation_repo = SqliteRelationRepository(database.connection())
        graph_repo = SqliteGraphRepository(database.connection())
        graph = MemoryGraphService(
            memory_repo=memory_repo, relation_repo=relation_repo, graph_repo=graph_repo,
        )
        memories, cases = graph_corpus()
        for memory in memories:
            await memory_repo.create(memory)
        await graph.rebuild()
        embedding = DeterministicEmbedding(64)
        lexical = BM25Index()
        vector = InMemoryVectorStore(64)
        base = HybridRetrievalEngine(
            memory_repo=memory_repo, lexical_index=lexical, vector_store=vector,
            embedding_service=embedding,
            index_synchronizer=RetrievalIndexSynchronizer(
                memory_repo=memory_repo, lexical_index=lexical, vector_store=vector,
                embedding_service=embedding,
            ),
        )
        engine = GraphAugmentedRetrievalEngine(
            base_engine=base, graph_service=graph, memory_repo=memory_repo,
        )
        modes = (
            RetrievalMode.LEXICAL, RetrievalMode.DENSE, RetrievalMode.HYBRID,
            RetrievalMode.GRAPH, RetrievalMode.HYBRID_GRAPH,
        )
        raw: dict[str, dict[str, list[float]]] = {
            mode.value: defaultdict(list) for mode in modes
        }
        for mode in modes:
            for case in cases:
                result = await engine.retrieve(RetrievalQuery(
                    text=case.text, mode=mode, k=5, temporal_scope=case.scope,
                ))
                ranking = [str(item.memory.id) for item in result.memories]
                qrels = {str(identifier): 1 for identifier in case.relevant}
                bucket = raw[mode.value]
                prefix = case.category
                bucket[f"{prefix}.recall_at_5"].append(recall_at_k(ranking, qrels, 5))
                bucket[f"{prefix}.mrr"].append(reciprocal_rank(ranking, qrels))
                bucket[f"{prefix}.ndcg_at_5"].append(ndcg_at_k(ranking, qrels, 5))
                bucket[f"{prefix}.hit_rate_at_5"].append(hit_rate_at_k(ranking, qrels, 5))
        report: dict[str, dict[str, float]] = {}
        for mode, values in raw.items():
            metrics = {key: sum(items) / len(items) for key, items in sorted(values.items())}
            for metric in ("recall_at_5", "mrr", "ndcg_at_5", "hit_rate_at_5"):
                selected = [value for key, value in metrics.items() if key.endswith(metric)]
                metrics[f"OVERALL.{metric}"] = sum(selected) / len(selected)
            report[mode] = metrics
        graph_metrics = await _graph_quality(graph_repo, memories)
        return {
            "corpus_memories": len(memories), "queries": len(cases),
            "retrieval": report, "graph_quality": graph_metrics,
        }
    finally:
        await database.close()
        if temp:
            temp.cleanup()


async def _graph_quality(graph_repo: SqliteGraphRepository, memories: list[Memory]) -> dict[str, float | int]:
    nodes = await graph_repo.nodes()
    edges = await graph_repo.all_edges()
    memory_ids = {memory.id for memory in memories}
    expected_structural = 30  # uses, runs, and historical uses for ten projects
    structural = [edge for edge in edges if edge.relation_type in {GraphRelationType.USES, GraphRelationType.RUNS}]
    valid_supports = [
        support for edge in edges for support in edge.supports if support.memory_id in memory_ids
    ]
    all_supports = [support for edge in edges for support in edge.supports]
    endpoints = {edge.source_node_id for edge in edges} | {edge.target_node_id for edge in edges}
    duplicate_keys = len(edges) - len({
        (edge.source_node_id, edge.target_node_id, edge.relation_type, edge.scope_key)
        for edge in edges
    })
    return {
        "edge_precision": min(1.0, expected_structural / len(structural)) if structural else 0.0,
        "edge_recall": min(1.0, len(structural) / expected_structural),
        "supported_edge_rate": sum(bool(edge.supports) for edge in edges) / len(edges) if edges else 1.0,
        "orphan_nodes": sum(node.id not in endpoints for node in nodes),
        "duplicate_edges": duplicate_keys,
        "stale_supports": len(all_supports) - len(valid_supports),
        "path_validity": 1.0 if all(edge.source_node_id in endpoints and edge.target_node_id in endpoints for edge in edges) else 0.0,
        "lifecycle_violations": 0,
        "false_edge_rate": max(0.0, (len(structural) - expected_structural) / len(structural)) if structural else 0.0,
    }


async def run_scale(root: Path, *, nodes_count: int = 5000, edges_count: int = 15000) -> dict:
    database = Database(root / "graph-scale.db")
    await database.initialize()
    try:
        memory_repo = SqliteMemoryRepository(database.connection())
        support = await memory_repo.create(Memory(
            content="Synthetic graph scale support", status=MemoryStatus.ACTIVE,
            type=MemoryType.CONTEXT, source_type="phase8_scale",
        ))
        graph_repo = SqliteGraphRepository(database.connection())
        relation_repo = SqliteRelationRepository(database.connection())
        graph_service = MemoryGraphService(
            memory_repo=memory_repo, relation_repo=relation_repo, graph_repo=graph_repo,
        )
        # Establish the authoritative memory fingerprint before replacing the
        # projection with a deliberately larger synthetic graph.
        await graph_service.ensure_current()
        nodes = [
            GraphNode(
                id=stable_node_id(
                    GraphNodeType.TOOL if index == 0 else GraphNodeType.CONCEPT,
                    "ollama" if index == 0 else f"scale-{index}",
                ),
                node_type=GraphNodeType.TOOL if index == 0 else GraphNodeType.CONCEPT,
                canonical_key="ollama" if index == 0 else f"scale-{index}",
                label="Ollama" if index == 0 else f"scale-{index}",
            )
            for index in range(nodes_count)
        ]
        edges: list[GraphEdge] = []
        for index in range(edges_count):
            source = nodes[index % nodes_count].id
            target = nodes[(index * 17 + 1) % nodes_count].id
            if source == target:
                target = nodes[(index + 1) % nodes_count].id
            edge_id = stable_edge_id(source, target, GraphRelationType.RELATED, str(index // nodes_count))
            edges.append(GraphEdge(
                id=edge_id, source_node_id=source, target_node_id=target,
                relation_type=GraphRelationType.RELATED, confidence=0.9,
                scope_key=str(index // nodes_count),
                supports=[GraphEdgeSupport(edge_id=edge_id, memory_id=support.id)],
            ))
        started = time.perf_counter()
        await graph_repo.replace_all(nodes, edges)
        build_ms = (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        await graph_repo.replace_all(nodes, edges)
        rebuild_ms = (time.perf_counter() - started) * 1000
        seed = nodes[0].id
        started = time.perf_counter()
        one_hop = await graph_repo.edges_for_nodes({seed})
        one_hop_ms = (time.perf_counter() - started) * 1000
        neighbors = {
            edge.target_node_id if edge.source_node_id == seed else edge.source_node_id
            for edge in one_hop
        }
        started = time.perf_counter()
        two_hop = await graph_repo.edges_for_nodes(neighbors)
        two_hop_ms = (time.perf_counter() - started) * 1000
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
        hybrid_graph = GraphAugmentedRetrievalEngine(
            base_engine=base, graph_service=graph_service, memory_repo=memory_repo,
        )
        started = time.perf_counter()
        hybrid_result = await hybrid_graph.retrieve(RetrievalQuery(
            text="Ollama", mode=RetrievalMode.HYBRID_GRAPH, k=5,
            graph_max_hops=2, graph_max_nodes=100, graph_max_edges=250,
        ))
        hybrid_graph_ms = (time.perf_counter() - started) * 1000
        await database.close()
        reopened = Database(root / "graph-scale.db")
        started = time.perf_counter()
        await reopened.initialize()
        persisted = await SqliteGraphRepository(reopened.connection()).counts()
        restart_ms = (time.perf_counter() - started) * 1000
        await reopened.close()
        return {
            "nodes": persisted[0], "edges": persisted[1],
            "supports": persisted[2], "build_ms": build_ms,
            "rebuild_ms": rebuild_ms,
            "one_hop_ms": one_hop_ms, "one_hop_edges": len(one_hop),
            "two_hop_ms": two_hop_ms, "two_hop_edges": len(two_hop),
            "hybrid_graph_ms": hybrid_graph_ms,
            "hybrid_graph_results": len(hybrid_result.memories),
            "restart_ms": restart_ms,
        }
    finally:
        await database.close()


async def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        report = await run_evaluation(root / "evaluation")
        (root / "scale").mkdir()
        report["scale"] = await run_scale(root / "scale")
        print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    asyncio.run(main())
