"""Deterministic local MCP adapter measurement helpers.

Callers provide the already-wired deterministic services used by tests; this
module never creates a provider or sends network traffic.

Label: LOCAL DEVELOPMENT MACHINE SYNTHETIC INFRASTRUCTURE BENCHMARK
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass, field
from typing import Any

from contextos.core.models import RetrievalQuery
from contextos.mcp.server import ContextOSMCPApplication, MCPPermissions


@dataclass(frozen=True)
class OperationResult:
    """Timing and size for a single benchmark operation."""
    operation: str
    iterations: int
    mean_ms: float
    median_ms: float
    p95_ms: float | None  # Only with sufficient samples (>= 20)
    serialized_bytes: int
    # Operation-specific metrics
    selected_memories: int | None = None
    compiled_facts: int | None = None
    compiled_tokens: int | None = None
    graph_nodes: int | None = None
    privacy_decision: str | None = None


@dataclass(frozen=True)
class MCPBenchmarkReport:
    """Complete Phase 10 benchmark report."""
    label: str = "LOCAL DEVELOPMENT MACHINE SYNTHETIC INFRASTRUCTURE BENCHMARK"
    direct_results: list[OperationResult] = field(default_factory=list)
    mcp_results: list[OperationResult] = field(default_factory=list)
    stdio_results: list[OperationResult] = field(default_factory=list)
    overheads: dict[str, float] = field(default_factory=dict)


def _p95(samples: list[float]) -> float | None:
    if len(samples) < 20:
        return None
    return sorted(samples)[int(len(samples) * 0.95)]


def _operation_result(
    operation: str, samples: list[float], response: dict[str, object],
) -> OperationResult:
    serialized = len(json.dumps(response, sort_keys=True, default=str).encode("utf-8"))
    return OperationResult(
        operation=operation,
        iterations=len(samples),
        mean_ms=statistics.mean(samples),
        median_ms=statistics.median(samples),
        p95_ms=_p95(samples),
        serialized_bytes=serialized,
        selected_memories=int(response.get("selected_memory_count", 0)) if "selected_memory_count" in response else (int(response.get("result_count", 0)) if response.get("ok") else None),
        compiled_facts=int(response.get("compiled_fact_count", 0)) if "compiled_fact_count" in response else None,
        compiled_tokens=int(response.get("token_count", 0)) if "token_count" in response else None,
        graph_nodes=int(response.get("graph_node_count", 0)) if "graph_node_count" in response else None,
        privacy_decision=str(response.get("error_code")) if response.get("error_code") == "PRIVACY_REJECTED" else None,
    )


async def _measure(coro_factory, iterations: int) -> tuple[list[float], Any]:
    """Run a coroutine factory N times and collect timings."""
    samples: list[float] = []
    last_result = None
    for _ in range(iterations):
        started = time.perf_counter()
        last_result = await coro_factory()
        elapsed = (time.perf_counter() - started) * 1000
        samples.append(elapsed)
    return samples, last_result


async def run_phase10_mcp_benchmark(
    services: dict[str, Any],
    query: str = "ContextOS memory",
    iterations: int = 5,
) -> MCPBenchmarkReport:
    """Measure direct retrieval versus the local adapter using identical services.

    Exercises: search, compile, graph, remember success, remember privacy
    rejection, and temporal lookup.
    """
    app = ContextOSMCPApplication(services, MCPPermissions(allow_write=True))
    direct_results: list[OperationResult] = []
    mcp_results: list[OperationResult] = []
    overheads: dict[str, float] = {}

    # --- Direct service measurements ---

    # Direct search
    samples, last = await _measure(
        lambda: services["retrieval"].retrieve(RetrievalQuery(text=query, k=5)),
        iterations,
    )
    direct_results.append(OperationResult(
        operation="search", iterations=len(samples),
        mean_ms=statistics.mean(samples), median_ms=statistics.median(samples),
        p95_ms=_p95(samples), serialized_bytes=0,
        selected_memories=len(last.memories) if last else 0,
    ))

    # --- MCP adapter measurements ---

    # MCP search
    samples, last = await _measure(
        lambda: app.invoke("contextos_search_memory", None,
                           lambda: app.search(query, 5, "hybrid", False)),
        iterations,
    )
    mcp_results.append(_operation_result("search", samples, last))
    overheads["search_adapter_ms"] = max(0.0, mcp_results[-1].mean_ms - direct_results[0].mean_ms)

    # MCP compile
    samples, last = await _measure(
        lambda: app.invoke("contextos_compile_context", None,
                           lambda: app.compile(query, 1000, "hybrid")),
        iterations,
    )
    mcp_results.append(_operation_result("compile", samples, last))

    # MCP graph
    samples, last = await _measure(
        lambda: app.invoke("contextos_graph_neighbors", None,
                           lambda: app.graph_neighbors(query, 1, 50, 100)),
        iterations,
    )
    mcp_results.append(_operation_result("graph", samples, last))

    # MCP remember (success)
    samples, last = await _measure(
        lambda: app.invoke("contextos_remember", None,
                           lambda: app.remember(f"Benchmark memory about {query}")),
        iterations,
    )
    mcp_results.append(_operation_result("remember_success", samples, last))

    # MCP temporal lookup
    samples, last = await _measure(
        lambda: app.invoke("contextos_current_state", None,
                           lambda: app.current_state("programming_language", "user", "global")),
        iterations,
    )
    mcp_results.append(_operation_result("temporal_lookup", samples, last))

    # MCP telemetry
    samples, last = await _measure(
        lambda: app.invoke("contextos_telemetry_summary", None,
                           app.telemetry_summary),
        iterations,
    )
    mcp_results.append(_operation_result("telemetry", samples, last))

    return MCPBenchmarkReport(
        direct_results=direct_results,
        mcp_results=mcp_results,
        overheads=overheads,
    )


async def main() -> None:
    """Run the canonical Phase 10 MCP benchmark against a temporary SQLite DB."""
    import asyncio
    import tempfile
    from pathlib import Path
    from mcp import Client
    from contextos.core.enums import SecretDetectionMode
    from contextos.embedding.deterministic import DeterministicEmbedding
    from contextos.mcp.server import MCPPermissions, create_mcp_server
    from contextos.services.compilation import QueryAwareContextCompiler
    from contextos.services.extraction import RuleBasedMemoryExtractor
    from contextos.services.graph import MemoryGraphService
    from contextos.services.graph_retrieval import GraphAugmentedRetrievalEngine
    from contextos.services.ingestion import IngestionPipeline
    from contextos.services.optimization import MemoryContextOptimizer
    from contextos.services.retrieval import HybridRetrievalEngine
    from contextos.services.retrieval_index import RetrievalIndexSynchronizer
    from contextos.services.secret_scanner import PatternSecretScanner
    from contextos.services.telemetry_query import TelemetryQueryService
    from contextos.services.temporal import TemporalMemoryService
    from contextos.services.token_counter import DeterministicWordTokenCounter
    from contextos.storage.database import Database
    from contextos.storage.event_repo import SqliteEventRepository
    from contextos.storage.graph_repo import SqliteGraphRepository
    from contextos.storage.lexical.bm25 import BM25Index
    from contextos.storage.memory_repo import SqliteMemoryRepository
    from contextos.storage.relation_repo import SqliteRelationRepository
    from contextos.storage.telemetry_repo import SqliteTelemetryRepository
    from contextos.storage.vector.in_memory import InMemoryVectorStore

    tmp_dir = Path(tempfile.mkdtemp())
    db_path = tmp_dir / "bench.db"
    db = Database(db_path)
    await db.initialize()
    conn = db.connection()
    memory_repo = SqliteMemoryRepository(conn)
    event_repo = SqliteEventRepository(conn)
    relation_repo = SqliteRelationRepository(conn)
    graph_repo = SqliteGraphRepository(conn)
    telemetry_repo = SqliteTelemetryRepository(conn)
    embedding = DeterministicEmbedding(16)
    lexical, vector = BM25Index(), InMemoryVectorStore(16)
    index = RetrievalIndexSynchronizer(
        memory_repo=memory_repo, embedding_service=embedding,
        vector_store=vector, lexical_index=lexical,
    )
    graph = MemoryGraphService(
        memory_repo=memory_repo, relation_repo=relation_repo,
        graph_repo=graph_repo,
    )
    retrieval = GraphAugmentedRetrievalEngine(
        base_engine=HybridRetrievalEngine(
            memory_repo=memory_repo, vector_store=vector,
            lexical_index=lexical, embedding_service=embedding,
            index_synchronizer=index,
        ),
        graph_service=graph, memory_repo=memory_repo,
    )
    token_counter = DeterministicWordTokenCounter()
    temporal = TemporalMemoryService(memory_repo)
    ingestion = IngestionPipeline(
        secret_scanner=PatternSecretScanner(),
        memory_extractor=RuleBasedMemoryExtractor(),
        memory_repo=memory_repo, event_repo=event_repo,
        embedding_service=embedding, vector_store=vector,
        lexical_index=lexical, token_counter=token_counter,
        secret_detection_mode=SecretDetectionMode.STRICT,
    )
    services = {
        "database": db,
        "retrieval": retrieval,
        "optimizer": MemoryContextOptimizer(token_counter=token_counter),
        "compilation": QueryAwareContextCompiler(token_counter=token_counter),
        "graph": graph,
        "temporal": temporal,
        "ingestion": ingestion,
        "telemetry_query": TelemetryQueryService(telemetry_repo),
    }

    # Seed data using MCP adapter
    server = create_mcp_server(services, MCPPermissions(allow_write=True))
    async with Client(server) as client:
        seed_data = [
            "I am working on Project Atlas. Project Atlas uses Python for machine learning.",
            "I am working on Project Beta. Project Beta uses Rust for systems programming.",
            "I currently use Ollama for local inference.",
            "I am working on Project Atlas. Project Atlas uses llama.cpp.",
            "I prefer concise answers for technical questions.",
            "I use Docker for containerization.",
            "I am working on Project Gamma. Project Gamma uses PostgreSQL.",
            "I am learning TypeScript for web development.",
        ]
        for text in seed_data:
            await client.call_tool("contextos_remember", {"text": text})

    # Run benchmark
    report = await run_phase10_mcp_benchmark(services, query="Atlas Python machine learning", iterations=5)

    # Print results
    print("=" * 70)
    print("LOCAL DEVELOPMENT MACHINE SYNTHETIC INFRASTRUCTURE BENCHMARK")
    print("=" * 70)
    print()
    print("DIRECT SERVICE RESULTS:")
    print("-" * 50)
    for r in report.direct_results:
        print(f"  {r.operation}:")
        print(f"    iterations:       {r.iterations}")
        print(f"    mean latency:     {r.mean_ms:.3f} ms")
        print(f"    median latency:   {r.median_ms:.3f} ms")
        if r.p95_ms is not None:
            print(f"    p95 latency:      {r.p95_ms:.3f} ms")
        print(f"    serialized bytes: {r.serialized_bytes}")
        if r.selected_memories is not None:
            print(f"    selected memories: {r.selected_memories}")
        print()

    print("MCP ADAPTER RESULTS:")
    print("-" * 50)
    for r in report.mcp_results:
        print(f"  {r.operation}:")
        print(f"    iterations:       {r.iterations}")
        print(f"    mean latency:     {r.mean_ms:.3f} ms")
        print(f"    median latency:   {r.median_ms:.3f} ms")
        if r.p95_ms is not None:
            print(f"    p95 latency:      {r.p95_ms:.3f} ms")
        print(f"    serialized bytes: {r.serialized_bytes}")
        if r.selected_memories is not None:
            print(f"    selected memories: {r.selected_memories}")
        if r.compiled_facts is not None:
            print(f"    compiled facts:    {r.compiled_facts}")
        if r.compiled_tokens is not None:
            print(f"    compiled tokens:   {r.compiled_tokens}")
        if r.graph_nodes is not None:
            print(f"    graph nodes:       {r.graph_nodes}")
        if r.privacy_decision is not None:
            print(f"    privacy decision:  {r.privacy_decision}")
        print()

    print("OVERHEAD CALCULATIONS:")
    print("-" * 50)
    for key, value in report.overheads.items():
        print(f"  {key}: {value:.3f} ms")
    print()

    await db.close()
    print("Benchmark complete.")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

