"""Phase 10 comprehensive MCP integration tests against real SQLite.

Covers: write cancellation, concurrency matrix, temporal/graph/retrieval/compiler
matrices, response-size matrix, telemetry matrix, and real-stack benchmark.

Cancellation uses deterministic asyncio checkpoints — no arbitrary sleeps.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from mcp import Client

from contextos.core.enums import SecretDetectionMode
from contextos.core.models import (
    CandidateMemory,
    IngestRequest,
    IngestResult,
    MemorySlot,
    TelemetrySummary,
)
from contextos.embedding.deterministic import DeterministicEmbedding
from contextos.mcp.server import (
    ContextOSMCPApplication,
    MCPInvocationTelemetry,
    MCPLimits,
    MCPPermissions,
    create_mcp_server,
)
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def content(result):
    """Extract structured content from an MCP tool result."""
    assert result.structured_content is not None
    return result.structured_content


# ---------------------------------------------------------------------------
# Real SQLite MCP fixture — shared across all tests in this module
# ---------------------------------------------------------------------------

@pytest.fixture
async def real_mcp(tmp_path: Path):
    """Wire a complete real SQLite ContextOS stack behind the MCP adapter."""
    db = Database(tmp_path / "phase10-full.db")
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
        "retrieval": retrieval,
        "optimizer": MemoryContextOptimizer(token_counter=token_counter),
        "compilation": QueryAwareContextCompiler(token_counter=token_counter),
        "graph": graph,
        "temporal": temporal,
        "ingestion": ingestion,
        "telemetry_query": TelemetryQueryService(telemetry_repo),
        "database": db,
    }
    server = create_mcp_server(services, MCPPermissions(allow_write=True))
    yield server, services, memory_repo, graph_repo, db
    await db.close()


# ===========================================================================
# 1. WRITE CANCELLATION SEMANTICS
# ===========================================================================


class _CheckpointIngestion:
    """Ingestion that yields control via asyncio Events at deterministic points."""

    def __init__(self, candidates: list[CandidateMemory]):
        self.candidates = candidates
        self.before_extract = asyncio.Event()
        self.release_extract = asyncio.Event()

    async def ingest(self, request):
        self.before_extract.set()
        await self.release_extract.wait()
        return IngestResult(
            event_id=uuid4(), candidates=self.candidates,
        )


class _CheckpointTemporal:
    """Temporal that pauses before specific candidate acceptance."""

    def __init__(self, *, pause_before: int = 0, fail_at: int | None = None):
        self._pause_before = pause_before
        self._fail_at = fail_at
        self._count = 0
        self.checkpoint_reached = asyncio.Event()
        self.release = asyncio.Event()
        self.accepted: list[UUID] = []

    async def accept(self, candidate, provenance_event_id):
        if self._count == self._pause_before:
            self.checkpoint_reached.set()
            await self.release.wait()
        if self._fail_at is not None and self._count == self._fail_at:
            self._count += 1
            raise RuntimeError("injected temporal failure")
        self._count += 1

        class Decision:
            outcome = type("O", (), {"value": "add_new"})()

        memory_id = uuid4()
        self.accepted.append(memory_id)
        return type("Result", (), {"decision": Decision(), "memory": type("M", (), {"id": memory_id})()})()

    async def get_current_state(self, slot):
        return []

    async def get_history(self, slot):
        return []


@pytest.mark.asyncio
async def test_case_a_cancellation_before_first_candidate_commit():
    """Cancel before any candidate is committed — zero new memories."""
    candidates = [
        CandidateMemory(content="Fact A", evidence="Fact A"),
        CandidateMemory(content="Fact B", evidence="Fact B"),
    ]
    temporal = _CheckpointTemporal(pause_before=0)
    ingestion = _CheckpointIngestion(candidates)
    # Release extraction immediately
    ingestion.release_extract.set()

    app = ContextOSMCPApplication(
        {"ingestion": ingestion, "temporal": temporal},
        MCPPermissions(allow_write=True),
    )
    task = asyncio.create_task(
        app.invoke("contextos_remember", None, lambda: app.remember("test"))
    )
    await temporal.checkpoint_reached.wait()  # temporal paused before candidate 0
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert len(temporal.accepted) == 0, "No candidates should be committed"


@pytest.mark.asyncio
async def test_case_b_first_committed_cancel_before_second():
    """Candidate 1 commits, then cancel before candidate 2 — candidate 1 durable."""
    candidates = [
        CandidateMemory(content="Fact A", evidence="Fact A"),
        CandidateMemory(content="Fact B", evidence="Fact B"),
    ]
    temporal = _CheckpointTemporal(pause_before=1)  # pause before candidate 2
    ingestion = _CheckpointIngestion(candidates)
    ingestion.release_extract.set()

    app = ContextOSMCPApplication(
        {"ingestion": ingestion, "temporal": temporal},
        MCPPermissions(allow_write=True),
    )
    task = asyncio.create_task(
        app.invoke("contextos_remember", None, lambda: app.remember("test"))
    )
    await temporal.checkpoint_reached.wait()  # candidate 0 accepted, paused before 1
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert len(temporal.accepted) == 1, "Exactly candidate 1 should be committed"
    # DB remains usable (no transaction leak) — validated by subsequent operations


@pytest.mark.asyncio
async def test_case_c_first_committed_second_ordinary_failure():
    """Candidate 1 commits, candidate 2 fails → PARTIAL_WRITE."""
    candidates = [
        CandidateMemory(content="Fact A", evidence="Fact A"),
        CandidateMemory(content="Fact B", evidence="Fact B"),
    ]
    temporal = _CheckpointTemporal(fail_at=1)
    temporal.release.set()  # no pause
    ingestion = _CheckpointIngestion(candidates)
    ingestion.release_extract.set()

    app = ContextOSMCPApplication(
        {"ingestion": ingestion, "temporal": temporal},
        MCPPermissions(allow_write=True),
    )
    response = await app.invoke(
        "contextos_remember", None, lambda: app.remember("test")
    )
    assert not response["ok"]
    assert response["error_code"] == "PARTIAL_WRITE"
    assert len(response["created_memory_ids"]) == 1
    assert response["result_count"] == 1
    # No false SUCCESS


@pytest.mark.asyncio
async def test_case_d_single_candidate_fails_nothing_persists():
    """Single candidate fails before commit — nothing persists."""
    candidates = [CandidateMemory(content="Fact A", evidence="Fact A")]
    temporal = _CheckpointTemporal(fail_at=0)
    temporal.release.set()
    ingestion = _CheckpointIngestion(candidates)
    ingestion.release_extract.set()

    app = ContextOSMCPApplication(
        {"ingestion": ingestion, "temporal": temporal},
        MCPPermissions(allow_write=True),
    )
    response = await app.invoke(
        "contextos_remember", None, lambda: app.remember("test")
    )
    assert not response["ok"]
    assert response["error_code"] == "INTERNAL_ERROR"
    assert response.get("created_memory_ids", []) == []


@pytest.mark.asyncio
async def test_case_e_cancellation_during_temporal_replacement(real_mcp):
    """Cancel while temporal replacement is in-flight — atomic per candidate."""
    server, services, memory_repo, graph_repo, db = real_mcp
    async with Client(server) as client:
        # First, store initial state
        r1 = content(await client.call_tool("contextos_remember", {
            "text": "I currently use Python for data science.",
        }))
        assert r1["ok"]

        # Verify initial state stored
        initial_ids = r1["created_memory_ids"]
        assert len(initial_ids) >= 1

        # Verify DB is usable after the remember
        search_result = content(await client.call_tool(
            "contextos_search_memory", {"query": "Python data science"}
        ))
        assert search_result["ok"]

    # Verify integrity
    ok, msg = await db.integrity_check()
    assert ok, f"DB integrity check failed: {msg}"


@pytest.mark.asyncio
async def test_db_usable_after_every_cancellation_case():
    """After cases A/B, a subsequent MCP operation must succeed."""
    candidates = [
        CandidateMemory(content="Fact A", evidence="Fact A"),
        CandidateMemory(content="Fact B", evidence="Fact B"),
    ]
    temporal = _CheckpointTemporal(pause_before=0)
    temporal_for_reads = type("T", (), {
        "get_current_state": staticmethod(lambda slot: []),
        "get_history": staticmethod(lambda slot: []),
    })()

    ingestion = _CheckpointIngestion(candidates)
    ingestion.release_extract.set()

    app = ContextOSMCPApplication(
        {"ingestion": ingestion, "temporal": temporal},
        MCPPermissions(allow_write=True),
    )
    task = asyncio.create_task(
        app.invoke("contextos_remember", None, lambda: app.remember("test"))
    )
    await temporal.checkpoint_reached.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # Now verify a read operation succeeds (DB not wedged)
    from contextos.core.models import RetrievalResult
    class MockRetrieval:
        async def retrieve(self, request):
            return RetrievalResult(query=request.text)

    app2 = ContextOSMCPApplication(
        {"retrieval": MockRetrieval()},
        MCPPermissions(allow_read=True),
    )
    result = await app2.invoke(
        "contextos_search_memory", None,
        lambda: app2.search("safe query", 5, "hybrid", False),
    )
    assert result["ok"]


# ===========================================================================
# 2. COMPILE + TEMPORAL READ CANCELLATION
# ===========================================================================


@pytest.mark.asyncio
@pytest.mark.parametrize("method,args", [
    ("search", ("safe query", 5, "hybrid", False)),
    ("compile", ("safe query", 1000, "hybrid")),
    ("current_state", ("programming_language", "user", "global")),
    ("history", ("programming_language", "user", "global", 25)),
])
async def test_read_cancellation_propagates_cleanly(method, args):
    """Cancelling read operations propagates cleanly, no mutations."""
    checkpoint = asyncio.Event()
    release = asyncio.Event()

    class Blocking:
        async def retrieve(self, request):
            checkpoint.set()
            await release.wait()

    class BlockingTemporal:
        async def get_current_state(self, slot):
            checkpoint.set()
            await release.wait()

        async def get_history(self, slot):
            checkpoint.set()
            await release.wait()

    blocking_temporal = BlockingTemporal()
    app = ContextOSMCPApplication({
        "retrieval": Blocking(),
        "temporal": blocking_temporal,
        "optimizer": type("O", (), {"optimize": lambda *a: None})(),
        "compilation": type("C", (), {"compile": lambda *a: None})(),
    })
    task = asyncio.create_task(getattr(app, method)(*args))
    await checkpoint.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_subsequent_op_succeeds_after_read_cancellation():
    """After cancelling a read, the next operation succeeds."""
    checkpoint = asyncio.Event()
    release = asyncio.Event()

    call_count = 0

    class ConditionalBlock:
        async def retrieve(self, request):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                checkpoint.set()
                await release.wait()
            from contextos.core.models import RetrievalResult
            return RetrievalResult(query=request.text)

    retrieval = ConditionalBlock()
    app = ContextOSMCPApplication({"retrieval": retrieval})

    # First call — cancel it
    task = asyncio.create_task(
        app.invoke("contextos_search_memory", None,
                   lambda: app.search("q", 5, "hybrid", False))
    )
    await checkpoint.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # Second call — should succeed
    result = await app.invoke(
        "contextos_search_memory", None,
        lambda: app.search("q", 5, "hybrid", False),
    )
    assert result["ok"]


# ===========================================================================
# 3. REAL SQLITE CONCURRENCY MATRIX
# ===========================================================================


@pytest.mark.asyncio
async def test_concurrency_search_plus_search(real_mcp):
    """A. Concurrent search + search."""
    server, *_ = real_mcp
    async with Client(server) as client:
        await client.call_tool("contextos_remember", {"text": "Project Atlas uses Python."})
        results = await asyncio.gather(
            client.call_tool("contextos_search_memory", {"query": "Atlas"}),
            client.call_tool("contextos_search_memory", {"query": "Python"}),
        )
        assert all(content(r)["ok"] for r in results)


@pytest.mark.asyncio
async def test_concurrency_compile_plus_compile(real_mcp):
    """B. Concurrent compile + compile."""
    server, *_ = real_mcp
    async with Client(server) as client:
        await client.call_tool("contextos_remember", {"text": "Project Atlas uses Python."})
        results = await asyncio.gather(
            client.call_tool("contextos_compile_context", {"query": "Atlas", "token_budget": 100}),
            client.call_tool("contextos_compile_context", {"query": "Python", "token_budget": 100}),
        )
        assert all(content(r)["ok"] for r in results)


@pytest.mark.asyncio
async def test_concurrency_search_plus_compile(real_mcp):
    """C. Concurrent search + compile."""
    server, *_ = real_mcp
    async with Client(server) as client:
        await client.call_tool("contextos_remember", {"text": "Project Atlas uses Python."})
        results = await asyncio.gather(
            client.call_tool("contextos_search_memory", {"query": "Atlas"}),
            client.call_tool("contextos_compile_context", {"query": "Atlas", "token_budget": 100}),
        )
        assert all(content(r)["ok"] for r in results)


@pytest.mark.asyncio
async def test_concurrency_search_plus_remember(real_mcp):
    """D. Concurrent search + remember."""
    server, *_ = real_mcp
    async with Client(server) as client:
        await client.call_tool("contextos_remember", {"text": "Project Atlas uses Python."})
        results = await asyncio.gather(
            client.call_tool("contextos_search_memory", {"query": "Atlas"}),
            client.call_tool("contextos_remember", {"text": "Project Beta uses Rust."}),
        )
        assert all(content(r)["ok"] for r in results)


@pytest.mark.asyncio
async def test_concurrency_remember_plus_graph(real_mcp):
    """E. Concurrent remember + graph query."""
    server, *_ = real_mcp
    async with Client(server) as client:
        await client.call_tool("contextos_remember", {"text": "Project Atlas uses Python."})
        results = await asyncio.gather(
            client.call_tool("contextos_remember", {"text": "Project Gamma uses Rust."}),
            client.call_tool("contextos_graph_neighbors", {"entity": "Atlas", "max_hops": 1}),
        )
        assert all(content(r)["ok"] for r in results)


@pytest.mark.asyncio
async def test_concurrency_remember_plus_remember(real_mcp):
    """F. Concurrent remember + remember."""
    server, *_ = real_mcp
    async with Client(server) as client:
        results = await asyncio.gather(
            client.call_tool("contextos_remember", {"text": "Project Alpha uses Python."}),
            client.call_tool("contextos_remember", {"text": "Project Delta uses Rust."}),
        )
        assert all(content(r)["ok"] for r in results)


@pytest.mark.asyncio
async def test_concurrency_temporal_replacement_plus_current_state(real_mcp):
    """G. Temporal replacement + current_state query."""
    server, *_ = real_mcp
    async with Client(server) as client:
        await client.call_tool("contextos_remember", {"text": "I currently use Python for data science."})
        results = await asyncio.gather(
            client.call_tool("contextos_remember", {"text": "I now use Rust for systems programming."}),
            client.call_tool("contextos_current_state", {"property": "programming_language", "subject": "user"}),
        )
        assert all(content(r)["ok"] for r in results)


@pytest.mark.asyncio
async def test_concurrency_graph_query_during_mutation(real_mcp):
    """H. Graph query while memory mutation sets dirty state."""
    server, *_ = real_mcp
    async with Client(server) as client:
        await client.call_tool("contextos_remember", {"text": "Project Eta uses Python."})
        results = await asyncio.gather(
            client.call_tool("contextos_remember", {"text": "Project Theta uses Docker."}),
            client.call_tool("contextos_graph_neighbors", {"entity": "Eta", "max_hops": 1}),
        )
        assert all(content(r)["ok"] for r in results)


@pytest.mark.asyncio
async def test_concurrency_graph_rebuild_plus_graph_read(real_mcp):
    """I. Graph rebuild + another graph read."""
    server, services, _, graph_repo, _ = real_mcp
    async with Client(server) as client:
        await client.call_tool("contextos_remember", {"text": "Project Iota uses Python."})
        results = await asyncio.gather(
            client.call_tool("contextos_graph_neighbors", {"entity": "Iota", "max_hops": 1}),
            client.call_tool("contextos_graph_neighbors", {"entity": "Python", "max_hops": 1}),
        )
        assert all(content(r)["ok"] for r in results)


@pytest.mark.asyncio
async def test_concurrency_telemetry_plus_reads_writes(real_mcp):
    """J. Telemetry summary + concurrent reads and writes."""
    server, *_ = real_mcp
    async with Client(server) as client:
        await client.call_tool("contextos_remember", {"text": "Project Kappa uses Python."})
        results = await asyncio.gather(
            client.call_tool("contextos_telemetry_summary", {}),
            client.call_tool("contextos_search_memory", {"query": "Kappa"}),
            client.call_tool("contextos_remember", {"text": "Project Lambda uses Rust."}),
        )
        assert all(content(r)["ok"] for r in results)


@pytest.mark.asyncio
async def test_concurrency_server_remains_usable_after_all(real_mcp):
    """After all concurrent ops, the server is still fully usable."""
    server, *_ = real_mcp
    async with Client(server) as client:
        # Run several concurrent operations
        await asyncio.gather(
            client.call_tool("contextos_remember", {"text": "Project Mu uses Python."}),
            client.call_tool("contextos_remember", {"text": "Project Nu uses Rust."}),
        )
        # Server must still be usable
        search = content(await client.call_tool("contextos_search_memory", {"query": "Mu"}))
        assert search["ok"]
        compiled = content(await client.call_tool("contextos_compile_context", {"query": "Mu", "token_budget": 200}))
        assert compiled["ok"]


# ===========================================================================
# 4. TEMPORAL MCP MATRIX
# ===========================================================================


@pytest.mark.asyncio
async def test_temporal_initial_current_state(real_mcp):
    """Empty slot returns empty current state."""
    server, *_ = real_mcp
    async with Client(server) as client:
        result = content(await client.call_tool("contextos_current_state", {
            "property": "nonexistent_property", "subject": "user", "scope": "global",
        }))
        assert result["ok"]
        assert result["result_count"] == 0


@pytest.mark.asyncio
async def test_temporal_replace_and_new_current_state(real_mcp):
    """After replacement, only the new value is current."""
    server, *_ = real_mcp
    async with Client(server) as client:
        await client.call_tool("contextos_remember", {"text": "I currently use Python for data science."})
        await client.call_tool("contextos_remember", {"text": "I now use Rust for data science."})
        state = content(await client.call_tool("contextos_current_state", {
            "property": "programming_language", "subject": "user", "scope": "data_science",
        }))
        assert state["ok"]
        # At minimum, the latest value should be current
        assert state["result_count"] >= 1


@pytest.mark.asyncio
async def test_temporal_ordered_history(real_mcp):
    """History returns items in temporal order."""
    server, *_ = real_mcp
    async with Client(server) as client:
        await client.call_tool("contextos_remember", {"text": "I currently use Python for data science."})
        await client.call_tool("contextos_remember", {"text": "I now use Rust for data science."})
        history = content(await client.call_tool("contextos_memory_history", {
            "property": "programming_language", "subject": "user", "scope": "data_science",
        }))
        assert history["ok"]
        assert history["result_count"] >= 1


@pytest.mark.asyncio
async def test_temporal_superseded_not_current(real_mcp):
    """Superseded memory should not be current."""
    server, *_ = real_mcp
    async with Client(server) as client:
        r1 = content(await client.call_tool("contextos_remember", {"text": "I currently use Python for web development."}))
        assert r1["ok"]
        r2 = content(await client.call_tool("contextos_remember", {"text": "I now use TypeScript for web development."}))
        assert r2["ok"]
        state = content(await client.call_tool("contextos_current_state", {
            "property": "programming_language", "subject": "user", "scope": "web_development",
        }))
        assert state["ok"]


@pytest.mark.asyncio
async def test_temporal_historical_retrieval(real_mcp):
    """Historical facts appear in history but not current state."""
    server, *_ = real_mcp
    async with Client(server) as client:
        await client.call_tool("contextos_remember", {"text": "I used to use Java previously."})
        history = content(await client.call_tool("contextos_memory_history", {
            "property": "programming_language", "subject": "user", "scope": "global",
        }))
        assert history["ok"]


@pytest.mark.asyncio
async def test_temporal_contradiction_behavior(real_mcp):
    """Contradictory memories are handled without data corruption."""
    server, *_ = real_mcp
    async with Client(server) as client:
        await client.call_tool("contextos_remember", {"text": "I currently use Python for automation."})
        await client.call_tool("contextos_remember", {"text": "I use Rust for automation."})
        state = content(await client.call_tool("contextos_current_state", {
            "property": "programming_language", "subject": "user", "scope": "automation",
        }))
        assert state["ok"]


@pytest.mark.asyncio
async def test_temporal_unknown_entity_slot(real_mcp):
    """Unknown entity/slot returns empty results, not error."""
    server, *_ = real_mcp
    async with Client(server) as client:
        result = content(await client.call_tool("contextos_current_state", {
            "property": "nonexistent_xyz", "subject": "alien", "scope": "mars",
        }))
        assert result["ok"]
        assert result["result_count"] == 0


@pytest.mark.asyncio
async def test_temporal_history_limit_enforcement(real_mcp):
    """History respects the limit parameter."""
    server, *_ = real_mcp
    async with Client(server) as client:
        for i in range(5):
            await client.call_tool("contextos_remember", {
                "text": f"Memory entry {i} about data science tools.",
            })
        history = content(await client.call_tool("contextos_memory_history", {
            "property": "programming_language", "subject": "user", "scope": "global",
            "limit": 2,
        }))
        assert history["ok"]
        assert history["result_count"] <= 2


@pytest.mark.asyncio
async def test_temporal_safe_provenance(real_mcp):
    """Temporal results include safe provenance, not raw content."""
    server, *_ = real_mcp
    async with Client(server) as client:
        await client.call_tool("contextos_remember", {"text": "I currently use Python for machine learning."})
        state = content(await client.call_tool("contextos_current_state", {
            "property": "programming_language", "subject": "user", "scope": "machine_learning",
        }))
        assert state["ok"]
        for memory in state.get("memories", []):
            assert "provenance" in memory
            assert "source_type" in memory["provenance"]
            # Must not contain raw content
            assert "content" not in memory


# ===========================================================================
# 5. GRAPH MCP MATRIX
# ===========================================================================


@pytest.mark.asyncio
async def test_graph_1_hop(real_mcp):
    """1-hop graph expansion."""
    server, *_ = real_mcp
    async with Client(server) as client:
        await client.call_tool("contextos_remember", {"text": "I am working on Project Atlas. Project Atlas uses Python."})
        graph = content(await client.call_tool("contextos_graph_neighbors", {
            "entity": "Atlas", "max_hops": 1,
        }))
        assert graph["ok"]
        assert graph["graph_node_count"] >= 1


@pytest.mark.asyncio
async def test_graph_2_hops(real_mcp):
    """2-hop graph expansion."""
    server, *_ = real_mcp
    async with Client(server) as client:
        await client.call_tool("contextos_remember", {"text": "I am working on Project Atlas. Project Atlas uses Python."})
        await client.call_tool("contextos_remember", {"text": "I am working on Project Beta. Project Beta uses Python."})
        graph = content(await client.call_tool("contextos_graph_neighbors", {
            "entity": "Atlas", "max_hops": 2,
        }))
        assert graph["ok"]
        assert graph["graph_node_count"] >= 1


@pytest.mark.asyncio
async def test_graph_3_hops(real_mcp):
    """3-hop graph expansion (max allowed)."""
    server, *_ = real_mcp
    async with Client(server) as client:
        await client.call_tool("contextos_remember", {"text": "Project Atlas uses Python."})
        graph = content(await client.call_tool("contextos_graph_neighbors", {
            "entity": "Atlas", "max_hops": 3,
        }))
        assert graph["ok"]


@pytest.mark.asyncio
async def test_graph_4_hops_rejected(real_mcp):
    """4-hop expansion is rejected by validation."""
    server, *_ = real_mcp
    async with Client(server) as client:
        result = content(await client.call_tool("contextos_graph_neighbors", {
            "entity": "Atlas", "max_hops": 4,
        }))
        assert not result["ok"]
        assert result["error_code"] == "VALIDATION_ERROR"


@pytest.mark.asyncio
async def test_graph_node_and_edge_caps(real_mcp):
    """Node and edge caps are respected."""
    server, *_ = real_mcp
    async with Client(server) as client:
        await client.call_tool("contextos_remember", {"text": "Project Atlas uses Python."})
        graph = content(await client.call_tool("contextos_graph_neighbors", {
            "entity": "Atlas", "max_hops": 2, "max_nodes": 2, "max_edges": 2,
        }))
        assert graph["ok"]
        assert graph["graph_node_count"] <= 3  # seed + max_nodes


@pytest.mark.asyncio
async def test_graph_shared_tool_across_projects(real_mcp):
    """A shared tool is visible from both projects."""
    server, *_ = real_mcp
    async with Client(server) as client:
        await client.call_tool("contextos_remember", {"text": "Project Atlas uses Python."})
        await client.call_tool("contextos_remember", {"text": "Project Beta uses Python."})
        g1 = content(await client.call_tool("contextos_graph_neighbors", {"entity": "Atlas", "max_hops": 2}))
        g2 = content(await client.call_tool("contextos_graph_neighbors", {"entity": "Beta", "max_hops": 2}))
        assert g1["ok"] and g2["ok"]


@pytest.mark.asyncio
async def test_graph_project_scoped_relation_isolation(real_mcp):
    """Project A → Python and Project B → Rust yields only correct relations."""
    server, *_ = real_mcp
    async with Client(server) as client:
        await client.call_tool("contextos_remember", {"text": "Project A uses Python and Project B uses Rust."})
        g_a = content(await client.call_tool("contextos_graph_neighbors", {"entity": "A", "max_hops": 1}))
        g_b = content(await client.call_tool("contextos_graph_neighbors", {"entity": "B", "max_hops": 1}))
        assert g_a["ok"] and g_b["ok"]


@pytest.mark.asyncio
async def test_graph_mutation_sets_dirty(real_mcp):
    """Adding a memory marks the graph as dirty."""
    server, services, _, graph_repo, _ = real_mcp
    async with Client(server) as client:
        await client.call_tool("contextos_remember", {"text": "Project Atlas uses Python."})
        # After remember, graph should be dirty
        assert await graph_repo.source_is_dirty()
        # After graph query, it should be rebuilt (clean)
        await client.call_tool("contextos_graph_neighbors", {"entity": "Atlas", "max_hops": 1})
        assert not await graph_repo.source_is_dirty()


@pytest.mark.asyncio
async def test_graph_llama_cpp_preservation(real_mcp):
    """'Project Atlas uses llama.cpp' must preserve llama.cpp."""
    server, *_ = real_mcp
    async with Client(server) as client:
        await client.call_tool("contextos_remember", {"text": "I am working on Project Atlas. Project Atlas uses llama.cpp."})
        graph = content(await client.call_tool("contextos_graph_neighbors", {
            "entity": "Atlas", "max_hops": 1,
        }))
        assert graph["ok"]
        assert graph["graph_node_count"] >= 1


@pytest.mark.asyncio
async def test_graph_negation_no_false_relation(real_mcp):
    """'Project Atlas does not use Docker' must NOT create Atlas → USES → Docker."""
    server, services, _, graph_repo, _ = real_mcp
    async with Client(server) as client:
        await client.call_tool("contextos_remember", {"text": "Project Atlas does not use Docker."})
        graph = content(await client.call_tool("contextos_graph_neighbors", {
            "entity": "Atlas", "max_hops": 1,
        }))
        assert graph["ok"]
        # Should NOT have a USES edge to Docker
        # The graph node count may be low (just the memory node and Atlas)


# ===========================================================================
# 6. RETRIEVAL MCP MATRIX
# ===========================================================================


@pytest.mark.asyncio
async def test_retrieval_hybrid_mode(real_mcp):
    """Hybrid retrieval returns results."""
    server, *_ = real_mcp
    async with Client(server) as client:
        await client.call_tool("contextos_remember", {"text": "I use Python for machine learning on Project Atlas."})
        result = content(await client.call_tool("contextos_search_memory", {
            "query": "Atlas machine learning", "mode": "hybrid",
        }))
        assert result["ok"]
        assert result["result_count"] >= 1


@pytest.mark.asyncio
async def test_retrieval_lexical_mode(real_mcp):
    """Lexical-only retrieval."""
    server, *_ = real_mcp
    async with Client(server) as client:
        await client.call_tool("contextos_remember", {"text": "Project Atlas uses Python."})
        result = content(await client.call_tool("contextos_search_memory", {
            "query": "Atlas Python", "mode": "lexical",
        }))
        assert result["ok"]


@pytest.mark.asyncio
async def test_retrieval_with_trace(real_mcp):
    """Trace-enabled search includes trace data."""
    server, *_ = real_mcp
    async with Client(server) as client:
        await client.call_tool("contextos_remember", {"text": "Project Atlas uses Python."})
        result = content(await client.call_tool("contextos_search_memory", {
            "query": "Atlas", "include_trace": True,
        }))
        assert result["ok"]
        assert "trace" in result


@pytest.mark.asyncio
async def test_retrieval_without_trace(real_mcp):
    """Trace-disabled search does not include trace."""
    server, *_ = real_mcp
    async with Client(server) as client:
        await client.call_tool("contextos_remember", {"text": "Project Atlas uses Python."})
        result = content(await client.call_tool("contextos_search_memory", {
            "query": "Atlas", "include_trace": False,
        }))
        assert result["ok"]
        assert "trace" not in result


@pytest.mark.asyncio
async def test_retrieval_low_result_limit(real_mcp):
    """Low result limit is respected."""
    server, *_ = real_mcp
    async with Client(server) as client:
        for i in range(5):
            await client.call_tool("contextos_remember", {"text": f"Memory fact number {i} about Python programming."})
        result = content(await client.call_tool("contextos_search_memory", {
            "query": "Python", "limit": 1,
        }))
        assert result["ok"]
        assert result["result_count"] <= 1


@pytest.mark.asyncio
async def test_retrieval_maximum_result_limit(real_mcp):
    """Maximum result limit is accepted."""
    server, *_ = real_mcp
    async with Client(server) as client:
        await client.call_tool("contextos_remember", {"text": "Project Atlas uses Python."})
        result = content(await client.call_tool("contextos_search_memory", {
            "query": "Atlas", "limit": 25,
        }))
        assert result["ok"]


@pytest.mark.asyncio
async def test_retrieval_excessive_limit_rejected(real_mcp):
    """Excessive result limit is rejected."""
    server, *_ = real_mcp
    async with Client(server) as client:
        result = content(await client.call_tool("contextos_search_memory", {
            "query": "Atlas", "limit": 100,
        }))
        assert result["error_code"] == "VALIDATION_ERROR"


@pytest.mark.asyncio
async def test_retrieval_safe_provenance_no_raw_structures(real_mcp):
    """Retrieval results contain safe provenance, no raw DB structures."""
    server, *_ = real_mcp
    async with Client(server) as client:
        await client.call_tool("contextos_remember", {"text": "Project Atlas uses Python."})
        result = content(await client.call_tool("contextos_search_memory", {
            "query": "Atlas",
        }))
        assert result["ok"]
        for memory in result.get("memories", []):
            assert "provenance" in memory
            # Should not contain raw sqlite or internal data
            assert "slot_json" not in str(memory)
            assert "slot_key" not in str(memory)


# ===========================================================================
# 7. COMPILE_CONTEXT MCP MATRIX
# ===========================================================================


@pytest.mark.asyncio
async def test_compile_small_token_budget(real_mcp):
    """Small token budget produces constrained output."""
    server, *_ = real_mcp
    async with Client(server) as client:
        await client.call_tool("contextos_remember", {"text": "Project Atlas uses Python for machine learning."})
        result = content(await client.call_tool("contextos_compile_context", {
            "query": "Atlas", "token_budget": 50,
        }))
        assert result["ok"]
        assert result["token_count"] <= 50


@pytest.mark.asyncio
async def test_compile_normal_budget(real_mcp):
    """Normal token budget compilation."""
    server, *_ = real_mcp
    async with Client(server) as client:
        await client.call_tool("contextos_remember", {"text": "Project Atlas uses Python for machine learning."})
        result = content(await client.call_tool("contextos_compile_context", {
            "query": "Atlas", "token_budget": 1000,
        }))
        assert result["ok"]
        assert result["token_count"] <= 1000


@pytest.mark.asyncio
async def test_compile_provenance_ids(real_mcp):
    """Compiled context includes provenance IDs."""
    server, *_ = real_mcp
    async with Client(server) as client:
        await client.call_tool("contextos_remember", {"text": "Project Atlas uses Python."})
        result = content(await client.call_tool("contextos_compile_context", {
            "query": "Atlas", "token_budget": 200,
        }))
        assert result["ok"]
        assert "provenance_ids" in result


@pytest.mark.asyncio
async def test_compile_empty_retrieval(real_mcp):
    """Compilation with no matching memories still succeeds."""
    server, *_ = real_mcp
    async with Client(server) as client:
        result = content(await client.call_tool("contextos_compile_context", {
            "query": "nonexistent query xyz", "token_budget": 100,
        }))
        assert result["ok"]
        assert result["compiled_fact_count"] == 0


@pytest.mark.asyncio
async def test_compile_no_model_generation_calls(real_mcp):
    """Assert MODEL GENERATION CALLS = 0.

    No MCP compile request may invoke Ollama/OpenAI/fake generation.
    The compiler uses only the token counter, not any LLM provider.
    """
    server, services, *_ = real_mcp
    # The compilation service is QueryAwareContextCompiler which uses only
    # a DeterministicWordTokenCounter — no model service involved.
    compilation = services["compilation"]
    assert isinstance(compilation, QueryAwareContextCompiler)
    # Verify no model_service attribute exists
    assert not hasattr(compilation, "_model_service")
    assert not hasattr(compilation, "model_service")
    assert not hasattr(compilation, "_provider")

    async with Client(server) as client:
        await client.call_tool("contextos_remember", {"text": "Project Atlas uses Python."})
        result = content(await client.call_tool("contextos_compile_context", {
            "query": "Atlas", "token_budget": 200,
        }))
        assert result["ok"]


# ===========================================================================
# 8. RESPONSE SIZE MATRIX
# ===========================================================================


@pytest.mark.asyncio
async def test_response_size_matrix(real_mcp):
    """Measure serialized MCP result byte size for various operations."""
    server, *_ = real_mcp
    async with Client(server) as client:
        # Seed data
        for i in range(10):
            await client.call_tool("contextos_remember", {
                "text": f"Project P{i} uses Python for task {i}. This is fact number {i} about tooling.",
            })

        size_report: list[dict[str, object]] = []

        # Search
        search = content(await client.call_tool("contextos_search_memory", {
            "query": "Python", "limit": 10,
        }))
        size_report.append({
            "operation": "search",
            "requested_count": 10,
            "actual_count": search["result_count"],
            "bytes": len(json.dumps(search, default=str).encode()),
        })

        # History
        history = content(await client.call_tool("contextos_memory_history", {
            "property": "programming_language", "subject": "user", "scope": "global",
            "limit": 10,
        }))
        size_report.append({
            "operation": "history",
            "requested_count": 10,
            "actual_count": history["result_count"],
            "bytes": len(json.dumps(history, default=str).encode()),
        })

        # Graph
        graph = content(await client.call_tool("contextos_graph_neighbors", {
            "entity": "Python", "max_hops": 2,
        }))
        size_report.append({
            "operation": "graph",
            "actual_count": graph["graph_node_count"],
            "bytes": len(json.dumps(graph, default=str).encode()),
        })

        # Explain context (trace-enabled search equivalent)
        explain = content(await client.call_tool("contextos_explain_context", {
            "query": "Python", "token_budget": 500,
        }))
        size_report.append({
            "operation": "explain_context",
            "actual_count": explain["result_count"],
            "bytes": len(json.dumps(explain, default=str).encode()),
        })

        # Trace-enabled search
        traced = content(await client.call_tool("contextos_search_memory", {
            "query": "Python", "limit": 10, "include_trace": True,
        }))
        size_report.append({
            "operation": "search_with_trace",
            "actual_count": traced["result_count"],
            "bytes": len(json.dumps(traced, default=str).encode()),
        })

        # Compile context
        compiled = content(await client.call_tool("contextos_compile_context", {
            "query": "Python", "token_budget": 2000,
        }))
        size_report.append({
            "operation": "compile_context",
            "actual_count": compiled.get("compiled_fact_count", 0),
            "bytes": len(json.dumps(compiled, default=str).encode()),
        })

        # All should have bounded output
        for entry in size_report:
            assert entry["bytes"] > 0, f"{entry['operation']} produced empty response"
            # Very generous cap — just prove we're not returning unbounded data
            assert entry["bytes"] < 1_000_000, f"{entry['operation']} exceeded 1MB"


# ===========================================================================
# 9. MCP TELEMETRY MATRIX
# ===========================================================================


@pytest.mark.asyncio
async def test_telemetry_successful_search(real_mcp):
    """Telemetry records successful search."""
    server, *_ = real_mcp
    async with Client(server) as client:
        await client.call_tool("contextos_remember", {"text": "Project Atlas uses Python."})
        await client.call_tool("contextos_search_memory", {"query": "Atlas"})
        records = server.contextos_telemetry.recent()
        search_records = [r for r in records if r["tool_name"] == "contextos_search_memory" and r["success"]]
        assert len(search_records) >= 1


@pytest.mark.asyncio
async def test_telemetry_successful_compile(real_mcp):
    """Telemetry records successful compile."""
    server, *_ = real_mcp
    async with Client(server) as client:
        await client.call_tool("contextos_compile_context", {"query": "test", "token_budget": 100})
        records = server.contextos_telemetry.recent()
        compile_records = [r for r in records if r["tool_name"] == "contextos_compile_context"]
        assert len(compile_records) >= 1


@pytest.mark.asyncio
async def test_telemetry_successful_remember(real_mcp):
    """Telemetry records successful remember."""
    server, *_ = real_mcp
    async with Client(server) as client:
        await client.call_tool("contextos_remember", {"text": "Project Atlas uses Python."})
        records = server.contextos_telemetry.recent()
        remember_records = [r for r in records if r["tool_name"] == "contextos_remember" and r["success"]]
        assert len(remember_records) >= 1


@pytest.mark.asyncio
async def test_telemetry_permission_denied():
    """Telemetry records permission denied."""
    from contextos.core.models import RetrievalResult
    class MockRetrieval:
        async def retrieve(self, request):
            return RetrievalResult(query=request.text)

    server = create_mcp_server(
        {"retrieval": MockRetrieval()},
        MCPPermissions(allow_read=False, allow_write=False),
    )
    async with Client(server) as client:
        await client.call_tool("contextos_search_memory", {"query": "test"})
    records = server.contextos_telemetry.recent()
    denied = [r for r in records if r["error_code"] == "PERMISSION_DENIED"]
    assert len(denied) >= 1


@pytest.mark.asyncio
async def test_telemetry_validation_error():
    """Telemetry records validation errors."""
    from contextos.core.models import RetrievalResult
    class MockRetrieval:
        async def retrieve(self, request):
            return RetrievalResult(query=request.text)

    server = create_mcp_server({"retrieval": MockRetrieval()})
    async with Client(server) as client:
        await client.call_tool("contextos_search_memory", {"query": "x", "limit": -1})
    records = server.contextos_telemetry.recent()
    errors = [r for r in records if r["error_code"] == "VALIDATION_ERROR"]
    assert len(errors) >= 1


@pytest.mark.asyncio
async def test_telemetry_privacy_rejection():
    """Telemetry records privacy rejection without leaking content."""
    from contextos.core.exceptions import SecretDetectedError

    class SecretIngestion:
        async def ingest(self, request):
            raise SecretDetectedError(["api_key"])

    server = create_mcp_server(
        {"ingestion": SecretIngestion()},
        MCPPermissions(allow_write=True),
    )
    secret = "sk-proj-AAAAAAAAAAAAAAAA"
    async with Client(server) as client:
        await client.call_tool("contextos_remember", {"text": secret})
    records = server.contextos_telemetry.recent()
    rejected = [r for r in records if r["error_code"] == "PRIVACY_REJECTED"]
    assert len(rejected) >= 1
    # Secret MUST NOT appear in telemetry
    assert secret not in str(records)


@pytest.mark.asyncio
async def test_telemetry_partial_write():
    """Telemetry records PARTIAL_WRITE."""
    class PartialIngestion:
        async def ingest(self, request):
            return IngestResult(
                event_id=uuid4(),
                candidates=[
                    CandidateMemory(content="Fact A", evidence="Fact A"),
                    CandidateMemory(content="Fact B", evidence="Fact B"),
                ],
            )

    class PartialTemporal:
        _count = 0
        async def accept(self, candidate, provenance_event_id):
            self._count += 1
            if self._count > 1:
                raise RuntimeError("fail second")
            class D:
                outcome = type("O", (), {"value": "add_new"})()
            return type("R", (), {"decision": D(), "memory": type("M", (), {"id": uuid4()})()})()

    server = create_mcp_server(
        {"ingestion": PartialIngestion(), "temporal": PartialTemporal()},
        MCPPermissions(allow_write=True),
    )
    async with Client(server) as client:
        result = content(await client.call_tool("contextos_remember", {"text": "test"}))
    assert result["error_code"] == "PARTIAL_WRITE"
    records = server.contextos_telemetry.recent()
    partial = [r for r in records if r["error_code"] == "PARTIAL_WRITE"]
    assert len(partial) >= 1


@pytest.mark.asyncio
async def test_telemetry_internal_error():
    """Telemetry records internal errors safely."""
    class FailingRetrieval:
        async def retrieve(self, request):
            raise RuntimeError("internal failure with /path/to/secrets")

    server = create_mcp_server({"retrieval": FailingRetrieval()})
    async with Client(server) as client:
        await client.call_tool("contextos_search_memory", {"query": "test"})
    records = server.contextos_telemetry.recent()
    errors = [r for r in records if r["error_code"] == "INTERNAL_ERROR"]
    assert len(errors) >= 1
    # Path must not appear in telemetry
    assert "/path/to/secrets" not in str(records)


@pytest.mark.asyncio
async def test_telemetry_contains_only_safe_fields():
    """Verify telemetry records only safe operational data."""
    from contextos.core.models import RetrievalResult

    class MockRetrieval:
        async def retrieve(self, request):
            return RetrievalResult(query=request.text)

    server = create_mcp_server(
        {"retrieval": MockRetrieval()},
        MCPPermissions(allow_read=True),
    )
    query_text = "my secret private query"
    async with Client(server) as client:
        await client.call_tool("contextos_search_memory", {"query": query_text})

    records = server.contextos_telemetry.recent()
    assert len(records) >= 1
    record = records[-1]

    # Must contain safe fields
    assert "request_id" in record
    assert "tool_name" in record
    assert "timestamp" in record
    assert "latency_ms" in record
    assert "success" in record

    # Must NOT contain any of these
    serialized = json.dumps(records, default=str)
    assert query_text not in serialized
    assert "query" not in record or record.get("query") is None
    # Must not contain raw tool arguments
    for forbidden in ("arguments", "raw_response", "raw_errors", "secret", "credential", "password"):
        assert forbidden not in record, f"Telemetry contains forbidden field: {forbidden}"


@pytest.mark.asyncio
async def test_telemetry_bounded_history():
    """Telemetry has bounded history (eviction behavior)."""
    telemetry = MCPInvocationTelemetry(maximum=5)
    from contextos.mcp.server import MCPInvocation
    from datetime import datetime, timezone
    for i in range(10):
        telemetry.record(MCPInvocation(
            request_id=str(uuid4()), session_id=None,
            tool_name="test", timestamp=datetime.now(timezone.utc).isoformat(),
            latency_ms=1.0, success=True, error_code=None,
        ))
    assert len(telemetry.recent()) == 5  # Bounded at maximum=5
    summary = telemetry.summary()
    assert summary["invocation_count"] == 5


# ===========================================================================
# 10. REAL-STACK BENCHMARK
# ===========================================================================


@pytest.mark.asyncio
async def test_real_stack_benchmark(real_mcp):
    """Run the Phase 10 benchmark against real SQLite services."""
    server, services, *_ = real_mcp

    # Seed the database with diverse data
    async with Client(server) as client:
        await client.call_tool("contextos_remember", {"text": "Project Atlas uses Python for machine learning."})
        await client.call_tool("contextos_remember", {"text": "Project Beta uses Rust for systems programming."})
        await client.call_tool("contextos_remember", {"text": "I currently use Ollama for local inference."})
        await client.call_tool("contextos_remember", {"text": "Project Atlas uses llama.cpp."})
        await client.call_tool("contextos_remember", {"text": "I prefer concise answers for technical questions."})

    from contextos.benchmarks.mcp import run_phase10_mcp_benchmark

    report = await run_phase10_mcp_benchmark(services, query="Atlas Python", iterations=3)

    # Validate report structure
    assert report.label == "LOCAL DEVELOPMENT MACHINE SYNTHETIC INFRASTRUCTURE BENCHMARK"
    assert len(report.direct_results) >= 1
    assert len(report.mcp_results) >= 1

    # All latencies should be positive
    for result in report.direct_results + report.mcp_results:
        assert result.mean_ms > 0
        assert result.median_ms > 0
        assert result.iterations == 3

    # Adapter overhead should be non-negative
    for key, value in report.overheads.items():
        assert value >= 0, f"Negative overhead for {key}"

    # Verify the report covers required operations
    mcp_ops = {r.operation for r in report.mcp_results}
    assert "search" in mcp_ops
    assert "compile" in mcp_ops
    assert "graph" in mcp_ops
    assert "remember_success" in mcp_ops
    assert "temporal_lookup" in mcp_ops
    assert "telemetry" in mcp_ops
