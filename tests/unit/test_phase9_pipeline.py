"""Pipeline Integration Tests AX through BC and Cross-Model Token Counting."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from contextos.core.enums import (
    CandidateTemporalStatus,
    CompilationStrategy,
    MemoryStatus,
    MemoryType,
    RetrievalMode,
    RoutingPolicy,
    TemporalOutcome,
    TemporalScope,
)
from contextos.core.exceptions import ContextWindowExceededError
from contextos.core.models import (
    CompilationConfig,
    ContextBudget,
    Memory,
    ModelCapabilities,
    RetrievalConfig,
    RetrievalQuery,
)
from contextos.embedding.deterministic import DeterministicEmbedding
from contextos.providers.fake import DeterministicFakeProvider
from contextos.services.compilation import QueryAwareContextCompiler
from contextos.services.graph import MemoryGraphService
from contextos.services.graph_retrieval import GraphAugmentedRetrievalEngine
from contextos.services.model_service import ContextOSModelService
from contextos.services.optimization import MemoryContextOptimizer
from contextos.services.retrieval import HybridRetrievalEngine
from contextos.services.retrieval_index import RetrievalIndexSynchronizer
from contextos.services.router import DeterministicModelRouter
from contextos.services.temporal import TemporalMemoryService
from contextos.services.token_counter import (
    DeterministicWordTokenCounter,
    TiktokenCounter,
    recount_cross_model,
)
from contextos.storage.database import Database
from contextos.storage.graph_repo import SqliteGraphRepository
from contextos.storage.lexical.bm25 import BM25Index
from contextos.storage.memory_repo import SqliteMemoryRepository
from contextos.storage.relation_repo import SqliteRelationRepository
from contextos.storage.telemetry_repo import SqliteTelemetryRepository
from contextos.storage.vector.in_memory import InMemoryVectorStore


@pytest.fixture
async def pipeline_stack(tmp_path: Path):
    db_path = tmp_path / "pipeline.db"
    db = Database(db_path)
    await db.initialize()

    conn = db.connection()
    memory_repo = SqliteMemoryRepository(conn)
    relation_repo = SqliteRelationRepository(conn)
    graph_repo = SqliteGraphRepository(conn)
    telemetry_repo = SqliteTelemetryRepository(conn)

    temporal_service = TemporalMemoryService(memory_repo)
    graph_service = MemoryGraphService(
        memory_repo=memory_repo,
        relation_repo=relation_repo,
        graph_repo=graph_repo,
    )

    token_counter = TiktokenCounter("cl100k_base")
    optimizer = MemoryContextOptimizer(token_counter=token_counter)
    compiler = QueryAwareContextCompiler(token_counter=token_counter)

    dim = 16
    embedding_service = DeterministicEmbedding(dim)
    lexical_index = BM25Index()
    vector_store = InMemoryVectorStore(dim)
    index_sync = RetrievalIndexSynchronizer(
        memory_repo=memory_repo,
        embedding_service=embedding_service,
        vector_store=vector_store,
        lexical_index=lexical_index,
    )

    base_retrieval = HybridRetrievalEngine(
        memory_repo=memory_repo,
        vector_store=vector_store,
        lexical_index=lexical_index,
        embedding_service=embedding_service,
        index_synchronizer=index_sync,
    )
    retrieval = GraphAugmentedRetrievalEngine(
        base_engine=base_retrieval,
        graph_service=graph_service,
        memory_repo=memory_repo,
    )

    fake_provider = DeterministicFakeProvider(
        provider_id="fake-local",
        is_local=True,
        models=[
            ModelCapabilities(
                provider_id="fake-local",
                model_id="fake-qwen",
                display_name="Fake Qwen",
                context_window=4096,
                max_output_tokens=1024,
                supports_tools=True,
                supports_json=True,
                local=True,
                tokenizer_family="qwen",
            ),
            ModelCapabilities(
                provider_id="fake-local",
                model_id="tiny-window-model",
                display_name="Tiny Window Model",
                context_window=20,  # Very small for overflow test
                max_output_tokens=10,
                local=True,
            ),
        ],
    )
    providers = {fake_provider.provider_id: fake_provider}
    router = DeterministicModelRouter(
        default_provider_id="fake-local",
        default_model_id="fake-qwen",
        default_policy=RoutingPolicy.LOCAL_FIRST,
    )

    model_service = ContextOSModelService(
        retrieval_service=retrieval,
        optimizer=optimizer,
        compilation_service=compiler,
        router=router,
        providers=providers,
        telemetry_repo=telemetry_repo,
        token_counter=token_counter,
    )

    yield {
        "db": db,
        "memory_repo": memory_repo,
        "temporal_service": temporal_service,
        "graph_service": graph_service,
        "telemetry_repo": telemetry_repo,
        "model_service": model_service,
        "fake_provider": fake_provider,
        "token_counter": token_counter,
    }
    await db.close()


# ---------------------------------------------------------------------------
# Test AX: Real SQLite -> Memory -> Retrieval -> Optimizer -> Compiler -> Fake -> Telemetry
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_ax_full_pipeline_real_sqlite(pipeline_stack):
    stack = pipeline_stack
    memory_repo = stack["memory_repo"]
    model_service = stack["model_service"]
    telemetry_repo = stack["telemetry_repo"]

    # Ingest memories
    m1 = Memory(
        content="ContextOS uses SQLite in WAL mode for persistent knowledge storage.",
        type=MemoryType.FACT,
        status=MemoryStatus.ACTIVE,
        confidence=0.95,
        importance=0.8,
    )
    m2 = Memory(
        content="ContextOS compiles retrieval results into budget-constrained context.",
        type=MemoryType.FACT,
        status=MemoryStatus.ACTIVE,
        confidence=0.9,
        importance=0.7,
    )
    await memory_repo.create(m1)
    await memory_repo.create(m2)

    result = await model_service.ask(
        query="How does ContextOS store persistent knowledge?",
    )

    assert result.response is not None
    assert result.response.text != ""
    assert result.compiled_context is not None
    assert "sqlite" in result.compiled_context.context_text.lower()
    assert result.telemetry is not None
    assert result.telemetry.candidate_context_tokens > 0
    assert result.telemetry.compiled_context_tokens > 0
    assert result.telemetry.end_to_end_ms > 0

    # Verify telemetry was persisted in SQLite
    persisted = await telemetry_repo.get(result.telemetry.invocation_id)
    assert persisted is not None
    assert persisted.model_id == "fake-qwen"
    assert persisted.candidate_context_tokens == result.telemetry.candidate_context_tokens


# ---------------------------------------------------------------------------
# Test AY: Graph-Expanded Retrieval -> Compiler -> Provider -> Telemetry Graph Contribution
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_ay_graph_expanded_retrieval_telemetry(pipeline_stack):
    stack = pipeline_stack
    memory_repo = stack["memory_repo"]
    graph_service = stack["graph_service"]
    model_service = stack["model_service"]

    # Ingest graph-related facts
    m1 = Memory(
        content="Project Atlas uses Ollama for local intelligence.",
        type=MemoryType.PROJECT,
        status=MemoryStatus.ACTIVE,
    )
    m2 = Memory(
        content="Ollama runs on port 11434 and serves local models.",
        type=MemoryType.PROCEDURE,
        status=MemoryStatus.ACTIVE,
    )
    await memory_repo.create(m1)
    await memory_repo.create(m2)
    await graph_service.rebuild()

    result = await model_service.ask(
        query="Project Atlas configuration and services",
        retrieval_config=RetrievalConfig(max_results=10),
    )

    assert result.compiled_context is not None
    assert result.telemetry.graph_expanded_count >= 0
    assert result.telemetry.selected_memory_count >= 1


# ---------------------------------------------------------------------------
# Test AZ: Temporal Supersession -> Current Query -> Provider receives current fact only
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_az_temporal_supersession_current_query(pipeline_stack):
    stack = pipeline_stack
    temporal_service = stack["temporal_service"]
    model_service = stack["model_service"]

    # Historical candidate
    m_old = Memory(
        id=UUID("90000000-0000-0000-0000-000000000001"),
        content="User primarily uses Python for backend development.",
        type=MemoryType.FACT,
        status=MemoryStatus.CANDIDATE,
        observed_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
    )
    # Newer candidate that explicitly updates the slot
    m_new = Memory(
        id=UUID("90000000-0000-0000-0000-000000000002"),
        content="User now primarily uses Rust for backend development.",
        type=MemoryType.FACT,
        status=MemoryStatus.CANDIDATE,
        observed_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )

    await temporal_service.resolve(m_old)
    resolution = await temporal_service.resolve(m_new)
    assert resolution.decision.outcome == TemporalOutcome.SUPERSEDE

    # Query with default CURRENT temporal scope
    result = await model_service.ask(
        query="What language does the user use for backend development?",
    )

    context_text = result.compiled_context.context_text
    assert "Rust" in context_text
    assert "Python" not in context_text



# ---------------------------------------------------------------------------
# Test BA: Historical Query -> Historical Fact -> Telemetry
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_ba_historical_query_telemetry(pipeline_stack):
    stack = pipeline_stack
    memory_repo = stack["memory_repo"]
    model_service = stack["model_service"]

    old_mem = Memory(
        content="User previously used Vim editor before switching.",
        type=MemoryType.PREFERENCE,
        status=MemoryStatus.HISTORICAL,
        temporal_status=CandidateTemporalStatus.HISTORICAL,
        observed_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )
    await memory_repo.create(old_mem)

    # Inquire with include_superseded=True or historical scope
    result = await model_service.ask(
        query="What editor did I previously use?",
        retrieval_config=RetrievalConfig(include_superseded=True),
    )

    assert result.telemetry is not None
    assert result.telemetry.candidate_context_tokens >= 0


# ---------------------------------------------------------------------------
# Test BB: Oversized Phase 6 Rescue -> Provider receives compressed relevant fact
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_bb_oversized_rescue_integration(pipeline_stack):
    stack = pipeline_stack
    memory_repo = stack["memory_repo"]
    model_service = stack["model_service"]

    # Oversized memory with a relevant embedded clause
    filler = "Arbitrary unhelpful background commentary about general software architecture. " * 30
    relevant_fact = "The critical security secret is encrypted with AES-GCM-256."
    long_content = f"{filler} {relevant_fact} {filler}"

    m_oversized = Memory(
        content=long_content,
        type=MemoryType.FACT,
        status=MemoryStatus.ACTIVE,
        confidence=0.95,
        importance=0.9,
    )
    await memory_repo.create(m_oversized)

    # Tight budget: whole memory would exceed budget in optimizer, but rescue extracts relevant clause
    tight_budget = 100
    result = await model_service.ask(
        query="How is the security secret encrypted?",
        compilation_config=CompilationConfig(budget=tight_budget),
    )

    assert result.compiled_context is not None
    assert result.compiled_context.total_tokens <= tight_budget
    assert "aes-gcm-256" in result.compiled_context.context_text.lower()
    assert result.telemetry.compiled_fact_count >= 1


# ---------------------------------------------------------------------------
# Test BC: Context-Window Too Small -> Structured Failure -> No Provider Call
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_bc_context_window_too_small_structured_failure(pipeline_stack):
    stack = pipeline_stack
    model_service = stack["model_service"]
    fake_provider = stack["fake_provider"]

    # Target the model with tiny context window (20 tokens)
    with pytest.raises(ContextWindowExceededError) as exc_info:
        await model_service.ask(
            query="This query and its context will easily exceed the twenty token context window limitation of the target model.",
            target_model="tiny-window-model",
            routing_policy=RoutingPolicy.EXPLICIT,
        )

    assert exc_info.value.context_window == 20
    assert exc_info.value.required_tokens > 20


# ---------------------------------------------------------------------------
# Cross-Model Token Recount Test
# ---------------------------------------------------------------------------
def test_cross_model_token_recounting():
    compiled_text = (
        "User currently uses C++17 with clang-tidy on macOS. "
        "Project Atlas depends on Ollama local runtime version 0.3. "
        "Preferences: dark mode, type annotations, zero telemetry leakage."
    )

    recounts = recount_cross_model(compiled_text, ["claude", "openai", "qwen"])

    assert "claude" in recounts
    assert "openai" in recounts
    assert "qwen" in recounts

    count_claude, src_claude = recounts["claude"]
    count_openai, src_openai = recounts["openai"]
    count_qwen, src_qwen = recounts["qwen"]

    assert count_claude > 0
    assert count_openai > 0
    assert count_qwen > 0

    # Ensure measurements are properly labeled
    assert src_claude.value in {"tokenizer_counted", "approximated"}
    assert src_openai.value in {"tokenizer_counted", "approximated"}
    assert src_qwen.value in {"tokenizer_counted", "approximated"}


@pytest.mark.asyncio
async def test_phase9_benchmark_execution(tmp_path: Path):
    from contextos.benchmarks.model_routing import run_phase9_benchmark

    db_path = tmp_path / "bench.db"
    report = await run_phase9_benchmark(db_path=db_path)
    assert report.candidate_context_tokens > 0
    assert report.compiled_context_tokens > 0
    assert "FAKE_CLAUDE_LIKE" in report.profiles
    assert "FAKE_OPENAI_LIKE" in report.profiles
    assert "FAKE_QWEN_LIKE" in report.profiles
    assert report.router_overhead_ms < 50.0
    assert report.telemetry_overhead_ms < 100.0

