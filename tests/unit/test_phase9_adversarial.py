"""Adversarial Review Tests for ContextOS Phase 9.

Validates truthfulness, token accounting, safety margins, secret sanitization,
failure resilience, migration from Phase 8 schema v4, and aggregation weighting.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from contextos.core.enums import (
    MemoryStatus,
    MemoryType,
    ModelFinishReason,
    RoutingPolicy,
    TemporalScope,
    TokenMeasurementSource,
)
from contextos.core.exceptions import (
    ContextWindowExceededError,
    MalformedProviderResponseError,
    ModelUnavailableError,
    ProviderAuthenticationError,
    ProviderUnavailableError,
)
from contextos.core.models import (
    CompilationConfig,
    CompiledContext,
    ContextBudget,
    Memory,
    ModelCapabilities,
    ModelInvocationTelemetry,
    ModelRequest,
    ModelResponse,
    RetrievalConfig,
    RetrievalResult,
    RetrievalTrace,
    RouteDecision,
    ScoredMemory,
    StageTrace,
)
from contextos.providers.fake import DeterministicFakeProvider
from contextos.providers.openai_compatible import OpenAICompatibleProvider
from contextos.providers.ollama import OllamaProvider
from contextos.services.compilation import QueryAwareContextCompiler
from contextos.services.model_service import ContextOSModelService
from contextos.services.optimization import MemoryContextOptimizer
from contextos.services.router import DeterministicModelRouter
from contextos.services.token_counter import (
    ClaudeProfileTokenCounter,
    DeterministicWordTokenCounter,
    QwenProfileTokenCounter,
    TiktokenCounter,
    get_token_counter_for_model,
)
from contextos.storage.database import Database, SCHEMA_SQL, MIGRATION_2_SQL, MIGRATION_3_SQL, MIGRATION_4_SQL
from contextos.storage.telemetry_repo import SqliteTelemetryRepository, sanitize_telemetry_metadata


# ===========================================================================
# Attack 1: Token Measurement Truthfulness
# ===========================================================================

def test_approximate_tokenizer_not_mislabeled_exact():
    """Heuristic/profile counters must emit APPROXIMATED, never TOKENIZER_COUNTED."""
    claude = ClaudeProfileTokenCounter()
    assert claude.measurement_source == TokenMeasurementSource.APPROXIMATED
    assert claude.count("Technical Claude-like context") > 0

    qwen = QwenProfileTokenCounter()
    assert qwen.measurement_source == TokenMeasurementSource.APPROXIMATED
    assert qwen.count("Technical Qwen-like context") > 0

    heuristic = DeterministicWordTokenCounter()
    assert heuristic.measurement_source == TokenMeasurementSource.APPROXIMATED

    # Only genuine tokenizers running real BPE encoding may emit TOKENIZER_COUNTED
    tiktoken_cnt = TiktokenCounter("cl100k_base")
    assert tiktoken_cnt.measurement_source == TokenMeasurementSource.TOKENIZER_COUNTED


# ===========================================================================
# Attack 2 & 3: Cross-Tokenizer Subtraction Prevention & Model-Specific Recounting
# ===========================================================================

@pytest.mark.asyncio
async def test_cross_tokenizer_subtraction_prevented():
    """Candidate and compiled counts must use the target model's tokenizer basis."""
    db_path = Path(tempfile.mkdtemp()) / "test_cross_tok.db"
    db = Database(db_path)
    await db.initialize()
    repo = SqliteTelemetryRepository(db.connection())

    tik_counter = TiktokenCounter("cl100k_base")
    optimizer = MemoryContextOptimizer(token_counter=tik_counter)
    compiler = QueryAwareContextCompiler(token_counter=tik_counter)

    qwen_counter = QwenProfileTokenCounter()

    text1 = "Complex CamelCaseIdentifier with technical_symbols $#@! and numbers 12345"
    text2 = "Another long string containing system preferences and memory state."

    # Build memories
    m1 = Memory(id=uuid4(), content=text1, status=MemoryStatus.ACTIVE)
    m2 = Memory(id=uuid4(), content=text2, status=MemoryStatus.ACTIVE)
    scored = [ScoredMemory(memory=m1, final_score=0.9), ScoredMemory(memory=m2, final_score=0.8)]

    class MockRetrieval:
        async def retrieve(self, q, c=None):
            return RetrievalResult(
                query=q,
                memories=scored,
                strategy_results={"lexical": scored, "dense": []},
                trace=RetrievalTrace(stages=[], total_latency_ms=1.0, total_candidates=2, total_results=2),
            )

    # Provider with Qwen model
    qwen_model = ModelCapabilities(
        provider_id="fake-qwen-prov",
        model_id="qwen-2.5-7b",
        display_name="Qwen 2.5",
        context_window=8192,
        tokenizer_family="qwen",
        local=True,
    )
    provider = DeterministicFakeProvider(
        provider_id="fake-qwen-prov",
        models=[qwen_model],
    )
    router = DeterministicModelRouter(
        default_provider_id="fake-qwen-prov",
        default_model_id="qwen-2.5-7b",
        default_policy=RoutingPolicy.FIXED_DEFAULT,
    )

    service = ContextOSModelService(
        retrieval_service=MockRetrieval(),
        optimizer=optimizer,
        compilation_service=compiler,
        router=router,
        providers={"fake-qwen-prov": provider},
        telemetry_repo=repo,
        token_counter=tik_counter,
    )

    result = await service.ask("Explain system state", routing_policy=RoutingPolicy.FIXED_DEFAULT)

    # Verify candidate tokens in telemetry were counted with QwenProfileTokenCounter, NOT tiktoken!
    expected_qwen_candidate = qwen_counter.count(text1) + qwen_counter.count(text2)
    expected_qwen_compiled = qwen_counter.count(result.compiled_context.context_text)

    assert result.telemetry.candidate_context_tokens == expected_qwen_candidate
    assert result.telemetry.compiled_context_tokens == expected_qwen_compiled
    assert result.telemetry.context_tokens_avoided == (expected_qwen_candidate - expected_qwen_compiled)
    expected_ratio = 1.0 - (expected_qwen_compiled / expected_qwen_candidate)
    assert abs(result.telemetry.reduction_ratio - expected_ratio) < 1e-6

    await db.close()


@pytest.mark.asyncio
async def test_cross_model_comparison_isolation():
    """Routing the same context to different models independently recounts tokens."""
    db_path = Path(tempfile.mkdtemp()) / "test_iso.db"
    db = Database(db_path)
    await db.initialize()
    repo = SqliteTelemetryRepository(db.connection())

    content = "User configures LLM with async httpx transport and sqlite-wal projection."
    m = Memory(id=uuid4(), content=content, status=MemoryStatus.ACTIVE)
    scored = [ScoredMemory(memory=m, final_score=0.9)]

    class MockRetrieval:
        async def retrieve(self, q, c=None):
            return RetrievalResult(
                query=q,
                memories=scored,
                strategy_results={"lexical": scored},
                trace=RetrievalTrace(stages=[], total_latency_ms=1.0, total_candidates=1, total_results=1),
            )

    tik_counter = TiktokenCounter("cl100k_base")
    optimizer = MemoryContextOptimizer(token_counter=tik_counter)
    compiler = QueryAwareContextCompiler(token_counter=tik_counter)

    claude_model = ModelCapabilities(
        provider_id="fake-claude-prov",
        model_id="claude-3-5-sonnet",
        display_name="Claude",
        context_window=10000,
        tokenizer_family="claude",
        local=False,
    )
    openai_model = ModelCapabilities(
        provider_id="fake-openai-prov",
        model_id="gpt-4o",
        display_name="GPT-4o",
        context_window=10000,
        tokenizer_family="o200k_base",
        local=False,
    )
    qwen_model = ModelCapabilities(
        provider_id="fake-qwen-prov",
        model_id="qwen-2.5",
        display_name="Qwen",
        context_window=10000,
        tokenizer_family="qwen",
        local=True,
    )

    prov_claude = DeterministicFakeProvider(provider_id="fake-claude-prov", models=[claude_model], is_local=False)
    prov_openai = DeterministicFakeProvider(provider_id="fake-openai-prov", models=[openai_model], is_local=False)
    prov_qwen = DeterministicFakeProvider(provider_id="fake-qwen-prov", models=[qwen_model], is_local=True)

    providers = {
        "fake-claude-prov": prov_claude,
        "fake-openai-prov": prov_openai,
        "fake-qwen-prov": prov_qwen,
    }
    router = DeterministicModelRouter()

    service = ContextOSModelService(
        retrieval_service=MockRetrieval(),
        optimizer=optimizer,
        compilation_service=compiler,
        router=router,
        providers=providers,
        telemetry_repo=repo,
        token_counter=tik_counter,
    )

    res_claude = await service.ask("query", target_provider="fake-claude-prov", target_model="claude-3-5-sonnet", routing_policy=RoutingPolicy.EXPLICIT)
    res_openai = await service.ask("query", target_provider="fake-openai-prov", target_model="gpt-4o", routing_policy=RoutingPolicy.EXPLICIT)
    res_qwen = await service.ask("query", target_provider="fake-qwen-prov", target_model="qwen-2.5", routing_policy=RoutingPolicy.EXPLICIT)

    # Counters
    c_cnt = ClaudeProfileTokenCounter()
    o_cnt = TiktokenCounter("o200k_base")
    q_cnt = QwenProfileTokenCounter()

    assert res_claude.telemetry.candidate_context_tokens == c_cnt.count(content)
    assert res_openai.telemetry.candidate_context_tokens == o_cnt.count(content)
    assert res_qwen.telemetry.candidate_context_tokens == q_cnt.count(content)

    await db.close()


# ===========================================================================
# Attack 4 & 6: Preflight vs Provider-Reported Tokens
# ===========================================================================

@pytest.mark.asyncio
async def test_preflight_vs_provider_reported_disagreement():
    """Preflight calculation must remain distinct from authoritative provider reported usage."""
    db_path = Path(tempfile.mkdtemp()) / "test_preflight.db"
    db = Database(db_path)
    await db.initialize()
    repo = SqliteTelemetryRepository(db.connection())

    model = ModelCapabilities(
        provider_id="fake",
        model_id="m1",
        display_name="M1",
        context_window=4096,
        tokenizer_family="cl100k_base",
    )
    provider = DeterministicFakeProvider(provider_id="fake", models=[model], report_usage=True)

    class MockRetrieval:
        async def retrieve(self, q, c=None):
            return RetrievalResult(
                query=q, memories=[], strategy_results={},
                trace=RetrievalTrace(stages=[], total_latency_ms=1.0, total_candidates=0, total_results=0),
            )

    tik_counter = TiktokenCounter("cl100k_base")
    optimizer = MemoryContextOptimizer(token_counter=tik_counter)
    compiler = QueryAwareContextCompiler(token_counter=tik_counter)

    router = DeterministicModelRouter(default_provider_id="fake", default_model_id="m1")

    service = ContextOSModelService(
        retrieval_service=MockRetrieval(),
        optimizer=optimizer,
        compilation_service=compiler,
        router=router,
        providers={"fake": provider},
        telemetry_repo=repo,
        token_counter=tik_counter,
    )

    result = await service.ask("Short question", routing_policy=RoutingPolicy.FIXED_DEFAULT)

    assert result.telemetry.preflight_input_tokens > 0
    assert result.telemetry.final_input_tokens == result.telemetry.preflight_input_tokens
    assert result.telemetry.provider_input_tokens > 0
    assert "preflight_input_tokens" in result.telemetry.model_dump()
    assert "provider_input_tokens" in result.telemetry.model_dump()

    await db.close()


# ===========================================================================
# Attack 5: Context-Window Safety Boundary
# ===========================================================================

@pytest.mark.asyncio
async def test_context_window_boundary_with_safety_margin():
    """Request near context-window boundary must be rejected due to framing safety buffer."""
    router = DeterministicModelRouter()
    model = ModelCapabilities(
        provider_id="local",
        model_id="tiny",
        display_name="Tiny",
        context_window=100,
        tokenizer_family="cl100k_base",
        local=True,
    )
    provider = DeterministicFakeProvider(provider_id="local", models=[model])
    providers = {"local": provider}

    prompt_text = "word " * 50
    req = ModelRequest(
        user_prompt=prompt_text,
        provider="local",
        model="tiny",
        max_output_tokens=40,
    )

    with pytest.raises(ContextWindowExceededError) as exc_info:
        await router.route(req, providers, policy=RoutingPolicy.EXPLICIT)
    assert exc_info.value.context_window == 100
    assert exc_info.value.required_tokens > 100


# ===========================================================================
# Attack 7 & 8: Secret Sanitization in Metadata and Error Messages
# ===========================================================================

def test_metadata_secret_injection_and_sanitization():
    """Sensitive keys and token values in metadata must be recursively redacted."""
    raw_meta = {
        "user_id": "u123",
        "api_key": "sk-1234567890abcdef12345678",
        "authorization": "Bearer secret_jwt_token_here",
        "nested": {
            "password": "super_secret_pw",
            "safe_tag": "v1.0",
            "deep_token": "token_xyz",
        },
        "items": [
            {"raw_prompt": "secret prompt containing sensitive code"},
            "safe string",
            "Found token: Bearer abcdef1234567890 in log",
        ],
    }

    sanitized = sanitize_telemetry_metadata(raw_meta)

    assert sanitized["user_id"] == "u123"
    assert sanitized["api_key"] == "[REDACTED]"
    assert sanitized["authorization"] == "[REDACTED]"
    assert sanitized["nested"]["password"] == "[REDACTED]"
    assert sanitized["nested"]["safe_tag"] == "v1.0"
    assert sanitized["nested"]["deep_token"] == "[REDACTED]"
    assert sanitized["items"][0]["raw_prompt"] == "[REDACTED]"
    assert "Bearer" not in sanitized["items"][2] or "[REDACTED_SECRET]" in sanitized["items"][2]


# ===========================================================================
# Attack 10: Local vs Remote Endpoint Classification
# ===========================================================================

def test_openai_compatible_locality_detection():
    """Verify private LAN and loopback addresses are classified as local, public as remote."""
    p1 = OpenAICompatibleProvider(base_url="http://127.0.0.1:8000/v1")
    assert p1.is_local is True

    p2 = OpenAICompatibleProvider(base_url="http://localhost:11434/v1")
    assert p2.is_local is True

    p3 = OpenAICompatibleProvider(base_url="http://[::1]:8000/v1")
    assert p3.is_local is True

    p4 = OpenAICompatibleProvider(base_url="http://192.168.1.50:8000/v1")
    assert p4.is_local is True

    p5 = OpenAICompatibleProvider(base_url="http://10.0.0.2:8000/v1")
    assert p5.is_local is True

    p6 = OpenAICompatibleProvider(base_url="http://172.20.1.1:8000/v1")
    assert p6.is_local is True

    p7 = OpenAICompatibleProvider(base_url="https://api.openai.com/v1")
    assert p7.is_local is False

    p8 = OpenAICompatibleProvider(base_url="https://inference.company.com/v1")
    assert p8.is_local is False

    p9 = OpenAICompatibleProvider(base_url="https://api.openai.com/v1", is_local=True)
    assert p9.is_local is True


# ===========================================================================
# Attack 11: No Silent Cloud Fallback
# ===========================================================================

@pytest.mark.asyncio
async def test_no_silent_cloud_fallback():
    """LOCAL_FIRST must fail if local is down and fallback is disabled."""
    router = DeterministicModelRouter()
    local_p = DeterministicFakeProvider(provider_id="loc", is_local=True)
    local_p.simulate_unhealthy = True

    remote_p = DeterministicFakeProvider(provider_id="rem", is_local=False)
    providers = {"loc": local_p, "rem": remote_p}

    req_no_fallback = ModelRequest(user_prompt="Hello", allow_fallback=False)
    with pytest.raises(ProviderUnavailableError) as exc_info:
        await router.route(req_no_fallback, providers, policy=RoutingPolicy.LOCAL_FIRST)
    assert "fallback not allowed" in str(exc_info.value).lower()

    req_with_fallback = ModelRequest(user_prompt="Hello", allow_fallback=True)
    decision = await router.route(req_with_fallback, providers, policy=RoutingPolicy.LOCAL_FIRST)
    assert decision.fallback_used is True
    assert decision.selected_provider == "rem"
    assert decision.initial_provider == "loc"
    assert decision.fallback_reason is not None


# ===========================================================================
# Attack 16: Malformed Provider Response
# ===========================================================================

@pytest.mark.asyncio
async def test_openai_compatible_malformed_json_response():
    """Non-JSON response must raise MalformedProviderResponseError."""
    import httpx

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>502 Bad Gateway from reverse proxy</html>")

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport, base_url="http://test-endpoint/v1")

    provider = OpenAICompatibleProvider(base_url="http://test-endpoint/v1", client=client)
    req = ModelRequest(user_prompt="Test")
    with pytest.raises(MalformedProviderResponseError):
        await provider.generate(req)


# ===========================================================================
# Attack 17: Telemetry Failure Semantics & Safe Failure Record
# ===========================================================================

@pytest.mark.asyncio
async def test_telemetry_db_failure_does_not_lose_model_response():
    """If telemetry insertion fails, ask() must still return the successful response."""
    class FailingRepo:
        async def record(self, tel):
            raise sqlite3.OperationalError("database disk image is malformed")

    provider = DeterministicFakeProvider(provider_id="fake")
    router = DeterministicModelRouter(default_provider_id="fake")

    class MockRetrieval:
        async def retrieve(self, q, c=None):
            return RetrievalResult(
                query=q, memories=[], strategy_results={},
                trace=RetrievalTrace(stages=[], total_latency_ms=1.0, total_candidates=0, total_results=0),
            )

    tik_counter = TiktokenCounter("cl100k_base")
    optimizer = MemoryContextOptimizer(token_counter=tik_counter)
    compiler = QueryAwareContextCompiler(token_counter=tik_counter)

    service = ContextOSModelService(
        retrieval_service=MockRetrieval(),
        optimizer=optimizer,
        compilation_service=compiler,
        router=router,
        providers={"fake": provider},
        telemetry_repo=FailingRepo(),
        token_counter=tik_counter,
    )

    result = await service.ask("What is 2+2?")
    assert result.response.text is not None
    assert len(result.response.text) > 0


@pytest.mark.asyncio
async def test_provider_failure_persists_safe_failure_telemetry():
    """If provider generation fails, a safe error telemetry record must be persisted."""
    db_path = Path(tempfile.mkdtemp()) / "test_fail_tel.db"
    db = Database(db_path)
    await db.initialize()
    repo = SqliteTelemetryRepository(db.connection())

    provider = DeterministicFakeProvider(provider_id="fake")
    provider.simulate_timeout = True
    router = DeterministicModelRouter(default_provider_id="fake")

    class MockRetrieval:
        async def retrieve(self, q, c=None):
            return RetrievalResult(
                query=q, memories=[], strategy_results={},
                trace=RetrievalTrace(stages=[], total_latency_ms=1.0, total_candidates=0, total_results=0),
            )

    tik_counter = TiktokenCounter("cl100k_base")
    optimizer = MemoryContextOptimizer(token_counter=tik_counter)
    compiler = QueryAwareContextCompiler(token_counter=tik_counter)

    service = ContextOSModelService(
        retrieval_service=MockRetrieval(),
        optimizer=optimizer,
        compilation_service=compiler,
        router=router,
        providers={"fake": provider},
        telemetry_repo=repo,
        token_counter=tik_counter,
    )

    from contextos.core.exceptions import ProviderTimeoutError
    with pytest.raises(ProviderTimeoutError):
        await service.ask("Trigger failure")

    assert await repo.count() == 1
    recent = await repo.list_recent(limit=1)
    record = recent[0]
    assert record.status == "error"
    assert record.finish_reason == ModelFinishReason.ERROR
    assert record.error_code == "ProviderTimeoutError"
    assert record.provider_output_tokens == 0

    await db.close()


# ===========================================================================
# Attack 18: Real Phase 8 Schema v4 -> v5 Migration
# ===========================================================================

@pytest.mark.asyncio
async def test_schema_v4_to_v5_migration_with_real_data():
    """Create a real Phase 8 schema v4 database with data, migrate to v5, verify data integrity."""
    db_path = Path(tempfile.mkdtemp()) / "mig_v4_v5.db"

    # 1. Manually build a Phase 8 (version 4) database
    db_path.parent.mkdir(parents=True, exist_ok=True)
    import aiosqlite
    async with aiosqlite.connect(str(db_path)) as c:
        await c.executescript(
            SCHEMA_SQL + "\n"
            + "INSERT INTO schema_version(version, description) VALUES (1, 'Initial');\n"
            + MIGRATION_2_SQL + "\n"
            + "INSERT INTO schema_version(version, description) VALUES (2, 'Temporal');\n"
            + MIGRATION_3_SQL + "\n"
            + "INSERT INTO schema_version(version, description) VALUES (3, 'Graph');\n"
            + MIGRATION_4_SQL + "\n"
            + "INSERT INTO schema_version(version, description) VALUES (4, 'Graph dirty');\n"
        )
        mem_id = str(uuid4())
        h = hashlib.sha256(b"ContextOS uses SQLite WAL mode").hexdigest()
        await c.execute(
            "INSERT INTO memories (id, content, content_hash, type, status, confidence, token_count, created_at, updated_at) "
            "VALUES (?, ?, ?, 'fact', 'active', 0.9, 10, '2026-01-01T00:00:00', '2026-01-01T00:00:00')",
            (mem_id, "ContextOS uses SQLite WAL mode", h),
        )
        node1_id = str(uuid4())
        node2_id = str(uuid4())
        await c.execute(
            "INSERT INTO graph_nodes (id, node_type, canonical_key, label, created_at, updated_at) "
            "VALUES (?, 'entity', 'contextos', 'ContextOS', '2026-01-01T00:00:00', '2026-01-01T00:00:00')",
            (node1_id,),
        )
        await c.execute(
            "INSERT INTO graph_nodes (id, node_type, canonical_key, label, created_at, updated_at) "
            "VALUES (?, 'technology', 'sqlite', 'SQLite', '2026-01-01T00:00:00', '2026-01-01T00:00:00')",
            (node2_id,),
        )
        edge_id = str(uuid4())
        await c.execute(
            "INSERT INTO graph_edges (id, source_node_id, target_node_id, relation_type, confidence, created_at, updated_at) "
            "VALUES (?, ?, ?, 'USES', 0.95, '2026-01-01T00:00:00', '2026-01-01T00:00:00')",
            (edge_id, node1_id, node2_id),
        )
        await c.execute(
            "INSERT INTO graph_edge_supports (edge_id, memory_id, confidence, created_at) "
            "VALUES (?, ?, 0.95, '2026-01-01T00:00:00')",
            (edge_id, mem_id),
        )
        await c.commit()

    # 2. Now open with Database.initialize() which must perform v4 -> v5 migration
    db = Database(db_path)
    await db.initialize()

    # 3. Verify schema version is 5
    async with db.connection().execute("SELECT MAX(version) FROM schema_version") as cursor:
        row = await cursor.fetchone()
        assert row[0] == 5

    # 4. Verify all Phase 8 data survived intact
    async with db.connection().execute("SELECT COUNT(*) FROM memories") as cursor:
        assert (await cursor.fetchone())[0] == 1
    async with db.connection().execute("SELECT COUNT(*) FROM graph_nodes") as cursor:
        assert (await cursor.fetchone())[0] == 2
    async with db.connection().execute("SELECT COUNT(*) FROM graph_edges") as cursor:
        assert (await cursor.fetchone())[0] == 1
    async with db.connection().execute("SELECT COUNT(*) FROM graph_edge_supports") as cursor:
        assert (await cursor.fetchone())[0] == 1
    async with db.connection().execute("SELECT COUNT(*) FROM graph_projection_state") as cursor:
        assert (await cursor.fetchone())[0] == 1

    # 5. Verify model_invocations table exists and is writable
    repo = SqliteTelemetryRepository(db.connection())
    assert await repo.count() == 0

    await db.close()


# ===========================================================================
# Attack 19: Telemetry Aggregation Weighting vs Arithmetic Mean
# ===========================================================================

@pytest.mark.asyncio
async def test_telemetry_aggregation_weighted_and_arithmetic_mean():
    """Verify both average_reduction_ratio (mean of ratios) and weighted_reduction_ratio are correct."""
    db_path = Path(tempfile.mkdtemp()) / "test_agg.db"
    db = Database(db_path)
    await db.initialize()
    repo = SqliteTelemetryRepository(db.connection())

    t1 = ModelInvocationTelemetry(
        invocation_id=uuid4(),
        provider_id="prov-a",
        model_id="model-1",
        is_local=True,
        candidate_context_tokens=100,
        compiled_context_tokens=50,
        context_tokens_avoided=50,
        reduction_ratio=0.50,
        provider_input_tokens=60,
        provider_output_tokens=20,
        provider_latency_ms=100.0,
    )
    t2 = ModelInvocationTelemetry(
        invocation_id=uuid4(),
        provider_id="prov-a",
        model_id="model-1",
        is_local=True,
        candidate_context_tokens=1000,
        compiled_context_tokens=100,
        context_tokens_avoided=900,
        reduction_ratio=0.90,
        provider_input_tokens=110,
        provider_output_tokens=50,
        provider_latency_ms=200.0,
    )
    await repo.record(t1)
    await repo.record(t2)

    summary = await repo.summary()

    # Arithmetic mean: (0.50 + 0.90) / 2 = 0.70
    assert abs(summary.average_reduction_ratio - 0.70) < 1e-4

    # Weighted reduction ratio: (50 + 900) / (100 + 1000) = 950 / 1100 = 0.863636...
    expected_weighted = 950.0 / 1100.0
    assert abs(summary.weighted_reduction_ratio - expected_weighted) < 1e-4

    p_info = summary.by_provider["prov-a"]
    assert abs(p_info["average_reduction_ratio"] - 0.70) < 1e-4
    assert abs(p_info["weighted_reduction_ratio"] - expected_weighted) < 1e-4

    await db.close()


# ===========================================================================
# Attack 20: Cost Metadata Absence
# ===========================================================================

def test_model_capabilities_cost_defaults_to_none():
    """Cost fields must default to None rather than hardcoded zero/commercial values."""
    caps = ModelCapabilities(
        provider_id="test",
        model_id="m",
        display_name="M",
        context_window=4096,
    )
    assert caps.cost_per_million_input is None
    assert caps.cost_per_million_output is None


# ===========================================================================
# Attack 21: Graph and Temporal Telemetry Derivation
# ===========================================================================

@pytest.mark.asyncio
async def test_graph_and_temporal_telemetry_trace_derivation():
    """graph_expanded_count and temporal_filtered_count must derive directly from retrieval trace."""
    db_path = Path(tempfile.mkdtemp()) / "test_gt_tel.db"
    db = Database(db_path)
    await db.initialize()
    repo = SqliteTelemetryRepository(db.connection())

    mem1 = Memory(id=uuid4(), content="Context 1", status=MemoryStatus.ACTIVE)
    mem2 = Memory(id=uuid4(), content="Context 2", status=MemoryStatus.ACTIVE)
    scored1 = ScoredMemory(memory=mem1, final_score=0.9, retrieval_sources=["lexical"])
    scored2 = ScoredMemory(memory=mem2, final_score=0.8, retrieval_sources=["graph"])

    class MockRetrieval:
        async def retrieve(self, q, c=None):
            return RetrievalResult(
                query=q,
                memories=[scored1, scored2],
                strategy_results={"lexical": [scored1], "graph": [scored2]},
                trace=RetrievalTrace(
                    stages=[
                        StageTrace(
                            stage_name="eligibility_filter",
                            input_count=10,
                            output_count=6,
                            duration_ms=1.5,
                        )
                    ],
                    total_latency_ms=3.0,
                    total_candidates=2,
                    total_results=2,
                ),
            )

    tik_counter = TiktokenCounter("cl100k_base")
    optimizer = MemoryContextOptimizer(token_counter=tik_counter)
    compiler = QueryAwareContextCompiler(token_counter=tik_counter)

    provider = DeterministicFakeProvider()
    router = DeterministicModelRouter()

    service = ContextOSModelService(
        retrieval_service=MockRetrieval(),
        optimizer=optimizer,
        compilation_service=compiler,
        router=router,
        providers={"fake": provider},
        telemetry_repo=repo,
        token_counter=tik_counter,
    )

    result = await service.ask("Graph query")

    assert result.telemetry.graph_expanded_count == 1
    assert result.telemetry.temporal_filtered_count == 4

    await db.close()


# ===========================================================================
# Attack 22: Quality Placeholder Fields Unset in ask()
# ===========================================================================

@pytest.mark.asyncio
async def test_quality_placeholders_stay_null_in_normal_ask():
    """answer_score, required_fact_coverage, and benchmark_id must be None in normal ask() calls."""
    db_path = Path(tempfile.mkdtemp()) / "test_qual.db"
    db = Database(db_path)
    await db.initialize()
    repo = SqliteTelemetryRepository(db.connection())

    provider = DeterministicFakeProvider()
    router = DeterministicModelRouter()

    class MockRetrieval:
        async def retrieve(self, q, c=None):
            return RetrievalResult(
                query=q, memories=[], strategy_results={},
                trace=RetrievalTrace(stages=[], total_latency_ms=1.0, total_candidates=0, total_results=0),
            )

    tik_counter = TiktokenCounter("cl100k_base")
    optimizer = MemoryContextOptimizer(token_counter=tik_counter)
    compiler = QueryAwareContextCompiler(token_counter=tik_counter)

    service = ContextOSModelService(
        retrieval_service=MockRetrieval(),
        optimizer=optimizer,
        compilation_service=compiler,
        router=router,
        providers={"fake": provider},
        telemetry_repo=repo,
        token_counter=tik_counter,
    )

    result = await service.ask("Normal user query")

    assert result.telemetry.answer_score is None
    assert result.telemetry.required_fact_coverage is None
    assert result.telemetry.benchmark_id is None

    await db.close()


# ===========================================================================
# Attack 7: Provider Error Secret Sanitization
# ===========================================================================

@pytest.mark.asyncio
async def test_provider_auth_error_no_secret_leakage():
    """Provider authentication errors must never echo the raw API key or token."""
    import httpx

    def handler(request: httpx.Request) -> httpx.Response:
        # Server echoes back the Authorization header in 401 error body
        return httpx.Response(
            401,
            json={"error": "Unauthorized: provided key sk-SECRET-12345678 is invalid"},
            headers={"Authorization": "Bearer sk-SECRET-12345678"},
        )

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport, base_url="https://api.example.com/v1")
    prov = OpenAICompatibleProvider(
        base_url="https://api.example.com/v1",
        api_key="sk-SECRET-12345678",
        client=client,
    )

    req = ModelRequest(user_prompt="Hello")
    with pytest.raises(ProviderAuthenticationError) as exc_info:
        await prov.generate(req)

    err_text = str(exc_info.value)
    assert "sk-SECRET-12345678" not in err_text
    assert "Bearer" not in err_text


# ===========================================================================
# Attack 13 & 14: Capability-Aware Routing, Disabled Models & Tie-Breaking
# ===========================================================================

@pytest.mark.asyncio
async def test_capability_aware_disabled_model_and_tie_breaking():
    """CAPABILITY_AWARE routing must skip disabled models, match capabilities, and break ties deterministically."""
    router = DeterministicModelRouter()

    # Disabled model with large window
    m_disabled = ModelCapabilities(
        provider_id="prov-1",
        model_id="disabled-giant",
        display_name="Giant (Disabled)",
        context_window=32768,
        supports_tools=True,
        enabled=False,
    )
    # Enabled model without tools
    m_no_tools = ModelCapabilities(
        provider_id="prov-1",
        model_id="small-no-tools",
        display_name="Small No Tools",
        context_window=4096,
        supports_tools=False,
        enabled=True,
    )
    # Enabled model with tools (context 8192)
    m_with_tools_1 = ModelCapabilities(
        provider_id="prov-1",
        model_id="medium-tools-local",
        display_name="Medium Tools Local",
        context_window=8192,
        supports_tools=True,
        local=True,
        enabled=True,
    )
    # Enabled model with tools (context 16384, remote)
    m_with_tools_2 = ModelCapabilities(
        provider_id="prov-2",
        model_id="large-tools-remote",
        display_name="Large Tools Remote",
        context_window=16384,
        supports_tools=True,
        local=False,
        enabled=True,
    )

    p1 = DeterministicFakeProvider(provider_id="prov-1", models=[m_disabled, m_no_tools, m_with_tools_1], is_local=True)
    p2 = DeterministicFakeProvider(provider_id="prov-2", models=[m_with_tools_2], is_local=False)
    providers = {"prov-1": p1, "prov-2": p2}

    req = ModelRequest(user_prompt="Run tool call", required_capabilities=["tools"])
    decision = await router.route(req, providers, policy=RoutingPolicy.CAPABILITY_AWARE)

    # Local is preferred over remote: medium-tools-local must be selected over large-tools-remote
    assert decision.selected_provider == "prov-1"
    assert decision.selected_model == "medium-tools-local"
    assert not decision.fallback_used


# ===========================================================================
# Attack 15: Ollama Missing Usage Truthful Source Label
# ===========================================================================

@pytest.mark.asyncio
async def test_ollama_missing_usage_source_label():
    """When Ollama omits prompt_eval_count, measurement source must match model tokenizer truth."""
    import httpx

    def handler(request: httpx.Request) -> httpx.Response:
        # Return response without prompt_eval_count (absent usage)
        return httpx.Response(
            200,
            json={"message": {"role": "assistant", "content": "Hello world from Ollama"}, "done": True},
        )

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1:11434")

    # Qwen model: counter is APPROXIMATED
    prov_qwen = OllamaProvider(default_model="qwen2.5:7b", client=client)
    resp_qwen = await prov_qwen.generate(ModelRequest(user_prompt="Hi", model="qwen2.5:7b"))
    assert resp_qwen.token_measurement_source == TokenMeasurementSource.APPROXIMATED

    # Llama/cl100k model: counter is TOKENIZER_COUNTED
    prov_llama = OllamaProvider(default_model="llama3.2:3b", client=client)
    resp_llama = await prov_llama.generate(ModelRequest(user_prompt="Hi", model="llama3.2:3b"))
    assert resp_llama.token_measurement_source == TokenMeasurementSource.TOKENIZER_COUNTED

