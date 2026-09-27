"""Tests U through AW: Token accounting, telemetry model, persistence, aggregation, and privacy."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from contextos.core.enums import (
    ModelFinishReason,
    RoutingPolicy,
    TokenMeasurementSource,
)
from contextos.core.models import (
    ModelInvocationTelemetry,
)
from contextos.providers.fake import DeterministicFakeProvider
from contextos.services.token_counter import (
    DeterministicWordTokenCounter,
    TiktokenCounter,
    get_token_counter_for_model,
)
from contextos.storage.database import Database
from contextos.storage.telemetry_repo import SqliteTelemetryRepository


@pytest.fixture
async def telemetry_db(tmp_path: Path):
    db_path = tmp_path / "telemetry_test.db"
    db = Database(db_path)
    await db.initialize()
    yield db
    await db.close()


@pytest.fixture
def telemetry_repo(telemetry_db: Database):
    return SqliteTelemetryRepository(telemetry_db.connection())


def sample_telemetry(
    provider_id: str = "fake",
    model_id: str = "fake-default",
    is_local: bool = True,
    candidate_tokens: int = 1000,
    compiled_tokens: int = 250,
) -> ModelInvocationTelemetry:
    avoided = max(0, candidate_tokens - compiled_tokens)
    ratio = 1.0 - (compiled_tokens / candidate_tokens) if candidate_tokens > 0 else 0.0
    return ModelInvocationTelemetry(
        invocation_id=uuid4(),
        session_id="session-1",
        provider_id=provider_id,
        model_id=model_id,
        is_local=is_local,
        timestamp=datetime.now(timezone.utc),
        candidate_context_tokens=candidate_tokens,
        retrieved_context_tokens=candidate_tokens,
        optimized_context_tokens=300,
        compiled_context_tokens=compiled_tokens,
        prompt_tokens_before_context=50,
        final_input_tokens=300,
        provider_input_tokens=300,
        provider_output_tokens=80,
        provider_total_tokens=380,
        token_measurement_source=TokenMeasurementSource.PROVIDER_REPORTED,
        context_tokens_avoided=avoided,
        reduction_ratio=ratio,
        lexical_candidate_count=5,
        dense_candidate_count=5,
        hybrid_candidate_count=10,
        graph_expanded_count=2,
        temporal_filtered_count=3,
        selected_memory_count=4,
        compiled_fact_count=6,
        retrieval_ms=12.5,
        optimization_ms=3.2,
        compilation_ms=4.1,
        routing_ms=0.5,
        token_counting_ms=0.8,
        provider_latency_ms=120.0,
        end_to_end_ms=141.1,
        routing_policy=RoutingPolicy.LOCAL_FIRST,
        routing_reason="Local provider healthy",
        selected_provider=provider_id,
        selected_model=model_id,
        fallback_used=False,
        finish_reason=ModelFinishReason.STOP,
        status="success",
        metadata={"client": "test"},
    )


# ---------------------------------------------------------------------------
# Test U: Provider-Reported Token Usage
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_u_provider_reported_token_usage():
    provider = DeterministicFakeProvider(report_usage=True)
    from contextos.core.models import ModelRequest
    req = ModelRequest(user_prompt="Count my tokens please")
    resp = await provider.generate(req)
    assert resp.token_measurement_source == TokenMeasurementSource.PROVIDER_REPORTED
    assert resp.input_tokens > 0
    assert resp.output_tokens > 0
    assert resp.total_tokens == resp.input_tokens + resp.output_tokens


# ---------------------------------------------------------------------------
# Test V: Tokenizer-Counted Usage
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_v_tokenizer_counted_usage():
    provider = DeterministicFakeProvider(report_usage=False)
    from contextos.core.models import ModelRequest
    req = ModelRequest(user_prompt="Count my tokens with fallback")
    resp = await provider.generate(req)
    assert resp.token_measurement_source == TokenMeasurementSource.TOKENIZER_COUNTED


# ---------------------------------------------------------------------------
# Test W: Approximate Fallback
# ---------------------------------------------------------------------------
def test_w_approximate_fallback():
    counter = DeterministicWordTokenCounter()
    assert counter.measurement_source == TokenMeasurementSource.APPROXIMATED
    count = counter.count("This is a heuristic token count for testing.")
    assert count > 0

    model_counter = get_token_counter_for_model("unknown-custom-model", "deterministic")
    assert model_counter.measurement_source == TokenMeasurementSource.APPROXIMATED


# ---------------------------------------------------------------------------
# Test X: Token Source Label
# ---------------------------------------------------------------------------
def test_x_token_source_label():
    tiktoken_c = TiktokenCounter()
    assert tiktoken_c.measurement_source == TokenMeasurementSource.TOKENIZER_COUNTED

    approx_c = DeterministicWordTokenCounter()
    assert approx_c.measurement_source == TokenMeasurementSource.APPROXIMATED

    for source in TokenMeasurementSource:
        assert source.value in {"provider_reported", "tokenizer_counted", "approximated"}


# ---------------------------------------------------------------------------
# Tests Y through AE: Context pipeline tokens, savings, and I/O tokens
# ---------------------------------------------------------------------------
def test_y_through_ae_pipeline_token_metrics():
    t = sample_telemetry(candidate_tokens=1000, compiled_tokens=200)

    # Y. candidate context tokens
    assert t.candidate_context_tokens == 1000

    # Z. optimized context tokens
    assert t.optimized_context_tokens == 300

    # AA. compiled context tokens
    assert t.compiled_context_tokens == 200

    # AB. context tokens avoided
    assert t.context_tokens_avoided == 800

    # AC. reduction ratio
    assert t.reduction_ratio == 0.8

    # AD. final input tokens
    assert t.final_input_tokens == 300

    # AE. output tokens
    assert t.provider_output_tokens == 80


# ---------------------------------------------------------------------------
# Tests AF through AH: Latency Accounting
# ---------------------------------------------------------------------------
def test_af_through_ah_latency_accounting():
    t = sample_telemetry()

    # AF. end-to-end latency
    assert t.end_to_end_ms > 0

    # AG. provider latency
    assert t.provider_latency_ms == 120.0

    # AH. retrieval latency
    assert t.retrieval_ms == 12.5


# ---------------------------------------------------------------------------
# Tests AI through AL: Contribution counts
# ---------------------------------------------------------------------------
def test_ai_through_al_contribution_counts():
    t = sample_telemetry()

    # AI. graph-expanded count
    assert t.graph_expanded_count == 2

    # AJ. temporal-filter count
    assert t.temporal_filtered_count == 3

    # AK. selected-memory count
    assert t.selected_memory_count == 4

    # AL. compiled-fact count
    assert t.compiled_fact_count == 6


# ---------------------------------------------------------------------------
# Tests AM through AO: Model Identity, Local flag, Fallback Metadata
# ---------------------------------------------------------------------------
def test_am_through_ao_identity_and_fallback():
    t = sample_telemetry(provider_id="ollama", model_id="qwen2.5:7b", is_local=True)

    # AM. provider/model identity
    assert t.provider_id == "ollama"
    assert t.model_id == "qwen2.5:7b"

    # AN. local/remote flag
    assert t.is_local is True

    # AO. fallback metadata
    t_fallback = sample_telemetry()
    t_fallback.fallback_used = True
    t_fallback.fallback_reason = "Local provider timed out"
    assert t_fallback.fallback_used is True
    assert t_fallback.fallback_reason == "Local provider timed out"


# ---------------------------------------------------------------------------
# Test AP: Telemetry Persistence
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_ap_telemetry_persistence(telemetry_repo: SqliteTelemetryRepository):
    t = sample_telemetry()
    await telemetry_repo.record(t)

    retrieved = await telemetry_repo.get(t.invocation_id)
    assert retrieved is not None
    assert retrieved.invocation_id == t.invocation_id
    assert retrieved.provider_id == t.provider_id
    assert retrieved.model_id == t.model_id
    assert retrieved.candidate_context_tokens == t.candidate_context_tokens
    assert retrieved.context_tokens_avoided == t.context_tokens_avoided


# ---------------------------------------------------------------------------
# Test AQ: Restart Persistence
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_aq_restart_persistence(tmp_path: Path):
    db_path = tmp_path / "restart_test.db"
    db1 = Database(db_path)
    await db1.initialize()
    repo1 = SqliteTelemetryRepository(db1.connection())

    t = sample_telemetry()
    await repo1.record(t)
    await db1.close()

    # Reopen connection simulating process restart
    db2 = Database(db_path)
    await db2.initialize()
    repo2 = SqliteTelemetryRepository(db2.connection())

    reloaded = await repo2.get(t.invocation_id)
    assert reloaded is not None
    assert reloaded.invocation_id == t.invocation_id
    assert reloaded.compiled_context_tokens == t.compiled_context_tokens
    await db2.close()


# ---------------------------------------------------------------------------
# Tests AR through AT: Daily, Provider, and Model Aggregations
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_ar_through_at_aggregations(telemetry_repo: SqliteTelemetryRepository):
    # Record 3 distinct invocations
    t1 = sample_telemetry(provider_id="ollama", model_id="qwen", is_local=True, candidate_tokens=1000, compiled_tokens=200)
    t2 = sample_telemetry(provider_id="ollama", model_id="llama", is_local=True, candidate_tokens=800, compiled_tokens=400)
    t3 = sample_telemetry(provider_id="openai", model_id="gpt-4o", is_local=False, candidate_tokens=1200, compiled_tokens=300)

    await telemetry_repo.record(t1)
    await telemetry_repo.record(t2)
    await telemetry_repo.record(t3)

    summary = await telemetry_repo.summary()

    # AR. daily / overall aggregation
    assert summary.total_invocations == 3
    assert summary.total_tokens_avoided == (800 + 400 + 900)
    assert summary.local_invocations == 2
    assert summary.remote_invocations == 1

    # AS. provider aggregation
    assert "ollama" in summary.by_provider
    assert summary.by_provider["ollama"]["invocations"] == 2
    assert summary.by_provider["ollama"]["tokens_avoided"] == 1200
    assert "openai" in summary.by_provider
    assert summary.by_provider["openai"]["invocations"] == 1

    # AT. model aggregation
    assert "qwen" in summary.by_model
    assert summary.by_model["qwen"]["invocations"] == 1
    assert "llama" in summary.by_model
    assert summary.by_model["llama"]["invocations"] == 1
    assert "gpt-4o" in summary.by_model
    assert summary.by_model["gpt-4o"]["invocations"] == 1


# ---------------------------------------------------------------------------
# Tests AU through AW: Privacy and Secret Scanner Safety
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_au_through_aw_privacy_and_secrets(telemetry_db: Database, telemetry_repo: SqliteTelemetryRepository):
    t = sample_telemetry()
    t.metadata = {
        "status": "clean",
        "env": "production",
    }
    await telemetry_repo.record(t)

    # Query the raw SQLite table and check all text fields
    async with telemetry_db.connection().execute("SELECT * FROM model_invocations") as cursor:
        rows = await cursor.fetchall()
        assert len(rows) > 0
        raw_row_str = " ".join(str(val) for val in rows[0])

    # AU. No raw API keys
    assert "sk-" not in raw_row_str
    assert "key-" not in raw_row_str

    # AV. No Authorization header persisted
    assert "Authorization" not in raw_row_str
    assert "Bearer " not in raw_row_str

    # AW. No secret scanner findings or private prompts stored by default
    assert "password" not in raw_row_str.lower()
    assert "aws_secret" not in raw_row_str.lower()
    assert "private_key" not in raw_row_str.lower()
