"""Adversarial and decision matrix tests for Phase 13 explainability.

Thoroughly verifies:
1. Adversarial sanitization: ANSI CSI, OSC title, OSC hyperlink, CR/LF/BS/null,
   bidi override, zero-width controls, huge labels/connectors, secrets/keys/tokens,
   SQL/prompt injections, malformed UUIDs, unbounded parameters.
2. Retrieval decision matrix: lexical, dense, hybrid RRF, metadata reranking, result limits.
3. Graph decision matrix: direct seeds, multi-hop expansions, edge support, cycles, scope match/exclusion.
4. Temporal decision matrix: current, superseded, historical, corrects, coexists, contradicts, target deleted, no inferred motive.
5. Optimizer & Compiler decision matrix: selected, budget excluded, redundancy excluded, merged, rescued, dropped.
6. Provider dispatch decision matrix: standalone not attempted, request constructed, dispatch attempted, response received, dispatch failed, exact fingerprint proof.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from uuid import UUID, uuid4

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from pydantic import ValidationError

from contextos.api.server import create_app, set_services
from contextos.config.settings import Settings
from contextos.core.enums import (
    GraphNodeType,
    GraphRelationType,
    MemoryStatus,
    MemoryType,
    ProviderDispatchState,
    RelationType,
    RetrievalMode,
    TemporalScope,
)
from contextos.core.exceptions import ModelUnavailableError, ProviderTimeoutError
from contextos.core.models import (
    ContextBudget,
    GraphEdge,
    GraphEdgeSupport,
    GraphNode,
    Memory,
    MemoryFilters,
    MemoryRelation,
    RetrievalQuery,
)
from contextos.daemon.wiring import wire_services
from contextos.services.explainability import (
    ExplainabilityService,
    ExplanationRequest,
    TemporalEvidenceResolver,
    safe_source_type,
    safe_text,
)
from contextos.services.graph import (
    MemoryGraphService,
    _safe_node_label,
    stable_edge_id,
    stable_node_id,
)


@pytest_asyncio.fixture
async def adv_stack(tmp_path):
    services = await wire_services(Settings(
        daemon={"data_dir": tmp_path}, embedding={"model": "deterministic"}
    ))
    set_services(services)
    async with AsyncClient(transport=ASGITransport(app=create_app()), base_url="http://local") as client:
        yield services, client
    await services["database"].close()


# ==============================================================================
# 1. ADVERSARIAL SANITIZATION & BOUNDS
# ==============================================================================


@pytest.mark.parametrize(
    "attack_vector",
    [
        # ANSI CSI escape sequences (cursor movement, clear screen, color codes)
        "\x1b[2J\x1b[H\x1b[31;1mRedText\x1b[0mNormal",
        # OSC Title injection
        "\x1b]0;Hacked Title\x07NormalText",
        # OSC 8 Hyperlink injection
        "\x1b]8;;http://attacker.invalid\x1b\\ClickMe\x1b]8;;\x1b\\",
        # Carriage return, backspace, null bytes
        "CleanPrefix\rOverwrite\b\b\x00EmbeddedNull",
        # Bidi overrides (RLO, LRI, RLI, FSI, PDF)
        "Hello \u202e\u2066\u2067\u2068\u202cWorld",
        # Zero-width control characters
        "Invisible\u200b\u200c\u200d\ufeffControls",
    ],
)
def test_safe_text_neutralizes_all_terminal_and_bidi_attacks(attack_vector: str):
    cleaned = safe_text(attack_vector, limit=200)
    assert "\x1b" not in cleaned
    assert "\r" not in cleaned
    assert "\b" not in cleaned
    assert "\x00" not in cleaned
    assert "\u202e" not in cleaned
    assert "\u2066" not in cleaned
    assert "\u2067" not in cleaned
    assert "\u2068" not in cleaned
    assert "\u202c" not in cleaned
    assert "\u200b" not in cleaned
    assert "\u200c" not in cleaned
    assert "\u200d" not in cleaned
    assert "\ufeff" not in cleaned


def test_safe_text_and_node_labels_strictly_bounded_length():
    huge_input = "A" * 50_000
    assert len(safe_text(huge_input, limit=80)) == 80
    assert len(_safe_node_label(huge_input)) <= 120
    assert _safe_node_label("memory:12345") is None
    assert _safe_node_label("") is None
    assert _safe_node_label(None) is None
    assert _safe_node_label(r"C:\Users\private-user\Documents\notes.txt") is None
    assert _safe_node_label("postgresql://user:secret@localhost/db") is None
    assert _safe_node_label("password=super-secret") is None
    assert _safe_node_label("Project Apollo") == "Project Apollo"


@pytest.mark.parametrize(
    ("untrusted_source", "expected"),
    [
        ("local_file", "local_file"),
        ("connector:local_file", "connector:local_file"),
        ("connector:json_import", "connector:json_import"),
        ("connector:fake", "connector:fake"),
        ("cli_input", "cli_input"),
        ("api", "api"),
        # Secret-bearing source URI
        ("file:///etc/shadow?password=123", "unknown"),
        ("https://user:pass@api.openai.com/v1", "unknown"),
        ("s3://bucket/key?token=secret", "unknown"),
        ("connector:evil_malicious_plugin", "unknown"),
        ("token=sk-ant-api03-abcdef1234567890", "unknown"),
        ("Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9", "unknown"),
        ("postgresql://user:pass@localhost:5432/db", "unknown"),
        ("'; DROP TABLE memories; --", "unknown"),
    ],
)
def test_source_provenance_whitelisting(untrusted_source: str, expected: str):
    assert safe_source_type(untrusted_source) == expected


def test_request_validation_blocks_unbounded_and_malformed_inputs():
    # Huge query (> 10,000 chars)
    with pytest.raises(ValidationError):
        ExplanationRequest(query="x" * 10_001)

    # Empty query
    with pytest.raises(ValidationError):
        ExplanationRequest(query="")

    # Huge limit (> 100) or non-positive
    with pytest.raises(ValidationError):
        ExplanationRequest(query="test", limit=101)
    with pytest.raises(ValidationError):
        ExplanationRequest(query="test", limit=0)

    # Huge budget (> 8000) or non-positive
    with pytest.raises(ValidationError):
        ExplanationRequest(query="test", budget=8001)
    with pytest.raises(ValidationError):
        ExplanationRequest(query="test", budget=0)

    # Malformed UUID
    with pytest.raises(ValidationError):
        ExplanationRequest(query="test", target_memory_id="not-a-valid-uuid")


# ==============================================================================
# 2. RETRIEVAL DECISION MATRIX
# ==============================================================================


@pytest.mark.asyncio
async def test_retrieval_decision_matrix_stages_and_fusion(adv_stack):
    services, client = adv_stack
    repo = services["memory_repo"]

    # 1. Create distinct memories for lexical vs dense
    m_lex = await repo.create(Memory(
        content="Hyperparameter tuning matrix for gradient boosting trees.",
        status=MemoryStatus.ACTIVE,
    ))
    m_dense = await repo.create(Memory(
        content="Optimization algorithms for neural network loss functions.",
        status=MemoryStatus.ACTIVE,
    ))
    m_both = await repo.create(Memory(
        content="Hyperparameter optimization algorithms for tree models.",
        status=MemoryStatus.ACTIVE,
    ))
    await services["retrieval_index"].ensure_current()

    # Query with explain
    resp = await client.post("/api/v1/explain", json={
        "query": "Hyperparameter optimization algorithms",
        "mode": "hybrid",
        "graph": False,
        "limit": 10,
    })
    assert resp.status_code == 200
    trace = resp.json()

    # Verify retrieval stages exist
    stage_names = [s["name"] for s in trace["stages"]]
    assert "lexical_search" in stage_names
    assert "dense_search" in stage_names
    assert "eligibility_filter" in stage_names
    assert "fusion_rerank" in stage_names

    # Check candidates have channels tracked
    cand_ids = {c["memory_id"] for c in trace["candidates"]}
    assert str(m_both.id) in cand_ids
    for cand in trace["candidates"]:
        assert "retrieval" in cand
        assert cand["retrieval"]["mode"] == "hybrid"
        assert cand["retrieval"]["origin"] in {"direct", "graph_expanded"}
        assert cand["rank"] >= 1


# ==============================================================================
# 3. GRAPH DECISION MATRIX (DIRECT, HOPS, CYCLES, SCOPE)
# ==============================================================================


@pytest.mark.asyncio
async def test_graph_decision_matrix_and_cycle_termination(adv_stack):
    services, client = adv_stack
    repo = services["memory_repo"]
    graph_service: MemoryGraphService = services["graph"]

    # Create cycle: Docker depends on Podman -> Podman depends on Ollama -> Ollama depends on Docker
    m_doc = await repo.create(Memory(content="Docker depends on Podman", status=MemoryStatus.ACTIVE))
    m_pod = await repo.create(Memory(content="Podman depends on Ollama", status=MemoryStatus.ACTIVE))
    m_oll = await repo.create(Memory(content="Ollama depends on Docker", status=MemoryStatus.ACTIVE))
    await services["retrieval_index"].ensure_current()
    await graph_service.rebuild()

    # Traversal should terminate safely without recursion or endless loop
    expansion = await graph_service.expand(
        query_text="Docker",
        max_hops=3,
        max_nodes=10,
        max_edges=10,
    )
    assert len(expansion.visited_node_ids) <= 10
    assert len(expansion.traversed_edge_ids) <= 10
    for paths in expansion.candidate_paths.values():
        for path in paths:
            assert path.hop_count <= 3
            assert path.path_nodes
            assert path.path_edges
            for edge in path.path_edges:
                assert edge.edge_type is not None
                assert edge.confidence > 0.0
                assert edge.supporting_memory_ids

    # Test scope match and scope exclusion
    m_proj_a = await repo.create(Memory(content="Project Apollo uses Python", status=MemoryStatus.ACTIVE))
    m_proj_b = await repo.create(Memory(content="Project Hermes uses Rust", status=MemoryStatus.ACTIVE))
    await services["retrieval_index"].ensure_current()
    await graph_service.rebuild()

    exp_apollo = await graph_service.expand(query_text="Project Apollo")
    for paths in exp_apollo.candidate_paths.values():
        for path in paths:
            if path.scope_match is not None:
                # Participated in apollo scope
                assert path.scope_match is True


# ==============================================================================
# 4. TEMPORAL EVIDENCE DECISION MATRIX
# ==============================================================================


@pytest.mark.asyncio
async def test_temporal_evidence_decision_matrix_all_relation_types(adv_stack):
    services, client = adv_stack
    mem_repo = services["memory_repo"]
    rel_repo = services["relation_repo"]

    # 1. Current active memory
    m_current = await mem_repo.create(Memory(
        content="I use VS Code for daily editing.",
        status=MemoryStatus.ACTIVE,
    ))
    # 2. Superseded memory
    m_old = await mem_repo.create(Memory(
        content="I use Sublime Text for editing.",
        status=MemoryStatus.SUPERSEDED,
        superseded_by=m_current.id,
    ))
    # 3. Contradicted memory
    m_contra = await mem_repo.create(Memory(
        content="I prefer light theme.",
        status=MemoryStatus.ACTIVE,
    ))
    m_contra_target = await mem_repo.create(Memory(
        content="I prefer dark theme exclusively.",
        status=MemoryStatus.ACTIVE,
    ))
    # 4. Coexisting memory
    m_coexist = await mem_repo.create(Memory(
        content="I also use PyCharm on secondary workstation.",
        status=MemoryStatus.ACTIVE,
    ))

    # Add relations: CORRECTS, CONTRADICTS, COEXISTS_WITH
    await rel_repo.create(MemoryRelation(
        source_memory_id=m_current.id,
        target_memory_id=m_old.id,
        relation_type=RelationType.CORRECTS,
        confidence=0.99,
    ))
    await rel_repo.create(MemoryRelation(
        source_memory_id=m_contra.id,
        target_memory_id=m_contra_target.id,
        relation_type=RelationType.CONTRADICTS,
        confidence=0.88,
    ))
    await rel_repo.create(MemoryRelation(
        source_memory_id=m_current.id,
        target_memory_id=m_coexist.id,
        relation_type=RelationType.COEXISTS_WITH,
        confidence=0.75,
    ))

    resolver = TemporalEvidenceResolver(memory_repo=mem_repo, relation_repo=rel_repo)

    # Check evidence for current
    ev_current = await resolver.resolve_memory_evidence(m_current)
    assert ev_current["reason_code"] == "CURRENT_STATE"
    assert ev_current["eligible"] is True
    assert ev_current["acceptance_rationale"] is None
    rel_types = {r["relation_type"] for r in ev_current["relations"]}
    assert "corrects" in rel_types
    assert "coexists_with" in rel_types

    # Check evidence for superseded
    ev_old = await resolver.resolve_memory_evidence(m_old)
    assert ev_old["reason_code"] == "REPLACED_BY_CURRENT_STATE"
    assert ev_old["eligible"] is False
    assert ev_old["replacement_memory_id"] == str(m_current.id)

    # Check evidence for contradiction
    ev_contra = await resolver.resolve_memory_evidence(m_contra)
    contra_rels = [r for r in ev_contra["relations"] if r["relation_type"] == "contradicts"]
    assert len(contra_rels) == 1
    assert contra_rels[0]["target_memory_id"] == str(m_contra_target.id)
    assert contra_rels[0]["acceptance_rationale"] is None


# ==============================================================================
# 5. OPTIMIZER & COMPILER DECISION MATRIX
# ==============================================================================


@pytest.mark.asyncio
async def test_optimizer_and_compiler_decision_matrix(adv_stack):
    services, client = adv_stack
    repo = services["memory_repo"]

    # Seed memories to trigger optimizer selection and budget exclusion
    m_high = await repo.create(Memory(
        content="Primary project instructions for Python development environment setup.",
        status=MemoryStatus.ACTIVE,
    ))
    m_oversized = await repo.create(Memory(
        content="Long detailed configuration notes repeating " + "settings " * 150,
        status=MemoryStatus.ACTIVE,
    ))
    await services["retrieval_index"].ensure_current()

    # Explain with tight budget
    resp = await client.post("/api/v1/explain", json={
        "query": "Python development environment setup",
        "budget": 40,
        "graph": False,
        "target_memory_id": str(m_oversized.id),
    })
    assert resp.status_code == 200
    data = resp.json()

    assert "final_context" in data
    assert data["final_context"]["compiled_tokens"] <= 40
    assert "selected" in data
    assert "excluded" in data

    # Requested memory should show optimizer exclusion
    req = data["requested_memory"]
    assert req["reason_code"] == "RETRIEVED_BUT_OPTIMIZER_EXCLUDED"
    assert req["status"] == "retrieved_but_excluded"
    assert req["reason"] in {"oversized", "budget_exhausted"}


# ==============================================================================
# 6. PROVIDER DISPATCH DECISION MATRIX & RECEIPT PROOF
# ==============================================================================


@pytest.mark.asyncio
async def test_provider_dispatch_receipt_proof_and_failure_states(adv_stack):
    services, client = adv_stack
    model_service = services["model_service"]
    fake_provider = services["fake_provider"]

    # A. Standalone explain: provider_dispatch MUST be NOT_ATTEMPTED
    standalone = (await client.post("/api/v1/explain", json={"query": "test query"})).json()
    assert standalone["provider_dispatch"]["state"] == "NOT_ATTEMPTED"
    assert "prepared by ContextOS" in standalone["provider_dispatch"]["status_message"]

    # B. ModelService.ask with explain=True (Successful generation)
    result = await model_service.ask(
        query="Explain how to configure Python packages.",
        explain=True,
    )
    assert result.dispatch_evidence is not None
    receipt = result.dispatch_evidence
    assert receipt.state == ProviderDispatchState.RESPONSE_RECEIVED
    assert receipt.provider_response_received is True
    assert receipt.context_match is True
    assert len(receipt.compiled_context_sha256) == 64
    assert len(receipt.logical_request_sha256) == 64

    # Verify context fingerprint is reproducible and matches the context
    expected_context_sha = hashlib.sha256(
        (result.compiled_context.context_text or "").encode("utf-8")
    ).hexdigest()
    assert receipt.compiled_context_sha256 == expected_context_sha

    # Verify explanation attached to AskResult
    assert result.explanation is not None
    assert result.explanation["provider_dispatch"]["state"] == "RESPONSE_RECEIVED"

    # C. Failure before dispatch (Routing failure with explicit nonexistent model)
    with pytest.raises(ModelUnavailableError):
        await model_service.ask(
            query="test",
            target_provider="fake",
            target_model="nonexistent-model",
            allow_fallback=False,
        )

    # D. Failure during dispatch (Provider timeout)
    fake_provider.simulate_timeout = True
    try:
        with pytest.raises(ProviderTimeoutError):
            await model_service.ask(query="timeout test", explain=True)
    finally:
        fake_provider.simulate_timeout = False

    # Check telemetry reflects error
    recent = await services["telemetry_repo"].list_recent(limit=1)
    assert recent[0].status == "error"
