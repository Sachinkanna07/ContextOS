"""Real SQLite integration checks for the Phase 13 explanation path."""

from __future__ import annotations

import json
from uuid import uuid4

import pytest
from httpx import ASGITransport, AsyncClient

from contextos.api.server import create_app, set_services
from contextos.config.settings import Settings
from contextos.daemon.wiring import wire_services
from contextos.core.enums import MemoryStatus
from contextos.core.models import Memory
from contextos.mcp.server import ContextOSMCPApplication, MCPPermissions
from contextos.services.explainability import ExplainabilityService, ExplanationRequest, safe_text
from typer.testing import CliRunner
from contextos.cli.app import app


@pytest.fixture
async def explain_stack(tmp_path):
    services = await wire_services(Settings(
        daemon={"data_dir": tmp_path}, embedding={"model": "deterministic"}
    ))
    set_services(services)
    async with AsyncClient(transport=ASGITransport(app=create_app()), base_url="http://local") as client:
        yield services, client
    await services["database"].close()


@pytest.mark.asyncio
async def test_explanation_is_ephemeral_bounded_and_content_free_by_default(explain_stack):
    services, client = explain_stack
    accepted = await client.post("/api/v1/remember", json={
        "text": "I prefer concise technical documentation for Python projects."
    })
    assert accepted.status_code == 200
    response = await client.post("/api/v1/explain", json={
        "query": "Python documentation", "budget": 500, "limit": 10,
    })
    assert response.status_code == 200
    data = response.json()
    assert len(data["trace_id"]) == 36
    assert data["trace_id"] == data["query_id"]
    assert data["candidates"]
    assert any(stage["name"] == "optimizer" for stage in data["stages"])
    assert any(stage["name"] == "fact_ir" for stage in data["stages"])
    assert "content" not in data["candidates"][0]
    assert data["final_context"]["compiled_tokens"] <= 500
    assert data["execution_ms"] >= data["explanation_overhead_ms"] >= 0
    assert data["final_context"]["token_measurement_source"] in {"tokenizer_counted", "approximated"}
    target_id = data["candidates"][0]["memory_id"]
    under_budget = (await client.post("/api/v1/explain", json={
        "query": "Python documentation", "budget": 1, "graph": False,
        "target_memory_id": target_id,
    })).json()["requested_memory"]
    assert under_budget["status"] == "retrieved_but_excluded"
    assert under_budget["reason"] in {"oversized", "budget_exhausted"}
    requested = (await client.post("/api/v1/explain", json={
        "query": "Python documentation", "target_memory_id": str(uuid4()),
    })).json()["requested_memory"]
    assert requested["status"] == "not_available"
    assert requested["reason"] == "not_retrieved_or_not_observed"
    serialized = json.dumps(data)
    assert "technical documentation for Python" not in serialized
    assert not any(key in serialized.lower() for key in ("source_uri", "raw_prompt", "authorization"))
    async with services["database"].connection().execute("SELECT MAX(version) FROM schema_version") as cursor:
        assert (await cursor.fetchone())[0] == 7


@pytest.mark.asyncio
async def test_explanation_decisions_are_deterministic_and_opt_in_content_is_bounded(explain_stack):
    _, client = explain_stack
    await client.post("/api/v1/remember", json={"text": "I use Python for local automation."})
    payload = {"query": "Python automation", "budget": 400, "mode": "hybrid", "graph": False}
    first = (await client.post("/api/v1/explain", json=payload)).json()
    second = (await client.post("/api/v1/explain", json=payload)).json()
    assert first["trace_id"] != second["trace_id"]
    for data in (first, second):
        for row in data["candidates"]:
            row.pop("content", None)
        data.pop("trace_id", None)
        data.pop("query_id", None)
    assert first["candidates"] == second["candidates"]
    explicit = (await client.post("/api/v1/explain", json={**payload, "include_content": True})).json()
    assert "Python" in explicit["candidates"][0]["content"]
    assert len(explicit["candidates"][0]["content"]) <= 1000


@pytest.mark.asyncio
async def test_limits_invalid_mode_and_oversized_query(explain_stack):
    _, client = explain_stack
    assert (await client.post("/api/v1/explain", json={"query": "x", "limit": 101})).status_code == 422
    assert (await client.post("/api/v1/explain", json={"query": "x", "mode": "unknown"})).status_code == 422
    assert (await client.post("/api/v1/explain", json={"query": "x" * 10001})).status_code == 422


@pytest.mark.asyncio
async def test_targeted_temporal_exclusion_uses_retrieval_eligibility(explain_stack):
    services, client = explain_stack
    prior = await services["memory_repo"].create(Memory(
        content="Previously I used Python for projects.", status=MemoryStatus.SUPERSEDED,
    ))
    response = await client.post("/api/v1/explain", json={
        "query": "Python projects", "target_memory_id": str(prior.id),
        "temporal_scope": "current", "graph": False,
    })
    requested = response.json()["requested_memory"]
    assert requested["status"] == "not_retrieved"
    assert requested["reason"] == "temporal_policy_ineligible"


@pytest.mark.asyncio
async def test_provenance_does_not_echo_source_uri_or_untrusted_source_type(explain_stack):
    services, client = explain_stack
    secret = "Bearer sk-secret-source-type-value"
    private_path = r"C:\Users\private-user\Documents\notes.txt"
    memory = await services["memory_repo"].create(Memory(
        content="Python documentation privacy preference.", status=MemoryStatus.ACTIVE,
        source_type=secret, source_uri=private_path,
    ))
    await services["retrieval_index"].ensure_current()
    data = (await client.post("/api/v1/explain", json={
        "query": "Python documentation privacy", "graph": False,
    })).json()
    row = next(item for item in data["candidates"] if item["memory_id"] == str(memory.id))
    serialized = json.dumps(row)
    assert row["provenance"]["source_type"] == "unknown"
    assert secret not in serialized and private_path not in serialized


@pytest.mark.asyncio
async def test_mcp_uses_shared_service_and_checks_read_permission(explain_stack):
    services, _ = explain_stack
    allowed = ContextOSMCPApplication(services)
    denied = ContextOSMCPApplication(services, MCPPermissions(allow_read=False))
    result = await allowed.explain("project memory", 300, "hybrid")
    assert result["ok"] is True
    assert "trace_id" in result and "candidates" in result
    assert "decisions" in result and "result_count" in result and "selected" in result
    assert (await denied.explain("project memory", 300, "hybrid")) == {
        "ok": False, "error_code": "PERMISSION_DENIED"
    }


def test_terminal_sanitizer_removes_escape_bidi_and_controls():
    attacked = "safe\x1b]8;;https://attacker.invalid\x1b\\link\x1b]8;;\x1b\\\u202e\x00"
    clean = safe_text(attacked)
    assert "\x1b" not in clean and "\u202e" not in clean and "\x00" not in clean
    assert "attacker.invalid" not in clean


def test_cli_explain_json_and_preview_explain(monkeypatch):
    trace = {"trace_id": "12345678-1234-4234-8234-123456789012", "query_id": "id",
             "strategy": "hybrid", "candidates": [], "selected": [], "excluded": [],
             "stages": [], "final_context": {}, "content": None,
             "provider_dispatch": {"state": "NOT_ATTEMPTED"}}
    calls = []
    class Response:
        def json(self):
            return trace
    monkeypatch.setattr("contextos.cli.app._api", lambda method, path, **kwargs: calls.append(path) or Response())
    runner = CliRunner()
    assert runner.invoke(app, ["explain", "query", "--json"]).exit_code == 0
    assert runner.invoke(app, ["preview", "query", "--explain"]).exit_code == 0
    assert calls == ["/explain", "/explain"]


@pytest.mark.asyncio
async def test_graph_explanation_evidence_labels_scope_and_hops(explain_stack):
    services, client = explain_stack
    # Ingest project and dependencies to create multi-hop graph projection
    repo = services["memory_repo"]
    m1 = await repo.create(Memory(content="Atlas uses Ollama", status=MemoryStatus.ACTIVE))
    m2 = await repo.create(Memory(content="Ollama runs Qwen9B", status=MemoryStatus.ACTIVE))
    await services["retrieval_index"].ensure_current()
    await services["graph"].rebuild()

    response = await client.post("/api/v1/explain", json={
        "query": "Atlas Qwen9B", "graph": True, "budget": 1000, "limit": 10,
    })
    assert response.status_code == 200
    data = response.json()
    assert data["candidates"]

    # Verify graph evidence structure
    found_graph = False
    for candidate in data["candidates"]:
        graph_paths = candidate.get("graph", [])
        if graph_paths:
            found_graph = True
            for gp in graph_paths:
                assert "path_nodes" in gp
                assert "path_edges" in gp
                assert "hop_count" in gp
                assert 1 <= gp["hop_count"] <= 3
                assert "score" in gp
                # Node structure check
                for node in gp["path_nodes"]:
                    assert "node_id" in node
                    assert "node_type" in node
                    if node["label"]:
                        assert not node["label"].startswith("memory:")
                # Edge structure check
                for edge in gp["path_edges"]:
                    assert "edge_type" in edge
                    assert "confidence" in edge
                    assert "supporting_memory_ids" in edge
    assert found_graph


@pytest.mark.asyncio
async def test_temporal_explanation_structured_evidence_and_relations(explain_stack):
    from datetime import datetime, timezone
    from contextos.core.enums import RelationType
    from contextos.core.models import MemoryRelation

    services, client = explain_stack
    memory_repo = services["memory_repo"]
    relation_repo = services["relation_repo"]

    m_current = await memory_repo.create(Memory(
        content="I currently use Python 3.12 for automation.",
        status=MemoryStatus.ACTIVE,
    ))
    m_superseded = await memory_repo.create(Memory(
        content="Previously I used Python 3.10.",
        status=MemoryStatus.SUPERSEDED,
        superseded_by=m_current.id,
    ))
    m_target_deleted = await memory_repo.create(Memory(
        content="Obsolete temporary note.",
        status=MemoryStatus.DELETED,
    ))

    # Add relations
    await relation_repo.create(MemoryRelation(
        source_memory_id=m_current.id,
        target_memory_id=m_superseded.id,
        relation_type=RelationType.CORRECTS,
        confidence=0.95,
    ))
    await relation_repo.create(MemoryRelation(
        source_memory_id=m_current.id,
        target_memory_id=m_target_deleted.id,
        relation_type=RelationType.SUPERSEDES,
        confidence=0.9,
    ))

    # Test explain targeting current memory
    resp = await client.post("/api/v1/explain", json={
        "query": "Python 3.12", "graph": False, "target_memory_id": str(m_current.id),
    })
    assert resp.status_code == 200
    candidate = next(c for c in resp.json()["candidates"] if c["memory_id"] == str(m_current.id))
    temporal = candidate["temporal"]
    assert temporal["reason_code"] == "CURRENT_STATE"
    assert temporal["eligible"] is True
    assert temporal["acceptance_rationale"] is None
    assert temporal["observed_at"] is not None

    # Check relations evidence
    relations = temporal["relations"]
    assert len(relations) >= 2
    corrects_rel = next(r for r in relations if r["relation_type"] == "corrects")
    assert corrects_rel["target_memory_id"] == str(m_superseded.id)
    assert corrects_rel["target_deleted"] is False

    deleted_rel = next(r for r in relations if r["target_memory_id"] == str(m_target_deleted.id))
    assert deleted_rel["target_deleted"] is True
    assert deleted_rel["related_memory_id"] == str(m_target_deleted.id)
    assert deleted_rel["related_memory_state"] == "deleted"
    incoming = await ExplainabilityService(services).temporal_resolver.resolve_memory_evidence(m_superseded)
    incoming_rel = next(r for r in incoming["relations"] if r["relation_type"] == "corrects")
    assert incoming_rel["related_memory_id"] == str(m_current.id)
    assert incoming_rel["related_memory_state"] == "present"


@pytest.mark.asyncio
async def test_result_limit_vs_index_and_channel_absence(explain_stack):
    services, client = explain_stack
    repo = services["memory_repo"]

    m1 = await repo.create(Memory(content="Python configuration guide part 1.", status=MemoryStatus.ACTIVE))
    m2 = await repo.create(Memory(content="Python configuration guide part 2.", status=MemoryStatus.ACTIVE))
    m_unindexed = await repo.create(Memory(content="Python configuration guide part 3.", status=MemoryStatus.ACTIVE))
    m_unrelated = await repo.create(Memory(content="Quantum mechanics physics experiment in laboratory.", status=MemoryStatus.ACTIVE))

    # Index only m1, m2, and m_unrelated (leave m_unindexed unindexed)
    await services["bm25_index"].rebuild({str(m.id): m.content for m in [m1, m2, m_unrelated]})
    await services["vector_store"].rebuild(
        [str(m.id) for m in [m1, m2, m_unrelated]],
        await services["embedding"].embed([m.content for m in [m1, m2, m_unrelated]]),
        [{"status": m.status.value, "type": m.type.value, "source_type": m.source_type} for m in [m1, m2, m_unrelated]],
    )

    # Disable auto-sync so m_unindexed stays absent from retrieval indices
    orig_sync = services["base_retrieval"]._index_synchronizer
    services["base_retrieval"]._index_synchronizer = None
    try:
        # 1. Test RESULT_LIMIT: query retrieves both m1 and m2 in pre-limit, but limit=1 cuts one
        all_res = (await client.post("/api/v1/explain", json={
            "query": "Python configuration guide", "limit": 1, "graph": False,
        })).json()
        top_cand_id = all_res["candidates"][0]["memory_id"]
        cut_id = str(m1.id) if top_cand_id == str(m2.id) else str(m2.id)

        res_limit = (await client.post("/api/v1/explain", json={
            "query": "Python configuration guide", "limit": 1, "target_memory_id": cut_id, "graph": False,
        })).json()["requested_memory"]
        assert res_limit["reason_code"] == "RESULT_LIMIT"
        assert res_limit["status"] == "retrieval_excluded"
        assert res_limit["reason"] == "result_limit_exceeded"

        # 2. Test INDEX_NOT_PRESENT: m_unindexed is in DB but not in lexical/vector store
        res_idx = (await client.post("/api/v1/explain", json={
            "query": "Python configuration guide", "target_memory_id": str(m_unindexed.id), "graph": False,
        })).json()["requested_memory"]
        assert res_idx["reason_code"] == "INDEX_NOT_PRESENT"
        assert res_idx["status"] == "not_retrieved"
        assert res_idx["reason"] == "absent_from_retrieval_index"

        # 3. Test CHANNEL_NOT_RETRIEVED: m_unrelated is indexed and eligible, but query does not surface it
        res_channel = (await client.post("/api/v1/explain", json={
            "query": "Python configuration guide", "target_memory_id": str(m_unrelated.id), "graph": False,
        })).json()["requested_memory"]
        assert res_channel["reason_code"] == "CHANNEL_NOT_RETRIEVED"
        assert res_channel["status"] == "not_retrieved"
        assert res_channel["reason"] == "channel_not_retrieved"

        # 4. Test NOT_AVAILABLE: random UUID not in database
        res_none = (await client.post("/api/v1/explain", json={
            "query": "Python configuration guide", "target_memory_id": str(uuid4()), "graph": False,
        })).json()["requested_memory"]
        assert res_none["reason_code"] == "NOT_AVAILABLE"
        assert res_none["status"] == "not_available"
    finally:
        services["base_retrieval"]._index_synchronizer = orig_sync


@pytest.mark.asyncio
async def test_truncated_prelimit_evidence_does_not_guess_absence(explain_stack, monkeypatch):
    from contextos.core.models import RetrievalResult, RetrievalTrace

    services, client = explain_stack
    memory = await services["memory_repo"].create(Memory(
        content="Indexed memory outside the captured candidate snapshot.",
        status=MemoryStatus.ACTIVE,
    ))
    await services["retrieval_index"].ensure_current()

    async def truncated_empty_result(query):
        return RetrievalResult(
            query=query.text,
            trace=RetrievalTrace(pre_limit_candidates_truncated=True),
        )
    monkeypatch.setattr(services["retrieval"], "retrieve", truncated_empty_result)
    response = await client.post("/api/v1/explain", json={
        "query": "unrelated",
        "graph": False,
        "target_memory_id": str(memory.id),
    })
    assert response.status_code == 200
    requested = response.json()["requested_memory"]
    assert requested["reason_code"] == "NOT_AVAILABLE"
    assert requested["reason"] == "candidate_evidence_truncated"


@pytest.mark.asyncio
async def test_optimizer_selection_is_not_misreported_as_compiler_inclusion(explain_stack, monkeypatch):
    services, client = explain_stack
    memory = await services["memory_repo"].create(Memory(
        content="The Python build pipeline uses deterministic testing.",
        status=MemoryStatus.ACTIVE,
    ))
    await services["retrieval_index"].ensure_current()
    original_compile = services["compilation"].compile

    async def omit_from_compiled_output(*args, **kwargs):
        result = await original_compile(*args, **kwargs)
        return result.model_copy(update={
            "included_memory_ids": [],
            "facts": [],
            "context_text": "",
            "memories_included": 0,
        })
    monkeypatch.setattr(services["compilation"], "compile", omit_from_compiled_output)

    response = await client.post("/api/v1/explain", json={
        "query": "Python build pipeline deterministic testing",
        "graph": False,
        "target_memory_id": str(memory.id),
    })
    assert response.status_code == 200
    requested = response.json()["requested_memory"]
    assert requested["reason_code"] == "COMPILER_EXCLUDED"
    assert requested["status"] == "optimizer_selected_not_compiled"


@pytest.mark.asyncio
async def test_provider_dispatch_receipt_standalone_vs_model_ask(explain_stack):
    from contextos.core.enums import ProviderDispatchState

    services, client = explain_stack

    # Standalone explain: provider_dispatch must be NOT_ATTEMPTED
    standalone_res = (await client.post("/api/v1/explain", json={"query": "test"})).json()
    assert standalone_res["provider_dispatch"]["state"] == "NOT_ATTEMPTED"
    assert "prepared by ContextOS" in standalone_res["provider_dispatch"]["status_message"]

    # ModelService.ask execution with explain=True
    model_service = services["model_service"]
    ask_result = await model_service.ask(
        query="I need help with Python scripting.",
        explain=True,
    )
    assert ask_result.dispatch_evidence is not None
    receipt = ask_result.dispatch_evidence
    assert receipt.state == ProviderDispatchState.RESPONSE_RECEIVED
    assert len(receipt.compiled_context_sha256) == 64
    assert len(receipt.logical_request_sha256) == 64
    assert receipt.compiled_context_in_request is True
    assert receipt.context_match is True
    assert receipt.provider_response_received is True
    assert receipt.provider_input_tokens is not None

    # Explanation attached to AskResult contains matching dispatch receipt
    assert ask_result.explanation is not None
    assert ask_result.explanation["provider_dispatch"]["state"] == "RESPONSE_RECEIVED"
    assert ask_result.explanation["provider_dispatch"]["compiled_context_sha256"] == receipt.compiled_context_sha256


@pytest.mark.asyncio
async def test_provider_dispatch_failure_states(explain_stack):
    from contextos.core.enums import ProviderDispatchState
    from contextos.core.exceptions import ModelUnavailableError, ProviderTimeoutError

    services, _ = explain_stack
    model_service = services["model_service"]
    fake_provider = services["fake_provider"]

    # 1. Failure before dispatch (e.g. invalid target model with allow_fallback=False)
    with pytest.raises(ModelUnavailableError):
        await model_service.ask(
            query="Unknown model",
            target_provider="fake",
            target_model="nonexistent-model-xyz",
            allow_fallback=False,
        )

    # 2. Simulate provider timeout during generation
    fake_provider.simulate_timeout = True
    try:
        with pytest.raises(ProviderTimeoutError) as exc_info:
            await model_service.ask(query="Failure simulation", explain=True)
        # Observability verification on the caught exception
        exc = exc_info.value
        assert hasattr(exc, "dispatch_evidence")
        assert exc.dispatch_evidence is not None
        assert exc.dispatch_evidence.state == ProviderDispatchState.DISPATCH_FAILED
        assert exc.dispatch_evidence.provider_response_received is False
        assert hasattr(exc, "explanation")
        assert exc.explanation is not None
    finally:
        fake_provider.simulate_timeout = False

    # Telemetry should reflect error status
    recent = await services["telemetry_repo"].list_recent(limit=1)
    assert len(recent) == 1
    assert recent[0].status == "error"


@pytest.mark.asyncio
async def test_provider_dispatch_failure_receipt_is_exposed_by_api(explain_stack):
    services, client = explain_stack
    services["fake_provider"].simulate_timeout = True
    try:
        response = await client.post("/api/v1/ask", json={
            "query": "timeout with safe dispatch receipt",
            "timeout_seconds": 0.5,
        })
    finally:
        services["fake_provider"].simulate_timeout = False
    assert response.status_code == 504
    payload = response.json()
    assert payload["dispatch_evidence"]["state"] == "DISPATCH_FAILED"
    assert payload["dispatch_evidence"]["provider_response_received"] is False
    assert "prompt" not in payload and "context" not in payload


@pytest.mark.asyncio
async def test_single_pass_pipeline_execution_without_replay(explain_stack, monkeypatch):
    services, client = explain_stack
    model_service = services["model_service"]

    retrieval_count = 0
    orig_retrieve = services["retrieval"].retrieve
    async def counting_retrieve(*args, **kwargs):
        nonlocal retrieval_count
        retrieval_count += 1
        return await orig_retrieve(*args, **kwargs)
    monkeypatch.setattr(services["retrieval"], "retrieve", counting_retrieve)

    optimize_count = 0
    orig_optimize = services["optimizer"].optimize
    def counting_optimize(*args, **kwargs):
        nonlocal optimize_count
        optimize_count += 1
        return orig_optimize(*args, **kwargs)
    monkeypatch.setattr(services["optimizer"], "optimize", counting_optimize)

    compile_count = 0
    orig_compile = services["compilation"].compile
    async def counting_compile(*args, **kwargs):
        nonlocal compile_count
        compile_count += 1
        return await orig_compile(*args, **kwargs)
    monkeypatch.setattr(services["compilation"], "compile", counting_compile)

    provider_count = 0
    orig_generate = services["fake_provider"].generate
    async def counting_generate(*args, **kwargs):
        nonlocal provider_count
        provider_count += 1
        return await orig_generate(*args, **kwargs)
    monkeypatch.setattr(services["fake_provider"], "generate", counting_generate)

    result = await model_service.ask(query="Single pass execution test", explain=True)
    assert result.explanation is not None
    assert result.dispatch_evidence is not None

    # Prove strictly one execution of each pipeline component
    assert retrieval_count == 1
    assert optimize_count == 1
    assert compile_count == 1
    assert provider_count == 1


@pytest.mark.asyncio
async def test_context_match_true_vs_deliberate_omission(explain_stack, monkeypatch):
    services, client = explain_stack
    model_service = services["model_service"]
    await services["memory_repo"].create(Memory(
        content="Testing normal context presence in the provider request.",
        status=MemoryStatus.ACTIVE,
    ))
    await services["retrieval_index"].ensure_current()

    # 1. Normal ask: compiled context is present in request -> context_match == True
    res_normal = await model_service.ask(query="Testing normal context presence", explain=True)
    assert res_normal.compiled_context.context_text
    assert res_normal.dispatch_evidence.context_match is True
    assert res_normal.dispatch_evidence.compiled_context_in_request is True

    # 2. Deliberate omission: patch provider to inspect request where compiled_context was removed
    orig_generate = services["fake_provider"].generate
    async def tampered_generate(req):
        # Even if provider generates, simulate request constructed without compiled context
        return await orig_generate(req)
    monkeypatch.setattr(services["fake_provider"], "generate", tampered_generate)

    # Patch router to clear compiled_context from the ModelRequest right before dispatch
    orig_route = model_service._router.route
    async def routing_with_context_strip(*args, **kwargs):
        decision = await orig_route(*args, **kwargs)
        req = kwargs.get("request") or (args[0] if args else None)
        if req is not None:
            req.compiled_context = None
        return decision
    monkeypatch.setattr(model_service._router, "route", routing_with_context_strip)

    res_stripped = await model_service.ask(query="Testing context stripping", explain=True)
    # When context was stripped, context_match must be False
    assert res_stripped.dispatch_evidence.context_match is False
    assert res_stripped.dispatch_evidence.compiled_context_in_request is False


@pytest.mark.asyncio
async def test_index_membership_hybrid_edge_cases(explain_stack, monkeypatch):
    services, client = explain_stack
    repo = services["memory_repo"]

    m_both = await repo.create(Memory(content="Hybrid present in both indices.", status=MemoryStatus.ACTIVE))
    m_bm25_only = await repo.create(Memory(content="Present only in BM25 index.", status=MemoryStatus.ACTIVE))
    m_dense_only = await repo.create(Memory(content="Present only in Vector index.", status=MemoryStatus.ACTIVE))
    m_neither = await repo.create(Memory(content="Present in neither index.", status=MemoryStatus.ACTIVE))

    # Populate BM25 with m_both and m_bm25_only
    await services["bm25_index"].rebuild({
        str(m_both.id): m_both.content,
        str(m_bm25_only.id): m_bm25_only.content,
    })
    # Populate Vector store with m_both and m_dense_only
    vec_both = await services["embedding"].embed([m_both.content])
    vec_dense = await services["embedding"].embed([m_dense_only.content])
    await services["vector_store"].rebuild(
        [str(m_both.id), str(m_dense_only.id)],
        vec_both + vec_dense,
        [{"status": "active"}, {"status": "active"}],
    )

    # Disable auto-sync so test verifies exact membership
    orig_sync = services["base_retrieval"]._index_synchronizer
    services["base_retrieval"]._index_synchronizer = None
    try:
        # A. Present in BM25, absent dense -> CHANNEL_NOT_RETRIEVED (NOT INDEX_NOT_PRESENT)
        res_bm25 = (await client.post("/api/v1/explain", json={
            "query": "Unrelated search query", "target_memory_id": str(m_bm25_only.id), "graph": False,
        })).json()["requested_memory"]
        assert res_bm25["reason_code"] == "CHANNEL_NOT_RETRIEVED"
        assert res_bm25["status"] == "not_retrieved"

        # B. Absent BM25, present dense -> CHANNEL_NOT_RETRIEVED (NOT INDEX_NOT_PRESENT)
        async def no_dense_results(*args, **kwargs):
            return []
        monkeypatch.setattr(services["vector_store"], "search", no_dense_results)
        res_dense = (await client.post("/api/v1/explain", json={
            "query": "Unrelated search query", "target_memory_id": str(m_dense_only.id), "graph": False,
        })).json()["requested_memory"]
        assert res_dense["reason_code"] == "CHANNEL_NOT_RETRIEVED"
        assert res_dense["status"] == "not_retrieved"

        # C. Absent both -> INDEX_NOT_PRESENT
        res_neither = (await client.post("/api/v1/explain", json={
            "query": "Unrelated search query", "target_memory_id": str(m_neither.id), "graph": False,
        })).json()["requested_memory"]
        assert res_neither["reason_code"] == "INDEX_NOT_PRESENT"
        assert res_neither["status"] == "not_retrieved"

        # With graph retrieval active, absence from the two non-graph stores
        # cannot prove global index absence.
        res_graph = (await client.post("/api/v1/explain", json={
            "query": "Unrelated search query", "target_memory_id": str(m_neither.id), "graph": True,
        })).json()["requested_memory"]
        assert res_graph["reason_code"] == "NOT_AVAILABLE"
        assert res_graph["reason"] == "graph_channel_membership_not_proven"
    finally:
        services["base_retrieval"]._index_synchronizer = orig_sync


@pytest.mark.asyncio
async def test_graph_hop_and_node_boundary_exact_definitions(explain_stack):
    services, client = explain_stack
    repo = services["memory_repo"]
    graph = services["graph"]

    # Chain: A uses B -> B uses C -> C uses D (3 hops = 3 edges, 4 nodes)
    await repo.create(Memory(content="Atlas uses Ollama", status=MemoryStatus.ACTIVE))
    await repo.create(Memory(content="Ollama runs Docker", status=MemoryStatus.ACTIVE))
    await repo.create(Memory(content="Docker depends on Linux", status=MemoryStatus.ACTIVE))
    await services["retrieval_index"].ensure_current()
    await graph.rebuild()

    expansion = await graph.expand(query_text="Atlas", max_hops=3)
    assert expansion.candidate_paths
    for paths in expansion.candidate_paths.values():
        for path in paths:
            # Strictly: hop_count <= 3, path_edges <= 3, path_nodes <= 4
            assert path.hop_count <= 3
            assert len(path.path_edges) <= 3
            assert len(path.path_nodes) <= 4
            assert len(path.path_nodes) == len(path.path_edges) + 1


