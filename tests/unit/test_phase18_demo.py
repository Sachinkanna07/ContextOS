"""The offline demo must actually traverse the promised boundaries."""

import pytest

from contextos.demo import run_demo


@pytest.mark.asyncio
async def test_demo_has_real_temporal_connector_graph_compiler_provider_and_privacy_evidence():
    result = await run_demo()
    assert result["temporal"] == {"previous_status": "superseded", "current_status": "active"}
    assert result["connector_first"]["accepted"] >= 1
    assert result["connector_second"]["unchanged"] >= 1
    assert result["graph"]["edges"] >= 1
    assert result["graph"]["expansion_candidates"] >= 1
    assert result["inspection"]["context_diff"]["facts_emitted"] >= 1
    assert result["explanation"]["provider_state"] == "NOT_ATTEMPTED"
    assert result["model"]["dispatch_state"] == "RESPONSE_RECEIVED"
    assert result["model"]["compiled_memory_ids"] >= 1
    assert result["model"]["compiled_tokens"] > 0
    assert result["model"]["tokens_avoided"] > 0
    assert result["model"]["provenance_coverage"] > 0
    assert result["telemetry_invocations"] == 1
    assert result["privacy_secret_rejected"] is True
    assert result["dashboard"]["active_memories"] >= 1
    assert result["dashboard"]["connector_tracked_items"] >= 1
    assert result["dashboard"]["recent_invocations"] == 1
