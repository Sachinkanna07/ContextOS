"""Tests A through T: Model Router, provider selection, fallback, error mapping, and safety."""

from __future__ import annotations

import httpx
import pytest

from contextos.core.enums import ModelFinishReason, RoutingPolicy, TokenMeasurementSource
from contextos.core.exceptions import (
    ContextWindowExceededError,
    MalformedProviderResponseError,
    ModelUnavailableError,
    ProviderAuthenticationError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    RoutingFailureError,
)
from contextos.core.models import (
    CompiledContext,
    ModelCapabilities,
    ModelRequest,
)
from contextos.providers.fake import DeterministicFakeProvider
from contextos.providers.ollama import OllamaProvider
from contextos.providers.openai_compatible import OpenAICompatibleProvider
from contextos.services.router import DeterministicModelRouter


@pytest.fixture
def fake_providers():
    local_fake = DeterministicFakeProvider(
        provider_id="fake-local",
        is_local=True,
        models=[
            ModelCapabilities(
                provider_id="fake-local",
                model_id="qwen-local",
                display_name="Qwen Local",
                context_window=4096,
                max_output_tokens=1024,
                supports_tools=True,
                supports_json=True,
                supports_vision=False,
                local=True,
                tokenizer_family="qwen",
                enabled=True,
            ),
            ModelCapabilities(
                provider_id="fake-local",
                model_id="llama-local",
                display_name="Llama Local",
                context_window=8192,
                max_output_tokens=2048,
                supports_tools=False,
                supports_json=True,
                supports_vision=False,
                local=True,
                tokenizer_family="cl100k_base",
                enabled=True,
            ),
        ],
    )
    remote_fake = DeterministicFakeProvider(
        provider_id="fake-remote",
        is_local=False,
        models=[
            ModelCapabilities(
                provider_id="fake-remote",
                model_id="claude-remote",
                display_name="Claude Remote",
                context_window=32768,
                max_output_tokens=4096,
                supports_tools=True,
                supports_json=True,
                supports_vision=True,
                local=False,
                tokenizer_family="claude",
                enabled=True,
            ),
        ],
    )
    return {
        "fake-local": local_fake,
        "fake-remote": remote_fake,
    }


@pytest.fixture
def router():
    return DeterministicModelRouter(
        default_provider_id="fake-local",
        default_model_id="llama-local",
        default_policy=RoutingPolicy.LOCAL_FIRST,
    )


# ---------------------------------------------------------------------------
# Test A: Explicit Provider Selection
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_a_explicit_provider_selection(router, fake_providers):
    req = ModelRequest(user_prompt="Hello", provider="fake-remote")
    decision = await router.route(req, fake_providers, policy=RoutingPolicy.EXPLICIT)
    assert decision.selected_provider == "fake-remote"
    assert decision.selected_model == "claude-remote"
    assert not decision.fallback_used


# ---------------------------------------------------------------------------
# Test B: Explicit Model Selection
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_b_explicit_model_selection(router, fake_providers):
    req = ModelRequest(user_prompt="Hello", provider="fake-local", model="qwen-local")
    decision = await router.route(req, fake_providers, policy=RoutingPolicy.EXPLICIT)
    assert decision.selected_provider == "fake-local"
    assert decision.selected_model == "qwen-local"


# ---------------------------------------------------------------------------
# Test C: Default Provider
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_c_default_provider(router, fake_providers):
    req = ModelRequest(user_prompt="Hello")
    decision = await router.route(req, fake_providers, policy=RoutingPolicy.FIXED_DEFAULT)
    assert decision.selected_provider == "fake-local"
    assert decision.selected_model == "llama-local"


# ---------------------------------------------------------------------------
# Test D: Local-First Selection
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_d_local_first_selection(router, fake_providers):
    req = ModelRequest(user_prompt="Explain C++ pointers")
    decision = await router.route(req, fake_providers, policy=RoutingPolicy.LOCAL_FIRST)
    assert decision.selected_provider == "fake-local"
    assert fake_providers[decision.selected_provider].is_local is True
    assert not decision.fallback_used


# ---------------------------------------------------------------------------
# Test E: Local-First No Fallback (Raises ProviderUnavailableError)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_e_local_first_no_fallback(router, fake_providers):
    fake_providers["fake-local"].simulate_unhealthy = True
    req = ModelRequest(user_prompt="Test query", allow_fallback=False)
    with pytest.raises(ProviderUnavailableError) as exc_info:
        await router.route(req, fake_providers, policy=RoutingPolicy.LOCAL_FIRST)
    assert "fallback not allowed" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# Test F: Local-First Explicit Fallback
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_f_local_first_explicit_fallback(router, fake_providers):
    fake_providers["fake-local"].simulate_unhealthy = True
    req = ModelRequest(user_prompt="Test query", allow_fallback=True)
    decision = await router.route(req, fake_providers, policy=RoutingPolicy.LOCAL_FIRST)
    assert decision.fallback_used is True
    assert decision.selected_provider == "fake-remote"
    assert decision.initial_provider == "fake-local"
    assert decision.fallback_reason is not None


# ---------------------------------------------------------------------------
# Test G: Unhealthy Provider
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_g_unhealthy_provider(router, fake_providers):
    fake_providers["fake-remote"].simulate_unhealthy = True
    req = ModelRequest(user_prompt="Test query", provider="fake-remote")
    with pytest.raises(ProviderUnavailableError):
        await router.route(req, fake_providers, policy=RoutingPolicy.EXPLICIT)


# ---------------------------------------------------------------------------
# Test H: Unavailable Model
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_h_unavailable_model(router, fake_providers):
    req = ModelRequest(user_prompt="Test query", provider="fake-local", model="nonexistent-model")
    with pytest.raises(ModelUnavailableError) as exc_info:
        await router.route(req, fake_providers, policy=RoutingPolicy.EXPLICIT)
    assert "nonexistent-model" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Test I: Deterministic Routing
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_i_deterministic_routing(router, fake_providers):
    req = ModelRequest(user_prompt="Consistent prompt")
    decision1 = await router.route(req, fake_providers, policy=RoutingPolicy.LOCAL_FIRST)
    decision2 = await router.route(req, fake_providers, policy=RoutingPolicy.LOCAL_FIRST)
    assert decision1.selected_provider == decision2.selected_provider
    assert decision1.selected_model == decision2.selected_model


# ---------------------------------------------------------------------------
# Test J: Routing Trace
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_j_routing_trace(router, fake_providers):
    req = ModelRequest(user_prompt="Trace test", required_capabilities=["tools", "vision"])
    decision = await router.route(req, fake_providers, policy=RoutingPolicy.CAPABILITY_AWARE)
    assert decision.policy == RoutingPolicy.CAPABILITY_AWARE
    assert len(decision.candidates_evaluated) > 0
    assert decision.selected_provider == "fake-remote"
    assert decision.routing_latency_ms >= 0.0


# ---------------------------------------------------------------------------
# Test K: Context-Window Fit
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_k_context_window_fit(router, fake_providers):
    short_context = CompiledContext(
        query="test",
        context_text="Brief facts about the project.",
        total_tokens=10,
        budget=1000,
        memories_considered=1,
        memories_included=1,
        memories_excluded=0,
        compression_ratio=1.0,
    )
    req = ModelRequest(user_prompt="Test fit", compiled_context=short_context)
    decision = await router.route(req, fake_providers, policy=RoutingPolicy.LOCAL_FIRST)
    assert decision.selected_model in {"qwen-local", "llama-local"}


# ---------------------------------------------------------------------------
# Test L: Context Overflow (No Silent Truncation)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_l_context_overflow(router, fake_providers):
    # Giant compiled context exceeding 4096 and 8192
    huge_text = "word " * 10000
    huge_context = CompiledContext(
        query="test",
        context_text=huge_text,
        total_tokens=10000,
        budget=15000,
        memories_considered=10,
        memories_included=10,
        memories_excluded=0,
        compression_ratio=1.0,
    )
    req = ModelRequest(
        user_prompt="Test overflow",
        compiled_context=huge_context,
        provider="fake-local",
        model="qwen-local",
    )
    with pytest.raises(ContextWindowExceededError) as exc_info:
        await router.route(req, fake_providers, policy=RoutingPolicy.EXPLICIT)
    assert exc_info.value.context_window == 4096
    assert exc_info.value.required_tokens > 4096


# ---------------------------------------------------------------------------
# Test M: Optional API Key
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_m_optional_api_key():
    # Local server without api_key
    provider_no_key = OpenAICompatibleProvider(
        base_url="http://127.0.0.1:8000/v1",
        api_key=None,
    )
    headers_no_key = provider_no_key._get_headers()
    assert "Authorization" not in headers_no_key

    # Cloud server with api_key
    provider_with_key = OpenAICompatibleProvider(
        base_url="https://api.openai.com/v1",
        api_key="sk-test-secret-12345",
    )
    headers_with_key = provider_with_key._get_headers()
    assert headers_with_key["Authorization"] == "Bearer sk-test-secret-12345"


# ---------------------------------------------------------------------------
# Test N: Remote Flag
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_n_remote_flag():
    local_p = OpenAICompatibleProvider(base_url="http://127.0.0.1:1234/v1")
    assert local_p.is_local is True

    remote_p = OpenAICompatibleProvider(base_url="https://api.deepseek.com/v1")
    assert remote_p.is_local is False


# ---------------------------------------------------------------------------
# Test O: Timeout Mapping
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_o_timeout_mapping():
    def timeout_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("Connection timed out", request=request)

    transport = httpx.MockTransport(timeout_handler)
    client = httpx.AsyncClient(transport=transport, base_url="http://test")
    provider = OpenAICompatibleProvider(client=client, timeout_seconds=1.0)

    req = ModelRequest(user_prompt="Timeout test", timeout_seconds=1.0)
    with pytest.raises(ProviderTimeoutError):
        await provider.generate(req)


# ---------------------------------------------------------------------------
# Test P: Authentication Error Mapping
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_p_authentication_error_mapping():
    def auth_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "Invalid API key"})

    transport = httpx.MockTransport(auth_handler)
    client = httpx.AsyncClient(transport=transport, base_url="http://test")
    provider = OpenAICompatibleProvider(
        client=client,
        api_key="super-secret-key-123",
    )

    req = ModelRequest(user_prompt="Auth test")
    with pytest.raises(ProviderAuthenticationError) as exc_info:
        await provider.generate(req)
    # Ensure secret is NOT in exception message
    assert "super-secret-key-123" not in str(exc_info.value)


# ---------------------------------------------------------------------------
# Test Q: Malformed Response Mapping
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_q_malformed_response():
    def malformed_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"unexpected_format": True})

    transport = httpx.MockTransport(malformed_handler)
    client = httpx.AsyncClient(transport=transport, base_url="http://test")
    provider = OpenAICompatibleProvider(client=client)

    req = ModelRequest(user_prompt="Malformed test")
    with pytest.raises(MalformedProviderResponseError):
        await provider.generate(req)


# ---------------------------------------------------------------------------
# Test R: Rate-Limit Mapping
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_r_rate_limit_mapping():
    def rate_limit_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, headers={"Retry-After": "5"}, json={"error": "Rate limit exceeded"})

    transport = httpx.MockTransport(rate_limit_handler)
    client = httpx.AsyncClient(transport=transport, base_url="http://test")
    provider = OpenAICompatibleProvider(client=client)

    req = ModelRequest(user_prompt="Rate limit test")
    with pytest.raises(ProviderRateLimitError) as exc_info:
        await provider.generate(req)
    assert exc_info.value.retry_after == 5.0


# ---------------------------------------------------------------------------
# Test S: Provider Failure
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_s_provider_failure():
    def failure_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "Internal Server Error"})

    transport = httpx.MockTransport(failure_handler)
    client = httpx.AsyncClient(transport=transport, base_url="http://test")
    provider = OpenAICompatibleProvider(client=client)

    req = ModelRequest(user_prompt="Failure test")
    with pytest.raises(ProviderUnavailableError):
        await provider.generate(req)


# ---------------------------------------------------------------------------
# Test T: No Secret Leakage in Errors
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_t_no_secret_leakage_in_errors():
    secret_key = "sk-live-998877665544332211"
    secret_url_param = "key=secret_param_value"

    def error_handler(request: httpx.Request) -> httpx.Response:
        # Simulate an upstream error that would echo the Authorization header if carelessly dumped
        return httpx.Response(401, json={"detail": "Unauthorized request"})

    transport = httpx.MockTransport(error_handler)
    client = httpx.AsyncClient(transport=transport, base_url="http://test")
    provider = OpenAICompatibleProvider(
        client=client,
        api_key=secret_key,
    )

    req = ModelRequest(user_prompt="Test secrets")
    with pytest.raises(ProviderAuthenticationError) as exc_info:
        await provider.generate(req)

    error_msg = str(exc_info.value)
    assert secret_key not in error_msg
    assert secret_url_param not in error_msg
    assert "Bearer" not in error_msg
