"""Native provider protocol and privacy contracts; no live credentials required."""

from __future__ import annotations

import json

import httpx
import pytest
from typer.testing import CliRunner

from contextos.api.server import create_app, set_services
from contextos.cli.app import app
from contextos.config.settings import ProviderConfig, ProvidersConfig, Settings
from contextos.core.enums import RoutingPolicy, TokenMeasurementSource
from contextos.core.exceptions import (
    MalformedProviderResponseError,
    ModelUnavailableError,
    ProviderAuthenticationError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from contextos.core.models import CompiledContext, ModelRequest
from contextos.daemon.wiring import wire_services
from contextos.providers.fake import DeterministicFakeProvider
from contextos.providers.frontier import AnthropicProvider, GeminiProvider, OpenAIProvider
from contextos.providers.ollama import OllamaProvider
from contextos.services.router import DeterministicModelRouter


def request() -> ModelRequest:
    context = CompiledContext(
        query="project",
        context_text="ContextOS. Java. Local-first.",
        total_tokens=6,
        budget=100,
        memories_considered=3,
        memories_included=3,
        memories_excluded=0,
        compression_ratio=1,
    )
    return ModelRequest(
        user_prompt="What project?",
        model="text-model",
        compiled_context=context,
        system_prompt="Be concise",
        max_output_tokens=30,
    )


CASES = [
    (
        OpenAIProvider,
        {
            "output": [
                {"type": "reasoning", "summary": "private"},
                {
                    "type": "message",
                    "content": [
                        {"type": "output_text", "text": "Context"},
                        {"type": "output_text", "text": "OS"},
                    ],
                },
            ],
            "status": "completed",
            "usage": {"input_tokens": 0, "output_tokens": 2, "total_tokens": 2},
            "id": "resp_123",
        },
        "/responses",
    ),
    (
        AnthropicProvider,
        {
            "content": [
                {"type": "thinking", "thinking": "private"},
                {"type": "text", "text": "Context"},
                {"type": "text", "text": "OS"},
            ],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 2, "cache_read_input_tokens": 2},
            "id": "msg_123",
        },
        "/messages",
    ),
    (
        GeminiProvider,
        {
            "steps": [
                {"type": "thought", "content": [{"text": "private"}]},
                {
                    "type": "model_output",
                    "content": [
                        {"type": "text", "text": "Context"},
                        {"type": "text", "text": "OS"},
                    ],
                },
            ],
            "status": "completed",
            "usage": {"total_input_tokens": 1, "total_output_tokens": 2, "total_tokens": 3},
            "id": "v1_123",
        },
        "/v1/interactions",
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("cls,body,path", CASES)
async def test_native_payload_response_and_usage(monkeypatch, cls, body, path):
    monkeypatch.setenv("RC4_TEST_KEY", "sk-test-secret-should-not-leak")
    seen = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        return httpx.Response(200, json=body)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = cls(api_key_env="RC4_TEST_KEY", client=client)
        response = await provider.generate(request())
    assert response.text == "ContextOS"
    assert response.token_measurement_source == TokenMeasurementSource.PROVIDER_REPORTED
    assert response.output_tokens == 2
    assert response.provider_id == provider.provider_id
    assert response.request_id
    assert str(seen[0].url).endswith(path)
    sent = json.loads(seen[0].content)
    assert "ContextOS. Java. Local-first." in json.dumps(sent)
    assert "Be concise" in json.dumps(sent)
    assert "sk-test-secret" not in json.dumps(sent)
    assert "private" not in response.model_dump_json()
    if cls is OpenAIProvider:
        assert "temperature" not in sent and sent["store"] is False
    if cls is GeminiProvider:
        assert sent["store"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize("cls,body,path", CASES)
async def test_missing_usage_is_approximated(monkeypatch, cls, body, path):
    monkeypatch.setenv("RC4_TEST_KEY", "secret")
    body = {**body, "usage": None}
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=body))
    ) as client:
        response = await cls("RC4_TEST_KEY", client=client).generate(request())
    assert response.input_tokens > 0
    assert response.token_measurement_source != TokenMeasurementSource.PROVIDER_REPORTED


@pytest.mark.asyncio
@pytest.mark.parametrize("cls,body,path", CASES)
@pytest.mark.parametrize(
    "status,error",
    [
        (401, ProviderAuthenticationError),
        (403, ProviderAuthenticationError),
        (404, ModelUnavailableError),
        (429, ProviderRateLimitError),
        (503, ProviderUnavailableError),
    ],
)
async def test_http_errors_are_typed_and_sanitized(monkeypatch, cls, body, path, status, error):
    secret = "sk-test-secret-should-not-leak"
    monkeypatch.setenv("RC4_TEST_KEY", secret)
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(
                status,
                json={"error": secret + " Authorization: Bearer " + secret},
                headers={"retry-after": "2"},
            )
        )
    ) as client:
        with pytest.raises(error) as raised:
            await cls("RC4_TEST_KEY", client=client).generate(request())
    assert secret not in str(raised.value)
    assert "Authorization" not in str(raised.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("cls,body,path", CASES)
async def test_malformed_response_and_missing_key(monkeypatch, cls, body, path):
    monkeypatch.delenv("RC4_TEST_KEY", raising=False)
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=body))
    ) as client:
        provider = cls("RC4_TEST_KEY", client=client)
        assert not await provider.health()
        with pytest.raises(ProviderAuthenticationError):
            await provider.generate(request())
    monkeypatch.setenv("RC4_TEST_KEY", "secret")
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, text="{invalid"))
    ) as client:
        with pytest.raises(MalformedProviderResponseError):
            await cls("RC4_TEST_KEY", client=client).generate(request())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "cls,inventory",
    [
        (OpenAIProvider, {"data": [{"id": "text-model"}]}),
        (
            AnthropicProvider,
            {"data": [{"id": "text-model", "max_input_tokens": 10000, "max_tokens": 500}]},
        ),
        (
            GeminiProvider,
            {
                "models": [
                    {
                        "name": "models/text-model",
                        "supportedGenerationMethods": ["generateContent"],
                        "inputTokenLimit": 10000,
                    }
                ]
            },
        ),
    ],
)
async def test_discovery_has_conservative_capabilities(monkeypatch, cls, inventory):
    monkeypatch.setenv("RC4_TEST_KEY", "secret")
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=inventory))
    ) as client:
        provider = cls("RC4_TEST_KEY", client=client)
        models = await provider.list_models()
    assert len(models) == 1 and models[0].model_id == "text-model"
    assert models[0].enabled and not models[0].local
    assert not models[0].supports_tools and not models[0].supports_json
    assert not models[0].supports_vision
    assert models[0].metadata["capability_source"] == "unknown"


@pytest.mark.asyncio
@pytest.mark.parametrize("cls,body,path", CASES)
async def test_timeout_and_connect_failure_are_typed(monkeypatch, cls, body, path):
    monkeypatch.setenv("RC4_TEST_KEY", "secret")

    def timed_out(_: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("private secret")

    def disconnected(_: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("private secret")

    for handler, error in (
        (timed_out, ProviderTimeoutError),
        (disconnected, ProviderUnavailableError),
    ):
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with pytest.raises(error) as raised:
                await cls("RC4_TEST_KEY", client=client).generate(request())
        assert "private secret" not in str(raised.value)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "cls,bad",
    [
        (OpenAIProvider, {"output": [{"type": "message", "content": "wrong"}]}),
        (AnthropicProvider, {"content": [{"type": "text", "text": {"bad": 1}}]}),
        (GeminiProvider, {"steps": [{"type": "model_output", "content": {"bad": 1}}]}),
    ],
)
async def test_malformed_structures_fail_closed(monkeypatch, cls, bad):
    monkeypatch.setenv("RC4_TEST_KEY", "secret")
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=bad))
    ) as client:
        with pytest.raises(MalformedProviderResponseError):
            await cls("RC4_TEST_KEY", client=client).generate(request())


@pytest.mark.asyncio
async def test_provider_request_id_cannot_carry_a_secret(monkeypatch):
    monkeypatch.setenv("RC4_TEST_KEY", "sk-test-secret-should-not-leak")
    body = {**CASES[0][1], "id": "sk-test-secret-should-not-leak"}
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=body))
    ) as client:
        response = await OpenAIProvider("RC4_TEST_KEY", client=client).generate(request())
    assert response.request_id is None
    assert "sk-test-secret" not in response.model_dump_json()


@pytest.mark.parametrize(
    "url",
    [
        "http://example.com/v1",
        "https://user:pass@example.com/v1",
        "https://example.com/v1?key=secret",
        "http://10.0.0.1/v1",
        "file:///etc/passwd",
        "https://example.com#fragment",
    ],
)
def test_unsafe_endpoint_rejected(url):
    with pytest.raises(ValueError):
        ProviderConfig(base_url=url)


def test_provider_ids_and_keys_are_validated():
    with pytest.raises(ValueError):
        ProvidersConfig(compatible={"openai": ProviderConfig(enabled=True)})
    with pytest.raises(ValueError):
        ProviderConfig(api_key_env="actual-secret; echo hello")


@pytest.mark.asyncio
async def test_unconfigured_remote_is_not_discoverable(tmp_path, monkeypatch):
    monkeypatch.delenv("RC4_TEST_KEY", raising=False)
    settings = Settings(
        daemon={"data_dir": tmp_path},
        providers={
            "openai": {"enabled": True, "api_key_env": "RC4_TEST_KEY"},
            "compatible": {
                "my-local": {"enabled": True, "base_url": "http://127.0.0.1:8000/v1"},
                "my-remote": {
                    "enabled": True,
                    "base_url": "https://example.com/v1",
                    "api_key_env": "RC4_TEST_KEY",
                },
            },
        },
    )
    services = await wire_services(settings)
    set_services(services)
    try:
        assert "openai" not in services["providers"]
        assert "my-remote" not in services["providers"]
        assert services["providers"]["my-local"].is_local
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_app()), base_url="http://localhost"
        ) as api:
            rows = (await api.get("/api/v1/models/providers")).json()
            by_id = {row["provider"]: row for row in rows}
            assert by_id["openai"]["credential"] == "missing"
            assert by_id["openai"]["status"] == "unconfigured"
            assert by_id["my-local"]["local"] is True
            assert by_id["my-remote"]["local"] is False
            assert "RC4_TEST_KEY" not in json.dumps(rows)
    finally:
        await services["database"].close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "field,value",
    [
        ("query", "x" * 20001),
        ("provider", "OpenAI"),
        ("provider", "../remote"),
        ("model", " "),
        ("max_output_tokens", 50000),
        ("timeout_seconds", 1000),
        ("temperature", 3),
        ("session_id", "bad\nlabel"),
    ],
)
async def test_ask_input_validation_excludes_echoed_values(field, value):
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=create_app()), base_url="http://localhost"
    ) as api:
        response = await api.post("/api/v1/ask", json={"query": "safe", field: value})
    assert response.status_code == 422
    if len(str(value).strip()) > 1:
        assert str(value) not in response.text


@pytest.mark.asyncio
async def test_remote_fallback_requires_second_opt_in():
    local = DeterministicFakeProvider(provider_id="local", is_local=True)
    remote = DeterministicFakeProvider(provider_id="remote", is_local=False)
    local.simulate_unhealthy = True
    router = DeterministicModelRouter(default_provider_id="local")
    with pytest.raises(ProviderUnavailableError):
        await router.route(
            ModelRequest(user_prompt="x", allow_fallback=True),
            {"local": local, "remote": remote},
            RoutingPolicy.LOCAL_FIRST,
        )
    decision = await router.route(
        ModelRequest(user_prompt="x", allow_fallback=True, allow_remote=True),
        {"local": local, "remote": remote},
        RoutingPolicy.LOCAL_FIRST,
    )
    assert decision.selected_provider == "remote" and decision.fallback_used
    explicit = await router.route(
        ModelRequest(user_prompt="x", provider="remote"),
        {"local": local, "remote": remote},
        RoutingPolicy.EXPLICIT,
    )
    assert explicit.selected_provider == "remote"


@pytest.mark.asyncio
async def test_one_store_compiles_for_four_provider_payloads(tmp_path, monkeypatch):
    monkeypatch.setenv("RC4_TEST_KEY", "secret")
    seen: dict[str, list[str]] = {name: [] for name in ("ollama", "openai", "anthropic", "gemini")}

    def mocked(name):
        def handler(req: httpx.Request) -> httpx.Response:
            path = req.url.path
            if name == "ollama" and path == "/api/tags":
                return httpx.Response(200, json={"models": [{"name": "qwen2.5-coder:7b"}]})
            if name == "ollama":
                seen[name].append(req.content.decode())
                return httpx.Response(
                    200,
                    json={
                        "message": {"content": "ContextOS Java"},
                        "done": True,
                        "prompt_eval_count": 20,
                        "eval_count": 2,
                    },
                )
            if path.endswith("/models"):
                if name == "gemini":
                    return httpx.Response(200, json={"models": [{"name": "models/text-model"}]})
                return httpx.Response(200, json={"data": [{"id": "text-model"}]})
            seen[name].append(req.content.decode())
            if name == "openai":
                return httpx.Response(200, json=CASES[0][1])
            if name == "anthropic":
                return httpx.Response(200, json=CASES[1][1])
            return httpx.Response(200, json=CASES[2][1])

        return handler

    services = await wire_services(Settings(daemon={"data_dir": tmp_path}))
    clients = {
        name: httpx.AsyncClient(transport=httpx.MockTransport(mocked(name)), base_url="http://mock")
        for name in seen
    }
    providers = {
        "ollama": OllamaProvider(client=clients["ollama"]),
        "openai": OpenAIProvider("RC4_TEST_KEY", client=clients["openai"]),
        "anthropic": AnthropicProvider("RC4_TEST_KEY", client=clients["anthropic"]),
        "gemini": GeminiProvider("RC4_TEST_KEY", client=clients["gemini"]),
    }
    for provider in providers.values():
        services["model_service"].register_provider(provider)
    services["providers"].update(providers)
    set_services(services)
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_app()), base_url="http://localhost"
        ) as api:
            for fact in (
                "I prefer concise technical explanations.",
                "My primary interview language is Java.",
                "I am building a local AI memory runtime called ContextOS.",
                "I prefer local-first AI tools.",
            ):
                response = await api.post("/api/v1/remember", json={"text": fact})
                assert response.status_code == 200, response.text
            stored_before = await services["memory_repo"].count()
            for name in providers:
                model = "qwen2.5-coder:7b" if name == "ollama" else "text-model"
                response = await api.post(
                    "/api/v1/ask",
                    json={
                        "query": "What project am I building and what language do I use?",
                        "provider": name,
                        "model": model,
                        "max_output_tokens": 64,
                    },
                )
                assert response.status_code == 200, response.text
                body = response.json()
                assert body["dispatch_evidence"]["compiled_context_in_request"]
                assert body["dispatch_evidence"]["provider_response_received"]
                assert body["telemetry"]["provider_id"] == name
                assert body["telemetry"]["status"] == "success"
                assert "ContextOS" in seen[name][0]
            assert await services["memory_repo"].count() == stored_before
            assert await services["telemetry_repo"].count() == 4
    finally:
        for client in clients.values():
            await client.aclose()
        await services["database"].close()


def test_cli_ask_hides_context_unless_requested(monkeypatch):
    class Reply:
        def json(self):
            return {
                "response": {"text": "ContextOS"},
                "compiled_context": {"context_text": "Private ContextOS memory"},
                "dispatch_evidence": {"provider_id": "ollama"},
            }

    sent = []

    def api(method, path, **kwargs):
        sent.append(kwargs["json"])
        return Reply()

    monkeypatch.setattr("contextos.cli.app._api", api)
    runner = CliRunner()
    plain = runner.invoke(app, ["ask", "What project?", "--provider", "ollama"])
    assert plain.exit_code == 0, plain.output
    assert "ContextOS" in plain.output and "Private" not in plain.output
    as_json = runner.invoke(app, ["ask", "What project?", "--json"])
    assert as_json.exit_code == 0 and "Private" not in as_json.output
    with_context = runner.invoke(app, ["ask", "What project?", "--show-context"])
    assert with_context.exit_code == 0 and "Private ContextOS memory" in with_context.output
    assert sent[0]["allow_remote"] is False and sent[0]["provider"] == "ollama"
