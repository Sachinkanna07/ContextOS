"""RC3 regressions for optional filters, bounded discovery and Ollama wire behavior."""

from __future__ import annotations

import asyncio
import json
import time

import httpx
import pytest
from typer.testing import CliRunner

from contextos.api.server import create_app, set_services
from contextos.cli.app import app
from contextos.config.settings import Settings
from contextos.core.enums import ModelFinishReason, TokenMeasurementSource
from contextos.core.exceptions import (
    MalformedProviderResponseError,
    ModelUnavailableError,
    ProviderAuthenticationError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from contextos.core.models import ModelCapabilities, ModelRequest
from contextos.daemon.wiring import wire_services
from contextos.providers.ollama import OllamaProvider
from contextos.providers.openai_compatible import OpenAICompatibleProvider
from contextos.services.model_discovery import ModelDiscovery


@pytest.mark.parametrize("command", ["stats", "monitor"])
@pytest.mark.parametrize(
    "filters",
    [{}, {"model": "qwen"}, {"provider": "ollama"}, {"model": "qwen", "provider": "ollama"}],
)
def test_cli_optional_filters(monkeypatch, command, filters):
    calls = []

    def request(method, path, **kwargs):
        calls.append((method, path, kwargs))
        return httpx.Response(200, json={})

    monkeypatch.setattr("contextos.cli.app._api", request)
    monkeypatch.setattr("contextos.cli.dashboard.render_dashboard", lambda *a, **kw: "dashboard")
    args = [command, *(["--samples", "1"] if command == "monitor" else ["--json"])]
    for name, value in filters.items():
        args.extend([f"--{name}", value])
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    assert calls == [("GET", "/dashboard", {"params": {"period": "all", **filters}})]
    assert "=&" not in str(httpx.QueryParams(calls[0][2]["params"]))


@pytest.mark.parametrize("command", ["stats", "monitor"])
@pytest.mark.parametrize("name", ["model", "provider"])
@pytest.mark.parametrize("value", ["", "   "])
def test_cli_rejects_blank_filters(monkeypatch, command, name, value):
    calls = []
    monkeypatch.setattr("contextos.cli.app._api", lambda *a, **kw: calls.append(kw))
    result = CliRunner().invoke(app, [command, f"--{name}", value])
    assert result.exit_code == 2
    assert "must not be empty" in result.output
    assert calls == []


def capability(provider="ollama", model="qwen"):
    return ModelCapabilities(
        provider_id=provider,
        model_id=model,
        display_name=model,
        context_window=8192,
        max_output_tokens=1024,
        local=True,
    )


class Provider:
    def __init__(self, inventory=None, error=None, gate=None):
        self.inventory = [capability()] if inventory is None else inventory
        self.error = error
        self.gate = gate
        self.calls = 0

    async def list_models(self):
        self.calls += 1
        if self.gate is not None:
            await self.gate.wait() if isinstance(self.gate, asyncio.Event) else await self.gate
        if self.error:
            raise self.error
        return self.inventory


async def test_discovery_healthy_over_one_second():
    future = asyncio.get_running_loop().create_future()
    timer = asyncio.get_running_loop().call_later(1.1, future.set_result, None)
    try:
        discovery = ModelDiscovery()
        provider = Provider(gate=future)
        started = time.monotonic()
        models = await discovery.list_models({"ollama": provider})
        assert 1.0 < time.monotonic() - started < 2.0
        assert models == [capability()]
    finally:
        timer.cancel()


async def test_discovery_mixed_timeout_dead_fast_and_recovery(monkeypatch):
    monkeypatch.setattr("contextos.services.model_discovery.DISCOVERY_TIMEOUT_SECONDS", 0.05)
    dead = Provider(error=ConnectionError("Bearer privatecredential"))
    blocked = Provider(gate=asyncio.Event())
    fast = Provider([capability("fake", "fake-default")])
    discovery = ModelDiscovery()
    providers = {"dead": dead, "timeout": blocked, "fake": fast}
    started = time.monotonic()
    assert await discovery.list_models(providers) == fast.inventory
    assert time.monotonic() - started < 0.5
    assert await discovery.list_models(providers) == fast.inventory
    assert (dead.calls, blocked.calls, fast.calls) == (1, 1, 1)
    dead.error = None
    # Advance cache time deterministically; no retry sleeps.
    now = time.monotonic()
    monkeypatch.setattr("contextos.services.model_discovery.monotonic", lambda: now + 2)
    assert await discovery.list_models(providers) == [*dead.inventory, *fast.inventory]
    assert (dead.calls, blocked.calls, fast.calls) == (2, 2, 1)


async def test_discovery_refresh_concurrency_and_provider_replacement(monkeypatch):
    discovery = ModelDiscovery()
    provider = Provider()
    providers = {"ollama": provider}
    results = await asyncio.gather(*(discovery.list_models(providers) for _ in range(8)))
    assert all(result == provider.inventory for result in results)
    assert provider.calls == 1
    provider.inventory = [capability(model="new-model")]
    now = time.monotonic()
    monkeypatch.setattr("contextos.services.model_discovery.monotonic", lambda: now + 11)
    assert await discovery.list_models(providers) == provider.inventory
    assert provider.calls == 2
    replacement = Provider([])
    assert await discovery.list_models({"ollama": replacement}) == []
    assert await discovery.list_models({}) == []


@pytest.mark.parametrize(
    "inventory", [None, {}, ["bad", capability().model_copy(update={"enabled": False})]]
)
async def test_discovery_fails_closed_on_invalid_or_disabled_models(inventory):
    provider = Provider()
    provider.inventory = inventory
    assert await ModelDiscovery().list_models({"provider": provider}) == []


@pytest.mark.parametrize(
    "adapter,key,path",
    [(OllamaProvider, "models", "/api/tags"), (OpenAICompatibleProvider, "data", "/models")],
)
async def test_inventory_single_request_no_health_cache_or_fabricated_models(adapter, key, path):
    calls = []
    status = 503

    def handler(request):
        calls.append(request.url.path)
        return httpx.Response(status, json={key: []})

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://localhost"
    ) as client:
        provider = adapter(client=client)
        with pytest.raises(ProviderUnavailableError):
            await provider.list_models()
        status = 200
        assert await provider.list_models() == []
    assert calls == [path, path]


@pytest.mark.parametrize(
    "status,error",
    [
        (404, ModelUnavailableError),
        (401, ProviderAuthenticationError),
        (403, ProviderAuthenticationError),
        (429, ProviderRateLimitError),
        (500, ProviderUnavailableError),
        (503, ProviderUnavailableError),
        (400, MalformedProviderResponseError),
    ],
)
async def test_ollama_http_error_types(status, error):
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda r: httpx.Response(status, json={"error": "private text"})
        ),
        base_url="http://localhost",
    ) as client:
        with pytest.raises(error) as caught:
            await OllamaProvider(client=client).generate(ModelRequest(user_prompt="hi"))
        assert "private text" not in str(caught.value)


@pytest.mark.parametrize(
    "failure,error",
    [
        (httpx.ReadTimeout("timeout"), ProviderTimeoutError),
        (httpx.ConnectError("connection"), ProviderUnavailableError),
    ],
)
async def test_ollama_transport_errors(failure, error):
    def handler(request):
        raise failure

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://localhost"
    ) as client:
        with pytest.raises(error):
            await OllamaProvider(client=client).generate(ModelRequest(user_prompt="hi"))


@pytest.mark.parametrize(
    "body",
    [
        b"invalid",
        b"[]",
        b'{"message":[]}',
        b'{"message":{"content":null}}',
        b'{"message":{"content":123}}',
        b'{"message":{"content":""},"done":false}',
        b'{"done":"true"}',
        b'{"done":true,"eval_count":-1}',
        b'{"done":true,"prompt_eval_count":true}',
        b'{"done":true,"eval_count":"5"}',
    ],
)
async def test_ollama_malformed_responses_are_typed(body):
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, content=body)),
        base_url="http://localhost",
    ) as client:
        with pytest.raises(MalformedProviderResponseError):
            await OllamaProvider(client=client).generate(ModelRequest(user_prompt="hi"))


@pytest.mark.parametrize("model", ["qwen2.5-coder:7b", "qwen3.5:9b"])
@pytest.mark.parametrize("reported", [True, False])
async def test_ollama_success_request_and_qwen_usage(model, reported):
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        body = {"message": {"content": "OK"}, "done": True}
        if reported:
            body.update(prompt_eval_count=83, eval_count=28)
        return httpx.Response(200, json=body)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://localhost"
    ) as client:
        provider = OllamaProvider(client=client)
        response = await provider.generate(
            ModelRequest(user_prompt="hi", model=model, system_prompt="test context")
        )
        assert response.text == "OK"
        assert requests[0]["messages"] == [
            {"role": "system", "content": "test context"},
            {"role": "user", "content": "hi"},
        ]
        assert requests[0]["stream"] is False
        assert "think" not in requests[0]
        if reported:
            assert (response.input_tokens, response.output_tokens, response.total_tokens) == (
                83,
                28,
                111,
            )
            assert response.token_measurement_source == TokenMeasurementSource.PROVIDER_REPORTED
        else:
            assert response.input_tokens == provider.count_tokens("test context\n\nhi", model)
            assert response.token_measurement_source == TokenMeasurementSource.APPROXIMATED


@pytest.mark.parametrize(
    "body",
    [
        {"message": {"content": ""}, "done": True, "prompt_eval_count": 0, "eval_count": 0},
        {
            "message": {"content": "", "thinking": "private reasoning"},
            "done": True,
            "done_reason": "length",
            "prompt_eval_count": 48,
            "eval_count": 256,
        },
    ],
)
async def test_ollama_done_empty_and_thinking_budget_responses(body):
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json=body)),
        base_url="http://localhost",
    ) as client:
        response = await OllamaProvider(client=client).generate(ModelRequest(user_prompt="hi"))
        assert response.text == ""
        assert response.token_measurement_source == TokenMeasurementSource.PROVIDER_REPORTED
        if body.get("done_reason") == "length":
            assert response.finish_reason == ModelFinishReason.LENGTH
            assert response.total_tokens == 304


async def test_ollama_external_cuda_error_diagnostic_cannot_leak_provider_text():
    raw = (
        "\x1b]0;attack\x07llama-server process has terminated: exit status 0xc0000409: "
        "CUDA error: shared object initialization failed\nBearer "
        + "x" * 80
        + " C:\\private\\model https://user:password@example.com"
    )
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda r: httpx.Response(500, json={"error": raw})),
        base_url="http://localhost",
    ) as client:
        with pytest.raises(ProviderUnavailableError) as caught:
            await OllamaProvider(client=client).generate(ModelRequest(user_prompt="hi"))
    message = str(caught.value)
    assert "exit status 0xc0000409" in message
    assert "CUDA shared object initialization failed" in message
    assert len(message) < 300
    for forbidden in ("Bearer", "private", "password", "example.com", "attack", "\x1b", "x" * 80):
        assert forbidden not in message


async def test_dashboard_inventory_matches_models_and_is_bounded(tmp_path, monkeypatch):
    monkeypatch.setattr("contextos.services.model_discovery.DISCOVERY_TIMEOUT_SECONDS", 0.05)
    services = await wire_services(Settings(daemon={"data_dir": tmp_path}))
    services["providers"] = {
        "ollama": Provider(),
        "fake": Provider([capability("fake", "fake-default")]),
        "dead": Provider(error=RuntimeError("private credential")),
        "timeout": Provider(gate=asyncio.Event()),
    }
    set_services(services)
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_app()), base_url="http://localhost"
        ) as api:
            started = time.monotonic()
            dashboard = await api.get("/api/v1/dashboard", params={"period": "all"})
            assert time.monotonic() - started < 0.5
            models = (await api.get("/api/v1/models")).json()
            assert (
                {(m["provider"], m["model"]) for m in dashboard.json()["models"]}
                == {(m["provider_id"], m["model_id"]) for m in models}
                == {("ollama", "qwen"), ("fake", "fake-default")}
            )
            assert "private credential" not in dashboard.text
            assert [m["simulated"] for m in dashboard.json()["models"]] == [False, True]
            for name in ("model", "provider"):
                for value in ("", "   "):
                    assert (
                        await api.get("/api/v1/dashboard", params={name: value})
                    ).status_code == 422
    finally:
        await services["database"].close()
