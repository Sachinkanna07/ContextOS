"""OpenAI-compatible provider adapter for local and remote endpoints."""

from __future__ import annotations

import ipaddress
import json
import logging
import os
import re
import time
from typing import Any
from urllib.parse import urlparse

import httpx

from contextos.config.settings import ProviderConfig
from contextos.core.enums import ModelFinishReason, TokenMeasurementSource
from contextos.core.exceptions import (
    MalformedProviderResponseError,
    ModelUnavailableError,
    ProviderAuthenticationError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from contextos.core.models import ModelCapabilities, ModelRequest, ModelResponse
from contextos.services.token_counter import get_token_counter_for_model

logger = logging.getLogger(__name__)


class OpenAICompatibleProvider:
    """Async adapter for generic OpenAI-compatible endpoints.

    Supports local runtimes (vLLM, LM Studio, llama.cpp server) and remote APIs.
    """

    def __init__(
        self,
        provider_id: str = "openai_compatible",
        base_url: str = "http://127.0.0.1:8000/v1",
        api_key: str | None = None,
        api_key_env: str = "",
        default_model: str = "default-model",
        timeout_seconds: float = 30.0,
        is_local: bool | None = None,
        health_cache_ttl_seconds: float = 30.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._provider_id = provider_id
        self._base_url = ProviderConfig.safe_base_url(base_url)
        self._api_key = api_key
        self._api_key_env = api_key_env
        self._default_model = default_model
        self._timeout_seconds = timeout_seconds
        self._health_ttl = health_cache_ttl_seconds
        self._client = client

        parsed = urlparse(self._base_url)
        hostname = (parsed.hostname or "").lower()
        if hostname in {"127.0.0.1", "localhost", "::1"}:
            loopback = True
        else:
            try:
                loopback = ipaddress.ip_address(hostname).is_loopback
            except ValueError:
                loopback = False
        # A caller may conservatively mark a loopback endpoint remote, never
        # upgrade a public endpoint to local and bypass remote consent.
        self._is_local = loopback and is_local is not False

        # Health caching
        self._last_health: bool = False
        self._last_health_ts: float = 0.0

    @property
    def provider_id(self) -> str:
        return self._provider_id

    @property
    def is_local(self) -> bool:
        return self._is_local

    @property
    def credential_available(self) -> bool:
        return self._is_local or not self._api_key_env or bool(os.environ.get(self._api_key_env, "").strip())

    def _get_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        key = os.environ.get(self._api_key_env, "").strip() if self._api_key_env else self._api_key
        if self._api_key_env and not self._is_local and not key:
            raise ProviderAuthenticationError(self._provider_id, "Credential unavailable")
        if key:
            headers["Authorization"] = f"Bearer {key}"
        return headers

    def _get_client(self, timeout: float | None = None) -> httpx.AsyncClient:
        if self._client is not None:
            return self._client
        t = timeout or self._timeout_seconds
        return httpx.AsyncClient(base_url=self._base_url, timeout=t, trust_env=False, follow_redirects=False)

    async def health(self) -> bool:
        """Bounded, cached health check against /models endpoint."""
        now = time.time()
        if (now - self._last_health_ts) < self._health_ttl:
            return self._last_health

        try:
            client = self._get_client(timeout=0.5)
            if self._client is not None:
                resp = await client.get("/models", headers=self._get_headers())
            else:
                async with client as c:
                    resp = await c.get(self._base_url + "/models", headers=self._get_headers())
            self._last_health = (resp.status_code == 200)
        except Exception:
            self._last_health = False

        self._last_health_ts = now
        return self._last_health

    async def list_models(self) -> list[ModelCapabilities]:
        """Fetch model inventory from /models."""
        try:
            client = self._get_client(timeout=5.0)
            if self._client is not None:
                resp = await client.get("/models", headers=self._get_headers())
            else:
                async with client as c:
                    resp = await c.get(self._base_url + "/models", headers=self._get_headers())

            if resp.status_code == 401 or resp.status_code == 403:
                raise ProviderAuthenticationError(self._provider_id)
            if resp.status_code != 200:
                raise ProviderUnavailableError(
                    self._provider_id, f"HTTP {resp.status_code} from /models"
                )
            data = resp.json()
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(self._provider_id, 5.0) from exc
        except (httpx.ConnectError, httpx.NetworkError) as exc:
            raise ProviderUnavailableError(self._provider_id, "Could not connect to provider endpoint") from exc
        except Exception as exc:
            if isinstance(exc, (ProviderAuthenticationError, ProviderTimeoutError, ProviderUnavailableError)):
                raise
            raise ProviderUnavailableError(self._provider_id, "Failed listing models") from exc

        if not isinstance(data, dict) or not isinstance(data.get("data"), list):
            raise MalformedProviderResponseError(self._provider_id, "Invalid model inventory")
        raw_models = data["data"]
        capabilities: list[ModelCapabilities] = []
        for m in raw_models:
            if not isinstance(m, dict) or not isinstance(m.get("id"), str):
                continue
            mid = m.get("id", "")
            if not mid:
                continue
            family = "cl100k_base"
            if "qwen" in mid.lower():
                family = "qwen"
            elif "claude" in mid.lower():
                family = "claude"

            capabilities.append(
                ModelCapabilities(
                    provider_id=self._provider_id,
                    model_id=mid,
                    display_name=mid,
                    context_window=32768 if "32k" in mid.lower() else 8192,
                    max_output_tokens=2048,
                    supports_tools=False,
                    supports_json=False,
                    supports_vision=False,
                    local=self._is_local,
                    tokenizer_family=family,
                    enabled=True,
                    metadata={"capability_source": "unknown"},
                )
            )

        return capabilities

    def count_tokens(self, text: str, model: str) -> int:
        counter = get_token_counter_for_model(model)
        return counter.count(text)

    async def generate(self, request: ModelRequest) -> ModelResponse:
        """Send chat completion request to /chat/completions."""
        model_name = request.model or self._default_model
        started = time.perf_counter()

        messages: list[dict[str, str]] = []
        system_content_parts: list[str] = []
        if request.system_prompt:
            system_content_parts.append(request.system_prompt)
        if request.compiled_context and request.compiled_context.context_text:
            system_content_parts.append(
                f"### Context Information:\n{request.compiled_context.context_text}"
            )

        if system_content_parts:
            messages.append({"role": "system", "content": "\n\n".join(system_content_parts)})

        messages.append({"role": "user", "content": request.user_prompt})

        payload: dict[str, Any] = {
            "model": model_name,
            "messages": messages,
        }
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.max_output_tokens:
            payload["max_tokens"] = request.max_output_tokens

        timeout_val = request.timeout_seconds or self._timeout_seconds

        try:
            client = self._get_client(timeout=timeout_val)
            if self._client is not None:
                resp = await client.post("/chat/completions", json=payload, headers=self._get_headers())
            else:
                async with client as c:
                    resp = await c.post(self._base_url + "/chat/completions", json=payload, headers=self._get_headers())

            if resp.status_code == 401 or resp.status_code == 403:
                # NEVER leak the API key or raw Authorization header in exception message
                raise ProviderAuthenticationError(self._provider_id, "Invalid credentials or unauthorized")
            elif resp.status_code == 404:
                raise ModelUnavailableError(model_name, self._provider_id)
            elif resp.status_code == 429:
                retry_after_str = resp.headers.get("retry-after")
                retry_after: float | None = None
                if retry_after_str:
                    try:
                        retry_after = float(retry_after_str)
                    except ValueError:
                        pass
                raise ProviderRateLimitError(self._provider_id, retry_after=retry_after)
            elif resp.status_code >= 500:
                raise ProviderUnavailableError(
                    self._provider_id, f"Server error HTTP {resp.status_code}"
                )
            elif resp.status_code != 200:
                raise MalformedProviderResponseError(
                    self._provider_id, f"Unexpected HTTP status {resp.status_code}"
                )

            try:
                data = resp.json()
            except (json.JSONDecodeError, ValueError) as exc:
                raise MalformedProviderResponseError(
                    self._provider_id, "Response body was not valid JSON"
                ) from exc
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(self._provider_id, timeout_val) from exc
        except (httpx.ConnectError, httpx.NetworkError) as exc:
            raise ProviderUnavailableError(self._provider_id, "Could not connect to endpoint") from exc
        except Exception as exc:
            if isinstance(
                exc,
                (
                    ModelUnavailableError,
                    ProviderAuthenticationError,
                    ProviderRateLimitError,
                    ProviderTimeoutError,
                    ProviderUnavailableError,
                    MalformedProviderResponseError,
                ),
            ):
                raise
            raise ProviderUnavailableError(self._provider_id, "Request failed") from exc

        if not isinstance(data, dict) or not isinstance(data.get("choices"), list) or not data["choices"]:
            raise MalformedProviderResponseError(self._provider_id, "Response contained no choices")
        choice = data["choices"][0]
        if not isinstance(choice, dict) or not isinstance(choice.get("message"), dict):
            raise MalformedProviderResponseError(self._provider_id, "Invalid choice")
        text = choice["message"].get("content", "")
        if not isinstance(text, str):
            raise MalformedProviderResponseError(self._provider_id, "Invalid message content")
        finish_reason_raw = choice.get("finish_reason", "stop")

        finish_reason = ModelFinishReason.STOP
        if finish_reason_raw == "length":
            finish_reason = ModelFinishReason.LENGTH
        elif finish_reason_raw in {"content_filter", "safety"}:
            finish_reason = ModelFinishReason.CONTENT_FILTER

        usage = data.get("usage")
        if usage is not None and not isinstance(usage, dict):
            raise MalformedProviderResponseError(self._provider_id, "Invalid token usage")
        usage = usage or {}
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        total_tokens = usage.get("total_tokens")
        for value in (prompt_tokens, completion_tokens, total_tokens):
            if value is not None and (type(value) is not int or value < 0):
                raise MalformedProviderResponseError(self._provider_id, "Invalid token usage")

        latency_ms = (time.perf_counter() - started) * 1000.0

        if prompt_tokens is not None and completion_tokens is not None:
            source = TokenMeasurementSource.PROVIDER_REPORTED
            in_tok = prompt_tokens
            out_tok = completion_tokens
            tot_tok = total_tokens if total_tokens is not None else in_tok + out_tok
        else:
            counter = get_token_counter_for_model(model_name)
            source = counter.measurement_source
            in_tok = counter.count("\n\n".join(message["content"] for message in messages))
            out_tok = counter.count(text)
            tot_tok = in_tok + out_tok

        return ModelResponse(
            text=text,
            model_id=model_name,
            provider_id=self._provider_id,
            input_tokens=in_tok,
            output_tokens=out_tok,
            total_tokens=tot_tok,
            latency_ms=latency_ms,
            finish_reason=finish_reason,
            token_measurement_source=source,
            raw_usage={"prompt_tokens": in_tok, "completion_tokens": out_tok,
                       "total_tokens": tot_tok} if source == TokenMeasurementSource.PROVIDER_REPORTED else None,
            request_id=(data.get("id") if isinstance(data.get("id"), str)
                        and re.fullmatch(r"chatcmpl-[A-Za-z0-9_-]{1,120}", data["id"])
                        else None),
        )
