"""Ollama provider adapter for local inference."""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import httpx

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


class OllamaProvider:
    """Async adapter for Ollama local inference API."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434",
        default_model: str = "llama3.2",
        timeout_seconds: float = 30.0,
        health_cache_ttl_seconds: float = 30.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._provider_id = "ollama"
        self._base_url = base_url.rstrip("/")
        self._default_model = default_model
        self._timeout_seconds = timeout_seconds
        self._health_ttl = health_cache_ttl_seconds
        self._client = client

        # Health caching to prevent expensive O(N) network calls per request
        self._last_health: bool = False
        self._last_health_ts: float = 0.0

    @property
    def provider_id(self) -> str:
        return self._provider_id

    @property
    def is_local(self) -> bool:
        return True

    def _get_client(self, timeout: float | None = None) -> httpx.AsyncClient:
        if self._client is not None:
            return self._client
        t = timeout or self._timeout_seconds
        return httpx.AsyncClient(base_url=self._base_url, timeout=t)

    async def health(self) -> bool:
        """Bounded, cached health check against Ollama /api/tags."""
        now = time.time()
        if (now - self._last_health_ts) < self._health_ttl:
            return self._last_health

        try:
            client = self._get_client(timeout=0.5)
            if self._client is not None:
                resp = await client.get("/api/tags")
            else:
                async with client as c:
                    resp = await c.get("/api/tags")
            self._last_health = (resp.status_code == 200)
        except Exception:
            self._last_health = False

        self._last_health_ts = now
        return self._last_health

    async def list_models(self) -> list[ModelCapabilities]:
        """Fetch model tags from Ollama."""
        if not await self.health():
            # Provider is offline/unhealthy; return disabled default descriptor
            return [
                ModelCapabilities(
                    provider_id=self._provider_id,
                    model_id=self._default_model,
                    display_name=self._default_model,
                    context_window=8192,
                    max_output_tokens=2048,
                    supports_tools=True,
                    supports_json=True,
                    supports_vision=False,
                    local=True,
                    tokenizer_family="llama",
                    enabled=False,
                )
            ]

        try:
            client = self._get_client(timeout=5.0)
            if self._client is not None:
                resp = await client.get("/api/tags")
            else:
                async with client as c:
                    resp = await c.get("/api/tags")

            if resp.status_code != 200:
                raise ProviderUnavailableError(
                    self._provider_id, f"HTTP {resp.status_code} from /api/tags"
                )
            data = resp.json()
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(self._provider_id, 5.0) from exc
        except (httpx.ConnectError, httpx.NetworkError) as exc:
            raise ProviderUnavailableError(self._provider_id, "Could not connect to Ollama") from exc
        except Exception as exc:
            if isinstance(exc, (ProviderTimeoutError, ProviderUnavailableError)):
                raise
            raise ProviderUnavailableError(self._provider_id, "Unexpected failure listing models") from exc

        raw_models = data.get("models", [])
        capabilities: list[ModelCapabilities] = []
        for m in raw_models:
            name = m.get("name", "")
            if not name:
                continue
            family = "qwen" if "qwen" in name.lower() else "llama"
            capabilities.append(
                ModelCapabilities(
                    provider_id=self._provider_id,
                    model_id=name,
                    display_name=name,
                    context_window=8192,
                    max_output_tokens=2048,
                    supports_tools=True,
                    supports_json=True,
                    supports_vision=False,
                    local=True,
                    tokenizer_family=family,
                    enabled=True,
                    metadata={"details": m.get("details", {})},
                )
            )

        if not capabilities:
            # Fallback entry if no models are downloaded yet
            capabilities.append(
                ModelCapabilities(
                    provider_id=self._provider_id,
                    model_id=self._default_model,
                    display_name=self._default_model,
                    context_window=8192,
                    max_output_tokens=2048,
                    supports_tools=True,
                    supports_json=True,
                    supports_vision=False,
                    local=True,
                    tokenizer_family="llama",
                    enabled=True,
                )
            )

        return capabilities

    def count_tokens(self, text: str, model: str) -> int:
        family = "qwen" if "qwen" in model.lower() else "cl100k_base"
        counter = get_token_counter_for_model(model, family)
        return counter.count(text)

    async def generate(self, request: ModelRequest) -> ModelResponse:
        """Send chat generation request to Ollama /api/chat."""
        model_name = request.model or self._default_model
        started = time.perf_counter()

        # Build messages payload
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

        payload = {
            "model": model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": request.temperature,
            },
        }
        if request.max_output_tokens:
            payload["options"]["num_predict"] = request.max_output_tokens

        timeout_val = request.timeout_seconds or self._timeout_seconds

        try:
            client = self._get_client(timeout=timeout_val)
            if self._client is not None:
                resp = await client.post("/api/chat", json=payload)
            else:
                async with client as c:
                    resp = await c.post("/api/chat", json=payload)

            if resp.status_code == 404:
                raise ModelUnavailableError(model_name, self._provider_id)
            elif resp.status_code == 401 or resp.status_code == 403:
                raise ProviderAuthenticationError(self._provider_id)
            elif resp.status_code == 429:
                raise ProviderRateLimitError(self._provider_id)
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
            raise ProviderUnavailableError(self._provider_id, "Could not connect to Ollama") from exc
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
            raise ProviderUnavailableError(self._provider_id, "Ollama request failed") from exc

        message = data.get("message", {})
        text = message.get("content", "")
        if not text and "response" in data:
            text = data["response"]

        if not text and not data.get("done", False):
            raise MalformedProviderResponseError(self._provider_id, "Empty response text")

        prompt_eval_count = data.get("prompt_eval_count", 0)
        eval_count = data.get("eval_count", 0)

        latency_ms = (time.perf_counter() - started) * 1000.0

        if prompt_eval_count > 0:
            source = TokenMeasurementSource.PROVIDER_REPORTED
            in_tok = prompt_eval_count
            out_tok = eval_count
            tot_tok = in_tok + out_tok
        else:
            family = "qwen" if "qwen" in model_name.lower() else "cl100k_base"
            counter = get_token_counter_for_model(model_name, family)
            source = counter.measurement_source
            in_tok = counter.count(request.user_prompt)
            out_tok = counter.count(text)
            tot_tok = in_tok + out_tok

        finish_reason = ModelFinishReason.STOP
        if data.get("done_reason") == "length":
            finish_reason = ModelFinishReason.LENGTH

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
            raw_usage={"prompt_eval_count": prompt_eval_count, "eval_count": eval_count},
            request_id=f"ollama-{int(started * 1000)}",
        )
