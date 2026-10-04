"""Native text adapters for current OpenAI, Claude, and Gemini HTTP APIs.

Only fixed protocol fields leave this module in errors and telemetry. Secrets are
looked up at request time and never kept in provider configuration or results.
"""

from __future__ import annotations

import json
import os
import re
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


def _count(value: Any) -> int | None:
    return value if type(value) is int and value >= 0 else None


class NativeRemoteProvider:
    provider_id: str = ""
    default_url: str = ""

    def __init__(
        self, api_key_env: str, default_model: str = "", client: httpx.AsyncClient | None = None
    ) -> None:
        self.api_key_env = api_key_env
        self.default_model = default_model
        self._client = client

    @property
    def is_local(self) -> bool:
        return False

    @property
    def credential_available(self) -> bool:
        return bool(os.environ.get(self.api_key_env, "").strip())

    def _headers(self) -> dict[str, str]:
        key = os.environ.get(self.api_key_env, "").strip()
        if not key:
            raise ProviderAuthenticationError(self.provider_id, "Credential unavailable")
        if self.provider_id == "gemini":
            return {"x-goog-api-key": key, "Content-Type": "application/json"}
        if self.provider_id == "anthropic":
            return {
                "Authorization": f"Bearer {key}",
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            }
        return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    async def _request(
        self, method: str, path: str, *, timeout: float, payload: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        headers = self._headers()
        url = self.default_url + path
        try:
            if self._client is None:
                async with httpx.AsyncClient(
                    timeout=timeout, trust_env=False, follow_redirects=False
                ) as client:
                    response = await client.request(method, url, headers=headers, json=payload)
            else:
                response = await self._client.request(
                    method, url, headers=headers, json=payload, timeout=timeout
                )
        except httpx.TimeoutException:
            raise ProviderTimeoutError(self.provider_id, timeout) from None
        except httpx.RequestError:
            raise ProviderUnavailableError(self.provider_id, "Connection failed") from None
        status = response.status_code
        if status in (401, 403):
            raise ProviderAuthenticationError(self.provider_id)
        if status == 404:
            raise ModelUnavailableError("requested model", self.provider_id)
        if status == 429:
            retry = response.headers.get("retry-after", "")
            delay: float | None
            try:
                delay = float(retry)
                if not 0 <= delay <= 3600:
                    delay = None
            except ValueError:
                delay = None
            raise ProviderRateLimitError(self.provider_id, retry_after=delay)
        if status >= 500 or 300 <= status < 400:
            raise ProviderUnavailableError(self.provider_id, f"HTTP {status}")
        if status != 200:
            raise MalformedProviderResponseError(self.provider_id, f"HTTP {status}")
        try:
            body = response.json()
        except (ValueError, json.JSONDecodeError):
            raise MalformedProviderResponseError(self.provider_id, "Invalid JSON") from None
        if not isinstance(body, dict):
            raise MalformedProviderResponseError(self.provider_id, "Invalid response object")
        return body

    async def health(self) -> bool:
        if not self.credential_available:
            return False
        try:
            return bool(await self.list_models())
        except Exception:
            return False

    async def list_models(self) -> list[ModelCapabilities]:
        raise NotImplementedError

    def count_tokens(self, text: str, model: str) -> int:
        return get_token_counter_for_model(model).count(text)

    def _response(
        self,
        request: ModelRequest,
        text: str,
        usage: Any,
        finish: ModelFinishReason,
        request_id: Any,
        started: float,
        input_key: str,
        output_key: str,
        total_key: str,
        input_extra: int = 0,
    ) -> ModelResponse:
        model = request.model or self.default_model
        if not isinstance(text, str):
            raise MalformedProviderResponseError(self.provider_id, "Invalid text content")
        if not text and finish not in (ModelFinishReason.LENGTH, ModelFinishReason.CONTENT_FILTER):
            raise MalformedProviderResponseError(self.provider_id, "Empty text content")
        counter = get_token_counter_for_model(model)
        safe_usage: dict[str, int] | None = None
        if isinstance(usage, dict):
            in_raw, out_raw = _count(usage.get(input_key)), _count(usage.get(output_key))
            total_raw = _count(usage.get(total_key))
            if in_raw is not None and out_raw is not None:
                in_tok, out_tok = in_raw + input_extra, out_raw
                total = total_raw if total_raw is not None else in_tok + out_tok
                source = TokenMeasurementSource.PROVIDER_REPORTED
                safe_usage = {
                    "input_tokens": in_tok,
                    "output_tokens": out_tok,
                    "total_tokens": total,
                }
            else:
                in_tok = out_tok = total = -1
        else:
            in_tok = out_tok = total = -1
        if in_tok < 0:
            context = request.compiled_context.context_text if request.compiled_context else ""
            in_tok = counter.count(
                "\n".join((request.system_prompt or "", context, request.user_prompt))
            )
            out_tok = counter.count(text)
            total = in_tok + out_tok
            source = counter.measurement_source
        return ModelResponse(
            text=text,
            model_id=model,
            provider_id=self.provider_id,
            input_tokens=in_tok,
            output_tokens=out_tok,
            total_tokens=total,
            latency_ms=(time.perf_counter() - started) * 1000,
            finish_reason=finish,
            token_measurement_source=source,
            raw_usage=safe_usage,
            request_id=(
                request_id
                if isinstance(request_id, str)
                and re.fullmatch(r"(?:resp|msg|v1)_[A-Za-z0-9_-]{1,120}", request_id)
                else None
            ),
        )

    def _capabilities(
        self, model_id: str, *, context: int = 8192, output: int = 2048, display: str | None = None
    ) -> ModelCapabilities:
        return ModelCapabilities(
            provider_id=self.provider_id,
            model_id=model_id,
            display_name=display or model_id,
            context_window=context,
            max_output_tokens=output,
            local=False,
            enabled=True,
            supports_tools=False,
            supports_json=False,
            supports_vision=False,
            metadata={"capability_source": "unknown"},
        )


class OpenAIProvider(NativeRemoteProvider):
    provider_id = "openai"
    default_url = "https://api.openai.com/v1"

    async def list_models(self) -> list[ModelCapabilities]:
        data = await self._request("GET", "/models", timeout=5.0)
        rows = data.get("data")
        if not isinstance(rows, list):
            raise MalformedProviderResponseError(self.provider_id, "Invalid model inventory")
        return [
            self._capabilities(row["id"])
            for row in rows
            if isinstance(row, dict) and isinstance(row.get("id"), str) and row["id"]
        ]

    async def generate(self, request: ModelRequest) -> ModelResponse:
        started = time.perf_counter()
        model = request.model or self.default_model
        if not model:
            raise ModelUnavailableError("configured model", self.provider_id)
        context = request.compiled_context.context_text if request.compiled_context else ""
        instructions = "\n\n".join(
            part
            for part in (
                request.system_prompt,
                f"### Context Information:\n{context}" if context else "",
            )
            if part
        )
        payload: dict[str, Any] = {
            "model": model,
            "input": request.user_prompt,
            "max_output_tokens": request.max_output_tokens,
            "store": False,
        }
        if instructions:
            payload["instructions"] = instructions
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        data = await self._request(
            "POST", "/responses", timeout=request.timeout_seconds, payload=payload
        )
        status = data.get("status")
        if status not in (None, "completed", "incomplete"):
            raise ProviderUnavailableError(self.provider_id, "Response did not complete")
        output = data.get("output")
        if not isinstance(output, list):
            raise MalformedProviderResponseError(self.provider_id, "Invalid output blocks")
        parts: list[str] = []
        for block in output:
            if not isinstance(block, dict):
                raise MalformedProviderResponseError(self.provider_id, "Invalid output block")
            if block.get("type") != "message":
                continue
            content = block.get("content")
            if not isinstance(content, list):
                raise MalformedProviderResponseError(self.provider_id, "Invalid content blocks")
            for item in content:
                if not isinstance(item, dict):
                    raise MalformedProviderResponseError(self.provider_id, "Invalid content block")
                if item.get("type") == "output_text":
                    if not isinstance(item.get("text"), str):
                        raise MalformedProviderResponseError(
                            self.provider_id, "Invalid output text"
                        )
                    parts.append(item["text"])
        reason = data.get("incomplete_details")
        finish = (
            ModelFinishReason.LENGTH
            if status == "incomplete"
            and isinstance(reason, dict)
            and reason.get("reason") == "max_output_tokens"
            else ModelFinishReason.STOP
        )
        return self._response(
            request,
            "".join(parts),
            data.get("usage"),
            finish,
            data.get("id"),
            started,
            "input_tokens",
            "output_tokens",
            "total_tokens",
        )


class AnthropicProvider(NativeRemoteProvider):
    provider_id = "anthropic"
    default_url = "https://api.anthropic.com/v1"

    async def list_models(self) -> list[ModelCapabilities]:
        data = await self._request("GET", "/models?limit=1000", timeout=5.0)
        rows = data.get("data")
        if not isinstance(rows, list):
            raise MalformedProviderResponseError(self.provider_id, "Invalid model inventory")
        models = []
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"]:
                continue
            context = _count(row.get("max_input_tokens")) or 8192
            output = _count(row.get("max_tokens")) or 2048
            models.append(
                self._capabilities(
                    row["id"],
                    context=context,
                    output=output,
                    display=row.get("display_name")
                    if isinstance(row.get("display_name"), str)
                    else None,
                )
            )
        return models

    async def generate(self, request: ModelRequest) -> ModelResponse:
        started = time.perf_counter()
        model = request.model or self.default_model
        if not model:
            raise ModelUnavailableError("configured model", self.provider_id)
        context = request.compiled_context.context_text if request.compiled_context else ""
        system = "\n\n".join(
            part
            for part in (
                request.system_prompt,
                f"### Context Information:\n{context}" if context else "",
            )
            if part
        )
        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": request.user_prompt}],
            "max_tokens": request.max_output_tokens,
        }
        if system:
            payload["system"] = system
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        data = await self._request(
            "POST", "/messages", timeout=request.timeout_seconds, payload=payload
        )
        content = data.get("content")
        if not isinstance(content, list):
            raise MalformedProviderResponseError(self.provider_id, "Invalid content blocks")
        parts = []
        for block in content:
            if not isinstance(block, dict) or not isinstance(block.get("type"), str):
                raise MalformedProviderResponseError(self.provider_id, "Invalid content block")
            if block["type"] == "text":
                if not isinstance(block.get("text"), str):
                    raise MalformedProviderResponseError(self.provider_id, "Invalid text block")
                parts.append(block["text"])
        stop = data.get("stop_reason")
        finish = (
            ModelFinishReason.LENGTH
            if stop == "max_tokens"
            else ModelFinishReason.CONTENT_FILTER
            if stop == "refusal"
            else ModelFinishReason.STOP
        )
        usage = data.get("usage")
        extra = 0
        if isinstance(usage, dict):
            extra = (_count(usage.get("cache_creation_input_tokens")) or 0) + (
                _count(usage.get("cache_read_input_tokens")) or 0
            )
        return self._response(
            request,
            "".join(parts),
            usage,
            finish,
            data.get("id"),
            started,
            "input_tokens",
            "output_tokens",
            "total_tokens",
            input_extra=extra,
        )


class GeminiProvider(NativeRemoteProvider):
    provider_id = "gemini"
    default_url = "https://generativelanguage.googleapis.com"

    async def list_models(self) -> list[ModelCapabilities]:
        data = await self._request("GET", "/v1beta/models?pageSize=1000", timeout=5.0)
        rows = data.get("models")
        if not isinstance(rows, list):
            raise MalformedProviderResponseError(self.provider_id, "Invalid model inventory")
        models = []
        for row in rows:
            if (
                not isinstance(row, dict)
                or not isinstance(row.get("name"), str)
                or not row["name"].startswith("models/")
            ):
                continue
            actions = row.get("supportedGenerationMethods", [])
            if actions and (not isinstance(actions, list) or "generateContent" not in actions):
                continue
            model = row["name"].removeprefix("models/")
            models.append(
                self._capabilities(
                    model,
                    context=_count(row.get("inputTokenLimit")) or 8192,
                    output=_count(row.get("outputTokenLimit")) or 2048,
                    display=row.get("displayName")
                    if isinstance(row.get("displayName"), str)
                    else None,
                )
            )
        return models

    async def generate(self, request: ModelRequest) -> ModelResponse:
        started = time.perf_counter()
        model = request.model or self.default_model
        if not model:
            raise ModelUnavailableError("configured model", self.provider_id)
        context = request.compiled_context.context_text if request.compiled_context else ""
        system = "\n\n".join(
            part
            for part in (
                request.system_prompt,
                f"### Context Information:\n{context}" if context else "",
            )
            if part
        )
        config: dict[str, Any] = {"max_output_tokens": request.max_output_tokens}
        if request.temperature is not None:
            config["temperature"] = request.temperature
        payload: dict[str, Any] = {
            "model": model,
            "input": request.user_prompt,
            "generation_config": config,
            "store": False,
        }
        if system:
            payload["system_instruction"] = system
        data = await self._request(
            "POST", "/v1/interactions", timeout=request.timeout_seconds, payload=payload
        )
        if data.get("status") not in (None, "completed"):
            raise ProviderUnavailableError(self.provider_id, "Interaction did not complete")
        steps = data.get("steps")
        if not isinstance(steps, list):
            raise MalformedProviderResponseError(self.provider_id, "Invalid interaction steps")
        parts = []
        for step in steps:
            if not isinstance(step, dict):
                raise MalformedProviderResponseError(self.provider_id, "Invalid interaction step")
            if step.get("type") != "model_output":
                continue
            content = step.get("content")
            if not isinstance(content, list):
                raise MalformedProviderResponseError(self.provider_id, "Invalid output content")
            for item in content:
                if not isinstance(item, dict):
                    raise MalformedProviderResponseError(self.provider_id, "Invalid output item")
                if item.get("type") == "text":
                    if not isinstance(item.get("text"), str):
                        raise MalformedProviderResponseError(
                            self.provider_id, "Invalid output text"
                        )
                    parts.append(item["text"])
        return self._response(
            request,
            "".join(parts),
            data.get("usage"),
            ModelFinishReason.STOP,
            data.get("id"),
            started,
            "total_input_tokens",
            "total_output_tokens",
            "total_tokens",
        )
