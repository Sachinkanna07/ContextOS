"""Deterministic Fake Provider for offline testing and reproducible benchmarking."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from typing import Any

from contextos.core.enums import ModelFinishReason, TokenMeasurementSource
from contextos.core.exceptions import (
    ContextWindowExceededError,
    MalformedProviderResponseError,
    ModelUnavailableError,
    ProviderAuthenticationError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from contextos.core.models import ModelCapabilities, ModelRequest, ModelResponse
from contextos.services.token_counter import get_token_counter_for_model


class DeterministicFakeProvider:
    """Offline, deterministic mock provider for testing and evaluation.

    Provides exact, predictable responses and error simulation modes.
    """

    def __init__(
        self,
        provider_id: str = "fake",
        is_local: bool = True,
        models: list[ModelCapabilities] | None = None,
        fixed_response: str | None = None,
        response_generator: Callable[[ModelRequest], str] | None = None,
        simulated_latency_ms: float = 2.0,
        report_usage: bool = True,
    ) -> None:
        self._provider_id = provider_id
        self._is_local = is_local
        self._fixed_response = fixed_response
        self._response_generator = response_generator
        self._simulated_latency_ms = simulated_latency_ms
        self._report_usage = report_usage

        # Simulation error controls
        self.simulate_timeout: bool = False
        self.simulate_rate_limit: bool = False
        self.simulate_auth_error: bool = False
        self.simulate_context_overflow: bool = False
        self.simulate_unhealthy: bool = False
        self.simulate_malformed: bool = False
        self.simulate_unavailable_model: bool = False
        self.simulate_provider_failure: bool = False
        self.rate_limit_retry_after: float = 1.5

        # Default model inventory if none provided
        if models is None:
            self._models = [
                ModelCapabilities(
                    provider_id=self._provider_id,
                    model_id="fake-default",
                    display_name="Fake Default Model",
                    context_window=8192,
                    max_output_tokens=2048,
                    supports_tools=True,
                    supports_json=True,
                    supports_vision=False,
                    local=self._is_local,
                    tokenizer_family="cl100k_base",
                    enabled=True,
                ),
                ModelCapabilities(
                    provider_id=self._provider_id,
                    model_id="fake-local-qwen",
                    display_name="Fake Local Qwen-like",
                    context_window=4096,
                    max_output_tokens=1024,
                    supports_tools=False,
                    supports_json=True,
                    supports_vision=False,
                    local=True,
                    tokenizer_family="qwen",
                    enabled=True,
                ),
                ModelCapabilities(
                    provider_id=self._provider_id,
                    model_id="fake-cloud-claude",
                    display_name="Fake Cloud Claude-like",
                    context_window=32768,
                    max_output_tokens=4096,
                    supports_tools=True,
                    supports_json=True,
                    supports_vision=True,
                    local=False,
                    tokenizer_family="claude",
                    enabled=True,
                ),
            ]
        else:
            self._models = list(models)

    @property
    def provider_id(self) -> str:
        return self._provider_id

    @property
    def is_local(self) -> bool:
        return self._is_local

    async def list_models(self) -> list[ModelCapabilities]:
        if self.simulate_provider_failure:
            raise ProviderUnavailableError(self._provider_id, "Failed to list models")
        return [m for m in self._models if m.enabled]

    async def health(self) -> bool:
        if self.simulate_unhealthy or self.simulate_provider_failure:
            return False
        return True

    def count_tokens(self, text: str, model: str) -> int:
        counter = get_token_counter_for_model(model)
        return counter.count(text)

    async def generate(self, request: ModelRequest) -> ModelResponse:
        start_time = time.perf_counter()

        # Simulate latency
        if self._simulated_latency_ms > 0:
            await asyncio.sleep(self._simulated_latency_ms / 1000.0)

        # Trigger simulated error modes if set
        if self.simulate_timeout:
            raise ProviderTimeoutError(self._provider_id, request.timeout_seconds)

        if self.simulate_auth_error:
            raise ProviderAuthenticationError(self._provider_id, "Invalid API token")

        if self.simulate_rate_limit:
            raise ProviderRateLimitError(self._provider_id, retry_after=self.rate_limit_retry_after)

        if self.simulate_unhealthy or self.simulate_provider_failure:
            raise ProviderUnavailableError(self._provider_id, "Service unavailable or crashed")

        if self.simulate_malformed:
            raise MalformedProviderResponseError(self._provider_id, "Missing choices/response key")

        # Check model availability
        target_model = request.model or self._models[0].model_id
        if self.simulate_unavailable_model:
            raise ModelUnavailableError(target_model, self._provider_id)

        matching = [m for m in self._models if m.model_id == target_model and m.enabled]
        if not matching:
            raise ModelUnavailableError(target_model, self._provider_id)
        model_meta = matching[0]

        # Calculate input tokens
        context_text = request.compiled_context.context_text if request.compiled_context else ""
        full_input = ""
        if request.system_prompt:
            full_input += request.system_prompt + "\n"
        if context_text:
            full_input += context_text + "\n"
        full_input += request.user_prompt

        input_tokens = self.count_tokens(full_input, model_meta.model_id)
        reserved_output = request.max_output_tokens or 1024

        # Validate context window
        if self.simulate_context_overflow or (input_tokens + reserved_output > model_meta.context_window):
            raise ContextWindowExceededError(
                model_id=model_meta.model_id,
                required_tokens=input_tokens + reserved_output,
                context_window=model_meta.context_window,
                prompt_tokens=self.count_tokens(request.user_prompt, model_meta.model_id),
                compiled_context_tokens=self.count_tokens(context_text, model_meta.model_id) if context_text else 0,
                reserved_output_tokens=reserved_output,
            )

        # Generate response text
        if self._response_generator is not None:
            text = self._response_generator(request)
        elif self._fixed_response is not None:
            text = self._fixed_response
        else:
            text = f"Deterministic response to '{request.user_prompt}' with context length {len(context_text)}."

        output_tokens = self.count_tokens(text, model_meta.model_id)
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        if self._report_usage:
            measurement_source = TokenMeasurementSource.PROVIDER_REPORTED
            rep_input = input_tokens
            rep_output = output_tokens
            rep_total = input_tokens + output_tokens
        else:
            counter = get_token_counter_for_model(model_meta.model_id, model_meta.tokenizer_family)
            measurement_source = counter.measurement_source
            rep_input = 0
            rep_output = 0
            rep_total = 0

        return ModelResponse(
            text=text,
            model_id=model_meta.model_id,
            provider_id=self._provider_id,
            input_tokens=rep_input,
            output_tokens=rep_output,
            total_tokens=rep_total,
            latency_ms=latency_ms,
            finish_reason=ModelFinishReason.STOP,
            token_measurement_source=measurement_source,
            raw_usage={"fake_counter": True, "input": input_tokens, "output": output_tokens},
            request_id=f"fake-{int(time.time() * 1000)}",
        )
