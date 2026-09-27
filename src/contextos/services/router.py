"""Deterministic Model Router for ContextOS."""

from __future__ import annotations

import logging
import time
from typing import Any

from contextos.core.enums import RoutingPolicy, TokenMeasurementSource
from contextos.core.exceptions import (
    ContextWindowExceededError,
    ModelUnavailableError,
    ProviderUnavailableError,
    RoutingFailureError,
)
from contextos.core.models import (
    ModelCapabilities,
    ModelRequest,
    RouteDecision,
)
from contextos.core.protocols import ModelProvider
from contextos.services.token_counter import TokenCounter, get_token_counter_for_model

logger = logging.getLogger(__name__)


class DeterministicModelRouter:
    """Selects provider and model deterministically based on policy and constraints."""

    def __init__(
        self,
        default_provider_id: str = "fake",
        default_model_id: str = "fake-default",
        default_policy: RoutingPolicy = RoutingPolicy.LOCAL_FIRST,
    ) -> None:
        self._default_provider_id = default_provider_id
        self._default_model_id = default_model_id
        self._default_policy = default_policy

    async def route(
        self,
        request: ModelRequest,
        providers: dict[str, ModelProvider],
        policy: RoutingPolicy | None = None,
    ) -> RouteDecision:
        """Route request to the most appropriate healthy provider and model."""
        start_time = time.perf_counter()
        chosen_policy = policy or request.routing_policy or self._default_policy

        if not providers:
            raise RoutingFailureError(chosen_policy.value, "No providers registered in system")

        candidates_evaluated: list[str] = []

        # -------------------------------------------------------------------
        # Policy: EXPLICIT
        # -------------------------------------------------------------------
        if chosen_policy == RoutingPolicy.EXPLICIT or (request.provider and chosen_policy != RoutingPolicy.FIXED_DEFAULT):
            prov_id = request.provider or self._default_provider_id
            if prov_id not in providers:
                raise ProviderUnavailableError(prov_id, f"Provider '{prov_id}' not found in registry")

            provider = providers[prov_id]
            is_healthy = await provider.health()
            if not is_healthy:
                raise ProviderUnavailableError(prov_id, f"Provider '{prov_id}' is unhealthy")

            models = await provider.list_models()
            target_model_id = request.model or (models[0].model_id if models else self._default_model_id)
            candidates_evaluated.append(f"{prov_id}/{target_model_id}")

            matched_model = next((m for m in models if m.model_id == target_model_id), None)
            if not matched_model:
                raise ModelUnavailableError(target_model_id, prov_id)

            self._validate_context_window(request, matched_model)

            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return RouteDecision(
                policy=chosen_policy,
                reason="Explicit provider/model selection requested and verified",
                candidates_evaluated=candidates_evaluated,
                selected_provider=prov_id,
                selected_model=target_model_id,
                fallback_used=False,
                routing_latency_ms=elapsed_ms,
            )

        # -------------------------------------------------------------------
        # Policy: FIXED_DEFAULT
        # -------------------------------------------------------------------
        if chosen_policy == RoutingPolicy.FIXED_DEFAULT:
            prov_id = self._default_provider_id
            if prov_id not in providers:
                raise ProviderUnavailableError(prov_id, "Default provider not registered")

            provider = providers[prov_id]
            if not await provider.health():
                raise ProviderUnavailableError(prov_id, "Default provider is unhealthy")

            models = await provider.list_models()
            matched_model = next((m for m in models if m.model_id == self._default_model_id), None)
            if not matched_model and models:
                matched_model = models[0]
            if not matched_model:
                raise ModelUnavailableError(self._default_model_id, prov_id)

            candidates_evaluated.append(f"{prov_id}/{matched_model.model_id}")
            self._validate_context_window(request, matched_model)

            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return RouteDecision(
                policy=chosen_policy,
                reason="Fixed default policy selected configured default model",
                candidates_evaluated=candidates_evaluated,
                selected_provider=prov_id,
                selected_model=matched_model.model_id,
                fallback_used=False,
                routing_latency_ms=elapsed_ms,
            )

        # -------------------------------------------------------------------
        # Policy: LOCAL_FIRST
        # -------------------------------------------------------------------
        if chosen_policy == RoutingPolicy.LOCAL_FIRST:
            local_providers = [p for p in providers.values() if p.is_local]
            selected_local: tuple[ModelProvider, ModelCapabilities] | None = None
            initial_provider_tried: str | None = None

            for prov in local_providers:
                initial_provider_tried = prov.provider_id
                try:
                    if not await prov.health():
                        candidates_evaluated.append(f"{prov.provider_id} (unhealthy)")
                        continue
                    prov_models = await prov.list_models()
                except Exception:
                    candidates_evaluated.append(f"{prov.provider_id} (health check failed)")
                    continue

                for model in prov_models:
                    candidates_evaluated.append(f"{prov.provider_id}/{model.model_id}")
                    if self._fits_context_and_capabilities(request, model):
                        selected_local = (prov, model)
                        break
                if selected_local:
                    break

            if selected_local:
                prov, model = selected_local
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                return RouteDecision(
                    policy=chosen_policy,
                    reason="Local-first policy found healthy local provider and model fitting context window",
                    candidates_evaluated=candidates_evaluated,
                    selected_provider=prov.provider_id,
                    selected_model=model.model_id,
                    fallback_used=False,
                    routing_latency_ms=elapsed_ms,
                )

            # Local provider not found or unhealthy
            if not request.allow_fallback:
                raise ProviderUnavailableError(
                    initial_provider_tried or "local",
                    "Local providers unavailable and fallback not allowed by request policy",
                )

            # Fallback to remote provider
            remote_providers = [p for p in providers.values() if not p.is_local]
            selected_remote: tuple[ModelProvider, ModelCapabilities] | None = None
            for prov in remote_providers:
                if not await prov.health():
                    candidates_evaluated.append(f"{prov.provider_id} (unhealthy)")
                    continue
                prov_models = await prov.list_models()
                for model in prov_models:
                    candidates_evaluated.append(f"{prov.provider_id}/{model.model_id}")
                    if self._fits_context_and_capabilities(request, model):
                        selected_remote = (prov, model)
                        break
                if selected_remote:
                    break

            if selected_remote:
                prov, model = selected_remote
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                return RouteDecision(
                    policy=chosen_policy,
                    reason="Local provider unavailable; successfully fell back to remote provider",
                    candidates_evaluated=candidates_evaluated,
                    selected_provider=prov.provider_id,
                    selected_model=model.model_id,
                    fallback_used=True,
                    initial_provider=initial_provider_tried,
                    fallback_reason="Local provider unavailable or failed health check",
                    routing_latency_ms=elapsed_ms,
                )

            raise RoutingFailureError(
                chosen_policy.value,
                "Neither local nor remote fallback providers could satisfy the request",
            )

        # -------------------------------------------------------------------
        # Policy: CAPABILITY_AWARE
        # -------------------------------------------------------------------
        if chosen_policy == RoutingPolicy.CAPABILITY_AWARE:
            all_candidates: list[tuple[ModelProvider, ModelCapabilities]] = []
            for prov in providers.values():
                if not await prov.health():
                    continue
                models = await prov.list_models()
                for model in models:
                    candidates_evaluated.append(f"{prov.provider_id}/{model.model_id}")
                    if self._fits_context_and_capabilities(request, model):
                        all_candidates.append((prov, model))

            if not all_candidates:
                raise RoutingFailureError(
                    chosen_policy.value,
                    f"No model satisfied required capabilities: {request.required_capabilities}",
                )

            # Prefer local if available, then by largest context window
            all_candidates.sort(key=lambda item: (not item[0].is_local, -item[1].context_window))
            best_prov, best_model = all_candidates[0]

            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return RouteDecision(
                policy=chosen_policy,
                reason="Capability-aware policy matched required capabilities and context fit",
                candidates_evaluated=candidates_evaluated,
                selected_provider=best_prov.provider_id,
                selected_model=best_model.model_id,
                fallback_used=False,
                routing_latency_ms=elapsed_ms,
            )

        raise RoutingFailureError(str(chosen_policy), f"Unsupported routing policy '{chosen_policy}'")

    def _compute_safety_margin(self, counter: TokenCounter) -> int:
        """Conservative buffer for chat framing (roles, formatting) and approximation variance."""
        base_framing = 16
        if counter.measurement_source == TokenMeasurementSource.APPROXIMATED:
            return base_framing + 32
        return base_framing

    def _fits_context_and_capabilities(
        self, request: ModelRequest, model: ModelCapabilities
    ) -> bool:
        """Check capability constraints and context window fit."""
        if not model.enabled:
            return False

        # Capability checks
        if "tools" in request.required_capabilities and not model.supports_tools:
            return False
        if "json" in request.required_capabilities and not model.supports_json:
            return False
        if "vision" in request.required_capabilities and not model.supports_vision:
            return False

        # Context window fit with safety margin for framing/approximation
        context_text = request.compiled_context.context_text if request.compiled_context else ""
        counter = get_token_counter_for_model(model.model_id, model.tokenizer_family)
        prompt_tokens = counter.count(request.user_prompt)
        if request.system_prompt:
            prompt_tokens += counter.count(request.system_prompt)
        context_tokens = counter.count(context_text) if context_text else 0
        reserved_output = request.max_output_tokens or 1024
        margin = self._compute_safety_margin(counter)

        total_needed = prompt_tokens + context_tokens + reserved_output + margin
        return total_needed <= model.context_window

    def _validate_context_window(
        self, request: ModelRequest, model: ModelCapabilities
    ) -> None:
        """Ensure total needed tokens fit within the model context window.

        Raises ContextWindowExceededError if it does not fit.
        """
        context_text = request.compiled_context.context_text if request.compiled_context else ""
        counter = get_token_counter_for_model(model.model_id, model.tokenizer_family)
        prompt_tokens = counter.count(request.user_prompt)
        if request.system_prompt:
            prompt_tokens += counter.count(request.system_prompt)
        context_tokens = counter.count(context_text) if context_text else 0
        reserved_output = request.max_output_tokens or 1024
        margin = self._compute_safety_margin(counter)

        total_needed = prompt_tokens + context_tokens + reserved_output + margin
        if total_needed > model.context_window:
            raise ContextWindowExceededError(
                model_id=model.model_id,
                required_tokens=total_needed,
                context_window=model.context_window,
                prompt_tokens=prompt_tokens,
                compiled_context_tokens=context_tokens,
                reserved_output_tokens=reserved_output,
            )
