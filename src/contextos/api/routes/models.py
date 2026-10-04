"""API routes for model inference, capabilities, and telemetry."""

from __future__ import annotations

import asyncio
import os
import re
from typing import cast

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field, field_validator

from contextos.api.server import get_service
from contextos.core.enums import RoutingPolicy
from contextos.core.exceptions import ProviderAuthenticationError
from contextos.core.models import (
    AskResult,
    CompilationConfig,
    ModelCapabilities,
    RetrievalConfig,
    TelemetrySummary,
)

router = APIRouter(tags=["models"])


class AskApiRequest(BaseModel):
    """Request payload for the high-level ContextOS ask endpoint."""

    query: str = Field(min_length=1, max_length=20000)
    system_prompt: str | None = None
    provider: str | None = Field(default=None, max_length=64)
    model: str | None = Field(default=None, max_length=256)
    routing_policy: RoutingPolicy | None = None
    allow_fallback: bool = False
    allow_remote: bool = False
    temperature: float | None = Field(default=None, ge=0.0, le=2.0, allow_inf_nan=False)
    max_output_tokens: int = Field(default=1024, ge=1, le=32768)
    timeout_seconds: float = Field(default=30.0, ge=0.5, le=600.0)
    retrieval_config: RetrievalConfig | None = None
    compilation_config: CompilationConfig | None = None
    session_id: str | None = Field(default=None, max_length=128)

    @field_validator("provider")
    @classmethod
    def valid_provider(cls, value: str | None) -> str | None:
        if value is not None and not re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", value):
            raise ValueError("Invalid provider identifier")
        return value

    @field_validator("model", "session_id")
    @classmethod
    def valid_label(cls, value: str | None) -> str | None:
        if value is not None and (not value.strip() or any(ord(ch) < 32 for ch in value)):
            raise ValueError("Invalid identifier")
        return value


@router.post("/ask", response_model=AskResult)
async def ask_model(request: AskApiRequest) -> AskResult:
    """Prepare context, route to chosen model, execute generation, and record telemetry."""
    model_service = get_service("model_service")
    return cast("AskResult", await model_service.ask(
        query=request.query,
        system_prompt=request.system_prompt,
        retrieval_config=request.retrieval_config,
        compilation_config=request.compilation_config,
        routing_policy=request.routing_policy,
        target_provider=request.provider,
        target_model=request.model,
        allow_fallback=request.allow_fallback,
        allow_remote=request.allow_remote,
        temperature=request.temperature,
        max_output_tokens=request.max_output_tokens,
        timeout_seconds=request.timeout_seconds,
        session_id=request.session_id,
    ))


@router.get("/models", response_model=list[ModelCapabilities])
async def list_models() -> list[ModelCapabilities]:
    """List all models offered across registered providers."""
    models: list[ModelCapabilities] = await get_service("model_discovery").list_models(
        get_service("providers"),
    )
    return models


@router.get("/models/providers")
async def provider_status() -> list[dict[str, object]]:
    """Safe provider availability snapshot; never serialize a key or URL."""
    configured = get_service("provider_settings")
    providers = get_service("providers")
    specs = {"ollama": (True, "", True), "openai_compatible": (True, "", True)}
    for name in ("openai", "anthropic", "gemini"):
        config = getattr(configured, name)
        specs[name] = (config.enabled, config.api_key_env, False)
    for name, config in configured.compatible.items():
        from urllib.parse import urlsplit
        host = (urlsplit(config.base_url).hostname or "").lower()
        specs[name] = (config.enabled, config.api_key_env,
                       host in {"localhost", "127.0.0.1", "::1"})

    async def inspect(name: str, enabled: bool, key_env: str,
                      configured_local: bool) -> dict[str, object]:
        provider = providers.get(name)
        local = provider.is_local if provider else configured_local
        credential = "n/a" if local or not key_env else (
            "present" if os.environ.get(key_env, "").strip() else "missing"
        )
        state = "disabled" if not enabled else "unconfigured" if provider is None else "unavailable"
        models = 0
        if provider is not None:
            try:
                inventory = await asyncio.wait_for(provider.list_models(), timeout=5.0)
                models = len([model for model in inventory if model.enabled])
                state = "healthy" if models else "empty"
            except ProviderAuthenticationError:
                state = "auth-error"
            except Exception:
                state = "unavailable"
        return {"provider": name, "enabled": enabled, "local": local,
                "credential": credential, "status": state, "models_found": models}

    return await asyncio.gather(*(inspect(name, enabled, key_env, local)
                                  for name, (enabled, key_env, local) in specs.items()))


@router.get("/telemetry/summary", response_model=TelemetrySummary)
async def get_telemetry_summary(
    provider_id: str | None = Query(default=None),
    model_id: str | None = Query(default=None),
) -> TelemetrySummary:
    """Get aggregated invocation metrics."""
    telemetry_query = get_service("telemetry_query")
    return cast(
        "TelemetrySummary",
        await telemetry_query.summary_range(provider_id=provider_id, model_id=model_id),
    )
