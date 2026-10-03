"""API routes for model inference, capabilities, and telemetry."""

from __future__ import annotations

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

from contextos.api.server import get_service
from contextos.core.enums import RoutingPolicy
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

    query: str = Field(min_length=1)
    system_prompt: str | None = None
    provider: str | None = None
    model: str | None = None
    routing_policy: RoutingPolicy | None = None
    allow_fallback: bool = False
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_output_tokens: int = Field(default=1024, ge=1)
    timeout_seconds: float = Field(default=30.0, ge=0.5, le=600.0)
    retrieval_config: RetrievalConfig | None = None
    compilation_config: CompilationConfig | None = None
    session_id: str | None = None


@router.post("/ask", response_model=AskResult)
async def ask_model(request: AskApiRequest) -> AskResult:
    """Prepare context, route to chosen model, execute generation, and record telemetry."""
    model_service = get_service("model_service")
    return await model_service.ask(
        query=request.query,
        system_prompt=request.system_prompt,
        retrieval_config=request.retrieval_config,
        compilation_config=request.compilation_config,
        routing_policy=request.routing_policy,
        target_provider=request.provider,
        target_model=request.model,
        allow_fallback=request.allow_fallback,
        temperature=request.temperature,
        max_output_tokens=request.max_output_tokens,
        timeout_seconds=request.timeout_seconds,
        session_id=request.session_id,
    )


@router.get("/models", response_model=list[ModelCapabilities])
async def list_models() -> list[ModelCapabilities]:
    """List all models offered across registered providers."""
    models: list[ModelCapabilities] = await get_service("model_discovery").list_models(
        get_service("providers"),
    )
    return models


@router.get("/telemetry/summary", response_model=TelemetrySummary)
async def get_telemetry_summary(
    provider_id: str | None = Query(default=None),
    model_id: str | None = Query(default=None),
) -> TelemetrySummary:
    """Get aggregated invocation metrics."""
    telemetry_query = get_service("telemetry_query")
    return await telemetry_query.summary_range(provider_id=provider_id, model_id=model_id)
