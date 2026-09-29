"""Telemetry Query Service for metrics aggregation and dashboard reporting."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from contextos.core.models import ModelInvocationTelemetry, TelemetrySummary
from contextos.core.protocols import TelemetryRepository


class TelemetryQueryService:
    """Provides high-level analytical queries over recorded model invocations."""

    def __init__(self, telemetry_repo: TelemetryRepository) -> None:
        self._repo = telemetry_repo

    async def record(self, telemetry: ModelInvocationTelemetry) -> None:
        """Record an invocation directly."""
        await self._repo.record(telemetry)

    async def get(self, invocation_id: UUID) -> ModelInvocationTelemetry | None:
        return await self._repo.get(invocation_id)

    async def list_recent(self, limit: int = 50, model_id: str | None = None) -> list[ModelInvocationTelemetry]:
        return await self._repo.list_recent(limit=limit, model_id=model_id)

    async def summary_today(self) -> TelemetrySummary:
        """Aggregate telemetry for today (UTC start of day to now)."""
        now = datetime.now(timezone.utc)
        start_of_day = datetime(now.year, now.month, now.day, tzinfo=timezone.utc)
        return await self._repo.summary(start=start_of_day, end=now)

    async def summary_range(
        self,
        start: datetime | None = None,
        end: datetime | None = None,
        provider_id: str | None = None,
        model_id: str | None = None,
        success_only: bool = False,
    ) -> TelemetrySummary:
        """Aggregate telemetry over an arbitrary time window and filters."""
        return await self._repo.summary(
            start=start, end=end, provider_id=provider_id, model_id=model_id,
            success_only=success_only,
        )

    async def by_provider(self, provider_id: str) -> TelemetrySummary:
        """Aggregate telemetry filtered by provider."""
        return await self._repo.summary(provider_id=provider_id)

    async def by_model(self, model_id: str) -> TelemetrySummary:
        """Aggregate telemetry filtered by model."""
        return await self._repo.summary(model_id=model_id)

    async def context_measurement_bases(self, model_id: str | None = None) -> list[dict[str, str]]:
        return await self._repo.context_measurement_bases(model_id)

    @staticmethod
    def format_terminal_mock(telemetry: ModelInvocationTelemetry) -> dict[str, Any]:
        """Produce a CLI-friendly data structure suitable for future Phase 12 terminal UI."""
        return {
            "provider": telemetry.provider_id,
            "model": telemetry.model_id,
            "is_local": telemetry.is_local,
            "candidate_context_tokens": telemetry.candidate_context_tokens,
            "compiled_context_tokens": telemetry.compiled_context_tokens,
            "context_tokens_avoided": telemetry.context_tokens_avoided,
            "reduction_ratio_percent": f"{telemetry.reduction_ratio * 100.0:.1f}%",
            "graph_expanded_memories": telemetry.graph_expanded_count,
            "temporal_filtered_memories": telemetry.temporal_filtered_count,
            "selected_memories": telemetry.selected_memory_count,
            "compiled_facts": telemetry.compiled_fact_count,
            "provider_input_tokens": telemetry.provider_input_tokens,
            "provider_output_tokens": telemetry.provider_output_tokens,
            "retrieval_ms": round(telemetry.retrieval_ms, 2),
            "optimization_ms": round(telemetry.optimization_ms, 2),
            "compilation_ms": round(telemetry.compilation_ms, 2),
            "routing_ms": round(telemetry.routing_ms, 2),
            "provider_ms": round(telemetry.provider_latency_ms, 2),
            "end_to_end_ms": round(telemetry.end_to_end_ms, 2),
            "token_measurement_source": telemetry.token_measurement_source.value,
            "routing_policy": telemetry.routing_policy.value,
            "routing_reason": telemetry.routing_reason,
            "fallback_used": telemetry.fallback_used,
        }
