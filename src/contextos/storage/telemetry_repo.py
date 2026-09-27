"""SQLite persistence for model invocation telemetry."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from typing import Any
from uuid import UUID

import aiosqlite

from contextos.core.enums import ModelFinishReason, RoutingPolicy, TokenMeasurementSource
from contextos.core.models import ModelInvocationTelemetry, TelemetrySummary

logger = logging.getLogger(__name__)

_SECRET_PATTERNS = [
    re.compile(r"Bearer\s+[A-Za-z0-9_\-\.~+/]+=*", re.IGNORECASE),
    re.compile(r"sk-[A-Za-z0-9_\-]{8,}", re.IGNORECASE),
    re.compile(r"gh[pousr]-[A-Za-z0-9_]{16,}", re.IGNORECASE),
    re.compile(r"(?:api[-_]?key|secret|password|token)\s*[:=]\s*['\"]?[A-Za-z0-9_\-\.~+/]+['\"]?", re.IGNORECASE),
]

_SENSITIVE_KEY_SUBSTRINGS = {
    "api_key",
    "apikey",
    "secret",
    "password",
    "token",
    "auth",
    "authorization",
    "raw_prompt",
    "raw_response",
    "cookie",
    "credential",
}


def sanitize_telemetry_metadata(val: Any) -> Any:
    """Recursively sanitize metadata dicts/lists to strip secrets and raw payloads."""
    if isinstance(val, dict):
        cleaned: dict[str, Any] = {}
        for k, v in val.items():
            key_lower = str(k).lower()
            if any(s in key_lower for s in _SENSITIVE_KEY_SUBSTRINGS):
                cleaned[str(k)] = "[REDACTED]"
            else:
                cleaned[str(k)] = sanitize_telemetry_metadata(v)
        return cleaned
    elif isinstance(val, list):
        return [sanitize_telemetry_metadata(item) for item in val]
    elif isinstance(val, str):
        sanitized_str = val
        for pat in _SECRET_PATTERNS:
            sanitized_str = pat.sub("[REDACTED_SECRET]", sanitized_str)
        return sanitized_str
    return val



class SqliteTelemetryRepository:
    """Stores and queries model invocation metrics in SQLite."""

    def __init__(self, conn: aiosqlite.Connection) -> None:
        self._conn = conn

    async def record(self, telemetry: ModelInvocationTelemetry) -> None:
        """Persist a single model invocation telemetry record."""
        row_id = str(telemetry.invocation_id)
        safe_meta = sanitize_telemetry_metadata(telemetry.metadata)
        metadata_json = json.dumps(safe_meta)
        timestamp_str = telemetry.timestamp.isoformat()

        sql = """
        INSERT INTO model_invocations (
            id,
            invocation_id,
            session_id,
            provider_id,
            model_id,
            is_local,
            timestamp,
            candidate_context_tokens,
            retrieved_context_tokens,
            optimized_context_tokens,
            compiled_context_tokens,
            prompt_tokens_before_context,
            final_input_tokens,
            provider_input_tokens,
            provider_output_tokens,
            provider_total_tokens,
            token_measurement_source,
            context_tokens_avoided,
            reduction_ratio,
            lexical_candidate_count,
            dense_candidate_count,
            hybrid_candidate_count,
            graph_expanded_count,
            temporal_filtered_count,
            selected_memory_count,
            compiled_fact_count,
            retrieval_ms,
            optimization_ms,
            compilation_ms,
            routing_ms,
            token_counting_ms,
            provider_latency_ms,
            end_to_end_ms,
            routing_policy,
            routing_reason,
            selected_provider,
            selected_model,
            fallback_used,
            fallback_reason,
            finish_reason,
            status,
            error_code,
            metadata
        ) VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?
        )
        """
        params = (
            row_id,
            str(telemetry.invocation_id),
            telemetry.session_id,
            telemetry.provider_id,
            telemetry.model_id,
            1 if telemetry.is_local else 0,
            timestamp_str,
            telemetry.candidate_context_tokens,
            telemetry.retrieved_context_tokens,
            telemetry.optimized_context_tokens,
            telemetry.compiled_context_tokens,
            telemetry.prompt_tokens_before_context,
            telemetry.final_input_tokens,
            telemetry.provider_input_tokens,
            telemetry.provider_output_tokens,
            telemetry.provider_total_tokens,
            telemetry.token_measurement_source.value,
            telemetry.context_tokens_avoided,
            telemetry.reduction_ratio,
            telemetry.lexical_candidate_count,
            telemetry.dense_candidate_count,
            telemetry.hybrid_candidate_count,
            telemetry.graph_expanded_count,
            telemetry.temporal_filtered_count,
            telemetry.selected_memory_count,
            telemetry.compiled_fact_count,
            telemetry.retrieval_ms,
            telemetry.optimization_ms,
            telemetry.compilation_ms,
            telemetry.routing_ms,
            telemetry.token_counting_ms,
            telemetry.provider_latency_ms,
            telemetry.end_to_end_ms,
            telemetry.routing_policy.value,
            telemetry.routing_reason,
            telemetry.selected_provider,
            telemetry.selected_model,
            1 if telemetry.fallback_used else 0,
            telemetry.fallback_reason,
            telemetry.finish_reason.value,
            telemetry.status,
            telemetry.error_code,
            metadata_json,
        )
        await self._conn.execute(sql, params)
        await self._conn.commit()

    async def get(self, invocation_id: UUID) -> ModelInvocationTelemetry | None:
        """Retrieve a telemetry record by invocation UUID."""
        sql = "SELECT * FROM model_invocations WHERE invocation_id = ?"
        async with self._conn.execute(sql, (str(invocation_id),)) as cursor:
            row = await cursor.fetchone()
            if row is None:
                return None
            return self._row_to_model(row)

    async def list_recent(self, limit: int = 50) -> list[ModelInvocationTelemetry]:
        """List recent invocations in descending chronological order."""
        sql = "SELECT * FROM model_invocations ORDER BY timestamp DESC LIMIT ?"
        async with self._conn.execute(sql, (limit,)) as cursor:
            rows = await cursor.fetchall()
            return [self._row_to_model(r) for r in rows]

    async def count(self) -> int:
        """Count total recorded invocations."""
        sql = "SELECT COUNT(*) FROM model_invocations"
        async with self._conn.execute(sql) as cursor:
            row = await cursor.fetchone()
            return row[0] if row else 0

    async def summary(
        self,
        start: datetime | None = None,
        end: datetime | None = None,
        provider_id: str | None = None,
        model_id: str | None = None,
    ) -> TelemetrySummary:
        """Compute aggregated token usage, avoidance, and latency statistics."""
        conditions: list[str] = []
        params: list[Any] = []

        if start is not None:
            conditions.append("timestamp >= ?")
            params.append(start.isoformat())
        if end is not None:
            conditions.append("timestamp <= ?")
            params.append(end.isoformat())
        if provider_id is not None:
            conditions.append("provider_id = ?")
            params.append(provider_id)
        if model_id is not None:
            conditions.append("model_id = ?")
            params.append(model_id)

        where_clause = f" WHERE {' AND '.join(conditions)}" if conditions else ""

        agg_sql = f"""
        SELECT
            COUNT(*),
            COALESCE(SUM(provider_input_tokens), 0),
            COALESCE(SUM(provider_output_tokens), 0),
            COALESCE(SUM(context_tokens_avoided), 0),
            COALESCE(AVG(reduction_ratio), 0.0),
            COALESCE(AVG(provider_latency_ms), 0.0),
            COALESCE(SUM(CASE WHEN is_local = 1 THEN 1 ELSE 0 END), 0),
            COALESCE(SUM(CASE WHEN is_local = 0 THEN 1 ELSE 0 END), 0),
            COALESCE(SUM(candidate_context_tokens), 0)
        FROM model_invocations{where_clause}
        """

        async with self._conn.execute(agg_sql, params) as cursor:
            row = await cursor.fetchone()

        total_invocations = row[0] if row else 0
        total_input_tokens = row[1] if row else 0
        total_output_tokens = row[2] if row else 0
        total_tokens_avoided = row[3] if row else 0
        average_reduction_ratio = float(row[4]) if row else 0.0
        average_provider_latency_ms = float(row[5]) if row else 0.0
        local_invocations = row[6] if row else 0
        remote_invocations = row[7] if row else 0
        total_candidate_tokens = row[8] if row else 0
        weighted_reduction_ratio = (
            float(total_tokens_avoided / total_candidate_tokens)
            if total_candidate_tokens > 0
            else 0.0
        )

        # By provider breakdown
        by_provider: dict[str, Any] = {}
        prov_sql = f"""
        SELECT
            provider_id,
            COUNT(*),
            COALESCE(SUM(provider_input_tokens), 0),
            COALESCE(SUM(provider_output_tokens), 0),
            COALESCE(SUM(context_tokens_avoided), 0),
            COALESCE(AVG(reduction_ratio), 0.0),
            COALESCE(AVG(provider_latency_ms), 0.0),
            COALESCE(SUM(candidate_context_tokens), 0)
        FROM model_invocations{where_clause}
        GROUP BY provider_id
        """
        async with self._conn.execute(prov_sql, params) as cursor:
            async for p_row in cursor:
                p_avoided = p_row[4]
                p_cand = p_row[7]
                by_provider[p_row[0]] = {
                    "invocations": p_row[1],
                    "input_tokens": p_row[2],
                    "output_tokens": p_row[3],
                    "tokens_avoided": p_avoided,
                    "average_reduction_ratio": float(p_row[5]),
                    "weighted_reduction_ratio": float(p_avoided / p_cand) if p_cand > 0 else 0.0,
                    "average_latency_ms": float(p_row[6]),
                }

        # By model breakdown
        by_model: dict[str, Any] = {}
        model_sql = f"""
        SELECT
            model_id,
            COUNT(*),
            COALESCE(SUM(provider_input_tokens), 0),
            COALESCE(SUM(provider_output_tokens), 0),
            COALESCE(SUM(context_tokens_avoided), 0),
            COALESCE(AVG(reduction_ratio), 0.0),
            COALESCE(AVG(provider_latency_ms), 0.0),
            COALESCE(SUM(candidate_context_tokens), 0)
        FROM model_invocations{where_clause}
        GROUP BY model_id
        """
        async with self._conn.execute(model_sql, params) as cursor:
            async for m_row in cursor:
                m_avoided = m_row[4]
                m_cand = m_row[7]
                by_model[m_row[0]] = {
                    "invocations": m_row[1],
                    "input_tokens": m_row[2],
                    "output_tokens": m_row[3],
                    "tokens_avoided": m_avoided,
                    "average_reduction_ratio": float(m_row[5]),
                    "weighted_reduction_ratio": float(m_avoided / m_cand) if m_cand > 0 else 0.0,
                    "average_latency_ms": float(m_row[6]),
                }

        return TelemetrySummary(
            total_invocations=total_invocations,
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            total_tokens_avoided=total_tokens_avoided,
            average_reduction_ratio=average_reduction_ratio,
            weighted_reduction_ratio=weighted_reduction_ratio,
            average_provider_latency_ms=average_provider_latency_ms,
            local_invocations=local_invocations,
            remote_invocations=remote_invocations,
            by_provider=by_provider,
            by_model=by_model,
        )

    def _row_to_model(self, row: Any) -> ModelInvocationTelemetry:
        metadata_raw = row["metadata"] if "metadata" in row.keys() else "{}"
        try:
            metadata = json.loads(metadata_raw) if isinstance(metadata_raw, str) else metadata_raw
        except Exception:
            metadata = {}

        return ModelInvocationTelemetry(
            invocation_id=UUID(row["invocation_id"]),
            session_id=row["session_id"],
            provider_id=row["provider_id"],
            model_id=row["model_id"],
            is_local=bool(row["is_local"]),
            timestamp=datetime.fromisoformat(row["timestamp"]),
            candidate_context_tokens=row["candidate_context_tokens"],
            retrieved_context_tokens=row["retrieved_context_tokens"],
            optimized_context_tokens=row["optimized_context_tokens"],
            compiled_context_tokens=row["compiled_context_tokens"],
            prompt_tokens_before_context=row["prompt_tokens_before_context"],
            preflight_input_tokens=row["final_input_tokens"],
            final_input_tokens=row["final_input_tokens"],
            provider_input_tokens=row["provider_input_tokens"],
            provider_output_tokens=row["provider_output_tokens"],
            provider_total_tokens=row["provider_total_tokens"],
            token_measurement_source=TokenMeasurementSource(row["token_measurement_source"]),
            context_tokens_avoided=row["context_tokens_avoided"],
            reduction_ratio=float(row["reduction_ratio"]),
            lexical_candidate_count=row["lexical_candidate_count"],
            dense_candidate_count=row["dense_candidate_count"],
            hybrid_candidate_count=row["hybrid_candidate_count"],
            graph_expanded_count=row["graph_expanded_count"],
            temporal_filtered_count=row["temporal_filtered_count"],
            selected_memory_count=row["selected_memory_count"],
            compiled_fact_count=row["compiled_fact_count"],
            retrieval_ms=float(row["retrieval_ms"]),
            optimization_ms=float(row["optimization_ms"]),
            compilation_ms=float(row["compilation_ms"]),
            routing_ms=float(row["routing_ms"]),
            token_counting_ms=float(row["token_counting_ms"]),
            provider_latency_ms=float(row["provider_latency_ms"]),
            end_to_end_ms=float(row["end_to_end_ms"]),
            routing_policy=RoutingPolicy(row["routing_policy"]),
            routing_reason=row["routing_reason"],
            selected_provider=row["selected_provider"],
            selected_model=row["selected_model"],
            fallback_used=bool(row["fallback_used"]),
            fallback_reason=row["fallback_reason"],
            finish_reason=ModelFinishReason(row["finish_reason"]),
            status=row["status"],
            error_code=row["error_code"],
            metadata=metadata if isinstance(metadata, dict) else {},
        )
