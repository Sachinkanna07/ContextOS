"""Deterministic ContextOS MCP subprocess fixture; stdout is owned by the SDK."""

from __future__ import annotations

from contextos.core.models import RetrievalResult
from contextos.mcp.server import create_mcp_server


class _Retrieval:
    async def retrieve(self, request):
        return RetrievalResult(query=request.text)


class _Telemetry:
    async def summary_today(self):
        from contextos.core.models import TelemetrySummary
        return TelemetrySummary()


if __name__ == "__main__":
    create_mcp_server({"retrieval": _Retrieval(), "telemetry_query": _Telemetry()}).run(transport="stdio")
