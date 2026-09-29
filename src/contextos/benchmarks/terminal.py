"""Reproducible local Phase 12 latency measurements; no token-saving claims."""

from __future__ import annotations

import asyncio
import json
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from httpx import ASGITransport, AsyncClient

from contextos.api.server import create_app, set_services
from contextos.config.settings import Settings
from contextos.daemon.wiring import wire_services


async def measure() -> dict[str, dict[str, float]]:
    with tempfile.TemporaryDirectory(prefix="contextos-terminal-bench-") as folder:
        settings = Settings(daemon={"data_dir": Path(folder)}, embedding={"model": "deterministic"})
        services = await wire_services(settings)
        set_services(services)
        metrics: dict[str, list[float]] = {name: [] for name in (
            "cli_startup_ms", "dashboard_refresh_ms", "telemetry_query_ms",
            "memory_search_ms", "monitoring_request_overhead_ms",
        )}
        try:
            async with AsyncClient(transport=ASGITransport(app=create_app()), base_url="http://localhost") as http:
                await http.post("/api/v1/remember", json={"text": "I prefer concise technical documentation."})
                for _ in range(5):
                    started = time.perf_counter()
                    result = subprocess.run([sys.executable, "-m", "contextos", "version"],
                                            capture_output=True, timeout=15, check=True)
                    assert result.returncode == 0
                    metrics["cli_startup_ms"].append((time.perf_counter() - started) * 1000)
                    started = time.perf_counter()
                    await http.get("/api/v1/dashboard")
                    metrics["dashboard_refresh_ms"].append((time.perf_counter() - started) * 1000)
                    started = time.perf_counter()
                    await http.get("/api/v1/telemetry/summary")
                    metrics["telemetry_query_ms"].append((time.perf_counter() - started) * 1000)
                    started = time.perf_counter()
                    await http.post("/api/v1/retrieve", json={"query": "technical documentation"})
                    metrics["memory_search_ms"].append((time.perf_counter() - started) * 1000)
                # Monitor overhead is its polling request relative to a bare telemetry query.
                metrics["monitoring_request_overhead_ms"] = [
                    max(0.0, dashboard - telemetry) for dashboard, telemetry in zip(
                        metrics["dashboard_refresh_ms"], metrics["telemetry_query_ms"]
                    )
                ]
        finally:
            await services["database"].close()
    return {key: {"median_ms": round(statistics.median(values), 3),
                  "max_ms": round(max(values), 3), "samples": len(values)}
            for key, values in metrics.items()}


if __name__ == "__main__":
    print(json.dumps(asyncio.run(measure()), indent=2))
