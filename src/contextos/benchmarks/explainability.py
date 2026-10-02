"""LOCAL DEVELOPMENT SYNTHETIC EXPLAINABILITY BENCHMARK."""

from __future__ import annotations

import asyncio
import json
import statistics
import tempfile
import time
from pathlib import Path

from contextos.config.settings import Settings
from contextos.core.models import CompilationConfig, ContextBudget, RetrievalQuery
from contextos.daemon.wiring import wire_services
from contextos.services.explainability import ExplainabilityService, ExplanationRequest
from contextos.core.models import Memory
from contextos.core.enums import MemoryStatus


def _p95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]


async def measure() -> dict:
    report: dict[str, object] = {
        "label": "LOCAL DEVELOPMENT SYNTHETIC EXPLAINABILITY BENCHMARK",
        "iterations": 5,
        "datasets": {},
    }
    for size in (100, 1000):
        with tempfile.TemporaryDirectory(prefix=f"contextos-explain-{size}-") as folder:
            services = await wire_services(Settings(
                daemon={"data_dir": Path(folder)}, embedding={"model": "deterministic"}
            ))
            try:
                for index in range(size):
                    await services["memory_repo"].create(Memory(
                        content=f"I use Python automation tool number {index} for local project task {index % 31}.",
                        status=MemoryStatus.ACTIVE,
                        source_type="benchmark",
                    ))
                await services["retrieval_index"].ensure_current()
                explanation_service = ExplainabilityService(services)
                retrieval_ms: list[float] = []
                full_explain_ms: list[float] = []
                compile_ms: list[float] = []
                trace_bytes: list[int] = []
                candidate_counts: list[int] = []
                trace_overhead_ms: list[float] = []
                measured_pipeline_ms: list[float] = []
                for _ in range(5):
                    query = "Python automation local project"
                    started = time.perf_counter()
                    retrieved = await services["retrieval"].retrieve(
                        RetrievalQuery(text=query, k=25)
                    )
                    retrieval_ms.append((time.perf_counter() - started) * 1000)
                    started = time.perf_counter()
                    explained = await explanation_service.explain(ExplanationRequest(
                        query=query, budget=1000, limit=25, graph=False,
                    ))
                    full_explain_ms.append((time.perf_counter() - started) * 1000)
                    candidate_counts.append(len(explained.candidates))
                    trace_bytes.append(len(explained.model_dump_json().encode("utf-8")))
                    trace_overhead_ms.append(explained.explanation_overhead_ms)
                    measured_pipeline_ms.append(explained.measured_pipeline_ms)

                    started = time.perf_counter()
                    compile_retrieved = await services["retrieval"].retrieve(RetrievalQuery(text=query, k=25))
                    selection = services["optimizer"].optimize(query, compile_retrieved.memories,
                                                                 ContextBudget(max_tokens=1000))
                    await services["compilation"].compile(
                        query, selection, CompilationConfig(budget=1000)
                    )
                    compile_ms.append((time.perf_counter() - started) * 1000)
                baseline = statistics.mean(compile_ms)
                explained_mean = statistics.mean(full_explain_ms)
                report["datasets"][str(size)] = {
                    "mean_ms": round(explained_mean, 3),
                    "median_ms": round(statistics.median(full_explain_ms), 3),
                    "p95_ms": round(_p95(full_explain_ms), 3),
                    "retrieval_without_explanation_mean_ms": round(statistics.mean(retrieval_ms), 3),
                    "compile_without_explanation_mean_ms": round(baseline, 3),
                    "compile_with_explanation_mean_ms": round(explained_mean, 3),
                    "measured_pipeline_mean_ms": round(statistics.mean(measured_pipeline_ms), 3),
                    "explanation_overhead_ms": round(statistics.mean(trace_overhead_ms), 3),
                    "explanation_overhead_percent": round(statistics.mean(trace_overhead_ms) / statistics.mean(measured_pipeline_ms) * 100, 2) if statistics.mean(measured_pipeline_ms) else 0.0,
                    "trace_size_bytes_mean": round(statistics.mean(trace_bytes)),
                    "candidate_count_mean": round(statistics.mean(candidate_counts), 2),
                    "samples": 5,
                }
            finally:
                await services["database"].close()
    return report


def main() -> None:
    print(json.dumps(asyncio.run(measure()), indent=2))


if __name__ == "__main__":
    main()
