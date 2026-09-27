"""Deterministic Phase 9 benchmark: Model routing, multi-model token comparison, and telemetry overhead."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from contextos.core.enums import (
    MemoryStatus,
    ModelFinishReason,
    RoutingPolicy,
    TokenMeasurementSource,
)
from contextos.core.models import (
    CompilationConfig,
    ContextBudget,
    Memory,
    ModelCapabilities,
    ModelInvocationTelemetry,
    ModelRequest,
    ScoredMemory,
)
from contextos.providers.fake import DeterministicFakeProvider
from contextos.services.compilation import QueryAwareContextCompiler
from contextos.services.optimization import MemoryContextOptimizer
from contextos.services.router import DeterministicModelRouter
from contextos.services.token_counter import (
    ClaudeProfileTokenCounter,
    QwenProfileTokenCounter,
    TiktokenCounter,
    get_token_counter_for_model,
    recount_cross_model,
)
from contextos.storage.database import Database
from contextos.storage.telemetry_repo import SqliteTelemetryRepository


@dataclass(frozen=True)
class ModelProfileResult:
    profile_name: str
    target_token_estimate: int
    measurement_source: str
    reduction_ratio: float
    context_tokens_avoided: int


@dataclass(frozen=True)
class Phase9BenchmarkReport:
    candidate_context_tokens: int
    compiled_context_tokens: int
    profiles: dict[str, ModelProfileResult]
    router_overhead_ms: float
    telemetry_overhead_ms: float
    token_counting_overhead_ms: float


def benchmark_memories() -> list[ScoredMemory]:
    """Diverse benchmark corpus containing code, technical preferences, and project facts."""
    texts = [
        "User currently uses Ollama for local model inference with llama3.2.",
        "User prefers concise technical responses with typed Python 3.12+ syntax and pydantic models.",
        "Project Atlas architecture uses SQLite with WAL mode, graph projection, and reciprocal rank fusion.",
        "User's machine is an M2 Max with 64GB unified memory running macOS Sonoma.",
        "User previously attempted running Qwen-72B locally, but inference failed due to high memory pressure.",
        "Development setup: pytest for testing, ruff for linting, and mypy in strict mode.",
        "ContextOS compilation pipeline achieves token reduction while preserving exact fact attribution.",
        "The local LLM endpoint is hosted at http://127.0.0.1:11434 with zero external telemetry transmission.",
    ]
    memories = [
        Memory(
            id=uuid4(),
            content=text,
            status=MemoryStatus.ACTIVE,
            token_count=len(text.split()),
        )
        for text in texts
    ]
    return [
        ScoredMemory(
            memory=m,
            final_score=0.95 - (i * 0.05),
            retrieval_sources=["lexical", "dense"] if i % 2 == 0 else ["lexical", "graph"],
        )
        for i, m in enumerate(memories)
    ]


async def run_phase9_benchmark(db_path: Path | None = None) -> Phase9BenchmarkReport:
    """Run deterministic Phase 9 benchmark and return structured metrics."""
    ref_counter = TiktokenCounter("cl100k_base")
    optimizer = MemoryContextOptimizer(token_counter=ref_counter)
    compiler = QueryAwareContextCompiler(token_counter=ref_counter)

    candidates = benchmark_memories()
    candidate_tokens = sum(ref_counter.count(c.memory.content) for c in candidates)

    # 1. Optimize and Compile
    query = "What is the recommended local model and hardware configuration for Project Atlas?"
    selection = optimizer.optimize(
        query=query,
        candidates=candidates,
        budget=ContextBudget(max_tokens=2000),
    )
    compiled = await compiler.compile(
        query=query,
        memories=selection,
        config=CompilationConfig(budget=1500),
    )
    compiled_tokens = compiled.total_tokens

    # 2. Multi-Model Token Recounting (Profiles)
    tc_start = time.perf_counter()
    recounts = recount_cross_model(
        compiled.context_text,
        ["FAKE_CLAUDE_LIKE", "FAKE_OPENAI_LIKE", "FAKE_QWEN_LIKE"],
    )
    tc_overhead = (time.perf_counter() - tc_start) * 1000.0

    profiles: dict[str, ModelProfileResult] = {}
    for prof_name, (count, src) in recounts.items():
        # Context tokens avoided from perspective of this model profile
        prof_counter = get_token_counter_for_model(prof_name, prof_name)
        cand_for_model = sum(prof_counter.count(c.memory.content) for c in candidates)
        avoided = max(0, cand_for_model - count)
        red_ratio = max(0.0, 1.0 - (count / cand_for_model)) if cand_for_model > 0 else 0.0
        profiles[prof_name] = ModelProfileResult(
            profile_name=prof_name,
            target_token_estimate=count,
            measurement_source=src.value,
            reduction_ratio=red_ratio,
            context_tokens_avoided=avoided,
        )

    # 3. Router Overhead Benchmark
    router = DeterministicModelRouter()
    fake_prov = DeterministicFakeProvider()
    providers = {fake_prov.provider_id: fake_prov}
    req = ModelRequest(user_prompt=query, compiled_context=compiled)

    r_start = time.perf_counter()
    iterations = 50
    for _ in range(iterations):
        await router.route(req, providers, policy=RoutingPolicy.LOCAL_FIRST)
    router_overhead = ((time.perf_counter() - r_start) * 1000.0) / iterations

    # 4. Telemetry Persistence Overhead Benchmark
    import tempfile
    test_db_path = db_path or Path(tempfile.mkdtemp()) / "bench_telemetry.db"
    db = Database(test_db_path)
    await db.initialize()
    repo = SqliteTelemetryRepository(db.connection())

    t_start = time.perf_counter()
    tel_iterations = 20
    for i in range(tel_iterations):
        sample_tel = ModelInvocationTelemetry(
            invocation_id=uuid4(),
            provider_id="fake",
            model_id="fake-default",
            is_local=True,
            candidate_context_tokens=candidate_tokens,
            compiled_context_tokens=compiled_tokens,
            context_tokens_avoided=max(0, candidate_tokens - compiled_tokens),
            reduction_ratio=1.0 - (compiled_tokens / candidate_tokens) if candidate_tokens > 0 else 0.0,
            token_measurement_source=TokenMeasurementSource.PROVIDER_REPORTED,
            routing_policy=RoutingPolicy.LOCAL_FIRST,
            routing_reason="benchmark",
            selected_provider="fake",
            selected_model="fake-default",
            finish_reason=ModelFinishReason.STOP,
        )
        await repo.record(sample_tel)
    telemetry_overhead = ((time.perf_counter() - t_start) * 1000.0) / tel_iterations
    await db.close()

    return Phase9BenchmarkReport(
        candidate_context_tokens=candidate_tokens,
        compiled_context_tokens=compiled_tokens,
        profiles=profiles,
        router_overhead_ms=router_overhead,
        telemetry_overhead_ms=telemetry_overhead,
        token_counting_overhead_ms=tc_overhead,
    )


if __name__ == "__main__":
    report = asyncio.run(run_phase9_benchmark())
    print("\n=== CONTEXTOS PHASE 9 BENCHMARK REPORT ===")
    print(f"Candidate Context Tokens: {report.candidate_context_tokens}")
    print(f"Compiled Context Tokens:  {report.compiled_context_tokens}")
    print(f"Router Overhead:          {report.router_overhead_ms:.3f} ms")
    print(f"Telemetry Write Overhead: {report.telemetry_overhead_ms:.3f} ms")
    print(f"Token Counting Overhead:  {report.token_counting_overhead_ms:.3f} ms")
    print("\n--- Target Model Tokenizer Profiles (Synthetic Profiles — NOT Actual Vendor Benchmarks) ---")
    for name, p in report.profiles.items():
        print(
            f"  {name:20s}: tokens={p.target_token_estimate:4d} | "
            f"reduction={p.reduction_ratio * 100.0:.1f}% | "
            f"avoided={p.context_tokens_avoided:4d} | source={p.measurement_source}"
        )
