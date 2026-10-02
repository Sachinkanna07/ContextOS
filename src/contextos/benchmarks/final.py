"""Local synthetic comparison of retrieval and context preparation strategies."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import platform
import statistics
import tempfile
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid5

import psutil  # type: ignore[import-untyped]

from contextos.config.settings import DaemonConfig, EmbeddingConfig, Settings
from contextos.core.enums import MemoryStatus, RetrievalMode
from contextos.core.models import CompilationConfig, ContextBudget, Memory, RetrievalQuery
from contextos.daemon.wiring import wire_services
from contextos.services.inspection import InspectionRequest


def ranking_metrics(ids: list[str], relevant: set[str]) -> dict[str, float]:
    """Binary relevance metrics from IDs specified before retrieval runs."""
    if not relevant:
        raise ValueError("Ground truth cannot be empty")
    metrics: dict[str, float] = {}
    for k in (1, 3, 5, 10):
        hits = sum(item in relevant for item in ids[:k])
        metrics[f"recall@{k}"] = hits / len(relevant)
        metrics[f"precision@{k}"] = hits / k
        metrics[f"hit@{k}"] = float(hits > 0)
    first = next((rank for rank, item in enumerate(ids, 1) if item in relevant), None)
    metrics["mrr"] = 1 / first if first else 0.0
    dcg = sum(
        (1 / math.log2(rank + 1)) for rank, item in enumerate(ids[:10], 1) if item in relevant
    )
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(len(relevant), 10) + 1))
    metrics["ndcg@10"] = dcg / ideal if ideal else 0.0
    return metrics


def distribution(samples: list[float]) -> dict[str, float | int]:
    ordered = sorted(samples)
    return {
        "mean_ms": round(statistics.mean(samples), 3),
        "median_ms": round(statistics.median(samples), 3),
        "p95_ms": round(ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))], 3),
        "samples": len(samples),
    }


async def _dataset(
    size: int, services: dict[str, Any]
) -> tuple[list[Memory], list[tuple[str, set[str]]], set[str]]:
    repo = services["memory_repo"]
    fixed = (
        ("Project Atlas uses Python for automation.", MemoryStatus.ACTIVE),
        ("Project Boreal uses Rust for its local compiler.", MemoryStatus.ACTIVE),
        ("Ollama runs on the local workstation.", MemoryStatus.ACTIVE),
        ("My current editor is VS Code.", MemoryStatus.ACTIVE),
        ("Previously my editor was Sublime Text.", MemoryStatus.SUPERSEDED),
        ("Atlas stores reproducible build settings.", MemoryStatus.ACTIVE),
        ("Boreal depends on Ollama for local inference.", MemoryStatus.ACTIVE),
        ("I use Python and Rust for different projects.", MemoryStatus.ACTIVE),
        ("A long project note: " + "integration detail " * 80, MemoryStatus.ACTIVE),
        ("Project Atlas uses Python for automation.", MemoryStatus.ACTIVE),
        ("Project Atlas previously used Java for automation.", MemoryStatus.SUPERSEDED),
        ("Project Cedar uses PostgreSQL for catalog queries.", MemoryStatus.ACTIVE),
        ("Project Cedar uses PostgreSQL for catalog queries.", MemoryStatus.ACTIVE),
        ("I prefer concise code review comments.", MemoryStatus.ACTIVE),
        ("I prefer detailed architecture review comments.", MemoryStatus.ACTIVE),
        ("I do not use Docker for Project Atlas deployment.", MemoryStatus.ACTIVE),
        ("Project Birch uses Qwen9B for local summaries.", MemoryStatus.ACTIVE),
        ("My travel preference is a window seat.", MemoryStatus.ACTIVE),
        ("I prefer an aisle seat on long flights.", MemoryStatus.ACTIVE),
        ("I attended Event Cedar in 2025.", MemoryStatus.CONTRADICTED),
    )
    memories: list[Memory] = []
    epoch = datetime(2026, 1, 1, tzinfo=UTC)
    for index, (content, status) in enumerate(fixed):
        instant = epoch + timedelta(seconds=index)
        memories.append(
            await repo.create(
                Memory(
                    id=uuid5(NAMESPACE_URL, f"contextos-final-{size}-{index}"),
                    content=content,
                    status=status,
                    source_type="benchmark",
                    created_at=instant,
                    updated_at=instant,
                    observed_at=instant,
                )
            )
        )
    for index in range(size - len(fixed)):
        category = ("preference", "project", "tool", "technical fact", "irrelevant")[index % 5]
        content = (
            f"Synthetic {category} item {index}: local workspace module "
            f"{index % 37} uses feature {index % 19}."
        )
        position = index + len(fixed)
        instant = epoch + timedelta(seconds=position)
        memories.append(
            await repo.create(
                Memory(
                    id=uuid5(NAMESPACE_URL, f"contextos-final-{size}-{position}"),
                    content=content,
                    status=MemoryStatus.ACTIVE,
                    source_type="benchmark",
                    created_at=instant,
                    updated_at=instant,
                    observed_at=instant,
                )
            )
        )
    queries = [
        ("Which language does Project Atlas use?", {str(memories[0].id), str(memories[9].id)}),
        ("What does Project Boreal use?", {str(memories[1].id), str(memories[6].id)}),
        ("Where does Ollama run?", {str(memories[2].id)}),
        ("What is my current editor?", {str(memories[3].id)}),
        ("Which database does Project Cedar use?", {str(memories[11].id), str(memories[12].id)}),
        ("What style of code review comments do I prefer?", {str(memories[13].id)}),
        ("What style of architecture review comments do I prefer?", {str(memories[14].id)}),
        ("Which model does Project Birch use for local summaries?", {str(memories[16].id)}),
        ("What is my travel seat preference?", {str(memories[17].id), str(memories[18].id)}),
        ("Do I use Docker for Atlas deployment?", {str(memories[15].id)}),
    ]
    return (
        memories,
        queries,
        {
            str(memories[4].id),
            str(memories[10].id),
            str(memories[19].id),
        },
    )


async def _measure_size(size: int, iterations: int) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix=f"contextos-final-{size}-") as folder:
        services = await wire_services(
            Settings(
                daemon=DaemonConfig(data_dir=Path(folder)),
                embedding=EmbeddingConfig(model="deterministic"),
            )
        )
        try:
            memories, queries, stale_ids = await _dataset(size, services)
            await services["retrieval_index"].ensure_current()
            lexical_count = await services["bm25_index"].count()
            dense_count = await services["vector_store"].count()
            if lexical_count != size or dense_count != size:
                raise RuntimeError(
                    f"Incomplete benchmark indexes: expected {size}, "
                    f"lexical={lexical_count}, dense={dense_count}"
                )
            graph_started = time.perf_counter()
            await services["graph"].rebuild()
            graph_rebuild_ms = (time.perf_counter() - graph_started) * 1000
            graph_nodes, graph_edges, _ = await services["graph_repo"].counts()
            counter = services["token_counter"]
            modes = {
                "vector": RetrievalMode.DENSE,
                "hybrid": RetrievalMode.HYBRID,
                "hybrid_graph": RetrievalMode.HYBRID_GRAPH,
                "contextos": RetrievalMode.HYBRID,
            }
            metrics: dict[str, list[dict[str, float]]] = {
                name: [] for name in ("full_history", *modes)
            }
            latencies: dict[str, list[float]] = {name: [] for name in modes}
            contextos_stage_ms: dict[str, list[float]] = {
                "retrieval": [],
                "optimizer": [],
                "compiler": [],
            }
            contextos_facts = {
                "emitted": 0,
                "excluded": 0,
                "duplicates_excluded": 0,
                "with_provenance": 0,
                "selected_memories": 0,
                "required_source_hits": 0,
                "required_source_total": 0,
                "stale_source_facts": 0,
            }
            token_rows: dict[str, list[dict[str, int]]] = {
                name: [] for name in ("full_history", *modes)
            }
            stale_inclusion: dict[str, int] = {name: 0 for name in ("full_history", *modes)}
            overlaps: list[float] = []
            inspector_latencies: list[float] = []
            for _ in range(iterations):
                for query, relevant in queries:
                    full_ids = [str(memory.id) for memory in memories]
                    metrics["full_history"].append(ranking_metrics(full_ids, relevant))
                    full_tokens = sum(counter.count(memory.content) for memory in memories)
                    token_rows["full_history"].append(
                        {"candidate": full_tokens, "compiled": full_tokens}
                    )
                    stale_inclusion["full_history"] += bool(stale_ids & set(full_ids))
                    result_ids: dict[str, list[str]] = {}
                    for name, mode in modes.items():
                        started = time.perf_counter()
                        result = await services["retrieval"].retrieve(
                            RetrievalQuery(
                                text=query,
                                mode=mode,
                                k=10,
                            )
                        )
                        candidate_tokens = sum(
                            counter.count(row.memory.content) for row in result.memories
                        )
                        compiled_tokens = candidate_tokens
                        optimized_tokens = candidate_tokens
                        if name == "contextos":
                            contextos_stage_ms["retrieval"].append(
                                (time.perf_counter() - started) * 1000
                            )
                            optimize_started = time.perf_counter()
                            selection = services["optimizer"].optimize(
                                query, result.memories, ContextBudget(max_tokens=200)
                            )
                            contextos_stage_ms["optimizer"].append(
                                (time.perf_counter() - optimize_started) * 1000
                            )
                            optimized_tokens = sum(
                                counter.count(row.memory.content)
                                for row in selection.selected_memories
                            )
                            compile_started = time.perf_counter()
                            compiled = await services["compilation"].compile(
                                query, selection, CompilationConfig(budget=200)
                            )
                            contextos_stage_ms["compiler"].append(
                                (time.perf_counter() - compile_started) * 1000
                            )
                            compiled_tokens = counter.count(compiled.context_text)
                            contextos_facts["emitted"] += len(compiled.facts)
                            contextos_facts["excluded"] += len(compiled.excluded_facts)
                            contextos_facts["duplicates_excluded"] += sum(
                                fact.reason.value == "duplicate" for fact in compiled.excluded_facts
                            )
                            contextos_facts["with_provenance"] += sum(
                                bool(fact.provenance_event_ids) for fact in compiled.facts
                            )
                            contextos_facts["selected_memories"] += len(selection.selected_memories)
                            fact_sources = {
                                str(source_id)
                                for fact in compiled.facts
                                for source_id in fact.source_memory_ids
                            }
                            contextos_facts["required_source_hits"] += len(fact_sources & relevant)
                            contextos_facts["required_source_total"] += len(relevant)
                            contextos_facts["stale_source_facts"] += sum(
                                bool(
                                    {str(source_id) for source_id in fact.source_memory_ids}
                                    & stale_ids
                                )
                                for fact in compiled.facts
                            )
                            included = set(compiled.included_memory_ids)
                            ids = [
                                str(row.memory.id)
                                for row in selection.selected_memories
                                if row.memory.id in included
                            ]
                        else:
                            ids = [str(row.memory.id) for row in result.memories]
                        latencies[name].append((time.perf_counter() - started) * 1000)
                        result_ids[name] = ids
                        metrics[name].append(ranking_metrics(ids, relevant))
                        token_rows[name].append(
                            {
                                "candidate": candidate_tokens,
                                "optimized": optimized_tokens,
                                "compiled": compiled_tokens,
                            }
                        )
                        stale_inclusion[name] += bool(stale_ids & set(ids))
                    overlaps.append(
                        len(set(result_ids["hybrid"]) & set(result_ids["hybrid_graph"])) / 10
                    )
                    inspect_started = time.perf_counter()
                    await services["inspector"].inspect(
                        InspectionRequest(
                            query=query,
                            limit=10,
                            budget=200,
                            graph=False,
                        )
                    )
                    inspector_latencies.append((time.perf_counter() - inspect_started) * 1000)
            aggregate: dict[str, Any] = {}
            for name, rows in metrics.items():
                candidate_sum = sum(row["candidate"] for row in token_rows[name])
                optimized_sum = sum(
                    row.get("optimized", row["candidate"]) for row in token_rows[name]
                )
                compiled_sum = sum(row["compiled"] for row in token_rows[name])
                aggregate[name] = {
                    "retrieval": {
                        key: round(statistics.mean(row[key] for row in rows), 4) for key in rows[0]
                    },
                    "candidate_tokens_total": candidate_sum,
                    "optimized_tokens_total": optimized_sum,
                    "compiled_tokens_total": compiled_sum,
                    "weighted_token_reduction": (
                        round(1 - compiled_sum / candidate_sum, 4) if candidate_sum else None
                    ),
                    "stale_inclusion_queries": stale_inclusion[name],
                    "latency": distribution(latencies[name]) if name in latencies else None,
                    "stage_latency": (
                        {
                            stage: distribution(samples)
                            for stage, samples in contextos_stage_ms.items()
                        }
                        if name == "contextos"
                        else None
                    ),
                    "context_evidence": (
                        {
                            **contextos_facts,
                            "provenance_coverage": (
                                contextos_facts["with_provenance"] / contextos_facts["emitted"]
                                if contextos_facts["emitted"]
                                else None
                            ),
                            "required_source_coverage": (
                                contextos_facts["required_source_hits"]
                                / contextos_facts["required_source_total"]
                                if contextos_facts["required_source_total"]
                                else None
                            ),
                            "budget_utilization": compiled_sum / (200 * len(rows)),
                        }
                        if name == "contextos"
                        else None
                    ),
                }
            return {
                "memory_count": size,
                "queries": len(queries),
                "iterations": iterations,
                "strategies": aggregate,
                "hybrid_graph_top10_overlap": round(statistics.mean(overlaps), 4),
                "graph_rebuild_ms": round(graph_rebuild_ms, 3),
                "inspector_latency": distribution(inspector_latencies),
                "graph_nodes": graph_nodes,
                "graph_edges": graph_edges,
                "lexical_index_count": lexical_count,
                "dense_index_count": dense_count,
                "sqlite_bytes": await services["database"].get_size_bytes(),
                "process_rss_bytes": psutil.Process().memory_info().rss,
                "token_measurement_source": counter.measurement_source.value,
                "tokenizer": counter.encoding_name,
                "answer_quality": None,
                "answer_quality_reason": "No answer model or independently graded answers",
                "temporal_relation_accuracy": None,
                "temporal_relation_reason": (
                    "This retrieval dataset does not grade relation classification"
                ),
                "provenance_note": "Direct synthetic fixture rows have no provenance event IDs",
            }
        finally:
            await services["database"].close()


async def measure(extended: bool = False, iterations: int = 2) -> dict[str, Any]:
    sizes = (100, 1000, 5000) if extended else (100, 1000)
    from contextos.benchmarks.temporal import run_evaluation

    return {
        "label": "LOCAL SYNTHETIC DETERMINISTIC BENCHMARK",
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "dataset_note": (
            "Fixed queries and relevance IDs are defined before retrieval; no model judge"
        ),
        "ground_truth_scope": (
            "Ten fixed questions over synthetic records; not a general quality claim"
        ),
        "corpora": {str(size): await _measure_size(size, iterations) for size in sizes},
        "temporal_evaluation": await run_evaluation(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extended", action="store_true", help="Also measure 5,000 memories")
    parser.add_argument("--iterations", type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.iterations <= 10:
        parser.error("iterations must be between 1 and 10")
    print(json.dumps(asyncio.run(measure(args.extended, args.iterations)), indent=2))


if __name__ == "__main__":
    main()
