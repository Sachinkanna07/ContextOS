"""Deterministic Phase 11 connector performance and unchanged skip benchmark.

Label: LOCAL DEVELOPMENT SYNTHETIC CONNECTOR BENCHMARK
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from contextos.connectors.fake import FakeConnector
from contextos.connectors.json_import import JsonImportConnector
from contextos.connectors.local_files import LocalFileConnector
from contextos.connectors.manager import ConnectorManager
from contextos.connectors.models import ConnectorItem, RetentionPolicy
from contextos.core.enums import SecretDetectionMode
from contextos.embedding.deterministic import DeterministicEmbedding
from contextos.services.extraction import RuleBasedMemoryExtractor
from contextos.services.ingestion import IngestionPipeline
from contextos.services.secret_scanner import PatternSecretScanner
from contextos.services.temporal import TemporalMemoryService
from contextos.services.token_counter import DeterministicWordTokenCounter
from contextos.storage.connector_repo import SqliteConnectorRepository
from contextos.storage.database import Database
from contextos.storage.event_repo import SqliteEventRepository
from contextos.storage.lexical.bm25 import BM25Index
from contextos.storage.memory_repo import SqliteMemoryRepository
from contextos.storage.vector.in_memory import InMemoryVectorStore


class CountingSecretScanner(PatternSecretScanner):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def scan(self, text: str, *args, **kwargs):
        self.calls += 1
        return super().scan(text, *args, **kwargs)


class CountingExtractor(RuleBasedMemoryExtractor):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def extract(self, text: str, *args, **kwargs):
        self.calls += 1
        return super().extract(text, *args, **kwargs)


class CountingTemporalService(TemporalMemoryService):
    def __init__(self, memory_repo) -> None:
        super().__init__(memory_repo)
        self.accept_calls = 0

    async def accept(self, candidate, *, provenance_event_id=None):
        self.accept_calls += 1
        return await super().accept(candidate, provenance_event_id=provenance_event_id)


@dataclass
class BenchmarkScenarioResult:
    name: str
    items_count: int
    duration_ms: float
    items_per_sec: float
    scanned: int
    accepted: int
    unchanged: int
    updated: int
    rejected: int
    failed: int
    privacy_calls: int
    extractor_calls: int
    temporal_calls: int


async def run_connector_benchmark() -> list[BenchmarkScenarioResult]:
    tmp_dir = Path(tempfile.mkdtemp())
    db_path = tmp_dir / "bench_connectors.db"
    db = Database(db_path)
    await db.initialize()
    conn = db.connection()

    memory_repo = SqliteMemoryRepository(conn)
    event_repo = SqliteEventRepository(conn)
    connector_repo = SqliteConnectorRepository(conn)

    scanner = CountingSecretScanner()
    extractor = CountingExtractor()
    token_counter = DeterministicWordTokenCounter()
    embedding = DeterministicEmbedding(16)
    lexical = BM25Index()
    vector = InMemoryVectorStore(16)

    ingestion = IngestionPipeline(
        secret_scanner=scanner,
        memory_extractor=extractor,
        memory_repo=memory_repo,
        event_repo=event_repo,
        embedding_service=embedding,
        vector_store=vector,
        lexical_index=lexical,
        token_counter=token_counter,
        secret_detection_mode=SecretDetectionMode.STRICT,
    )
    temporal = CountingTemporalService(memory_repo)
    manager = ConnectorManager(
        state_repo=connector_repo,
        ingestion=ingestion,
        temporal=temporal,
        retention_policy=RetentionPolicy.KEEP_DERIVED_MEMORY,
        memory_repo=memory_repo,
    )

    results: list[BenchmarkScenarioResult] = []

    def reset_counters():
        scanner.calls = 0
        extractor.calls = 0
        temporal.accept_calls = 0

    # --- Scenario 1: Initial Sync (100 items) ---
    items_100 = [
        ConnectorItem(
            external_id=f"doc_{i}",
            source_type="fake",
            source_uri=f"fake://doc_{i}",
            content=f"I am working on Project Atlas module {i}. Project Atlas uses Python.",
            revision=f"rev_1_{i}",
        )
        for i in range(100)
    ]
    connector_100 = FakeConnector("fake-100", items_100)
    manager.register(connector_100)

    reset_counters()
    started = time.perf_counter()
    sync_res = await manager.sync("fake-100")
    dur_ms = (time.perf_counter() - started) * 1000
    results.append(
        BenchmarkScenarioResult(
            name="Initial Sync (100 items)",
            items_count=100,
            duration_ms=dur_ms,
            items_per_sec=100 / (dur_ms / 1000) if dur_ms > 0 else 0,
            scanned=sync_res.scanned,
            accepted=sync_res.accepted,
            unchanged=sync_res.unchanged,
            updated=sync_res.updated,
            rejected=sync_res.rejected,
            failed=sync_res.failed,
            privacy_calls=scanner.calls,
            extractor_calls=extractor.calls,
            temporal_calls=temporal.accept_calls,
        )
    )

    # --- Scenario 2: Unchanged Second Sync (100 items) ---
    reset_counters()
    started = time.perf_counter()
    sync_res = await manager.sync("fake-100")
    dur_ms = (time.perf_counter() - started) * 1000
    results.append(
        BenchmarkScenarioResult(
            name="Unchanged Second Sync (100 items)",
            items_count=100,
            duration_ms=dur_ms,
            items_per_sec=100 / (dur_ms / 1000) if dur_ms > 0 else 0,
            scanned=sync_res.scanned,
            accepted=sync_res.accepted,
            unchanged=sync_res.unchanged,
            updated=sync_res.updated,
            rejected=sync_res.rejected,
            failed=sync_res.failed,
            privacy_calls=scanner.calls,
            extractor_calls=extractor.calls,
            temporal_calls=temporal.accept_calls,
        )
    )

    # --- Scenario 3: 10% Changed Sync (100 items, 10 modified) ---
    changed_items_100 = list(items_100)
    for i in range(10):
        changed_items_100[i] = ConnectorItem(
            external_id=f"doc_{i}",
            source_type="fake",
            source_uri=f"fake://doc_{i}",
            content=f"I am working on Project Atlas module {i}. Now using Rust.",
            revision=f"rev_2_{i}",
        )
    connector_100.items = changed_items_100

    reset_counters()
    started = time.perf_counter()
    sync_res = await manager.sync("fake-100")
    dur_ms = (time.perf_counter() - started) * 1000
    results.append(
        BenchmarkScenarioResult(
            name="10% Changed Sync (100 items)",
            items_count=100,
            duration_ms=dur_ms,
            items_per_sec=100 / (dur_ms / 1000) if dur_ms > 0 else 0,
            scanned=sync_res.scanned,
            accepted=sync_res.accepted,
            unchanged=sync_res.unchanged,
            updated=sync_res.updated,
            rejected=sync_res.rejected,
            failed=sync_res.failed,
            privacy_calls=scanner.calls,
            extractor_calls=extractor.calls,
            temporal_calls=temporal.accept_calls,
        )
    )

    # --- Scenario 4: Privacy Rejection Batch (20 items with secrets) ---
    secret_items = [
        ConnectorItem(
            external_id=f"sec_{i}",
            source_type="fake",
            source_uri=f"fake://sec_{i}",
            content=f"I prefer secret key: sk-proj-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
            revision="rev_sec",
        )
        for i in range(20)
    ]
    connector_sec = FakeConnector("fake-sec", secret_items)
    manager.register(connector_sec)

    reset_counters()
    started = time.perf_counter()
    sync_res = await manager.sync("fake-sec")
    dur_ms = (time.perf_counter() - started) * 1000
    results.append(
        BenchmarkScenarioResult(
            name="Privacy Rejection Batch (20 items)",
            items_count=20,
            duration_ms=dur_ms,
            items_per_sec=20 / (dur_ms / 1000) if dur_ms > 0 else 0,
            scanned=sync_res.scanned,
            accepted=sync_res.accepted,
            unchanged=sync_res.unchanged,
            updated=sync_res.updated,
            rejected=sync_res.rejected,
            failed=sync_res.failed,
            privacy_calls=scanner.calls,
            extractor_calls=extractor.calls,
            temporal_calls=temporal.accept_calls,
        )
    )

    # --- Scenario 5: JSONL Import Scan (100 items) ---
    jsonl_file = tmp_dir / "import.jsonl"
    jsonl_lines = [
        json.dumps({"id": f"j_{i}", "content": f"I am working on Project Gamma {i}. Project Gamma uses PostgreSQL."})
        for i in range(100)
    ]
    jsonl_file.write_text("\n".join(jsonl_lines), encoding="utf-8")
    json_conn = JsonImportConnector("json-100", jsonl_file)
    manager.register(json_conn)

    reset_counters()
    started = time.perf_counter()
    sync_res = await manager.sync("json-100")
    dur_ms = (time.perf_counter() - started) * 1000
    results.append(
        BenchmarkScenarioResult(
            name="JSONL Import Scan (100 items)",
            items_count=100,
            duration_ms=dur_ms,
            items_per_sec=100 / (dur_ms / 1000) if dur_ms > 0 else 0,
            scanned=sync_res.scanned,
            accepted=sync_res.accepted,
            unchanged=sync_res.unchanged,
            updated=sync_res.updated,
            rejected=sync_res.rejected,
            failed=sync_res.failed,
            privacy_calls=scanner.calls,
            extractor_calls=extractor.calls,
            temporal_calls=temporal.accept_calls,
        )
    )

    # --- Scenario 6: Local File Scan (50 files) ---
    files_dir = tmp_dir / "files"
    files_dir.mkdir()
    for i in range(50):
        (files_dir / f"note_{i}.txt").write_text(
            f"I am working on Project Beta module {i}. Project Beta uses Rust.", encoding="utf-8"
        )
    file_conn = LocalFileConnector("files-50", [files_dir])
    manager.register(file_conn)

    reset_counters()
    started = time.perf_counter()
    sync_res = await manager.sync("files-50")
    dur_ms = (time.perf_counter() - started) * 1000
    results.append(
        BenchmarkScenarioResult(
            name="Local File Scan (50 files)",
            items_count=50,
            duration_ms=dur_ms,
            items_per_sec=50 / (dur_ms / 1000) if dur_ms > 0 else 0,
            scanned=sync_res.scanned,
            accepted=sync_res.accepted,
            unchanged=sync_res.unchanged,
            updated=sync_res.updated,
            rejected=sync_res.rejected,
            failed=sync_res.failed,
            privacy_calls=scanner.calls,
            extractor_calls=extractor.calls,
            temporal_calls=temporal.accept_calls,
        )
    )

    # --- Scenario 7: Scale Initial Sync (1000 items) ---
    items_1000 = [
        ConnectorItem(
            external_id=f"big_{i}",
            source_type="fake",
            source_uri=f"fake://big_{i}",
            content=f"I am working on Project Scale component {i}. Uses TypeScript.",
            revision=f"rev_scale_{i}",
        )
        for i in range(1000)
    ]
    connector_1000 = FakeConnector("fake-1000", items_1000)
    manager.register(connector_1000)

    reset_counters()
    started = time.perf_counter()
    sync_res = await manager.sync("fake-1000")
    dur_ms = (time.perf_counter() - started) * 1000
    results.append(
        BenchmarkScenarioResult(
            name="Initial Sync (1000 items)",
            items_count=1000,
            duration_ms=dur_ms,
            items_per_sec=1000 / (dur_ms / 1000) if dur_ms > 0 else 0,
            scanned=sync_res.scanned,
            accepted=sync_res.accepted,
            unchanged=sync_res.unchanged,
            updated=sync_res.updated,
            rejected=sync_res.rejected,
            failed=sync_res.failed,
            privacy_calls=scanner.calls,
            extractor_calls=extractor.calls,
            temporal_calls=temporal.accept_calls,
        )
    )

    # --- Scenario 8: Scale Unchanged Second Sync (1000 items) ---
    reset_counters()
    started = time.perf_counter()
    sync_res = await manager.sync("fake-1000")
    dur_ms = (time.perf_counter() - started) * 1000
    results.append(
        BenchmarkScenarioResult(
            name="Unchanged Second Sync (1000 items)",
            items_count=1000,
            duration_ms=dur_ms,
            items_per_sec=1000 / (dur_ms / 1000) if dur_ms > 0 else 0,
            scanned=sync_res.scanned,
            accepted=sync_res.accepted,
            unchanged=sync_res.unchanged,
            updated=sync_res.updated,
            rejected=sync_res.rejected,
            failed=sync_res.failed,
            privacy_calls=scanner.calls,
            extractor_calls=extractor.calls,
            temporal_calls=temporal.accept_calls,
        )
    )

    await db.close()
    return results


def main() -> None:
    results = asyncio.run(run_connector_benchmark())

    print("=" * 80)
    print("LOCAL DEVELOPMENT SYNTHETIC CONNECTOR BENCHMARK")
    print("=" * 80)
    print()
    print(
        f"{'Scenario':<38} | {'Items':<6} | {'Time (ms)':<9} | {'Items/sec':<10} | "
        f"{'Accepted':<8} | {'Unchanged':<9} | {'Pipeline Calls (Scanner/Extract/Temporal)'}"
    )
    print("-" * 115)

    for r in results:
        pipe_str = f"{r.privacy_calls}/{r.extractor_calls}/{r.temporal_calls}"
        print(
            f"{r.name:<38} | {r.items_count:<6} | {r.duration_ms:>9.2f} | "
            f"{r.items_per_sec:>10.1f} | {r.accepted:<8} | {r.unchanged:<9} | {pipe_str}"
        )

    print()
    print("UNCHANGED SKIP INVARIANT SUMMARY:")
    print("-" * 50)
    unchanged_100 = next(r for r in results if r.name == "Unchanged Second Sync (100 items)")
    unchanged_1000 = next(r for r in results if r.name == "Unchanged Second Sync (1000 items)")
    print(
        f"100 Unchanged Sync:  Unchanged={unchanged_100.unchanged}/100, "
        f"Pipeline Calls (Scanner/Extract/Temporal) = {unchanged_100.privacy_calls}/{unchanged_100.extractor_calls}/{unchanged_100.temporal_calls}"
    )
    print(
        f"1000 Unchanged Sync: Unchanged={unchanged_1000.unchanged}/1000, "
        f"Pipeline Calls (Scanner/Extract/Temporal) = {unchanged_1000.privacy_calls}/{unchanged_1000.extractor_calls}/{unchanged_1000.temporal_calls}"
    )
    print("* Instrumentation Note: Scanner calls count internal PatternSecretScanner checks across content and candidates (14 per item).")
    print()
    print("Benchmark complete.")


if __name__ == "__main__":
    main()
