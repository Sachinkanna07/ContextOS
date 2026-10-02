"""Isolated temporal peer lookup benchmark for local profiling."""

from __future__ import annotations

import asyncio
import json
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any

from contextos.core.enums import MemoryStatus
from contextos.core.models import Memory, MemorySlot
from contextos.services.temporal import TemporalMemoryService
from contextos.storage.database import Database
from contextos.storage.memory_repo import SqliteMemoryRepository


async def measure(seed_count: int = 600, iterations: int = 25) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="contextos-temporal-profile-") as folder:
        database = Database(Path(folder) / "temporal.db")
        await database.initialize()
        try:
            repo = SqliteMemoryRepository(database.connection())
            service = TemporalMemoryService(repo)
            for index in range(seed_count):
                await repo.create(
                    Memory(
                        content=f"Synthetic tooling fact {index}.",
                        status=MemoryStatus.ACTIVE,
                        slot=MemorySlot(
                            subject="user", property="tool_usage", scope=f"scope_{index}"
                        ),
                    )
                )
            cursor = await database.connection().execute(
                "EXPLAIN QUERY PLAN SELECT * FROM memories WHERE status = ? "
                "AND slot_key IS NOT NULL AND slot_key != ? "
                "AND json_extract(slot_json, '$.subject') = ? "
                "AND json_extract(slot_json, '$.property') = ? "
                "ORDER BY COALESCE(valid_from, observed_at, created_at) DESC, "
                "observed_at DESC, id DESC LIMIT 1",
                ("active", "unused", "user", "tool_usage"),
            )
            plan = [row[3] for row in await cursor.fetchall()]
            samples = []
            outcomes = []
            for index in range(iterations):
                candidate = Memory(
                    content=f"Synthetic new tooling fact {index}.",
                    slot=MemorySlot(
                        subject="user", property="tool_usage", scope=f"new_scope_{index}"
                    ),
                )
                started = time.perf_counter()
                decision = await service.decide(candidate)
                samples.append((time.perf_counter() - started) * 1000)
                outcomes.append(decision.outcome.value)
            return {
                "seed_count": seed_count,
                "iterations": iterations,
                "mean_ms": round(statistics.mean(samples), 3),
                "median_ms": round(statistics.median(samples), 3),
                "p95_ms": round(
                    sorted(samples)[min(len(samples) - 1, int(0.95 * len(samples)))], 3
                ),
                "outcomes": sorted(set(outcomes)),
                "query_plan": plan,
            }
        finally:
            await database.close()


if __name__ == "__main__":
    print(json.dumps(asyncio.run(measure()), indent=2))
