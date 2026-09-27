"""Deterministic Phase 4 -> 5 -> 6 oversized-rescue smoke scenario."""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from uuid import UUID

from contextos.core.enums import MemoryStatus, MemoryType
from contextos.core.models import CompilationConfig, ContextBudget, Memory, RetrievalQuery
from contextos.embedding.deterministic import DeterministicEmbedding
from contextos.services.compilation import QueryAwareContextCompiler
from contextos.services.optimization import MemoryContextOptimizer
from contextos.services.retrieval import HybridRetrievalEngine
from contextos.services.retrieval_index import RetrievalIndexSynchronizer
from contextos.services.token_counter import DeterministicWordTokenCounter
from contextos.storage.database import Database
from contextos.storage.lexical.bm25 import BM25Index
from contextos.storage.memory_repo import SqliteMemoryRepository
from contextos.storage.vector.in_memory import InMemoryVectorStore


def _memory(number: int, content: str, status: MemoryStatus = MemoryStatus.ACTIVE) -> Memory:
    return Memory(
        id=UUID(f"80000000-0000-0000-0000-{number:012d}"),
        content=content,
        status=status,
        type=MemoryType.FACT,
        provenance_event_id=UUID(f"81000000-0000-0000-0000-{number:012d}"),
        confidence=0.95,
        importance=0.9,
    )


async def run_smoke() -> dict[str, object]:
    padding = " ".join(f"diagnostic{index}" for index in range(320)) + "."
    unrelated = " ".join(f"gardening{index}" for index in range(320)) + "."
    memories = [
        _memory(1, "Ollama is the selected local model runner."),
        _memory(
            2,
            "Qwen30B failed on the local machine because available VRAM was insufficient. "
            + padding,
        ),
        _memory(3, "Qwen9B works on the local machine."),
        _memory(4, "A historical garden watering archive. " + unrelated, MemoryStatus.HISTORICAL),
    ]
    with tempfile.TemporaryDirectory(prefix="contextos-rescue-") as directory:
        database_path = Path(directory) / "contextos.db"
        database = Database(database_path)
        await database.initialize()
        try:
            repository = SqliteMemoryRepository(database.connection())
            for memory in memories:
                await repository.create(memory)

            embedding = DeterministicEmbedding(64)
            lexical = BM25Index()
            vector = InMemoryVectorStore(embedding.dimension)
            synchronizer = RetrievalIndexSynchronizer(
                memory_repo=repository,
                lexical_index=lexical,
                vector_store=vector,
                embedding_service=embedding,
            )
            retrieval = HybridRetrievalEngine(
                memory_repo=repository,
                lexical_index=lexical,
                vector_store=vector,
                embedding_service=embedding,
                index_synchronizer=synchronizer,
            )
            query = "local model machine attempts"
            retrieved = await retrieval.retrieve(RetrievalQuery(text=query, k=10))
            counter = DeterministicWordTokenCounter()
            optimizer = MemoryContextOptimizer(token_counter=counter)
            selection = optimizer.optimize(
                query, retrieved.memories, ContextBudget(max_tokens=24)
            )
            compiler = QueryAwareContextCompiler(token_counter=counter)
            compiled = await compiler.compile(
                query, selection, CompilationConfig(budget=48)
            )
            return {
                "database_on_disk": database_path.exists(),
                "retrieved_ids": [str(value.memory.id) for value in retrieved.memories],
                "selected_ids": [
                    str(value.memory.id) for value in selection.selected_memories
                ],
                "rescue_ids": [
                    str(value.memory.id)
                    for value in selection.compiler_rescue_candidates
                ],
                "emitted_facts": [
                    {
                        "text": fact.text,
                        "input_kind": fact.input_kind.value,
                        "source_memory_ids": [
                            str(value) for value in fact.source_memory_ids
                        ],
                        "provenance_event_ids": [
                            str(value) for value in fact.provenance_event_ids
                        ],
                    }
                    for fact in compiled.facts
                ],
                "tokens": compiled.total_tokens,
                "budget": compiled.budget,
                "oversized_rescue_inputs": compiled.trace.oversized_rescue_inputs,
                "rescued_facts_included": compiled.trace.rescued_facts_included,
                "unsupported_fact_rate": compiled.unsupported_fact_rate,
                "historical_unrelated_rescued": memories[3].id in {
                    value.memory.id for value in selection.compiler_rescue_candidates
                },
            }
        finally:
            await database.close()


def main() -> None:
    print(json.dumps(asyncio.run(run_smoke()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
