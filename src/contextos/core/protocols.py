"""Component protocols (interfaces) for ContextOS.

Every major component communicates through a Protocol defined here.
Concrete implementations are injected at startup via daemon/wiring.py.

Protocol rules:
- All methods that touch I/O are async.
- Protocols are minimal — they define the contract, not convenience methods.
- Return types are domain models from core.models, never raw dicts or tuples.
- Protocols are testable: contract test suites verify any implementation.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable
from uuid import UUID

from contextos.core.enums import MemoryStatus, MemoryType, OptimizationStrategy, SourceRole
from contextos.core.models import (
    CandidateMemory,
    CompiledContext,
    CompilationConfig,
    ContextBudget,
    EventFilters,
    IngestRequest,
    IngestResult,
    LexicalResult,
    Memory,
    MemoryFilters,
    MemoryRelation,
    MemoryUpdate,
    RawEvent,
    RetrievalConfig,
    RetrievalQuery,
    RetrievalResult,
    SelectionResult,
    ScanResult,
    ScoredMemory,
    StageTrace,
    TemporalDecision,
    TemporalResolutionResult,
    MemorySlot,
    VectorResult,
)


# ---------------------------------------------------------------------------
# Storage Protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class MemoryRepository(Protocol):
    """Persistence layer for Memory objects (SQLite)."""

    async def get(self, memory_id: UUID) -> Memory | None: ...

    async def list(self, filters: MemoryFilters) -> list[Memory]: ...

    async def create(self, memory: Memory) -> Memory: ...

    async def update(self, memory_id: UUID, update: MemoryUpdate, expected_version: int) -> Memory:
        """Update a memory. Raises ConcurrencyError if version doesn't match."""
        ...

    async def update_status(
        self, memory_id: UUID, new_status: MemoryStatus, expected_version: int
    ) -> Memory: ...

    async def supersede(
        self, old_id: UUID, successor: Memory, expected_version: int
    ) -> Memory: ...

    async def update_access(self, memory_id: UUID) -> None:
        """Increment access_count and update last_accessed_at."""
        ...

    async def delete(self, memory_id: UUID) -> None:
        """Remove from database entirely (hard delete at storage level)."""
        ...

    async def count(self, filters: MemoryFilters | None = None) -> int: ...

    async def get_by_hash(self, content_hash: str) -> Memory | None:
        """Find a memory by its content hash. Used for exact dedup."""
        ...

    async def list_by_slot(self, slot_key: str) -> list[Memory]: ...

    async def list_temporal(self, *, limit: int = 500) -> list[Memory]: ...

    async def apply_temporal_decision(
        self, candidate: Memory, decision: TemporalDecision
    ) -> TemporalResolutionResult: ...


@runtime_checkable
class EventRepository(Protocol):
    """Append-only persistence for RawEvent objects."""

    async def append(self, event: RawEvent) -> None: ...

    async def get(self, event_id: UUID) -> RawEvent | None: ...

    async def list(self, filters: EventFilters) -> list[RawEvent]: ...

    async def count(self, filters: EventFilters | None = None) -> int: ...


@runtime_checkable
class RelationRepository(Protocol):
    """Persistence for MemoryRelation objects."""

    async def create(self, relation: MemoryRelation) -> MemoryRelation: ...

    async def get_relations(
        self, memory_id: UUID, direction: str = "both"
    ) -> list[MemoryRelation]:
        """Get relations for a memory.

        direction: 'outgoing', 'incoming', or 'both'.
        """
        ...

    async def delete_for_memory(self, memory_id: UUID) -> int:
        """Delete all relations involving this memory. Returns count deleted."""
        ...


@runtime_checkable
class VectorStore(Protocol):
    """Vector storage and similarity search."""

    async def add(
        self, ids: list[str], vectors: list[list[float]], metadata: list[dict[str, Any]]
    ) -> None: ...

    async def search(
        self,
        vector: list[float],
        top_k: int = 20,
        filters: dict[str, Any] | None = None,
    ) -> list[VectorResult]: ...

    async def delete(self, ids: list[str]) -> None: ...

    async def count(self) -> int: ...

    async def rebuild(
        self, ids: list[str], vectors: list[list[float]], metadata: list[dict[str, Any]]
    ) -> None: ...


@runtime_checkable
class LexicalIndex(Protocol):
    """Full-text / BM25 lexical search index."""

    async def index(self, doc_id: str, text: str, metadata: dict[str, Any] | None = None) -> None:
        ...

    async def search(
        self,
        query: str,
        top_k: int = 20,
        filters: dict[str, Any] | None = None,
    ) -> list[LexicalResult]: ...

    async def delete(self, doc_id: str) -> None: ...

    async def count(self) -> int: ...

    async def rebuild(self, documents: dict[str, str]) -> None:
        """Rebuild the entire index from a dict of {id: text}."""
        ...


# ---------------------------------------------------------------------------
# Service Protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class IngestionService(Protocol):
    """End-to-end ingestion pipeline: validate → scan → store → extract → index."""

    async def ingest(self, request: IngestRequest) -> IngestResult: ...


@runtime_checkable
class MemoryService(Protocol):
    """Business logic for memory management, including lifecycle transitions."""

    async def get(self, memory_id: UUID) -> Memory | None: ...

    async def list(self, filters: MemoryFilters) -> list[Memory]: ...

    async def search(self, query: str, limit: int = 50) -> list[ScoredMemory]: ...

    async def update(self, memory_id: UUID, update: MemoryUpdate) -> Memory: ...

    async def transition(
        self, memory_id: UUID, new_status: MemoryStatus, reason: str = ""
    ) -> Memory:
        """Transition a memory to a new lifecycle state.

        Raises InvalidTransitionError if the transition is not allowed.
        """
        ...

    async def delete(self, memory_id: UUID) -> None:
        """Soft delete: transition to DELETED, remove from indices."""
        ...

    async def purge(self, memory_id: UUID) -> None:
        """Hard delete: destroy from all stores."""
        ...

    async def history(self, topic: str) -> list[Memory]:
        """Temporal evolution of memories related to a topic."""
        ...


@runtime_checkable
class RetrievalService(Protocol):
    """Multi-strategy retrieval with fusion, reranking, and dedup."""

    async def retrieve(
        self, query: str | RetrievalQuery, config: RetrievalConfig | None = None
    ) -> RetrievalResult: ...


@runtime_checkable
class CompilationService(Protocol):
    """Budget-constrained context compilation from retrieved memories."""

    async def compile(
        self,
        query: str,
        memories: list[ScoredMemory] | SelectionResult,
        config: CompilationConfig | None = None,
    ) -> CompiledContext: ...


@runtime_checkable
class TokenAwareOptimizer(Protocol):
    """Select whole retrieved memories within a memory-context budget."""

    def optimize(
        self,
        query: str,
        candidates: list[ScoredMemory],
        budget: ContextBudget,
        strategy: OptimizationStrategy = OptimizationStrategy.CONTEXTOS,
    ) -> SelectionResult: ...


@runtime_checkable
class TemporalResolver(Protocol):
    """Resolve accepted candidates into deterministic temporal timelines."""

    async def resolve(self, candidate: Memory) -> TemporalResolutionResult: ...

    async def get_current_state(self, slot: MemorySlot | str) -> list[Memory]: ...

    async def get_history(self, slot: MemorySlot | str) -> list[Memory]: ...

    async def get_previous(self, memory_id: UUID) -> Memory | None: ...


# ---------------------------------------------------------------------------
# Infrastructure Protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class EmbeddingService(Protocol):
    """Generate embeddings from text. Abstracts over model choice."""

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts."""
        ...

    async def embed_query(self, query: str) -> list[float]:
        """Embed a single query. May use a different prefix/strategy than documents."""
        ...

    @property
    def dimension(self) -> int:
        """Embedding vector dimension."""
        ...

    @property
    def model_name(self) -> str:
        """Name of the loaded model."""
        ...


@runtime_checkable
class SecretScanner(Protocol):
    """Detect secrets and sensitive content in text."""

    def scan(self, text: str) -> ScanResult: ...

    def redact(self, text: str) -> tuple[str, ScanResult]:
        """Scan and return (redacted_text, scan_result)."""
        ...


@runtime_checkable
class MemoryExtractor(Protocol):
    """Extract discrete memories from raw text."""

    async def extract(
        self,
        text: str,
        source_type: str = "cli_input",
        source_uri: str | None = None,
        suggested_type: MemoryType | None = None,
        tags: list[str] | None = None,
        source_role: SourceRole = SourceRole.USER,
        confirmed_user_information: bool = False,
    ) -> list[CandidateMemory]: ...


@runtime_checkable
class DuplicateDetector(Protocol):
    """Detect duplicates and near-duplicates against existing memories."""

    async def check(
        self, candidate: CandidateMemory, existing_memories: list[Memory] | None = None
    ) -> DuplicateCheckResult: ...


class DuplicateCheckResult:
    """Result of a duplicate check."""

    __slots__ = ("is_duplicate", "is_near_duplicate", "matched_memory_id", "similarity_score")

    def __init__(
        self,
        is_duplicate: bool = False,
        is_near_duplicate: bool = False,
        matched_memory_id: UUID | None = None,
        similarity_score: float = 0.0,
    ) -> None:
        self.is_duplicate = is_duplicate
        self.is_near_duplicate = is_near_duplicate
        self.matched_memory_id = matched_memory_id
        self.similarity_score = similarity_score


@runtime_checkable
class TokenCounter(Protocol):
    """Count tokens in text. Abstracts over tokenizer choice."""

    def count(self, text: str) -> int: ...

    def count_batch(self, texts: list[str]) -> list[int]: ...

    @property
    def encoding_name(self) -> str: ...


# ---------------------------------------------------------------------------
# Observability Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class TraceCollector(Protocol):
    """Collect and store pipeline traces."""

    async def record(self, trace_id: str, stage: StageTrace) -> None: ...

    async def get_traces(self, limit: int = 100) -> list[dict[str, Any]]: ...

    async def get_stats(self) -> dict[str, Any]: ...
