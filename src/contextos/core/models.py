"""Core domain models for ContextOS.

All domain objects are Pydantic models for:
- Validation at construction time
- Serialization to/from JSON and SQLite
- Automatic schema generation for API docs

These models are the canonical representation of data flowing through the system.
Storage layers convert to/from these models; business logic operates on them.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator, model_validator

from contextos.core.enums import (
    CandidateAction,
    CandidateTemporalStatus,
    CompilationStrategy,
    CompilerInputKind,
    CompressionLevel,
    ExclusionReason,
    EventType,
    MemoryStatus,
    MemoryType,
    OptimizationStrategy,
    PrivacyClassification,
    PrivacyDecision,
    PrivacyLevel,
    PrivacySeverity,
    FactExclusionReason,
    RelationType,
    RetrievalMode,
    SecretType,
    SourceRole,
    SourceTrust,
    TemporalScope,
)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return timezone-aware UTC now. Avoids naive datetime issues."""
    return datetime.now(timezone.utc)


def _content_hash(content: str) -> str:
    """Compute a normalized content hash for deduplication.

    Normalization: strip, collapse whitespace, lowercase.
    Using SHA-256 truncated to 16 hex chars (64 bits) — collision probability
    is negligible for our scale (< 1M memories).
    """
    normalized = " ".join(content.strip().lower().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------


class Memory(BaseModel):
    """A discrete unit of user knowledge extracted from raw input.

    Memories are the primary data objects in ContextOS. They have lifecycle
    state, confidence, importance, provenance, and privacy classification.
    """

    model_config = ConfigDict(from_attributes=True)

    id: UUID = Field(default_factory=uuid4)
    content: str = Field(min_length=1, max_length=10_000)
    content_hash: str = Field(default="")
    type: MemoryType = Field(default=MemoryType.CONTEXT)
    source_type: str = Field(default="cli_input")
    source_uri: str | None = Field(default=None)
    provenance_event_id: UUID | None = Field(default=None)
    status: MemoryStatus = Field(default=MemoryStatus.CANDIDATE)
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    privacy_level: PrivacyLevel = Field(default=PrivacyLevel.PERSONAL)
    token_count: int = Field(default=0, ge=0)
    embedding_id: str | None = Field(default=None)
    superseded_by: UUID | None = Field(default=None)
    supersedes: UUID | None = Field(default=None)
    access_count: int = Field(default=0, ge=0)
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)
    last_accessed_at: datetime | None = Field(default=None)
    expires_at: datetime | None = Field(default=None)
    version: int = Field(default=1, ge=1)
    tags: list[str] = Field(default_factory=list)

    @field_validator("content")
    @classmethod
    def nonblank_content(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Memory content cannot be blank")
        return value

    @field_validator("created_at", "updated_at", "last_accessed_at", "expires_at")
    @classmethod
    def aware_timestamp(cls, value: datetime | None) -> datetime | None:
        if value is not None and value.utcoffset() is None:
            raise ValueError("Memory timestamps must include a timezone")
        return value

    def model_post_init(self, _context: Any) -> None:
        """Keep the derived hash consistent with the actual content."""
        self.content_hash = _content_hash(self.content)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_retrievable(self) -> bool:
        """Whether this memory should appear in retrieval results."""
        return self.status in {
            MemoryStatus.ACTIVE,
            MemoryStatus.SUPERSEDED,
            MemoryStatus.CONTRADICTED,
            MemoryStatus.EXPIRED,
        }

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_compilable(self) -> bool:
        """Whether this memory can be included in compiled context."""
        return self.status == MemoryStatus.ACTIVE


class MemoryUpdate(BaseModel):
    """Fields that can be updated on an existing memory.

    Only non-None fields are applied. This is a partial update model.
    """

    content: str | None = Field(default=None, min_length=1, max_length=10_000)
    type: MemoryType | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    importance: float | None = Field(default=None, ge=0.0, le=1.0)
    privacy_level: PrivacyLevel | None = None
    expires_at: datetime | None = None
    tags: list[str] | None = None

    @field_validator("content")
    @classmethod
    def nonblank_content(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("Memory content cannot be blank")
        return value

    @field_validator("expires_at")
    @classmethod
    def aware_expiry(cls, value: datetime | None) -> datetime | None:
        if value is not None and value.utcoffset() is None:
            raise ValueError("Memory expiry must include a timezone")
        return value


# ---------------------------------------------------------------------------
# Memory Relations
# ---------------------------------------------------------------------------


class MemoryRelation(BaseModel):
    """A typed, directional relationship between two memories."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID = Field(default_factory=uuid4)
    source_memory_id: UUID
    target_memory_id: UUID
    relation_type: RelationType
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=_utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Raw Events
# ---------------------------------------------------------------------------


class RawEvent(BaseModel):
    """An immutable record of something that happened in ContextOS.

    Events form the append-only audit log. Memories are derived from events,
    but events are never modified (except by explicit purge).
    """

    model_config = ConfigDict(from_attributes=True)

    id: UUID = Field(default_factory=uuid4)
    event_type: EventType
    timestamp: datetime = Field(default_factory=_utcnow)
    source_type: str = Field(default="system")
    source_uri: str | None = Field(default=None)
    content: str | None = Field(default=None)
    content_hash: str | None = Field(default=None)
    metadata: dict[str, Any] = Field(default_factory=dict)
    privacy_scan_result: dict[str, Any] | None = Field(default=None)
    memory_ids: list[UUID] = Field(default_factory=list)

    def model_post_init(self, _context: Any) -> None:
        """Compute content_hash for ingest events if content is present."""
        if self.content and not self.content_hash:
            self.content_hash = _content_hash(self.content)


# ---------------------------------------------------------------------------
# Secret Scanner Results
# ---------------------------------------------------------------------------


class SecretMatch(BaseModel):
    """A single secret detected in scanned text."""

    secret_type: SecretType
    start: int  # Character offset in original text
    end: int
    matched_text: str = Field(default="", repr=False)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class ScanResult(BaseModel):
    """Result of scanning text for secrets."""

    has_secrets: bool = False
    matches: list[SecretMatch] = Field(default_factory=list)
    scanned_length: int = 0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def secret_types_found(self) -> list[SecretType]:
        """Unique secret types detected."""
        return list({m.secret_type for m in self.matches})


class PrivacyFinding(BaseModel):
    """Secret finding safe to persist; it never contains the matched value."""

    category: SecretType
    severity: PrivacySeverity
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    detector: str = Field(min_length=1)
    location: str = Field(default="content", min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    safe_preview: str = Field(min_length=1)
    fingerprint: str = Field(min_length=64, max_length=64)

    @model_validator(mode="after")
    def finding_span_is_valid(self) -> PrivacyFinding:
        if self.end <= self.start:
            raise ValueError("Finding end must be greater than start")
        return self


class PrivacyAssessment(BaseModel):
    """Deterministic privacy decision containing only sanitized data."""

    decision: PrivacyDecision
    classification: PrivacyClassification
    source_trust: SourceTrust
    findings: list[PrivacyFinding] = Field(default_factory=list)
    sanitized_text: str = ""
    scanned_length: int = Field(ge=0)
    input_hash: str = Field(min_length=64, max_length=64)
    untrusted_instruction_detected: bool = False

    @computed_field  # type: ignore[prop-decorator]
    @property
    def has_findings(self) -> bool:
        return bool(self.findings)


# ---------------------------------------------------------------------------
# Retrieval Results
# ---------------------------------------------------------------------------


class ScoredMemory(BaseModel):
    """A memory with retrieval scores attached."""

    memory: Memory
    final_score: float = Field(ge=0.0)
    vector_score: float | None = None
    bm25_score: float | None = None
    rrf_rank: int = Field(default=0, ge=0)
    rerank_score: float | None = None
    rank: int = Field(default=0, ge=0)
    lexical_rank: int | None = Field(default=None, ge=1)
    dense_rank: int | None = Field(default=None, ge=1)
    metadata_adjustment: float = Field(default=0.0, ge=0.0)
    retrieval_sources: list[str] = Field(default_factory=list)


class StageTrace(BaseModel):
    """Trace data for a single pipeline stage."""

    stage_name: str
    input_count: int = 0
    output_count: int = 0
    latency_ms: float = 0.0
    input_tokens: int | None = None
    output_tokens: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalTrace(BaseModel):
    """Full trace of a retrieval operation."""

    stages: list[StageTrace] = Field(default_factory=list)
    total_latency_ms: float = 0.0
    total_candidates: int = 0
    total_results: int = 0


class RetrievalResult(BaseModel):
    """Complete result of a retrieval query."""

    query: str
    memories: list[ScoredMemory] = Field(default_factory=list)
    strategy_results: dict[str, list[ScoredMemory]] = Field(default_factory=dict)
    trace: RetrievalTrace = Field(default_factory=RetrievalTrace)


class RetrievalQuery(BaseModel):
    """Validated, side-effect-free retrieval request."""

    text: str = Field(min_length=1, max_length=10_000)
    k: int = Field(default=10, ge=1, le=200)
    mode: RetrievalMode = RetrievalMode.HYBRID
    temporal_scope: TemporalScope = TemporalScope.CURRENT
    allowed_memory_types: set[MemoryType] | None = None
    allowed_statuses: set[MemoryStatus] | None = None
    source_types: set[str] | None = None
    tags: set[str] | None = None
    created_after: datetime | None = None
    created_before: datetime | None = None
    min_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    min_importance: float | None = Field(default=None, ge=0.0, le=1.0)
    include_trace: bool = True
    apply_metadata_rerank: bool = True

    @field_validator("text")
    @classmethod
    def nonblank_query(cls, value: str) -> str:
        value = " ".join(value.split())
        if not value:
            raise ValueError("Retrieval query cannot be blank")
        return value

    @field_validator("created_after", "created_before")
    @classmethod
    def aware_retrieval_timestamp(cls, value: datetime | None) -> datetime | None:
        if value is not None and value.utcoffset() is None:
            raise ValueError("Retrieval timestamps must include a timezone")
        return value

    @model_validator(mode="after")
    def valid_time_range(self) -> RetrievalQuery:
        if self.created_after and self.created_before and self.created_after > self.created_before:
            raise ValueError("created_after must not be later than created_before")
        forbidden = {
            MemoryStatus.CANDIDATE,
            MemoryStatus.DELETED,
            MemoryStatus.PURGED,
            MemoryStatus.MERGED,
        }
        if self.allowed_statuses and self.allowed_statuses & forbidden:
            raise ValueError("Unaccepted or deleted memories are not retrievable")
        return self


# ---------------------------------------------------------------------------
# Token-aware selection
# ---------------------------------------------------------------------------


class ContextBudget(BaseModel):
    """Token allocation available to memory context only."""

    max_tokens: int = Field(ge=0)
    reserved_tokens: int = Field(default=0, ge=0)
    overhead_per_memory: int = Field(default=2, ge=0)

    @model_validator(mode="after")
    def reservation_fits(self) -> ContextBudget:
        if self.reserved_tokens > self.max_tokens:
            raise ValueError("reserved_tokens cannot exceed max_tokens")
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def available_tokens(self) -> int:
        return self.max_tokens - self.reserved_tokens


class CandidateDecision(BaseModel):
    """Inspectable optimization decision without raw memory content."""

    memory_id: UUID
    token_cost: int = Field(ge=0)
    content_tokens: int = Field(ge=0)
    overhead_tokens: int = Field(ge=0)
    retrieval_score: float = Field(ge=0.0)
    normalized_relevance: float = Field(ge=0.0, le=1.0)
    importance_contribution: float = Field(ge=0.0, le=0.1)
    confidence_contribution: float = Field(ge=0.0, le=0.1)
    support_contribution: float = Field(ge=0.0, le=0.05)
    lifecycle_multiplier: float = Field(ge=0.0, le=1.0)
    base_utility: float = Field(ge=0.0, le=1.0)
    marginal_utility: float = Field(ge=0.0)
    redundancy: float = Field(ge=0.0, le=1.0)
    selected: bool = False
    exclusion_reason: ExclusionReason | None = None
    redundant_with: UUID | None = None


class OptimizationTrace(BaseModel):
    """Structured account of every token-aware selection decision."""

    strategy: OptimizationStrategy
    candidate_count: int = Field(ge=0)
    eligible_count: int = Field(ge=0)
    selected_count: int = Field(ge=0)
    compiler_rescue_candidate_count: int = Field(default=0, ge=0)
    budget_tokens: int = Field(ge=0)
    reserved_tokens: int = Field(ge=0)
    available_tokens: int = Field(ge=0)
    tokens_used: int = Field(ge=0)
    remaining_tokens: int = Field(ge=0)
    latency_ms: float = Field(ge=0.0)
    decisions: list[CandidateDecision] = Field(default_factory=list)


class SelectionResult(BaseModel):
    """Whole-memory selection produced after retrieval and before compilation."""

    strategy: OptimizationStrategy
    selected_memories: list[ScoredMemory] = Field(default_factory=list)
    compiler_rescue_candidates: list[ScoredMemory] = Field(default_factory=list)
    total_tokens: int = Field(ge=0)
    content_tokens: int = Field(ge=0)
    overhead_tokens: int = Field(ge=0)
    budget: ContextBudget
    remaining_tokens: int = Field(ge=0)
    utilization: float = Field(ge=0.0, le=1.0)
    trace: OptimizationTrace


# ---------------------------------------------------------------------------
# Compilation Results
# ---------------------------------------------------------------------------


class CompilationTrace(BaseModel):
    """Trace data for context compilation."""

    stages: list[StageTrace] = Field(default_factory=list)
    memories_considered: int = 0
    memories_included: int = 0
    memories_excluded: int = 0
    normal_selected_inputs: int = Field(default=0, ge=0)
    oversized_rescue_inputs: int = Field(default=0, ge=0)
    rescued_facts_included: int = Field(default=0, ge=0)
    value_densities: dict[str, float] = Field(default_factory=dict)
    facts_created: int = Field(default=0, ge=0)
    facts_included: int = Field(default=0, ge=0)
    facts_excluded: int = Field(default=0, ge=0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    provenance_coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    total_latency_ms: float = 0.0


class ContextFact(BaseModel):
    """Fact-level compiler IR with complete source attribution."""

    fact_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    source_memory_ids: list[UUID] = Field(min_length=1)
    input_kind: CompilerInputKind = CompilerInputKind.NORMAL_SELECTED
    provenance_event_ids: list[UUID] = Field(default_factory=list)
    memory_type: MemoryType
    temporal_status: CandidateTemporalStatus = CandidateTemporalStatus.UNSPECIFIED
    confidence: float = Field(ge=0.0, le=1.0)
    importance: float = Field(ge=0.0, le=1.0)
    negated: bool = False
    uncertain: bool = False
    causal: bool = False
    query_relevance: float = Field(default=0.0, ge=0.0, le=1.0)
    token_cost: int = Field(default=0, ge=0)


class ExcludedContextFact(BaseModel):
    """Supported fact omitted from serialization with an explicit reason."""

    fact_id: str
    source_memory_ids: list[UUID] = Field(min_length=1)
    input_kind: CompilerInputKind = CompilerInputKind.NORMAL_SELECTED
    reason: FactExclusionReason
    token_cost: int = Field(default=0, ge=0)


class CompiledContext(BaseModel):
    """The final compiled context ready to send to an LLM."""

    query: str
    context_text: str
    total_tokens: int = Field(ge=0)
    budget: int = Field(ge=0)
    memories_considered: int = Field(ge=0)
    memories_included: int = Field(ge=0)
    memories_excluded: int = Field(ge=0)
    compression_ratio: float = Field(ge=0.0)
    included_memory_ids: list[UUID] = Field(default_factory=list)
    included_fact_ids: list[str] = Field(default_factory=list)
    facts: list[ContextFact] = Field(default_factory=list)
    excluded_facts: list[ExcludedContextFact] = Field(default_factory=list)
    provenance_map: dict[str, list[UUID]] = Field(default_factory=dict)
    input_tokens: int = Field(default=0, ge=0)
    utilization: float = Field(default=0.0, ge=0.0, le=1.0)
    unsupported_fact_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    provenance_coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    strategy: CompilationStrategy = CompilationStrategy.CONTEXTOS_COMPILER
    compression_level: CompressionLevel = CompressionLevel.LIGHT
    trace: CompilationTrace = Field(default_factory=CompilationTrace)


# ---------------------------------------------------------------------------
# Ingestion Models
# ---------------------------------------------------------------------------


class IngestRequest(BaseModel):
    """Request to ingest content into ContextOS."""

    model_config = ConfigDict(hide_input_in_errors=True)

    content: str = Field(min_length=1, repr=False)
    source_type: str = Field(default="cli_input")
    source_uri: str | None = Field(default=None, repr=False)
    source_role: SourceRole = SourceRole.USER
    confirmed_user_information: bool = False
    memory_type: MemoryType | None = None
    tags: list[str] = Field(default_factory=list, repr=False)
    skip_secret_scan: bool = Field(default=False)


class CandidateMemory(BaseModel):
    """Unaccepted memory candidate inferred from one raw input."""

    content: str = Field(min_length=1, max_length=10_000)
    memory_type: MemoryType = Field(default=MemoryType.CONTEXT)
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    temporal_status: CandidateTemporalStatus = CandidateTemporalStatus.UNSPECIFIED
    temporal_hint: str | None = None
    action_hint: CandidateAction = CandidateAction.ADD
    source_type: str = Field(default="cli_input", min_length=1)
    source_uri: str | None = None
    source_role: SourceRole = SourceRole.USER
    evidence: str = Field(min_length=1)
    evidence_start: int | None = Field(default=None, ge=0)
    evidence_end: int | None = Field(default=None, ge=0)
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    privacy_assessment: PrivacyAssessment | None = None

    @field_validator("content", "evidence", "source_type")
    @classmethod
    def candidate_strings_are_not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Candidate text fields cannot be blank")
        return value

    @model_validator(mode="after")
    def evidence_span_is_valid(self) -> CandidateMemory:
        if (self.evidence_start is None) != (self.evidence_end is None):
            raise ValueError("Evidence start and end must be provided together")
        if self.evidence_start is not None and self.evidence_end <= self.evidence_start:
            raise ValueError("Evidence end must be greater than evidence start")
        return self

    @property
    def type(self) -> MemoryType:
        """Compatibility accessor for callers that previously used ExtractedMemory.type."""
        return self.memory_type


# Compatibility name for integrations written against the initial baseline.
ExtractedMemory = CandidateMemory


class IngestResult(BaseModel):
    """Result of an ingestion operation."""

    event_id: UUID
    candidates: list[CandidateMemory] = Field(default_factory=list)
    privacy_assessment: PrivacyAssessment | None = None
    blocked_candidate_assessments: list[PrivacyAssessment] = Field(default_factory=list)
    memories_created: list[UUID] = Field(default_factory=list)
    memories_updated: list[UUID] = Field(default_factory=list)
    memories_merged: list[UUID] = Field(default_factory=list)
    secrets_detected: bool = False
    secrets_redacted: bool = False
    warnings: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Retrieval / Compilation Config
# ---------------------------------------------------------------------------


class RetrievalConfig(BaseModel):
    """Configuration for a retrieval query."""

    vector_top_k: int = Field(default=20, ge=1, le=200)
    bm25_top_k: int = Field(default=20, ge=1, le=200)
    rrf_k: int = Field(default=60, ge=1)
    include_superseded: bool = False
    include_contradicted: bool = True
    include_expired: bool = False
    max_results: int = Field(default=50, ge=1, le=200)
    min_score: float = Field(default=0.0, ge=0.0)


class CompilationConfig(BaseModel):
    """Configuration for context compilation."""

    budget: int = Field(default=4000, ge=0, le=32_000)
    strategy: CompilationStrategy = CompilationStrategy.CONTEXTOS_COMPILER
    compression_level: CompressionLevel = CompressionLevel.LIGHT
    include_sources: bool = False
    include_confidence: bool = False
    format: str = Field(default="text")  # "text" or "json"


# ---------------------------------------------------------------------------
# System Status
# ---------------------------------------------------------------------------


class SystemStatus(BaseModel):
    """Current system health and statistics."""

    daemon_running: bool = False
    pid: int | None = None
    uptime_seconds: float = 0.0
    total_memories: int = 0
    active_memories: int = 0
    total_events: int = 0
    embedding_model: str = ""
    embedding_model_loaded: bool = False
    vector_index_size: int = 0
    bm25_index_size: int = 0
    database_size_bytes: int = 0
    data_directory: str = ""


class TokenStats(BaseModel):
    """Aggregate token statistics."""

    total_tokens_stored: int = 0
    total_compilations: int = 0
    total_tokens_compiled: int = 0
    total_tokens_saved: int = 0
    average_compression_ratio: float = 0.0
    tokens_per_memory: float = 0.0


# ---------------------------------------------------------------------------
# Vector / Lexical search result types used by storage protocols
# ---------------------------------------------------------------------------


class VectorResult(BaseModel):
    """Result from vector store search."""

    id: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class LexicalResult(BaseModel):
    """Result from BM25 / lexical search."""

    id: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------


class MemoryFilters(BaseModel):
    """Filters for listing/querying memories."""

    status: MemoryStatus | None = None
    type: MemoryType | None = None
    privacy_level: PrivacyLevel | None = None
    source_type: str | None = None
    tags: list[str] | None = None
    created_after: datetime | None = None
    created_before: datetime | None = None
    min_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    min_importance: float | None = Field(default=None, ge=0.0, le=1.0)
    limit: int = Field(default=50, ge=1, le=500)
    offset: int = Field(default=0, ge=0)


class EventFilters(BaseModel):
    """Filters for querying events."""

    event_type: EventType | None = None
    source_type: str | None = None
    after: datetime | None = None
    before: datetime | None = None
    memory_id: UUID | None = None
    limit: int = Field(default=50, ge=1, le=500)
    offset: int = Field(default=0, ge=0)
