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
from builtins import property as builtin_property
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
    GraphNodeType,
    GraphRelationType,
    ModelFinishReason,
    ProviderDispatchState,
    ProviderType,
    RelationType,
    RetrievalMode,
    RoutingPolicy,
    SecretType,
    SourceRole,
    SourceTrust,
    TemporalOutcome,
    TemporalPrecision,
    TemporalScope,
    TokenMeasurementSource,
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


class MemorySlot(BaseModel):
    """Inspectable identity for a logical user-memory property and scope."""

    subject: str = Field(default="user", min_length=1)
    property: str = Field(min_length=1)
    scope: str = Field(default="global", min_length=1)
    entity: str | None = None
    qualifiers: tuple[str, ...] = ()

    @field_validator("subject", "property", "scope", "entity")
    @classmethod
    def normalize_component(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = "_".join(value.strip().casefold().split())
        if not normalized:
            raise ValueError("Memory slot components cannot be blank")
        return normalized

    @computed_field  # type: ignore[prop-decorator]
    @builtin_property
    def key(self) -> str:
        parts = [self.subject, self.property, self.scope, self.entity or "-"]
        if self.qualifiers:
            parts.append(",".join(sorted(self.qualifiers)))
        return "/".join(parts)


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
    observed_at: datetime = Field(default_factory=_utcnow)
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    temporal_precision: TemporalPrecision = TemporalPrecision.UNKNOWN
    temporal_status: CandidateTemporalStatus = CandidateTemporalStatus.UNSPECIFIED
    temporal_expression: str | None = None
    slot: MemorySlot | None = None
    uncertain: bool = False
    negated: bool = False
    resolution_reason: str | None = None
    resolution_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
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

    @field_validator(
        "created_at", "updated_at", "last_accessed_at", "expires_at",
        "observed_at", "valid_from", "valid_to",
    )
    @classmethod
    def aware_timestamp(cls, value: datetime | None) -> datetime | None:
        if value is not None and value.utcoffset() is None:
            raise ValueError("Memory timestamps must include a timezone")
        return value

    @model_validator(mode="after")
    def temporal_interval_and_links_are_valid(self) -> Memory:
        if self.valid_from and self.valid_to and self.valid_to < self.valid_from:
            raise ValueError("valid_to must not be earlier than valid_from")
        if self.superseded_by == self.id or self.supersedes == self.id:
            raise ValueError("A memory cannot supersede itself")
        return self

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

    @model_validator(mode="after")
    def relation_is_not_self_referential(self) -> MemoryRelation:
        if self.source_memory_id == self.target_memory_id:
            raise ValueError("A memory relation cannot target itself")
        return self


class GraphNode(BaseModel):
    """Stable node in the rebuildable property graph."""

    id: UUID
    node_type: GraphNodeType
    canonical_key: str = Field(min_length=1)
    label: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)


class GraphEdgeSupport(BaseModel):
    """A memory that provides provenance for a derived graph edge."""

    edge_id: UUID
    memory_id: UUID
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    provenance_event_id: UUID | None = None
    created_at: datetime = Field(default_factory=_utcnow)


class GraphEdge(BaseModel):
    """Deduplicated typed edge with one or more independent supports."""

    id: UUID
    source_node_id: UUID
    target_node_id: UUID
    relation_type: GraphRelationType
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    directed: bool = True
    scope_key: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    supports: list[GraphEdgeSupport] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)

    @model_validator(mode="after")
    def edge_is_not_self_referential(self) -> GraphEdge:
        if self.source_node_id == self.target_node_id:
            raise ValueError("A graph edge cannot target itself")
        return self


class GraphPathNode(BaseModel):
    """Structured graph node evidence along a traversed path."""

    node_id: UUID
    node_type: GraphNodeType
    label: str | None = None
    project_scope: str | None = None


class GraphPathEdge(BaseModel):
    """Structured graph edge evidence along a traversed path."""

    edge_type: GraphRelationType
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    supporting_memory_ids: list[UUID] = Field(default_factory=list)
    project_scope: str | None = None


class GraphCandidateEvidence(BaseModel):
    """Structured explanation connecting a seed memory to a graph candidate."""

    seed_memory_id: UUID | None = None
    candidate_memory_id: UUID
    hop_count: int = Field(ge=1, le=3)
    graph_score: float = Field(ge=0.0, le=1.0)
    path_nodes: list[GraphPathNode] = Field(default_factory=list)
    path_edges: list[GraphPathEdge] = Field(default_factory=list)
    scope_match: bool | None = None


class GraphPath(BaseModel):
    """Content-free explanation for one graph-derived memory candidate."""

    seed_node_ids: list[UUID]
    node_ids: list[UUID]
    node_types: list[GraphNodeType]
    edge_ids: list[UUID]
    edge_types: list[GraphRelationType]
    hop_count: int = Field(ge=1, le=3)
    graph_contribution: float = Field(ge=0.0, le=1.0)
    source_memory_ids: list[UUID] = Field(default_factory=list)
    path_nodes: list[GraphPathNode] = Field(default_factory=list)
    path_edges: list[GraphPathEdge] = Field(default_factory=list)
    scope_match: bool | None = None


class GraphExpansion(BaseModel):
    """Bounded graph traversal result and its safe trace."""

    seed_node_ids: list[UUID] = Field(default_factory=list)
    candidate_scores: dict[UUID, float] = Field(default_factory=dict)
    candidate_paths: dict[UUID, list[GraphPath]] = Field(default_factory=dict)
    visited_node_ids: list[UUID] = Field(default_factory=list)
    traversed_edge_ids: list[UUID] = Field(default_factory=list)


class TemporalChange(BaseModel):
    """Value-free lifecycle mutation included in a temporal trace."""

    memory_id: UUID
    from_status: MemoryStatus | None = None
    to_status: MemoryStatus


class TemporalDecision(BaseModel):
    """Inspectable resolution plan produced before any database mutation."""

    candidate_id: UUID
    slot: MemorySlot
    outcome: TemporalOutcome
    compared_memory_ids: list[UUID] = Field(default_factory=list)
    related_memory_id: UUID | None = None
    evidence: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    changes: list[TemporalChange] = Field(default_factory=list)


class TemporalResolutionResult(BaseModel):
    """Persisted result of one atomic temporal resolution."""

    decision: TemporalDecision
    memory: Memory
    affected_memories: list[Memory] = Field(default_factory=list)
    relations: list[MemoryRelation] = Field(default_factory=list)


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
    graph_score: float | None = Field(default=None, ge=0.0, le=1.0)
    graph_rank: int | None = Field(default=None, ge=1)
    graph_paths: list[GraphPath] = Field(default_factory=list)


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
    lexical_candidate_ids: list[str] = Field(default_factory=list)
    dense_candidate_ids: list[str] = Field(default_factory=list)
    pre_limit_candidate_ids: list[str] = Field(default_factory=list)
    channel_candidates_truncated: bool = False
    pre_limit_candidates_truncated: bool = False


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
    graph_max_hops: int = Field(default=2, ge=1, le=3)
    graph_min_confidence: float = Field(default=0.6, ge=0.0, le=1.0)
    graph_max_nodes: int = Field(default=100, ge=1, le=1000)
    graph_max_edges: int = Field(default=250, ge=1, le=2500)

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
    temporal_precision: TemporalPrecision = TemporalPrecision.UNKNOWN
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    observed_at: datetime | None = None
    slot: MemorySlot | None = None
    uncertain: bool = False
    negated: bool = False
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
        if (
            self.evidence_start is not None
            and self.evidence_end is not None
            and self.evidence_end <= self.evidence_start
        ):
            raise ValueError("Evidence end must be greater than evidence start")
        if self.valid_from and self.valid_to and self.valid_to < self.valid_from:
            raise ValueError("valid_to must not be earlier than valid_from")
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
    include_contradicted: bool = False
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
    total_compilations: int | None = None
    total_tokens_compiled: int | None = None
    total_tokens_saved: int | None = None
    average_compression_ratio: float | None = None
    tokens_per_memory: float = 0.0
    measurement_basis: str | None = None


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


# ---------------------------------------------------------------------------
# Phase 9: Model Capabilities, Requests, Responses, and Routing
# ---------------------------------------------------------------------------


class ModelCapabilities(BaseModel):
    """Structured, inspectable capability metadata for a model."""

    provider_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    display_name: str = Field(min_length=1)
    context_window: int = Field(ge=1)
    max_output_tokens: int = Field(default=2048, ge=1)
    supports_tools: bool = False
    supports_json: bool = False
    supports_vision: bool = False
    local: bool = True
    tokenizer_family: str = Field(default="cl100k_base")
    enabled: bool = True
    cost_per_million_input: float | None = Field(default=None, ge=0.0)
    cost_per_million_output: float | None = Field(default=None, ge=0.0)
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ModelRequest(BaseModel):
    """Provider-neutral model execution request."""

    user_prompt: str = Field(min_length=1)
    model: str | None = None
    provider: str | None = None
    system_prompt: str | None = None
    compiled_context: CompiledContext | None = None
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_output_tokens: int | None = Field(default=1024, ge=1)
    timeout_seconds: float = Field(default=30.0, ge=0.5, le=600.0)
    routing_policy: RoutingPolicy | None = None
    allow_fallback: bool = False
    required_capabilities: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ModelResponse(BaseModel):
    """Standardized response from downstream model generation."""

    text: str
    model_id: str
    provider_id: str
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    latency_ms: float = Field(default=0.0, ge=0.0)
    finish_reason: ModelFinishReason = ModelFinishReason.STOP
    token_measurement_source: TokenMeasurementSource = TokenMeasurementSource.PROVIDER_REPORTED
    raw_usage: dict[str, Any] | None = None
    request_id: str | None = None
    error: str | None = None


class RouteDecision(BaseModel):
    """Traceable decision record produced by the model router."""

    policy: RoutingPolicy
    reason: str
    candidates_evaluated: list[str] = Field(default_factory=list)
    selected_provider: str
    selected_model: str
    fallback_used: bool = False
    initial_provider: str | None = None
    fallback_reason: str | None = None
    routing_latency_ms: float = Field(default=0.0, ge=0.0)


# ---------------------------------------------------------------------------
# Phase 9: Model Invocation Telemetry & Aggregation
# ---------------------------------------------------------------------------


class ModelInvocationTelemetry(BaseModel):
    """Structured, privacy-safe record of a single model invocation."""

    # Identifiers
    invocation_id: UUID = Field(default_factory=uuid4)
    session_id: str | None = None
    provider_id: str
    model_id: str
    is_local: bool
    timestamp: datetime = Field(default_factory=_utcnow)

    # Context Pipeline Tokens
    candidate_context_tokens: int = Field(default=0, ge=0)
    retrieved_context_tokens: int = Field(default=0, ge=0)
    optimized_context_tokens: int = Field(default=0, ge=0)
    compiled_context_tokens: int = Field(default=0, ge=0)
    prompt_tokens_before_context: int = Field(default=0, ge=0)
    preflight_input_tokens: int = Field(default=0, ge=0)
    final_input_tokens: int = Field(default=0, ge=0)

    # Provider Usage
    provider_input_tokens: int = Field(default=0, ge=0)
    provider_output_tokens: int = Field(default=0, ge=0)
    provider_total_tokens: int = Field(default=0, ge=0)
    token_measurement_source: TokenMeasurementSource = TokenMeasurementSource.PROVIDER_REPORTED
    context_token_measurement_source: TokenMeasurementSource | None = None
    context_tokenizer: str | None = None

    # Savings & Reduction
    estimated_full_history_tokens: int | None = None
    context_tokens_avoided: int = Field(default=0, ge=0)
    reduction_ratio: float = Field(default=0.0, ge=0.0, le=1.0)

    # Retrieval / Augmentation Contributions
    lexical_candidate_count: int = Field(default=0, ge=0)
    dense_candidate_count: int = Field(default=0, ge=0)
    hybrid_candidate_count: int = Field(default=0, ge=0)
    graph_expanded_count: int = Field(default=0, ge=0)
    temporal_filtered_count: int = Field(default=0, ge=0)
    selected_memory_count: int = Field(default=0, ge=0)
    compiled_fact_count: int = Field(default=0, ge=0)

    # Stage Timings (milliseconds)
    retrieval_ms: float = Field(default=0.0, ge=0.0)
    optimization_ms: float = Field(default=0.0, ge=0.0)
    compilation_ms: float = Field(default=0.0, ge=0.0)
    routing_ms: float = Field(default=0.0, ge=0.0)
    token_counting_ms: float = Field(default=0.0, ge=0.0)
    provider_latency_ms: float = Field(default=0.0, ge=0.0)
    end_to_end_ms: float = Field(default=0.0, ge=0.0)

    # Routing Audit
    routing_policy: RoutingPolicy = RoutingPolicy.LOCAL_FIRST
    routing_reason: str = ""
    selected_provider: str = ""
    selected_model: str = ""
    fallback_used: bool = False
    fallback_reason: str | None = None

    # Status & Evaluation Placeholders
    finish_reason: ModelFinishReason = ModelFinishReason.STOP
    status: str = "success"
    error_code: str | None = None
    answer_score: float | None = None
    required_fact_coverage: float | None = None
    benchmark_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class TelemetrySummary(BaseModel):
    """Aggregated invocation metrics for CLI and dashboard consumption."""

    total_invocations: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens_avoided: int = 0
    average_reduction_ratio: float = 0.0
    weighted_reduction_ratio: float = 0.0
    average_provider_latency_ms: float = 0.0
    local_invocations: int = 0
    remote_invocations: int = 0
    by_provider: dict[str, Any] = Field(default_factory=dict)
    by_model: dict[str, Any] = Field(default_factory=dict)


class ProviderDispatchEvidence(BaseModel):
    """Receipt proving exact downstream request construction and dispatch state."""

    provider_id: str
    model_id: str
    state: ProviderDispatchState
    compiled_context_sha256: str
    logical_request_sha256: str | None = None
    compiled_context_in_request: bool = False
    preflight_input_tokens: int = 0
    provider_input_tokens: int | None = None
    provider_response_received: bool = False
    measurement_source: str | None = None
    context_match: bool = False


class AskResult(BaseModel):
    """Result of an end-to-end ContextOSModelService ask execution."""

    response: ModelResponse
    compiled_context: CompiledContext
    route_decision: RouteDecision
    telemetry: ModelInvocationTelemetry
    dispatch_evidence: ProviderDispatchEvidence | None = None
    explanation: dict[str, Any] | None = None
