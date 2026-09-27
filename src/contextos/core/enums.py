"""Core domain enumerations for ContextOS.

These enums define the controlled vocabularies used across the system.
Every enum value is a lowercase string for consistent serialization and storage.
"""

from enum import StrEnum


class MemoryStatus(StrEnum):
    """Lifecycle states for memories.

    State machine transitions are enforced by MemoryService.transition().
    See Phase 0 spec Section 8 for the full state diagram.
    """

    CANDIDATE = "candidate"
    """Freshly extracted, not yet validated against existing memories."""

    ACTIVE = "active"
    """Validated, current, trusted memory. Retrievable and compilable."""

    SUPERSEDED = "superseded"
    """Replaced by a newer memory. Retains link to successor."""

    CONTRADICTED = "contradicted"
    """Conflicts with another memory. Pending resolution."""

    EXPIRED = "expired"
    """Past its relevance window (e.g., temporal memories)."""

    HISTORICAL = "historical"
    """Archived for provenance. Not actively retrieved."""

    DELETED = "deleted"
    """Soft-deleted. Not retrievable. Retained for audit/undo."""

    PURGED = "purged"
    """Hard-deleted. Data destroyed from all stores. Transient state."""

    MERGED = "merged"
    """Candidate was absorbed into an existing ACTIVE memory. Transient."""


# Valid state transitions: {from_state: {allowed_to_states}}
# Enforced at the service layer, not the enum layer.
VALID_TRANSITIONS: dict[MemoryStatus, set[MemoryStatus]] = {
    MemoryStatus.CANDIDATE: {
        MemoryStatus.ACTIVE,
        MemoryStatus.MERGED,
        MemoryStatus.CONTRADICTED,
    },
    MemoryStatus.ACTIVE: {
        MemoryStatus.SUPERSEDED,
        MemoryStatus.CONTRADICTED,
        MemoryStatus.EXPIRED,
        MemoryStatus.DELETED,
        MemoryStatus.ACTIVE,  # Reinforcement (confidence increase)
    },
    MemoryStatus.CONTRADICTED: {
        MemoryStatus.ACTIVE,
        MemoryStatus.SUPERSEDED,
        MemoryStatus.DELETED,
    },
    MemoryStatus.SUPERSEDED: {
        MemoryStatus.HISTORICAL,
    },
    MemoryStatus.EXPIRED: {
        MemoryStatus.HISTORICAL,
    },
    MemoryStatus.HISTORICAL: {
        MemoryStatus.DELETED,
    },
    MemoryStatus.DELETED: {
        MemoryStatus.PURGED,
    },
    # Terminal states — no transitions out
    MemoryStatus.PURGED: set(),
    MemoryStatus.MERGED: set(),
}


class MemoryType(StrEnum):
    """Categories of memories.

    These are not mutually exclusive in reality, but each memory gets
    exactly one primary type. This aids filtering and retrieval weighting.
    """

    PREFERENCE = "preference"
    """User preference: 'I prefer dark mode', 'I like Python over Java'."""

    FACT = "fact"
    """Factual statement about the user: 'I work at Acme Corp'."""

    SKILL = "skill"
    """Skill or competency: 'I'm proficient in Rust'."""

    PROJECT = "project"
    """Project information: 'I'm building a CLI tool called foo'."""

    RELATIONSHIP = "relationship"
    """Interpersonal relationship: 'Alice is my tech lead'."""

    PROCEDURE = "procedure"
    """How-to or workflow: 'To deploy, I run make deploy'."""

    OPINION = "opinion"
    """Subjective judgment: 'I think ORMs are overengineered'."""

    TEMPORAL = "temporal"
    """Time-bound context: 'I'm on vacation next week'."""

    GOAL = "goal"
    """Aspiration or objective: 'I want to learn Kubernetes'."""

    CONTEXT = "context"
    """General contextual information that doesn't fit other categories."""


class RetrievalMode(StrEnum):
    """Available retrieval strategies."""

    LEXICAL = "lexical"
    DENSE = "dense"
    HYBRID = "hybrid"
    GRAPH = "graph"
    HYBRID_GRAPH = "hybrid_graph"


class TemporalScope(StrEnum):
    """Lifecycle window considered by a retrieval query."""

    CURRENT = "current"
    HISTORICAL = "historical"
    ALL = "all"


class OptimizationStrategy(StrEnum):
    """Algorithms available for memory-context selection."""

    TOP_RANK = "top_rank"
    TOP_RANK_STOP = "top_rank"
    """Explicit alias for the original stop-at-first-nonfit baseline."""

    TOP_RANK_SKIP = "top_rank_skip"
    GREEDY = "greedy"
    CONTEXTOS = "contextos"


class ExclusionReason(StrEnum):
    """Why an optimizer candidate was not selected."""

    BUDGET_EXHAUSTED = "budget_exhausted"
    DUPLICATE_ID = "duplicate_id"
    INVALID_LIFECYCLE = "invalid_lifecycle"
    LOW_RELEVANCE = "low_relevance"
    OVERSIZED = "oversized"
    REDUNDANT = "redundant"


class CompilationStrategy(StrEnum):
    """Representation strategies used after memory selection."""

    RAW_CONCAT = "raw_concat"
    DEDUP_ONLY = "dedup_only"
    CONTEXTOS_COMPILER = "contextos_compiler"


class CompressionLevel(StrEnum):
    """Amount of deterministic, extractive fact reduction."""

    NONE = "none"
    LIGHT = "light"
    AGGRESSIVE = "aggressive"


class FactExclusionReason(StrEnum):
    """Why a supported source fact was not emitted."""

    BUDGET = "budget"
    DUPLICATE = "duplicate"
    EMPTY = "empty"
    PRIVACY_RESTRICTED = "privacy_restricted"
    QUERY_IRRELEVANT = "query_irrelevant"


class CompilerInputKind(StrEnum):
    """How a memory reached the fact compiler."""

    NORMAL_SELECTED = "normal_selected"
    OVERSIZED_RESCUE = "oversized_rescue"


class CandidateTemporalStatus(StrEnum):
    """Temporal interpretation inferred during candidate extraction."""

    CURRENT = "current"
    HISTORICAL = "historical"
    FUTURE = "future"
    UNSPECIFIED = "unspecified"


class TemporalPrecision(StrEnum):
    """Precision of a claimed effective time without false timestamp accuracy."""

    EXACT = "exact"
    DATE = "date"
    MONTH = "month"
    YEAR = "year"
    RELATIVE = "relative"
    UNKNOWN = "unknown"


class TemporalOutcome(StrEnum):
    """Deterministic result of resolving an accepted memory candidate."""

    ADD_NEW = "add_new"
    DUPLICATE = "duplicate"
    COEXIST = "coexist"
    SUPERSEDE = "supersede"
    CONTRADICT = "contradict"
    CORRECT = "correct"
    NO_CHANGE = "no_change"


class CandidateAction(StrEnum):
    """Non-binding hint for a later candidate validation stage."""

    ADD = "add"
    SUPERSEDE = "supersede"


class SourceRole(StrEnum):
    """Who authored the text presented to the extractor."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class PrivacyLevel(StrEnum):
    """Privacy classification for memories.

    Controls whether a memory can be sent to external LLM providers.
    """

    PUBLIC = "public"
    """Can be sent to any LLM. No personal information."""

    PERSONAL = "personal"
    """Personal data. Can be sent to LLM but is PII-adjacent."""

    SENSITIVE = "sensitive"
    """Only sent with explicit user consent per query."""

    RESTRICTED = "restricted"
    """Never sent externally. Local retrieval only."""


class EventType(StrEnum):
    """Types of events in the append-only event store."""

    INGEST = "ingest"
    """User ingested content into the system."""

    MEMORY_CREATED = "memory_created"
    """A new memory was extracted and stored."""

    MEMORY_UPDATED = "memory_updated"
    """Memory content or metadata was modified."""

    MEMORY_TRANSITION = "memory_transition"
    """Memory lifecycle state changed."""

    MEMORY_DELETED = "memory_deleted"
    """Memory was soft-deleted."""

    MEMORY_PURGED = "memory_purged"
    """Memory was hard-deleted (irreversible)."""

    SECRET_DETECTED = "secret_detected"
    """Secret scanner found sensitive content in input."""

    CONFLICT_DETECTED = "conflict_detected"
    """Duplicate or contradiction detected during ingestion."""

    CONFLICT_RESOLVED = "conflict_resolved"
    """A previously detected conflict was resolved."""

    RETRIEVAL = "retrieval"
    """A retrieval query was executed."""

    COMPILATION = "compilation"
    """Context was compiled for LLM consumption."""


class RelationType(StrEnum):
    """Types of relationships between memories."""

    SUPERSEDES = "supersedes"
    """Source memory replaces target memory."""

    CONTRADICTS = "contradicts"
    """Source memory conflicts with target memory."""

    CORRECTS = "corrects"
    """Source memory explicitly corrects target memory."""

    COEXISTS_WITH = "coexists_with"
    """Memories share a property but apply to compatible scopes."""

    DUPLICATE_OF = "duplicate_of"
    """Source candidate repeats an existing memory without changing state."""

    SUPPORTS = "supports"
    """Source memory reinforces target memory."""

    RELATED = "related"
    """Topical or contextual relation."""

    DERIVED_FROM = "derived_from"
    """Source memory was extracted/summarized from target."""

    PART_OF = "part_of"
    """Source memory is a component of target."""


class GraphNodeType(StrEnum):
    """Node kinds stored in the rebuildable memory graph."""

    MEMORY = "memory"
    PROJECT = "project"
    TOOL = "tool"
    CONCEPT = "concept"


class GraphRelationType(StrEnum):
    """Conservative edge vocabulary for explicit graph facts."""

    ABOUT = "about"
    MENTIONS = "mentions"
    USES = "uses"
    RUNS = "runs"
    DEPENDS_ON = "depends_on"
    PART_OF = "part_of"
    BELONGS_TO = "belongs_to"
    WORKS_ON = "works_on"
    SUPERSEDES = "supersedes"
    CONTRADICTS = "contradicts"
    CORRECTS = "corrects"
    COEXISTS_WITH = "coexists_with"
    DUPLICATE_OF = "duplicate_of"
    SUPPORTS = "supports"
    RELATED = "related"
    DERIVED_FROM = "derived_from"


class SecretDetectionMode(StrEnum):
    """How the secret scanner handles detected secrets."""

    STRICT = "strict"
    """Reject the entire input if any secret is detected."""

    REDACT = "redact"
    """Replace detected secrets with [REDACTED:type] placeholders."""

    WARN = "warn"
    """Store with warning flag. Memory marked as RESTRICTED."""


class SecretType(StrEnum):
    """Types of secrets the scanner can detect."""

    AWS_ACCESS_KEY = "aws_access_key"
    AWS_SECRET_KEY = "aws_secret_key"
    GITHUB_TOKEN = "github_token"
    OPENAI_API_KEY = "openai_api_key"
    ANTHROPIC_API_KEY = "anthropic_api_key"
    GENERIC_API_KEY = "generic_api_key"
    PRIVATE_KEY = "private_key"
    SSH_PRIVATE_KEY = "ssh_private_key"
    PASSWORD = "password"
    CONNECTION_STRING = "connection_string"
    JWT = "jwt"
    HIGH_ENTROPY = "high_entropy"
    GOOGLE_API_KEY = "google_api_key"
    SLACK_TOKEN = "slack_token"
    STRIPE_KEY = "stripe_key"
    BEARER_TOKEN = "bearer_token"
    AUTHORIZATION_HEADER = "authorization_header"
    OTP = "otp"
    SESSION_COOKIE = "session_cookie"
    ACCESS_TOKEN = "access_token"


class PrivacySeverity(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class PrivacyDecision(StrEnum):
    ALLOW = "allow"
    REDACT = "redact"
    REJECT = "reject"
    QUARANTINE = "quarantine"


class PrivacyClassification(StrEnum):
    SAFE = "safe"
    SENSITIVE = "sensitive"
    SECRET = "secret"
    BLOCKED = "blocked"


class SourceTrust(StrEnum):
    DIRECT_USER = "direct_user"
    LOCAL_TRUSTED_CONNECTOR = "local_trusted_connector"
    EXTERNAL_WEBPAGE = "external_webpage"
    IMPORTED_DOCUMENT = "imported_document"
    TOOL_OUTPUT = "tool_output"
    MODEL_OUTPUT = "model_output"


# ---------------------------------------------------------------------------
# Phase 9: Model Runtime, Routing, and Telemetry
# ---------------------------------------------------------------------------


class RoutingPolicy(StrEnum):
    """Policies guiding deterministic model routing decisions."""

    EXPLICIT = "explicit"
    """Target provider and/or model explicitly requested."""

    LOCAL_FIRST = "local_first"
    """Prefer healthy local runtime; fallback to cloud only if configured."""

    FIXED_DEFAULT = "fixed_default"
    """Route strictly to the configured default provider/model."""

    CAPABILITY_AWARE = "capability_aware"
    """Select model meeting required capabilities and context window fit."""


class TokenMeasurementSource(StrEnum):
    """Provenance/measurement method for token counts."""

    PROVIDER_REPORTED = "provider_reported"
    """Exact usage returned in the provider's API response."""

    TOKENIZER_COUNTED = "tokenizer_counted"
    """Locally calculated via exact or target-model tokenizer."""

    APPROXIMATED = "approximated"
    """Calculated via deterministic heuristic/approximation."""


class ModelFinishReason(StrEnum):
    """Reason why downstream generation completed."""

    STOP = "stop"
    LENGTH = "length"
    TIMEOUT = "timeout"
    ERROR = "error"
    CONTENT_FILTER = "content_filter"
    UNKNOWN = "unknown"


class ProviderType(StrEnum):
    """Supported provider adapter classes."""

    FAKE = "fake"
    OLLAMA = "ollama"
    OPENAI_COMPATIBLE = "openai_compatible"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
