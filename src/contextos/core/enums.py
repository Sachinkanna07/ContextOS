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

    SUPPORTS = "supports"
    """Source memory reinforces target memory."""

    RELATED = "related"
    """Topical or contextual relation."""

    DERIVED_FROM = "derived_from"
    """Source memory was extracted/summarized from target."""

    PART_OF = "part_of"
    """Source memory is a component of target."""


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
