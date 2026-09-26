"""Core domain package for ContextOS.

Re-exports the most commonly used types for convenience.
"""

from contextos.core.enums import (
    CandidateAction,
    CandidateTemporalStatus,
    EventType,
    MemoryStatus,
    MemoryType,
    PrivacyLevel,
    RelationType,
    SecretDetectionMode,
    SecretType,
    SourceRole,
)
from contextos.core.exceptions import (
    ConcurrencyError,
    ContextOSError,
    InvalidTransitionError,
    MemoryNotFoundError,
    SecretDetectedError,
)
from contextos.core.models import (
    CandidateMemory,
    CompiledContext,
    CompilationConfig,
    ExtractedMemory,
    IngestRequest,
    IngestResult,
    Memory,
    MemoryFilters,
    MemoryRelation,
    MemoryUpdate,
    RawEvent,
    RetrievalConfig,
    RetrievalResult,
    ScanResult,
    ScoredMemory,
)

__all__ = [
    # Enums
    "CandidateAction",
    "CandidateTemporalStatus",
    "EventType",
    "MemoryStatus",
    "MemoryType",
    "PrivacyLevel",
    "RelationType",
    "SecretDetectionMode",
    "SecretType",
    "SourceRole",
    # Exceptions
    "ConcurrencyError",
    "ContextOSError",
    "InvalidTransitionError",
    "MemoryNotFoundError",
    "SecretDetectedError",
    # Models
    "CandidateMemory",
    "CompiledContext",
    "CompilationConfig",
    "ExtractedMemory",
    "IngestRequest",
    "IngestResult",
    "Memory",
    "MemoryFilters",
    "MemoryRelation",
    "MemoryUpdate",
    "RawEvent",
    "RetrievalConfig",
    "RetrievalResult",
    "ScanResult",
    "ScoredMemory",
]
