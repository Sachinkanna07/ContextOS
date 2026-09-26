"""Core domain package for ContextOS.

Re-exports the most commonly used types for convenience.
"""

from contextos.core.enums import (
    EventType,
    MemoryStatus,
    MemoryType,
    PrivacyLevel,
    RelationType,
    SecretDetectionMode,
    SecretType,
)
from contextos.core.exceptions import (
    ConcurrencyError,
    ContextOSError,
    InvalidTransitionError,
    MemoryNotFoundError,
    SecretDetectedError,
)
from contextos.core.models import (
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
    "EventType",
    "MemoryStatus",
    "MemoryType",
    "PrivacyLevel",
    "RelationType",
    "SecretDetectionMode",
    "SecretType",
    # Exceptions
    "ConcurrencyError",
    "ContextOSError",
    "InvalidTransitionError",
    "MemoryNotFoundError",
    "SecretDetectedError",
    # Models
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
