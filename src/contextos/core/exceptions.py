"""Domain-specific exceptions for ContextOS.

All exceptions inherit from ContextOSError so callers can catch
the entire hierarchy with a single except clause when needed.
"""


class ContextOSError(Exception):
    """Base exception for all ContextOS errors."""


# --- Lifecycle Errors ---


class InvalidTransitionError(ContextOSError):
    """Raised when a memory lifecycle transition is not allowed."""

    def __init__(self, memory_id: str, from_status: str, to_status: str) -> None:
        self.memory_id = memory_id
        self.from_status = from_status
        self.to_status = to_status
        super().__init__(
            f"Invalid transition for memory {memory_id}: "
            f"{from_status} → {to_status}"
        )


class MemoryNotFoundError(ContextOSError):
    """Raised when a memory ID does not exist."""

    def __init__(self, memory_id: str) -> None:
        self.memory_id = memory_id
        super().__init__(f"Memory not found: {memory_id}")


class DuplicateMemoryError(ContextOSError):
    """Raised when a memory ID is already stored."""

    def __init__(self, memory_id: str) -> None:
        self.memory_id = memory_id
        super().__init__(f"Memory ID already exists: {memory_id}")


class EventNotFoundError(ContextOSError):
    """Raised when an event ID does not exist."""

    def __init__(self, event_id: str) -> None:
        self.event_id = event_id
        super().__init__(f"Event not found: {event_id}")


# --- Ingestion Errors ---


class SecretDetectedError(ContextOSError):
    """Raised when the secret scanner detects sensitive content in strict mode."""

    def __init__(self, secret_types: list[str], message: str = "") -> None:
        self.secret_types = secret_types
        detail = f"Detected secrets: {', '.join(secret_types)}"
        if message:
            detail += f". {message}"
        super().__init__(detail)


class IngestionError(ContextOSError):
    """Raised when the ingestion pipeline fails."""


class ExtractionError(ContextOSError):
    """Raised when memory extraction from raw text fails."""


# --- Storage Errors ---


class StorageError(ContextOSError):
    """Base exception for storage-layer failures."""


class DatabaseError(StorageError):
    """Raised for SQLite database errors."""


class VectorStoreError(StorageError):
    """Raised for vector store errors."""


class IndexError(StorageError):
    """Raised for BM25 index errors."""


class ConcurrencyError(StorageError):
    """Raised when an optimistic concurrency check fails (version mismatch)."""

    def __init__(self, memory_id: str, expected_version: int, actual_version: int) -> None:
        self.memory_id = memory_id
        self.expected_version = expected_version
        self.actual_version = actual_version
        super().__init__(
            f"Concurrency conflict for memory {memory_id}: "
            f"expected version {expected_version}, found {actual_version}"
        )


class MigrationError(StorageError):
    """Raised when a database migration fails."""


# --- Retrieval/Compilation Errors ---


class RetrievalError(ContextOSError):
    """Raised when the retrieval pipeline fails."""


class CompilationError(ContextOSError):
    """Raised when context compilation fails."""


class TokenBudgetExceededError(CompilationError):
    """Raised when no meaningful context fits within the token budget."""

    def __init__(self, budget: int, minimum_required: int) -> None:
        self.budget = budget
        self.minimum_required = minimum_required
        super().__init__(
            f"Token budget {budget} is too small. "
            f"Minimum required: {minimum_required}"
        )


# --- Embedding Errors ---


class EmbeddingError(ContextOSError):
    """Raised when embedding generation fails."""


class EmbeddingModelNotLoadedError(EmbeddingError):
    """Raised when the embedding model is not available."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        super().__init__(f"Embedding model not loaded: {model_name}")


# --- Daemon Errors ---


class DaemonError(ContextOSError):
    """Base exception for daemon management errors."""


class DaemonAlreadyRunningError(DaemonError):
    """Raised when attempting to start a daemon that is already running."""

    def __init__(self, pid: int) -> None:
        self.pid = pid
        super().__init__(f"ContextOS daemon is already running (PID: {pid})")


class DaemonNotRunningError(DaemonError):
    """Raised when attempting to interact with a daemon that is not running."""

    def __init__(self) -> None:
        super().__init__("ContextOS daemon is not running. Start it with: contextos start")


# --- Configuration Errors ---


class ConfigError(ContextOSError):
    """Raised for configuration-related errors."""


class ConfigKeyError(ConfigError):
    """Raised when a configuration key is invalid."""

    def __init__(self, key: str) -> None:
        self.key = key
        super().__init__(f"Unknown configuration key: {key}")
