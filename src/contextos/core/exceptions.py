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


class DaemonLockTimeoutError(DaemonError):
    """Raised when acquiring the daemon lifecycle lock times out."""


# --- Configuration Errors ---


class ConfigError(ContextOSError):
    """Raised for configuration-related errors."""


class ConfigKeyError(ConfigError):
    """Raised when a configuration key is invalid."""

    def __init__(self, key: str) -> None:
        self.key = key
        super().__init__(f"Unknown configuration key: {key}")


# --- Model Runtime & Routing Errors (Phase 9) ---


class ModelRuntimeError(ContextOSError):
    """Base exception for all model runtime, provider, and router errors."""


class ProviderUnavailableError(ModelRuntimeError):
    """Raised when a requested provider runtime is unreachable or unhealthy."""

    def __init__(self, provider_id: str, message: str = "") -> None:
        self.provider_id = provider_id
        detail = f"Provider '{provider_id}' is unavailable"
        if message:
            detail += f": {message}"
        super().__init__(detail)


class ModelUnavailableError(ModelRuntimeError):
    """Raised when a requested model is not offered or disabled by the provider."""

    def __init__(self, model_id: str, provider_id: str = "") -> None:
        self.model_id = model_id
        self.provider_id = provider_id
        msg = f"Model '{model_id}' is unavailable"
        if provider_id:
            msg += f" on provider '{provider_id}'"
        super().__init__(msg)


class ContextWindowExceededError(ModelRuntimeError):
    """Raised when prompt and context exceed the provider/model context window."""

    def __init__(
        self,
        model_id: str,
        required_tokens: int,
        context_window: int,
        prompt_tokens: int = 0,
        compiled_context_tokens: int = 0,
        reserved_output_tokens: int = 0,
    ) -> None:
        self.model_id = model_id
        self.required_tokens = required_tokens
        self.context_window = context_window
        self.prompt_tokens = prompt_tokens
        self.compiled_context_tokens = compiled_context_tokens
        self.reserved_output_tokens = reserved_output_tokens
        super().__init__(
            f"Context window exceeded for model '{model_id}': "
            f"required {required_tokens} tokens (prompt: {prompt_tokens}, "
            f"context: {compiled_context_tokens}, reserved output: {reserved_output_tokens}), "
            f"but context window is {context_window} tokens."
        )


class ProviderTimeoutError(ModelRuntimeError):
    """Raised when a provider request exceeds its allotted timeout."""

    def __init__(self, provider_id: str, timeout_seconds: float) -> None:
        self.provider_id = provider_id
        self.timeout_seconds = timeout_seconds
        super().__init__(
            f"Provider '{provider_id}' timed out after {timeout_seconds:.1f} seconds"
        )


class ProviderAuthenticationError(ModelRuntimeError):
    """Raised when provider authentication fails. Never echoes credentials."""

    def __init__(self, provider_id: str, message: str = "Authentication failed") -> None:
        self.provider_id = provider_id
        super().__init__(f"Provider '{provider_id}' authentication error: {message}")


class ProviderRateLimitError(ModelRuntimeError):
    """Raised when provider returns a rate limit / 429 response."""

    def __init__(self, provider_id: str, retry_after: float | None = None) -> None:
        self.provider_id = provider_id
        self.retry_after = retry_after
        msg = f"Provider '{provider_id}' rate limit reached"
        if retry_after is not None:
            msg += f" (retry after {retry_after:.1f}s)"
        super().__init__(msg)


class MalformedProviderResponseError(ModelRuntimeError):
    """Raised when provider returns an unparseable or unexpected payload structure."""

    def __init__(self, provider_id: str, reason: str = "") -> None:
        self.provider_id = provider_id
        msg = f"Malformed response from provider '{provider_id}'"
        if reason:
            msg += f": {reason}"
        super().__init__(msg)


class RoutingFailureError(ModelRuntimeError):
    """Raised when the router cannot select a model under current policy and constraints."""

    def __init__(self, policy: str, reason: str) -> None:
        self.policy = policy
        self.reason = reason
        super().__init__(f"Routing failure under policy '{policy}': {reason}")
