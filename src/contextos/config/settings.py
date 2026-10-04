"""Configuration management for ContextOS.

Uses Pydantic Settings for config validation with TOML file support.
Config is loaded from ~/.config/contextos/config.toml (XDG-compliant).
"""

from __future__ import annotations

import ipaddress
import platform
import re
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings


def _default_data_dir() -> Path:
    """Platform-appropriate default data directory."""
    system = platform.system()
    if system == "Windows":
        base = Path.home() / "AppData" / "Local" / "contextos"
    elif system == "Darwin":
        base = Path.home() / "Library" / "Application Support" / "contextos"
    else:
        # Linux / other Unix — XDG
        xdg_data = Path.home() / ".local" / "share"
        base = xdg_data / "contextos"
    return base


def _default_config_dir() -> Path:
    """Platform-appropriate default config directory."""
    system = platform.system()
    if system == "Windows":
        return Path.home() / "AppData" / "Local" / "contextos"
    elif system == "Darwin":
        return Path.home() / "Library" / "Application Support" / "contextos"
    else:
        xdg_config = Path.home() / ".config"
        return xdg_config / "contextos"


class DaemonConfig(BaseSettings):
    """Daemon process configuration."""
    host: str = "127.0.0.1"
    port: int = 52411
    log_level: str = "info"
    data_dir: Path = Field(default_factory=_default_data_dir)
    config_dir: Path = Field(default_factory=_default_config_dir)
    readiness_timeout: float = Field(default=30.0, ge=1, le=300, allow_inf_nan=False)
    lock_timeout: float = Field(default=45.0, ge=1, le=600, allow_inf_nan=False)

    @model_validator(mode="after")
    def bounded_startup(self) -> DaemonConfig:
        if self.lock_timeout <= self.readiness_timeout + 5:
            raise ValueError("lock_timeout must exceed readiness_timeout by more than 5 seconds")
        return self

    @field_validator("host")
    @classmethod
    def loopback_only(cls, value: str) -> str:
        if value not in {"127.0.0.1", "::1", "localhost"}:
            raise ValueError("ContextOS daemon must bind to loopback")
        return value


class EmbeddingConfig(BaseSettings):
    """Embedding model configuration."""
    model: str = "deterministic"
    device: str = "cpu"
    batch_size: int = 32


class TokenCounterConfig(BaseSettings):
    """Offline approximation by default; exact tokenizers require explicit setup."""

    encoding: Literal["deterministic", "cl100k_base", "o200k_base"] = "deterministic"


class RetrievalConfig(BaseSettings):
    """Default retrieval parameters."""
    vector_top_k: int = 20
    bm25_top_k: int = 20
    rrf_k: int = 60
    default_budget: int = 4000


class PrivacyConfig(BaseSettings):
    """Privacy and security configuration."""
    secret_detection: str = "strict"  # "strict", "redact", "warn"
    default_privacy_level: str = "personal"
    entropy_threshold: float = 4.5


class LLMConfig(BaseSettings):
    """LLM provider configuration."""
    provider: str = "none"
    model: str = ""
    api_key_env: str = ""


_PROVIDER_ID = re.compile(r"[a-z][a-z0-9_-]{0,63}\Z")
_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


class ProviderConfig(BaseSettings):
    """Provider settings contain an environment variable name, never its value."""

    enabled: bool = False
    api_key_env: str = ""
    default_model: str = ""
    base_url: str = ""

    @field_validator("api_key_env")
    @classmethod
    def safe_environment_name(cls, value: str) -> str:
        if value and not _ENV_NAME.fullmatch(value):
            raise ValueError("api_key_env must be an environment variable name")
        return value

    @field_validator("base_url")
    @classmethod
    def safe_base_url(cls, value: str) -> str:
        if not value:
            return value
        parsed = urlsplit(value)
        if (parsed.scheme not in {"http", "https"} or not parsed.hostname
                or parsed.username or parsed.password or parsed.query or parsed.fragment
                or any(ch.isspace() for ch in value)):
            raise ValueError("base_url must be a credential-free HTTP(S) URL")
        try:
            port = parsed.port
        except ValueError as exc:
            raise ValueError("base_url has an invalid port") from exc
        host = parsed.hostname.lower()
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            address = None
        loopback = host == "localhost" or bool(address and address.is_loopback)
        if parsed.scheme == "http" and not loopback:
            raise ValueError("plain HTTP is allowed only for loopback endpoints")
        if address and not address.is_loopback:
            raise ValueError("private or literal non-loopback IP endpoints are unsupported")
        if port == 0:
            raise ValueError("base_url port must be nonzero")
        return value.rstrip("/")


class ProvidersConfig(BaseSettings):
    openai: ProviderConfig = Field(
        default_factory=lambda: ProviderConfig(api_key_env="OPENAI_API_KEY")
    )
    anthropic: ProviderConfig = Field(
        default_factory=lambda: ProviderConfig(api_key_env="ANTHROPIC_API_KEY")
    )
    gemini: ProviderConfig = Field(
        default_factory=lambda: ProviderConfig(api_key_env="GEMINI_API_KEY")
    )
    compatible: dict[str, ProviderConfig] = Field(default_factory=dict)

    @field_validator("compatible")
    @classmethod
    def safe_ids(cls, value: dict[str, ProviderConfig]) -> dict[str, ProviderConfig]:
        reserved = {"ollama", "openai", "anthropic", "gemini", "fake", "openai_compatible"}
        if len(value) > 20 or any(
            not _PROVIDER_ID.fullmatch(key) or key in reserved for key in value
        ):
            raise ValueError("Invalid or reserved compatible provider identifier")
        return value


class MCPConfig(BaseSettings):
    """Local MCP exposure; writes and destructive operations fail closed."""

    enabled: bool = False
    transport: str = "stdio"
    allow_read: bool = True
    allow_write: bool = False
    allow_delete: bool = False
    allow_telemetry: bool = True
    max_input_chars: int = Field(default=10_000, ge=1, le=100_000)
    max_search_results: int = Field(default=25, ge=1, le=200)
    max_history_entries: int = Field(default=50, ge=1, le=200)
    max_graph_nodes: int = Field(default=100, ge=1, le=1_000)
    max_graph_edges: int = Field(default=250, ge=1, le=2_500)
    max_compilation_tokens: int = Field(default=8_000, ge=1, le=32_000)


class ConnectorConfig(BaseSettings):
    """Explicit local sources only; no discovered paths or credentials."""

    local_files: dict[str, list[Path]] = Field(default_factory=dict)
    json_imports: dict[str, Path] = Field(default_factory=dict)

    @field_validator("local_files", "json_imports")
    @classmethod
    def valid_ids(
        cls, value: dict[str, list[Path]] | dict[str, Path]
    ) -> dict[str, list[Path]] | dict[str, Path]:
        if len(value) > 20 or any(not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", key) for key in value):
            raise ValueError("Connector IDs must be short alphanumeric identifiers")
        return value


class Settings(BaseSettings):
    """Root settings for ContextOS."""
    daemon: DaemonConfig = Field(default_factory=DaemonConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    token_counter: TokenCounterConfig = Field(default_factory=TokenCounterConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    privacy: PrivacyConfig = Field(default_factory=PrivacyConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    providers: ProvidersConfig = Field(default_factory=ProvidersConfig)
    mcp: MCPConfig = Field(default_factory=MCPConfig)
    connectors: ConnectorConfig = Field(default_factory=ConnectorConfig)


def load_settings() -> Settings:
    """Load settings from config file, falling back to defaults."""
    config_dir = _default_config_dir()
    config_file = config_dir / "config.toml"

    if config_file.exists():
        try:
            import tomllib
            with open(config_file, "rb") as f:
                data = tomllib.load(f)
            return Settings(**data)
        except Exception as exc:
            raise ValueError(f"Invalid ContextOS configuration: {config_file}") from exc

    return Settings()
