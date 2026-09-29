"""Configuration management for ContextOS.

Uses Pydantic Settings for config validation with TOML file support.
Config is loaded from ~/.config/contextos/config.toml (XDG-compliant).
"""

from __future__ import annotations

import platform
import re
from pathlib import Path

from pydantic import Field, field_validator
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

    @field_validator("host")
    @classmethod
    def loopback_only(cls, value: str) -> str:
        if value not in {"127.0.0.1", "::1", "localhost"}:
            raise ValueError("ContextOS daemon must bind to loopback")
        return value


class EmbeddingConfig(BaseSettings):
    """Embedding model configuration."""
    model: str = "all-MiniLM-L6-v2"
    device: str = "cpu"
    batch_size: int = 32


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
    def valid_ids(cls, value):
        if len(value) > 20 or any(not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", key) for key in value):
            raise ValueError("Connector IDs must be short alphanumeric identifiers")
        return value


class Settings(BaseSettings):
    """Root settings for ContextOS."""
    daemon: DaemonConfig = Field(default_factory=DaemonConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    privacy: PrivacyConfig = Field(default_factory=PrivacyConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
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
