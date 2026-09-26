"""Configuration management for ContextOS.

Uses Pydantic Settings for config validation with TOML file support.
Config is loaded from ~/.config/contextos/config.toml (XDG-compliant).
"""

from __future__ import annotations

import platform
from pathlib import Path

from pydantic import Field
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


class Settings(BaseSettings):
    """Root settings for ContextOS."""
    daemon: DaemonConfig = Field(default_factory=DaemonConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    privacy: PrivacyConfig = Field(default_factory=PrivacyConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)


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
        except Exception:
            import logging
            logging.getLogger(__name__).warning(
                "Failed to load config from %s, using defaults", config_file
            )

    return Settings()
