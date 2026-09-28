"""Bounded provider-neutral connector contracts."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class RetentionPolicy(StrEnum):
    KEEP_DERIVED_MEMORY = "keep_derived_memory"
    EXPIRE_ON_SOURCE_DELETE = "expire_on_source_delete"


class ConnectorItem(BaseModel):
    """Normalized untrusted input. It is never a final memory."""
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)
    external_id: str = Field(min_length=1, max_length=512)
    source_type: str = Field(min_length=1, max_length=64)
    source_uri: str = Field(min_length=1, max_length=2048)
    content: str = Field(min_length=1, max_length=100_000, repr=False)
    revision: str = Field(min_length=1, max_length=256)
    created_at: datetime | None = None
    updated_at: datetime | None = None
    title: str | None = Field(default=None, max_length=512)
    metadata: dict[str, str | int | float | bool | None] = Field(default_factory=dict, max_length=32)
    deleted: bool = False

    @field_validator("external_id", "source_type", "source_uri", "revision")
    @classmethod
    def safe_identifier(cls, value: str) -> str:
        if "\x00" in value or any(ord(char) < 32 for char in value):
            raise ValueError("control characters are not allowed")
        return value.strip()


class ConnectorSyncState(BaseModel):
    connector_id: str
    connector_type: str
    cursor: str | None = None
    last_success_at: datetime | None = None
    last_attempt_at: datetime | None = None
    enabled: bool = True
    status: str = "idle"
    error_code: str | None = None


class ConnectorSyncResult(BaseModel):
    connector_id: str
    status: str
    scanned: int = 0
    accepted: int = 0
    rejected: int = 0
    unchanged: int = 0
    updated: int = 0
    deleted: int = 0
    failed: int = 0
    next_cursor: str | None = None
    duration_ms: float = 0.0
    error_code: str | None = None
