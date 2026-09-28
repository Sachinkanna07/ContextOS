"""Small connector protocol; connectors return normalized untrusted items."""
from __future__ import annotations
from typing import Protocol
from contextos.connectors.models import ConnectorItem

class Connector(Protocol):
    connector_id: str
    source_type: str
    async def health(self) -> bool: ...
    async def scan(self, cursor: str | None) -> tuple[list[ConnectorItem], str | None]: ...
    async def close(self) -> None: ...
