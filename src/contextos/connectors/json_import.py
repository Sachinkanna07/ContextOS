"""Strict generic JSON/JSONL import connector."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from contextos.connectors.models import ConnectorItem


class JsonImportConnector:
    source_type = "json_import"

    def __init__(self, connector_id: str, path: Path, max_bytes: int = 1_000_000) -> None:
        self.connector_id = connector_id
        self._path = path.resolve()
        self._max_bytes = max_bytes

    async def health(self) -> bool:
        return self._path.exists() and self._path.is_file()

    async def close(self) -> None:
        pass

    async def scan(self, cursor: str | None) -> tuple[list[ConnectorItem], str | None]:
        if self._path.stat().st_size > self._max_bytes:
            raise ValueError("ITEM_TOO_LARGE")

        text = self._path.read_text(encoding="utf-8", errors="strict")
        if self._path.suffix.lower() == ".jsonl":
            raw = [json.loads(line) for line in text.splitlines() if line.strip()]
        else:
            raw = json.loads(text)

        if isinstance(raw, dict):
            raw = [raw]
        if not isinstance(raw, list):
            raise ValueError("invalid import root")

        ids = set()
        items = []
        for value in raw:
            if not isinstance(value, dict) or set(value) - {
                "id",
                "content",
                "timestamp",
                "updated_at",
                "source",
                "title",
                "metadata",
                "revision",
            }:
                raise ValueError("invalid item schema")

            external_id = value.get("id")
            content = value.get("content")
            if not isinstance(external_id, str) or not isinstance(content, str) or not external_id.strip() or not content.strip():
                raise ValueError("missing or invalid id or content")
            if external_id in ids:
                raise ValueError("duplicate id")
            ids.add(external_id)

            if len(content) > 100_000:
                raise ValueError("ITEM_TOO_LARGE")

            # Check for control characters
            if any(ord(c) < 32 for c in external_id) or any(ord(c) < 32 and c not in "\n\r\t" for c in content):
                raise ValueError("control characters not allowed")

            metadata = value.get("metadata", {})
            if not isinstance(metadata, dict) or any(isinstance(v, (dict, list)) for v in metadata.values()):
                raise ValueError("invalid metadata: nested structures rejected")

            # Check for secret metadata
            secret_tokens = ("secret", "password", "api_key", "token", "sk-", "bearer")
            for k, v in metadata.items():
                k_lower = str(k).lower()
                v_lower = str(v).lower()
                if any(sec in k_lower or sec in v_lower for sec in secret_tokens):
                    raise ValueError("secret detected in metadata")

            # Timestamp parsing
            created_at = None
            if "timestamp" in value and value["timestamp"] is not None:
                ts = value["timestamp"]
                try:
                    if isinstance(ts, (int, float)):
                        created_at = datetime.fromtimestamp(ts, tz=timezone.utc)
                    elif isinstance(ts, str):
                        created_at = datetime.fromisoformat(ts)
                    else:
                        raise ValueError("invalid timestamp")
                except Exception as exc:
                    raise ValueError(f"invalid timestamp: {exc}") from exc

            updated_at = None
            if "updated_at" in value and value["updated_at"] is not None:
                ts = value["updated_at"]
                try:
                    if isinstance(ts, (int, float)):
                        updated_at = datetime.fromtimestamp(ts, tz=timezone.utc)
                    elif isinstance(ts, str):
                        updated_at = datetime.fromisoformat(ts)
                    else:
                        raise ValueError("invalid timestamp")
                except Exception as exc:
                    raise ValueError(f"invalid timestamp: {exc}") from exc

            revision = value.get("revision") or hashlib.sha256(content.encode()).hexdigest()
            items.append(
                ConnectorItem(
                    external_id=external_id,
                    source_type=self.source_type,
                    source_uri=f"json://{self.connector_id}/{external_id}",
                    content=content,
                    revision=revision,
                    title=value.get("title"),
                    metadata=metadata,
                    created_at=created_at,
                    updated_at=updated_at,
                )
            )

        return items, str(len(items))
