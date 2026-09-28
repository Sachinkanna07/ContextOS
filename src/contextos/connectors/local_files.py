"""Explicit-root UTF-8 local file connector."""
from __future__ import annotations
import hashlib
from pathlib import Path
from contextos.connectors.models import ConnectorItem

import os

def _is_reserved(path: Path) -> bool:
    if hasattr(os.path, "isreserved"):
        return os.path.isreserved(str(path))
    name = path.name.upper()
    stem = path.stem.upper()
    reserved = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}
    return name in reserved or stem in reserved

class LocalFileConnector:
    source_type = "local_file"

    def __init__(self, connector_id: str, roots: list[Path], max_bytes: int = 1_000_000) -> None:
        self.connector_id = connector_id
        resolved_roots = []
        for r in roots:
            s = str(r)
            if s.startswith(r"\\") or s.startswith("//"):
                raise ValueError(f"UNC and network paths are not allowed: {s}")
            resolved = r.resolve()
            if str(resolved).startswith(r"\\") or str(resolved).startswith("//"):
                raise ValueError(f"UNC and network paths are not allowed: {resolved}")
            resolved_roots.append(resolved)
        self._roots = resolved_roots
        self._max_bytes = max_bytes

    async def health(self) -> bool:
        return all(root.exists() and root.is_dir() for root in self._roots)

    async def close(self) -> None:
        pass

    def _allowed(self, path: Path) -> bool:
        try:
            s = str(path)
            if s.startswith(r"\\") or s.startswith("//") or _is_reserved(path):
                return False
            resolved = path.resolve(strict=True)
            res_str = str(resolved)
            if res_str.startswith(r"\\") or res_str.startswith("//") or _is_reserved(resolved):
                return False
            return any(resolved.is_relative_to(root) for root in self._roots)
        except (OSError, RuntimeError, ValueError):
            return False

    async def scan(self, cursor: str | None) -> tuple[list[ConnectorItem], str | None]:
        from datetime import datetime, timezone

        paths = []
        for root in self._roots:
            if not root.exists():
                continue
            for path in root.rglob("*"):
                try:
                    if path.is_file() and path.suffix.lower() in {".txt", ".md", ".json", ".jsonl"} and self._allowed(path):
                        paths.append(path.resolve())
                except (OSError, UnicodeError):
                    continue
        items = []
        for path in sorted(set(paths), key=lambda item: str(item).casefold()):
            try:
                # TOCTOU re-check before opening
                if not self._allowed(path):
                    continue
                st = path.stat()
                size = st.st_size
                if size > self._max_bytes or size == 0:
                    continue
                # Bounded read to prevent memory exhaustion
                with open(path, "r", encoding="utf-8", errors="strict") as f:
                    content = f.read(self._max_bytes + 1)
                if len(content) > self._max_bytes or not content.strip():
                    continue
            except (OSError, UnicodeError):
                continue
            digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
            try:
                relative = next(
                    path.relative_to(root).as_posix()
                    for root in self._roots
                    if path.is_relative_to(root)
                )
            except Exception:
                continue
            items.append(
                ConnectorItem(
                    external_id=relative,
                    source_type=self.source_type,
                    source_uri=f"file:///{relative}",
                    content=content,
                    revision=digest,
                    updated_at=datetime.fromtimestamp(st.st_mtime, tz=timezone.utc),
                )
            )
        return items, str(len(items))
