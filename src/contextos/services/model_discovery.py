"""Shared bounded inventory discovery for the dashboard and model list."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from time import monotonic
from typing import Any

from contextos.core.models import ModelCapabilities

DISCOVERY_TIMEOUT_SECONDS = 5.0
SUCCESS_CACHE_SECONDS = 10.0
FAILURE_CACHE_SECONDS = 1.0


@dataclass
class _Inventory:
    provider: Any
    expires: float = 0.0
    models: list[ModelCapabilities] = field(default_factory=list)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class ModelDiscovery:
    """Cache independently per provider and coalesce concurrent refreshes.

    Five seconds covers the adapters' inventory request budget and measured
    local client setup (>1 second). Failed/empty inventories retry after one
    second rather than freezing a transient outage for the success cache TTL.
    """

    def __init__(self) -> None:
        self._inventories: dict[str, _Inventory] = {}

    async def list_models(self, providers: dict[str, Any]) -> list[ModelCapabilities]:
        snapshot = list(providers.items())
        for key in self._inventories.keys() - providers.keys():
            del self._inventories[key]

        async def discover(key: str, provider: Any) -> list[ModelCapabilities]:
            entry = self._inventories.get(key)
            if entry is None or entry.provider is not provider:
                entry = _Inventory(provider)
                self._inventories[key] = entry
            async with entry.lock:
                if monotonic() < entry.expires:
                    return entry.models
                try:
                    inventory = await asyncio.wait_for(
                        provider.list_models(),
                        timeout=DISCOVERY_TIMEOUT_SECONDS,
                    )
                    entry.models = (
                        [
                            item
                            for item in inventory
                            if isinstance(item, ModelCapabilities) and item.enabled
                        ]
                        if isinstance(inventory, list)
                        else []
                    )
                except Exception:
                    # Never expose provider errors or retain stale available models.
                    entry.models = []
                ttl = SUCCESS_CACHE_SECONDS if entry.models else FAILURE_CACHE_SECONDS
                entry.expires = monotonic() + ttl
                return entry.models

        inventories = await asyncio.gather(*(discover(key, p) for key, p in snapshot))
        return [model for inventory in inventories for model in inventory]
