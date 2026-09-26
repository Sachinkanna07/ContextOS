"""Small deterministic embedding adapter for offline tests and benchmarks."""

from __future__ import annotations

import hashlib
import math
import re


_SYNONYMS = {
    "ai": "model",
    "artificial": "model",
    "inference": "runtime",
    "engine": "runtime",
    "programming": "language",
    "coding": "language",
    "formerly": "previous",
    "prior": "previous",
    "before": "previous",
    "style": "response",
    "brief": "concise",
    "short": "concise",
    "large": "larger",
    "ram": "memory",
    "resources": "memory",
    "crashed": "failed",
}
_STOP_WORDS = {
    "a", "an", "and", "did", "do", "does", "for", "has", "have", "i", "in",
    "is", "it", "of", "on", "the", "to", "user", "was", "what", "when", "which",
    "with",
}


class DeterministicEmbedding:
    """Hashed bag-of-concepts embedding with no model or network dependency."""

    def __init__(self, dimension: int = 128) -> None:
        if dimension < 8:
            raise ValueError("dimension must be at least 8")
        self._dimension = dimension

    def _vectorize(self, text: str) -> list[float]:
        vector = [0.0] * self._dimension
        for token in re.findall(r"[a-z0-9]+(?:[+#._-][a-z0-9]+)*", text.casefold()):
            if token in _STOP_WORDS:
                continue
            if token in {"focused", "focuses", "focusing"}:
                token = "focus"
            elif (
                token != "focus"
                and token.endswith("s")
                and len(token) > 4
                and not token.endswith("ss")
            ):
                token = token[:-1]
            token = _SYNONYMS.get(token, token)
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self._dimension
            vector[index] += 1.0
        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector] if norm else vector

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vectorize(text) for text in texts]

    async def embed_query(self, query: str) -> list[float]:
        return self._vectorize(query)

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return "deterministic-hashed-concepts"
