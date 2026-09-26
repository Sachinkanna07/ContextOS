"""Deterministic in-memory BM25 lexical index."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

from contextos.core.models import LexicalResult

_STOP_WORDS = {
    "a", "an", "and", "did", "do", "does", "for", "has", "have", "i", "in",
    "is", "it", "of", "on", "the", "to", "user", "was", "what", "when", "which",
    "with",
}


def tokenize(text: str) -> list[str]:
    """Normalize case and punctuation while retaining technical identifiers."""
    tokens = re.findall(r"[a-z0-9]+(?:[+#._-][a-z0-9]+)*", text.casefold())
    normalized: list[str] = []
    for token in tokens:
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
        normalized.append(token)
    return normalized


class BM25Index:
    """BM25Okapi-style lexical ranking with deterministic ties."""

    def __init__(self, *, k1: float = 1.5, b: float = 0.75) -> None:
        self._k1 = k1
        self._b = b
        self._documents: dict[str, list[str]] = {}
        self._metadata: dict[str, dict[str, Any]] = {}

    async def index(
        self, doc_id: str, text: str, metadata: dict[str, Any] | None = None
    ) -> None:
        tokens = tokenize(text)
        if tokens:
            self._documents[doc_id] = tokens
            self._metadata[doc_id] = dict(metadata or {})
        else:
            await self.delete(doc_id)

    async def search(
        self,
        query: str,
        top_k: int = 20,
        filters: dict[str, Any] | None = None,
    ) -> list[LexicalResult]:
        terms = tokenize(query)
        if not terms or not self._documents or top_k <= 0:
            return []
        candidates = [doc_id for doc_id in self._documents if self._matches(doc_id, filters)]
        if not candidates:
            return []
        corpus_size = len(self._documents)
        average_length = sum(map(len, self._documents.values())) / corpus_size
        document_frequency = {
            term: sum(term in tokens for tokens in self._documents.values())
            for term in set(terms)
        }
        ranked: list[tuple[str, float]] = []
        for doc_id in candidates:
            tokens = self._documents[doc_id]
            frequencies = Counter(tokens)
            score = 0.0
            for term in terms:
                frequency = frequencies[term]
                if not frequency:
                    continue
                df = document_frequency[term]
                # Positive Robertson/Sparck Jones IDF, as used by Lucene.
                inverse_document_frequency = math.log(
                    1.0 + (corpus_size - df + 0.5) / (df + 0.5)
                )
                denominator = frequency + self._k1 * (
                    1.0 - self._b + self._b * len(tokens) / average_length
                )
                score += inverse_document_frequency * frequency * (self._k1 + 1.0) / denominator
            if score > 0.0:
                ranked.append((doc_id, score))
        ranked.sort(key=lambda item: (-item[1], item[0]))
        return [
            LexicalResult(id=doc_id, score=score, metadata=self._metadata[doc_id])
            for doc_id, score in ranked[:top_k]
        ]

    async def delete(self, doc_id: str) -> None:
        self._documents.pop(doc_id, None)
        self._metadata.pop(doc_id, None)

    async def count(self) -> int:
        return len(self._documents)

    async def rebuild(self, documents: dict[str, str]) -> None:
        self._documents = {
            doc_id: tokens for doc_id, text in documents.items() if (tokens := tokenize(text))
        }
        self._metadata = {doc_id: {} for doc_id in self._documents}

    async def rebuild_with_metadata(
        self, documents: dict[str, tuple[str, dict[str, Any]]]
    ) -> None:
        self._documents.clear()
        self._metadata.clear()
        for doc_id, (text, metadata) in documents.items():
            await self.index(doc_id, text, metadata)

    def _matches(self, doc_id: str, filters: dict[str, Any] | None) -> bool:
        if not filters:
            return True
        metadata = self._metadata.get(doc_id, {})
        return all(metadata.get(key) == value for key, value in filters.items())
