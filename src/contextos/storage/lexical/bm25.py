"""BM25 lexical search index for ContextOS.

In-memory BM25 index backed by rank-bm25. Index state is rebuilt from
the database on startup. This is acceptable for Phase 1 scale (< 100K memories).

For Phase 2, consider switching to tantivy-py for persistent indexing.
"""

from __future__ import annotations

import logging
from typing import Any

from rank_bm25 import BM25Okapi

from contextos.core.models import LexicalResult

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer.

    Lowercases, strips punctuation, removes very short tokens.
    No stemming — we keep it simple for Phase 1.
    """
    import re

    # Lowercase and split on non-alphanumeric
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    # Remove single-character tokens
    return [t for t in tokens if len(t) > 1]


class BM25Index:
    """In-memory BM25 index.

    Implements the LexicalIndex protocol.
    """

    def __init__(self) -> None:
        self._doc_ids: list[str] = []
        self._corpus: list[list[str]] = []
        self._metadata: dict[str, dict[str, Any]] = {}
        self._index: BM25Okapi | None = None

    def _rebuild_index(self) -> None:
        """Rebuild the BM25 index from the corpus."""
        if self._corpus:
            self._index = BM25Okapi(self._corpus)
        else:
            self._index = None

    async def index(
        self, doc_id: str, text: str, metadata: dict[str, Any] | None = None
    ) -> None:
        """Add or update a document in the index."""
        tokens = _tokenize(text)
        if not tokens:
            return

        # Check if doc already exists
        if doc_id in self._metadata:
            # Update: remove old, add new
            idx = self._doc_ids.index(doc_id)
            self._doc_ids.pop(idx)
            self._corpus.pop(idx)

        self._doc_ids.append(doc_id)
        self._corpus.append(tokens)
        self._metadata[doc_id] = metadata or {}

        self._rebuild_index()

    async def search(
        self,
        query: str,
        top_k: int = 20,
        filters: dict[str, Any] | None = None,
    ) -> list[LexicalResult]:
        """Search the index with BM25 scoring."""
        if self._index is None or not self._doc_ids:
            return []

        tokens = _tokenize(query)
        if not tokens:
            return []

        scores = self._index.get_scores(tokens)

        # Pair scores with doc_ids and sort
        scored_docs: list[tuple[str, float]] = [
            (doc_id, float(score))
            for doc_id, score in zip(self._doc_ids, scores)
            if score > 0
        ]

        # Apply filters
        if filters:
            scored_docs = [
                (doc_id, score)
                for doc_id, score in scored_docs
                if self._match_filters(doc_id, filters)
            ]

        scored_docs.sort(key=lambda x: x[1], reverse=True)

        results: list[LexicalResult] = []
        for doc_id, score in scored_docs[:top_k]:
            results.append(LexicalResult(
                id=doc_id,
                score=score,
                metadata=self._metadata.get(doc_id, {}),
            ))

        return results

    async def delete(self, doc_id: str) -> None:
        """Remove a document from the index."""
        if doc_id not in self._metadata:
            return

        idx = self._doc_ids.index(doc_id)
        self._doc_ids.pop(idx)
        self._corpus.pop(idx)
        del self._metadata[doc_id]

        self._rebuild_index()

    async def count(self) -> int:
        """Return the number of documents in the index."""
        return len(self._doc_ids)

    async def rebuild(self, documents: dict[str, str]) -> None:
        """Rebuild the entire index from scratch."""
        self._doc_ids.clear()
        self._corpus.clear()
        self._metadata.clear()

        for doc_id, text in documents.items():
            tokens = _tokenize(text)
            if tokens:
                self._doc_ids.append(doc_id)
                self._corpus.append(tokens)
                self._metadata[doc_id] = {}

        self._rebuild_index()
        logger.info("BM25 index rebuilt with %d documents", len(self._doc_ids))

    def _match_filters(self, doc_id: str, filters: dict[str, Any]) -> bool:
        """Check if a document's metadata matches the given filters."""
        meta = self._metadata.get(doc_id, {})
        return all(meta.get(k) == v for k, v in filters.items())
