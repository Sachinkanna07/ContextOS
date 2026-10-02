"""In-memory vector store using numpy for ContextOS.

Phase 1 implementation: simple brute-force cosine similarity search.
No ANN index — at Phase 1 scale (< 10K vectors), brute force is fast enough
and avoids external library complexity.

This will be replaced by sqlite-vec or LanceDB when we need ANN performance.
It satisfies the VectorStore protocol and passes the same contract tests.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from contextos.core.models import VectorResult

logger = logging.getLogger(__name__)


class InMemoryVectorStore:
    """Brute-force cosine similarity vector store.

    Implements the VectorStore protocol.

    Suitable for Phase 1 (< 10K vectors). O(n) search, but n is small.
    """

    def __init__(self, dimension: int = 384) -> None:
        self._dimension = dimension
        self._ids: list[str] = []
        self._vectors: np.ndarray = np.empty((0, dimension), dtype=np.float32)
        self._metadata: dict[str, dict[str, Any]] = {}

    def contains(self, doc_id: str | Any) -> bool:
        """Check whether a document ID is present in the vector store."""
        return str(doc_id) in self._metadata

    def __contains__(self, doc_id: str | Any) -> bool:
        return str(doc_id) in self._metadata

    async def add(
        self,
        ids: list[str],
        vectors: list[list[float]],
        metadata: list[dict[str, Any]],
    ) -> None:
        """Add vectors to the store."""
        if not ids:
            return
        if len(ids) != len(vectors) or len(ids) != len(metadata):
            raise ValueError("ids, vectors, and metadata must have equal lengths")

        new_vectors = np.array(vectors, dtype=np.float32)
        if new_vectors.ndim == 1:
            new_vectors = new_vectors.reshape(1, -1)
        if new_vectors.ndim != 2 or new_vectors.shape[1] != self._dimension:
            raise ValueError(
                f"Expected vectors with dimension {self._dimension}, got shape {new_vectors.shape}"
            )
        if not np.isfinite(new_vectors).all():
            raise ValueError("Vectors must contain only finite values")

        # Normalize for cosine similarity
        norms = np.linalg.norm(new_vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0  # Avoid division by zero
        new_vectors = new_vectors / norms

        for i, doc_id in enumerate(ids):
            if doc_id in self._metadata:
                # Update: replace existing
                idx = self._ids.index(doc_id)
                self._vectors[idx] = new_vectors[i]
                self._metadata[doc_id] = metadata[i] if i < len(metadata) else {}
            else:
                # Add new
                self._ids.append(doc_id)
                self._vectors = np.vstack([self._vectors, new_vectors[i:i + 1]])
                self._metadata[doc_id] = metadata[i] if i < len(metadata) else {}

    async def search(
        self,
        vector: list[float],
        top_k: int = 20,
        filters: dict[str, Any] | None = None,
    ) -> list[VectorResult]:
        """Search by cosine similarity."""
        if len(self._ids) == 0:
            return []

        query = np.array(vector, dtype=np.float32)
        if query.ndim != 1 or query.shape[0] != self._dimension:
            raise ValueError(
                f"Expected query dimension {self._dimension}, got shape {query.shape}"
            )
        if not np.isfinite(query).all():
            raise ValueError("Query vector must contain only finite values")
        norm = np.linalg.norm(query)
        if norm == 0:
            return []
        query = query / norm

        # Cosine similarity (vectors are pre-normalized)
        similarities = self._vectors @ query

        # Build results
        indices = sorted(
            range(len(self._ids)),
            key=lambda index: (-float(similarities[index]), self._ids[index]),
        )
        results: list[VectorResult] = []

        for idx in indices:
            if len(results) >= top_k:
                break

            doc_id = self._ids[idx]
            score = float(similarities[idx])

            if score <= 0:
                continue

            # Apply filters
            if filters:
                meta = self._metadata.get(doc_id, {})
                if not all(meta.get(k) == v for k, v in filters.items()):
                    continue

            results.append(VectorResult(
                id=doc_id,
                score=score,
                metadata=self._metadata.get(doc_id, {}),
            ))

        return results

    async def delete(self, ids: list[str]) -> None:
        """Remove vectors from the store."""
        for doc_id in ids:
            if doc_id in self._metadata:
                idx = self._ids.index(doc_id)
                self._ids.pop(idx)
                self._vectors = np.delete(self._vectors, idx, axis=0)
                del self._metadata[doc_id]

    async def count(self) -> int:
        return len(self._ids)

    async def rebuild(
        self,
        ids: list[str],
        vectors: list[list[float]],
        metadata: list[dict[str, Any]],
    ) -> None:
        """Replace the complete index after validating the new corpus."""
        replacement = InMemoryVectorStore(self._dimension)
        await replacement.add(ids, vectors, metadata)
        self._ids = replacement._ids
        self._vectors = replacement._vectors
        self._metadata = replacement._metadata
