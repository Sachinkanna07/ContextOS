"""Sentence-transformers embedding service for ContextOS.

Wraps sentence-transformers for local embedding generation.
Default model: all-MiniLM-L6-v2 (384 dimensions, ~22M parameters, fast).
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class SentenceTransformerEmbedding:
    """Local embedding using sentence-transformers.

    Implements the EmbeddingService protocol.
    Lazy-loads the model on first use to avoid blocking startup.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "cpu",
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._model = None
        self._dimension: int | None = None

    def _load_model(self) -> None:
        """Lazily load the model."""
        if self._model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading embedding model: %s (device: %s)", self._model_name, self._device)
            self._model = SentenceTransformer(self._model_name, device=self._device)
            self._dimension = self._model.get_sentence_embedding_dimension()
            logger.info(
                "Embedding model loaded: %s (dim=%d)", self._model_name, self._dimension
            )
        except ImportError:
            raise RuntimeError(
                "sentence-transformers is required for SentenceTransformerEmbedding. "
                "Install ContextOS with the embeddings extra: pip install 'contextos[embeddings]'"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load embedding model '{self._model_name}': {e}") from e

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts."""
        self._load_model()
        assert self._model is not None

        embeddings = self._model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        return embeddings.tolist()

    async def embed_query(self, query: str) -> list[float]:
        """Embed a single query."""
        results = await self.embed([query])
        return results[0]

    @property
    def dimension(self) -> int:
        """Embedding vector dimension."""
        self._load_model()
        assert self._dimension is not None
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model_name
