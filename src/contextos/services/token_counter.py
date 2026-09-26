"""Token counting service for ContextOS.

Uses tiktoken with cl100k_base encoding (GPT-4/GPT-3.5 tokenizer)
as the reference tokenizer for token accounting.
"""

from __future__ import annotations

import tiktoken


class TiktokenCounter:
    """Token counter using tiktoken.

    Implements the TokenCounter protocol.
    """

    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        self._encoding_name = encoding_name
        self._encoder = tiktoken.get_encoding(encoding_name)

    def count(self, text: str) -> int:
        """Count tokens in text."""
        if not text:
            return 0
        return len(self._encoder.encode(text))

    def count_batch(self, texts: list[str]) -> list[int]:
        """Count tokens for a batch of texts."""
        return [self.count(t) for t in texts]

    @property
    def encoding_name(self) -> str:
        return self._encoding_name
