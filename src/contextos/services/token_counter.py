"""Token counting service for ContextOS.

Supports exact provider usage, exact local tokenizers (tiktoken), model-family
profiles (Claude-like, Qwen-like), and deterministic approximation fallbacks.
Every token count is explicitly labeled with its measurement source:
- PROVIDER_REPORTED
- TOKENIZER_COUNTED
- APPROXIMATED
"""

from __future__ import annotations

import re
from typing import Protocol, runtime_checkable

import tiktoken

from contextos.core.enums import TokenMeasurementSource


@runtime_checkable
class TokenCounter(Protocol):
    """Protocol for token counting."""

    def count(self, text: str) -> int: ...

    def count_batch(self, texts: list[str]) -> list[int]: ...

    @property
    def encoding_name(self) -> str: ...

    @property
    def measurement_source(self) -> TokenMeasurementSource: ...


class TiktokenCounter:
    """Token counter using tiktoken (OpenAI models).

    Implements TokenCounter with TOKENIZER_COUNTED source.
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

    @property
    def measurement_source(self) -> TokenMeasurementSource:
        return TokenMeasurementSource.TOKENIZER_COUNTED


class ClaudeProfileTokenCounter:
    """Deterministic token counter profile modeled after Claude/Anthropic tokenization.

    Uses Anthropic-like subword heuristics and whitespace/punctuation splitting.
    Implements TokenCounter with APPROXIMATED source (calibrated synthetic profile).
    """

    _STD_PATTERN = re.compile(
        r"[A-Za-z0-9]+(?:'[A-Za-z]+)?|[^\w\s]|\s+",
    )

    def __init__(self, model_name: str = "claude-profile") -> None:
        self._model_name = model_name

    def count(self, text: str) -> int:
        if not text:
            return 0
        # Deterministic Claude-like subword accounting:
        # Standard words, punctuation, and subword chunks for long technical terms
        matches = self._STD_PATTERN.findall(text)
        total = 0
        for token in matches:
            if not token.strip():
                # Whitespace tokens
                total += max(1, len(token) // 4)
            elif len(token) > 6:
                # Subword split for long tokens
                total += (len(token) + 3) // 4
            else:
                total += 1
        return total

    def count_batch(self, texts: list[str]) -> list[int]:
        return [self.count(t) for t in texts]

    @property
    def encoding_name(self) -> str:
        return f"profile-{self._model_name}"

    @property
    def measurement_source(self) -> TokenMeasurementSource:
        return TokenMeasurementSource.APPROXIMATED


class QwenProfileTokenCounter:
    """Deterministic token counter profile modeled after Qwen 152k vocabulary.

    Qwen's vocabulary compresses technical terms and code more densely while
    treating punctuation and camelCase identifiers with byte-level BPE splits.
    Implements TokenCounter with APPROXIMATED source (calibrated synthetic profile).
    """

    _PATTERN = re.compile(
        r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\b)|[0-9]+|[^\w\s]|\s+",
    )

    def __init__(self, model_name: str = "qwen-profile") -> None:
        self._model_name = model_name

    def count(self, text: str) -> int:
        if not text:
            return 0
        matches = self._PATTERN.findall(text)
        total = 0
        for token in matches:
            if not token.strip():
                total += max(1, len(token) // 5)
            elif len(token) > 8:
                total += (len(token) + 4) // 5
            else:
                total += 1
        return total

    def count_batch(self, texts: list[str]) -> list[int]:
        return [self.count(t) for t in texts]

    @property
    def encoding_name(self) -> str:
        return f"profile-{self._model_name}"

    @property
    def measurement_source(self) -> TokenMeasurementSource:
        return TokenMeasurementSource.APPROXIMATED


class DeterministicWordTokenCounter:
    """Offline word-token approximation fallback.

    Explicitly labeled with APPROXIMATED source.
    """

    _TOKEN_PATTERN = re.compile(
        r"[A-Za-z0-9]+(?:(?:[+#._-]+)[A-Za-z0-9]+)*|[^\w\s]"
    )

    def count(self, text: str) -> int:
        if not text:
            return 0
        return len(self._TOKEN_PATTERN.findall(text))

    def count_batch(self, texts: list[str]) -> list[int]:
        return [self.count(text) for text in texts]

    @property
    def encoding_name(self) -> str:
        return "deterministic-word-approximation"

    @property
    def measurement_source(self) -> TokenMeasurementSource:
        return TokenMeasurementSource.APPROXIMATED


def get_token_counter_for_model(
    model_id: str,
    tokenizer_family: str | None = None,
) -> TokenCounter:
    """Select the appropriate token counter for a model ID or family."""
    fam = (tokenizer_family or "").lower()
    mid = model_id.lower()

    # FakeProvider's default is an offline simulation, not a real BPE tokenizer.
    if fam in {"deterministic", "approximation"} or mid == "fake-default":
        return DeterministicWordTokenCounter()

    if "claude" in fam or "claude" in mid or "anthropic" in fam:
        return ClaudeProfileTokenCounter(model_name=model_id)

    if "qwen" in fam or "qwen" in mid:
        return QwenProfileTokenCounter(model_name=model_id)

    if "cl100k" in fam or "o200k" in fam or "gpt" in mid or "openai" in fam:
        encoding = "o200k_base" if "o200k" in fam or "omni" in mid or "gpt-4o" in mid else "cl100k_base"
        try:
            return TiktokenCounter(encoding_name=encoding)
        except Exception:
            return TiktokenCounter("cl100k_base")

    # Default fallback: try tiktoken cl100k_base
    try:
        return TiktokenCounter("cl100k_base")
    except Exception:
        return DeterministicWordTokenCounter()


def recount_cross_model(
    text: str,
    target_models_or_families: list[str],
) -> dict[str, tuple[int, TokenMeasurementSource]]:
    """Recount the SAME compiled context text across distinct model profiles.

    Returns a dict mapping model_or_family to (token_count, measurement_source).
    """
    results: dict[str, tuple[int, TokenMeasurementSource]] = {}
    for target in target_models_or_families:
        counter = get_token_counter_for_model(target, target)
        count = counter.count(text)
        results[target] = (count, counter.measurement_source)
    return results
