"""Rule-based memory extractor for ContextOS.

Phase 1 implementation: extracts discrete memories from raw text using
heuristic rules. No LLM dependency.

Strategy:
1. Split input into sentences/segments.
2. Classify each segment by memory type using keyword patterns.
3. Filter out segments that are too short, too generic, or not memory-worthy.
4. Assign confidence and importance scores based on linguistic signals.

This is deliberately simple and will be the baseline for evaluating
LLM-based extraction in Phase 2.
"""

from __future__ import annotations

import re

from contextos.core.enums import MemoryType
from contextos.core.models import ExtractedMemory


# ---------------------------------------------------------------------------
# Type Classification Patterns
# ---------------------------------------------------------------------------

# Patterns that suggest a specific memory type.
# Each tuple: (compiled_regex, memory_type, importance_boost)
TYPE_PATTERNS: list[tuple[re.Pattern[str], MemoryType, float]] = [
    # Preferences
    (re.compile(r"\b(?:i prefer|i like|i use|i always|i never|my favorite|i choose|i go with)\b", re.I), MemoryType.PREFERENCE, 0.1),
    # Skills
    (re.compile(r"\b(?:i(?:'m| am) (?:proficient|experienced|skilled|good|fluent) (?:in|at|with))\b", re.I), MemoryType.SKILL, 0.1),
    (re.compile(r"\b(?:i (?:know|can|understand|have experience with))\b", re.I), MemoryType.SKILL, 0.05),
    # Facts
    (re.compile(r"\b(?:i (?:work|live|study|teach|manage|lead) (?:at|in|for))\b", re.I), MemoryType.FACT, 0.1),
    (re.compile(r"\b(?:my (?:name|email|phone|address|title|role|job|company) (?:is|:))\b", re.I), MemoryType.FACT, 0.15),
    # Projects
    (re.compile(r"\b(?:i(?:'m| am) (?:building|working on|developing|creating|maintaining))\b", re.I), MemoryType.PROJECT, 0.1),
    (re.compile(r"\b(?:my project|our project|the project|current project)\b", re.I), MemoryType.PROJECT, 0.05),
    # Relationships
    (re.compile(r"\b(?:\w+ is my (?:manager|boss|lead|mentor|colleague|friend|partner|wife|husband))\b", re.I), MemoryType.RELATIONSHIP, 0.1),
    # Procedures
    (re.compile(r"\b(?:to (?:deploy|build|test|run|install|setup|configure),? i)\b", re.I), MemoryType.PROCEDURE, 0.1),
    (re.compile(r"\b(?:my (?:workflow|process|routine|setup) (?:is|for|:))\b", re.I), MemoryType.PROCEDURE, 0.1),
    # Opinions
    (re.compile(r"\b(?:i (?:think|believe|feel|find) (?:that)?)\b", re.I), MemoryType.OPINION, 0.0),
    (re.compile(r"\b(?:in my (?:opinion|experience|view))\b", re.I), MemoryType.OPINION, 0.0),
    # Goals
    (re.compile(r"\b(?:i want to|i(?:'d| would) like to|i plan to|i(?:'m| am) going to|my goal is)\b", re.I), MemoryType.GOAL, 0.05),
    # Temporal
    (re.compile(r"\b(?:this (?:week|month|sprint|quarter)|today|tomorrow|next (?:week|month)|currently|right now)\b", re.I), MemoryType.TEMPORAL, -0.1),
]

# Signals that increase confidence
CONFIDENCE_BOOSTERS: list[tuple[re.Pattern[str], float]] = [
    (re.compile(r"\b(?:always|never|definitely|absolutely|certainly)\b", re.I), 0.1),
    (re.compile(r"\b(?:i've been|for years|for a long time|since \d{4})\b", re.I), 0.1),
]

# Signals that decrease confidence
CONFIDENCE_DAMPENERS: list[tuple[re.Pattern[str], float]] = [
    (re.compile(r"\b(?:maybe|perhaps|might|sometimes|occasionally|i guess|not sure)\b", re.I), -0.15),
    (re.compile(r"\b(?:used to|previously|back when|in the past)\b", re.I), -0.1),
]

# Minimum useful content length (characters)
MIN_SEGMENT_LENGTH = 10

# Segments matching these patterns are not memory-worthy
SKIP_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^(?:ok|okay|sure|yes|no|thanks|thank you|got it|understood|hmm|huh)\s*[.!?]?$", re.I),
    re.compile(r"^(?:hi|hello|hey|good morning|good evening)\b", re.I),
    re.compile(r"^\s*$"),
]


# ---------------------------------------------------------------------------
# Text Segmentation
# ---------------------------------------------------------------------------


def _segment_text(text: str) -> list[str]:
    """Split text into meaningful segments for extraction.

    Strategy:
    1. Split on paragraph breaks (double newlines).
    2. For long paragraphs, split on sentence boundaries.
    3. Preserve segments that contain complete thoughts.
    """
    # First split on paragraph breaks
    paragraphs = re.split(r"\n\s*\n", text.strip())

    segments: list[str] = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # Split on sentence boundaries so distinct thoughts become distinct memories.
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", para)
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                segments.append(sentence)

    return segments


# ---------------------------------------------------------------------------
# Extractor Implementation
# ---------------------------------------------------------------------------


class RuleBasedMemoryExtractor:
    """Phase 1 memory extractor using heuristic rules.

    Implements the MemoryExtractor protocol.
    """

    def __init__(
        self,
        *,
        min_confidence: float = 0.3,
        default_importance: float = 0.5,
        default_confidence: float = 0.7,
    ) -> None:
        self._min_confidence = min_confidence
        self._default_importance = default_importance
        self._default_confidence = default_confidence

    async def extract(
        self,
        text: str,
        source_type: str = "cli_input",
        source_uri: str | None = None,
        suggested_type: MemoryType | None = None,
        tags: list[str] | None = None,
    ) -> list[ExtractedMemory]:
        """Extract discrete memories from raw text."""
        if not text or not text.strip():
            return []

        segments = _segment_text(text)
        memories: list[ExtractedMemory] = []

        for segment in segments:
            segment = segment.strip()

            # Skip non-memory-worthy segments
            if len(segment) < MIN_SEGMENT_LENGTH:
                continue
            if any(p.match(segment) for p in SKIP_PATTERNS):
                continue

            # Classify type
            memory_type, importance_delta = self._classify_type(segment, suggested_type)

            # Score confidence
            confidence = self._score_confidence(segment)

            # Score importance
            importance = min(1.0, max(0.0, self._default_importance + importance_delta))

            # Skip low-confidence extractions
            if confidence < self._min_confidence:
                continue

            memories.append(
                ExtractedMemory(
                    content=segment,
                    type=memory_type,
                    confidence=confidence,
                    importance=importance,
                    tags=tags or [],
                )
            )

        return memories

    def _classify_type(
        self, text: str, suggested_type: MemoryType | None
    ) -> tuple[MemoryType, float]:
        """Classify a text segment into a memory type.

        Returns (type, importance_delta).
        """
        if suggested_type is not None:
            return suggested_type, 0.0

        best_type = MemoryType.CONTEXT
        best_importance_delta = 0.0

        for pattern, mem_type, importance_delta in TYPE_PATTERNS:
            if pattern.search(text):
                best_type = mem_type
                best_importance_delta = importance_delta
                break  # First match wins (patterns are ordered by priority)

        return best_type, best_importance_delta

    def _score_confidence(self, text: str) -> float:
        """Score confidence of a text segment based on linguistic signals."""
        confidence = self._default_confidence

        for pattern, boost in CONFIDENCE_BOOSTERS:
            if pattern.search(text):
                confidence += boost

        for pattern, dampener in CONFIDENCE_DAMPENERS:
            if pattern.search(text):
                confidence += dampener  # dampener is negative

        return min(1.0, max(0.0, confidence))
