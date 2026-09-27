"""Deterministic candidate extraction for ContextOS Phase 2.

The extractor is deliberately local and side-effect free. It turns raw user
text into unaccepted CandidateMemory objects; it never writes long-term memory.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from contextos.core.enums import (
    CandidateAction,
    CandidateTemporalStatus,
    MemoryType,
    SourceRole,
)
from contextos.core.models import CandidateMemory


MAX_INPUT_CHARS = 100_000
MAX_CANDIDATES = 100
MAX_CLAUSE_CHARS = 9_000


@dataclass(frozen=True)
class _Clause:
    text: str
    evidence: str
    start: int | None = None
    end: int | None = None


_TYPE_RULES: tuple[tuple[MemoryType, re.Pattern[str]], ...] = (
    (MemoryType.PREFERENCE, re.compile(
        r"\b(?:prefer|preference|favorite|always use|never use|keep (?:your )?answers)\b", re.I
    )),
    (MemoryType.GOAL, re.compile(
        r"\b(?:want to|would like to|plan(?:ning)? to|going to|goal|prepar(?:e|ing) for|"
        r"focus(?:ing)?(?: mainly)? on|switch(?:ing)? (?:from|to)|"
        r"(?:i(?:'ll| will| might| may)|maybe i(?:'ll| will)) learn|might learn|may learn|will learn)\b",
        re.I,
    )),
    (MemoryType.PROJECT, re.compile(
        r"\b(?:building|working on|developing|creating|maintaining|my project|current project)\b", re.I
    )),
    (MemoryType.SKILL, re.compile(
        r"\b(?:proficient|experienced|skilled|fluent|know|learning|learned|experience with)\b", re.I
    )),
    (MemoryType.RELATIONSHIP, re.compile(
        r"\b(?:manager|boss|lead|mentor|colleague|friend|partner|wife|husband)\b", re.I
    )),
    (MemoryType.PROCEDURE, re.compile(
        r"\b(?:workflow|process|routine|setup|to (?:deploy|build|test|run|install|configure))\b", re.I
    )),
    (MemoryType.OPINION, re.compile(r"\b(?:i think|i believe|in my opinion|in my view)\b", re.I)),
    (MemoryType.FACT, re.compile(
        r"\b(?:work(?:ing)? at|live in|study at|teach at|my (?:name|role|job|company)|i use)\b", re.I
    )),
    (MemoryType.TEMPORAL, re.compile(
        r"\b(?:today|tomorrow|this (?:week|month|sprint)|next (?:week|month)|right now)\b", re.I
    )),
)

_FILLER = re.compile(
    r"^(?:ok(?:ay)?|sure|yes|no|thanks|thank you|nice|lol|continue|got it|understood|"
    r"hi|hello|hey|hmm|huh)[.!?]*$",
    re.I,
)
_CASUAL = re.compile(r"^(?:the )?weather (?:looks|is) (?:good|nice|great)[.!?]*$", re.I)
_PERSONAL_SIGNAL = re.compile(r"\b(?:i|i'm|i've|i'll|i'd|my|me|user|the user)\b", re.I)
_UNCERTAIN = re.compile(r"\b(?:maybe|might|may|perhaps|possibly|i guess|not sure)\b", re.I)
_HISTORICAL = re.compile(r"\b(?:used to|previously|in the past|before)\b", re.I)
_CHANGE = re.compile(r"\b(?:stopped|no longer|not anymore|anymore|switching from|used to)\b", re.I)
_FUTURE = re.compile(
    r"\b(?:plan(?:ning)? to|going to|will|might|may|next (?:week|month|year)|later|someday)\b",
    re.I,
)
_CURRENT = re.compile(r"\b(?:now|currently|right now|recently|today)\b", re.I)
_NEGATION = re.compile(r"\b(?:don't|do not|doesn't|does not|never|no longer|stopped)\b", re.I)


def _normalize_input(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("’", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"[ \t]+", " ", text).strip()


def _locate(raw_text: str, evidence: str) -> tuple[int | None, int | None]:
    start = raw_text.casefold().find(evidence.casefold())
    return (start, start + len(evidence)) if start >= 0 else (None, None)


def _expand_switch_statement(sentence: str, raw_text: str) -> list[_Clause] | None:
    match = re.fullmatch(
        r"i(?:'m| am) switching from (?P<old>.+?) to (?P<new>.+?)[.!?]?",
        sentence.strip(),
        re.I,
    )
    if not match:
        return None
    start, end = _locate(raw_text, sentence)
    return [
        _Clause(f"I used to focus on {match.group('old')}", sentence, start, end),
        _Clause(f"I am switching to {match.group('new')}", sentence, start, end),
    ]


def _candidate_clauses(text: str) -> list[_Clause]:
    clauses: list[_Clause] = []
    sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        expanded = _expand_switch_statement(sentence, text)
        if expanded is not None:
            clauses.extend(expanded)
            continue

        contrast_parts = re.split(r"\s*,?\s*\b(?:but|however)\b\s*", sentence, flags=re.I)
        for part in contrast_parts:
            atomic_parts = re.split(
                r"\s*;\s*|\s+\band\b\s+(?=(?:i\b|i'm\b|i am\b|i've\b|i'll\b|i'd\b|my\b))",
                part,
                flags=re.I,
            )
            for atomic in atomic_parts:
                atomic = atomic.strip(" ,")
                if not atomic:
                    continue
                if len(atomic) > MAX_CLAUSE_CHARS:
                    atomic = atomic[:MAX_CLAUSE_CHARS].rstrip()
                start, end = _locate(text, atomic)
                clauses.append(_Clause(atomic, atomic, start, end))
    return clauses


def _is_memory_worthy(text: str, suggested_type: MemoryType | None) -> bool:
    stripped = text.strip()
    if not stripped or _FILLER.fullmatch(stripped) or _CASUAL.fullmatch(stripped):
        return False
    if len(stripped.strip(".!? ")) < 4:
        return False
    return suggested_type is not None or bool(_PERSONAL_SIGNAL.search(stripped))


def _classify(text: str, suggested_type: MemoryType | None) -> MemoryType:
    if suggested_type is not None:
        return suggested_type
    for memory_type, pattern in _TYPE_RULES:
        if pattern.search(text):
            return memory_type
    return MemoryType.CONTEXT


def _temporal_analysis(text: str) -> tuple[CandidateTemporalStatus, str | None, CandidateAction]:
    hint_match: re.Match[str] | None
    if (hint_match := _HISTORICAL.search(text)) is not None:
        return CandidateTemporalStatus.HISTORICAL, hint_match.group(0), CandidateAction.SUPERSEDE
    if (hint_match := _CHANGE.search(text)) is not None:
        return CandidateTemporalStatus.HISTORICAL, hint_match.group(0), CandidateAction.SUPERSEDE
    if (hint_match := _FUTURE.search(text)) is not None:
        return CandidateTemporalStatus.FUTURE, hint_match.group(0), CandidateAction.ADD
    if (hint_match := _CURRENT.search(text)) is not None:
        return CandidateTemporalStatus.CURRENT, hint_match.group(0), CandidateAction.ADD
    return CandidateTemporalStatus.CURRENT, None, CandidateAction.ADD


def _canonicalize(text: str) -> str:
    value = text.strip().rstrip(".!?").strip()
    value = re.sub(r"^also\s+", "", value, flags=re.I)
    value = re.sub(r"^i also\s+", "I ", value, flags=re.I)
    value = re.sub(
        r"^keep (?:your )?answers (?:short|concise)$",
        "User prefers concise answers",
        value,
        flags=re.I,
    )
    value = re.sub(r"^maybe\s+i(?:'ll| will)\s+", "User may ", value, flags=re.I)
    value = re.sub(r"^now\s+i(?:'m| am)\s+", "User is now ", value, flags=re.I)
    replacements = (
        (r"^i don't\s+", "User does not "),
        (r"^i do not\s+", "User does not "),
        (r"^i prefer\s+", "User prefers "),
        (r"^i use\s+", "User uses "),
        (r"^i want\s+", "User wants "),
        (r"^i plan\s+", "User plans "),
        (r"^i like\s+", "User likes "),
        (r"^i work\s+", "User works "),
        (r"^i think\s+", "User thinks "),
        (r"^i(?:'m| am)\s+", "User is "),
        (r"^i(?:'ve| have)\s+", "User has "),
        (r"^i(?:'ll| will)\s+", "User will "),
        (r"^i(?:'d| would)\s+", "User would "),
        (r"^i\s+", "User "),
        (r"^my\s+", "User's "),
    )
    for pattern, replacement in replacements:
        updated = re.sub(pattern, replacement, value, count=1, flags=re.I)
        if updated != value:
            value = updated
            break
    value = re.sub(r"\bshort answers\b", "concise answers", value, flags=re.I)
    return re.sub(r"\s+", " ", value).strip()


def _confidence(text: str) -> float:
    score = 0.88
    if _UNCERTAIN.search(text):
        score -= 0.32
    if re.search(r"\b(?:think|guess|could)\b", text, re.I):
        score -= 0.12
    if re.search(r"\b(?:always|never|definitely|absolutely|certainly)\b", text, re.I):
        score += 0.07
    return round(min(1.0, max(0.0, score)), 2)


def _importance(memory_type: MemoryType, text: str) -> float:
    scores = {
        MemoryType.GOAL: 0.85,
        MemoryType.PROJECT: 0.85,
        MemoryType.PREFERENCE: 0.8,
        MemoryType.PROCEDURE: 0.75,
        MemoryType.SKILL: 0.7,
        MemoryType.FACT: 0.65,
        MemoryType.RELATIONSHIP: 0.65,
        MemoryType.OPINION: 0.55,
        MemoryType.CONTEXT: 0.5,
        MemoryType.TEMPORAL: 0.4,
    }
    score = scores[memory_type]
    if re.search(r"\b(?:today|tomorrow|this week|right now)\b", text, re.I):
        score -= 0.15
    return round(min(1.0, max(0.0, score)), 2)


def _dedup_key(candidate: CandidateMemory) -> str:
    return re.sub(r"[^a-z0-9]+", " ", candidate.content.casefold()).strip()


class RuleBasedMemoryExtractor:
    """Side-effect-free deterministic baseline implementing MemoryExtractor."""

    def __init__(self, *, min_confidence: float = 0.3, max_candidates: int = MAX_CANDIDATES) -> None:
        self._min_confidence = min_confidence
        self._max_candidates = max_candidates

    async def extract(
        self,
        text: str,
        source_type: str = "cli_input",
        source_uri: str | None = None,
        suggested_type: MemoryType | None = None,
        tags: list[str] | None = None,
        source_role: SourceRole = SourceRole.USER,
        confirmed_user_information: bool = False,
    ) -> list[CandidateMemory]:
        if not text or not text.strip():
            return []
        if source_role != SourceRole.USER and not confirmed_user_information:
            return []

        normalized = _normalize_input(text[:MAX_INPUT_CHARS])
        input_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        candidates: list[CandidateMemory] = []
        seen: set[str] = set()

        for clause in _candidate_clauses(normalized):
            if len(candidates) >= self._max_candidates:
                break
            if not _is_memory_worthy(clause.text, suggested_type):
                continue
            memory_type = _classify(clause.text, suggested_type)
            temporal_status, temporal_hint, action_hint = _temporal_analysis(clause.text)
            confidence = _confidence(clause.text)
            if confidence < self._min_confidence:
                continue
            candidate = CandidateMemory(
                content=_canonicalize(clause.text),
                memory_type=memory_type,
                confidence=confidence,
                importance=_importance(memory_type, clause.text),
                temporal_status=temporal_status,
                temporal_hint=temporal_hint,
                action_hint=action_hint,
                uncertain=bool(_UNCERTAIN.search(clause.text)),
                negated=bool(_NEGATION.search(clause.text)),
                source_type=source_type,
                source_uri=source_uri,
                source_role=source_role,
                evidence=clause.evidence,
                evidence_start=clause.start,
                evidence_end=clause.end,
                tags=list(tags or []),
                metadata={
                    "input_hash": input_hash,
                    "negated": bool(_NEGATION.search(clause.text)),
                    "confirmed_user_information": confirmed_user_information,
                    "input_truncated": len(text) > MAX_INPUT_CHARS,
                    "extractor": "deterministic-v2",
                },
            )
            key = _dedup_key(candidate)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(candidate)

        return candidates
