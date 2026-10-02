"""Deterministic temporal memory identity, resolution, and timeline lookup."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from uuid import UUID

from contextos.core.enums import (
    CandidateTemporalStatus,
    MemoryStatus,
    MemoryType,
    TemporalOutcome,
    TemporalPrecision,
)
from contextos.core.exceptions import InvalidTransitionError
from contextos.core.models import (
    CandidateMemory,
    Memory,
    MemorySlot,
    TemporalChange,
    TemporalDecision,
    TemporalResolutionResult,
)
from contextos.core.protocols import MemoryRepository
from contextos.services.optimization import information_tokens


_UNCERTAIN = re.compile(
    r"\b(?:might|may|maybe|perhaps|possibly|could|i think|trying|tried|experimenting)\b",
    re.I,
)
_FUTURE = re.compile(
    r"\b(?:might learn|may learn|plan(?:s)? to|next year|later|will learn|intend(?:s)? to)\b",
    re.I,
)
_HISTORICAL = re.compile(
    r"\b(?:used to|previously|formerly|before|in the past)\b", re.I
)
_CORRECTION = re.compile(r"\b(?:correction|actually|to correct|rather than)\b", re.I)
_CHANGE = re.compile(
    r"\b(?:now|currently|no longer|stopped|switched from|changed to|instead|anymore|used to prefer)\b",
    re.I,
)
_NEGATION = re.compile(
    r"\b(?:do not|does not|did not|don't|doesn't|didn't|never|no longer|stopped)\b",
    re.I,
)
_LANGUAGES: tuple[tuple[str, str], ...] = (
    ("c++17", "c++17"), ("c++", "c++"), ("python", "python"),
    ("rust", "rust"), ("javascript", "javascript"), ("typescript", "typescript"),
    ("java", "java"), ("golang", "go"), ("go", "go"),
)
_MODELS: tuple[str, ...] = ("ollama", "localai", "qwen30b", "qwen14b", "qwen9b")
_MONTHS = (
    "january|february|march|april|may|june|july|august|september|october|"
    "november|december"
)


class TemporalSlotAnalyzer:
    """Derive inspectable slots and temporal metadata from bounded cue rules."""

    def prepare(self, memory: Memory) -> Memory:
        text = memory.content
        status = memory.temporal_status
        if status == CandidateTemporalStatus.UNSPECIFIED:
            if _FUTURE.search(text):
                status = CandidateTemporalStatus.FUTURE
            elif _HISTORICAL.search(text):
                status = CandidateTemporalStatus.HISTORICAL
            elif _CHANGE.search(text):
                status = CandidateTemporalStatus.CURRENT

        precision, expression, inferred_from = self._temporal_expression(text)
        if memory.temporal_precision != TemporalPrecision.UNKNOWN:
            precision = memory.temporal_precision
        valid_from = memory.valid_from or inferred_from
        return memory.model_copy(update={
            "slot": memory.slot or self.slot_for(text, memory.type),
            "temporal_status": status,
            "temporal_precision": precision,
            "temporal_expression": memory.temporal_expression or expression,
            "valid_from": valid_from,
            "uncertain": memory.uncertain or bool(_UNCERTAIN.search(text)),
            "negated": memory.negated or bool(_NEGATION.search(text)),
        })

    def slot_for(self, text: str, memory_type: MemoryType) -> MemorySlot:
        normalized = text.casefold()
        language = self._first_value(normalized, _LANGUAGES)
        if language:
            if re.search(r"\b(?:beginner|proficient|advanced|expert)\b", normalized):
                return MemorySlot(
                    subject="user", property="programming_skill",
                    scope=language, entity=language,
                )
            return MemorySlot(
                subject="user",
                property="programming_language",
                scope=self._language_scope(normalized),
            )
        if "ram" in normalized and re.search(r"\b\d+\s*gb\b", normalized):
            device = self._device_entity(normalized)
            return MemorySlot(
                subject="machine", property="ram_capacity",
                scope=device or "global", entity=device,
            )
        if "docker" in normalized:
            scope, entity = self._project_scope(normalized)
            return MemorySlot(
                subject="user", property="tool_usage", scope=scope, entity=entity,
                qualifiers=("docker",),
            )
        if any(model in normalized for model in _MODELS):
            return MemorySlot(subject="user", property="local_model", scope="local_machine")
        if re.search(r"\b(?:attended|attend|did not attend|didn't attend)\b", normalized):
            match = re.search(r"\bevent\s+([a-z0-9_-]+)", normalized)
            entity = f"event_{match.group(1)}" if match else "event"
            return MemorySlot(
                subject="user", property="attendance", scope=entity, entity=entity
            )
        if any(word in normalized for word in ("concise", "short answers", "detailed answers")):
            context = self._response_context(normalized)
            return MemorySlot(
                subject="user", property="response_style",
                scope=context or "global", entity=context,
            )
        if "project" in normalized and any(
            word in normalized for word in ("active", "paused", "complete", "cancelled")
        ):
            scope, entity = self._project_scope(normalized)
            return MemorySlot(
                subject="user", property="project_status", scope=scope, entity=entity
            )
        concepts = sorted(information_tokens(text))
        property_name = memory_type.value
        scope = "_".join(concepts[:3]) if concepts else "global"
        return MemorySlot(subject="user", property=property_name, scope=scope)

    def value_for(self, memory: Memory) -> str:
        normalized = memory.content.casefold()
        if memory.slot and memory.slot.property == "programming_skill":
            for level in ("beginner", "learning", "proficient", "advanced", "expert"):
                if level in normalized:
                    return level
        language = self._first_value(normalized, _LANGUAGES)
        if language:
            return language
        ram = re.search(r"\b(\d+)\s*gb\s*ram\b", normalized)
        if ram:
            return f"{ram.group(1)}gb"
        if memory.slot and memory.slot.property == "tool_usage":
            return memory.slot.qualifiers[0] if memory.slot.qualifiers else "tool"
        if memory.slot and memory.slot.property == "attendance":
            return "attended"
        for model in _MODELS:
            if model in normalized:
                return model
        if "concise" in normalized or "short answers" in normalized:
            return "concise"
        if "detailed answers" in normalized:
            return "detailed"
        status = next(
            (word for word in ("active", "paused", "complete", "cancelled")
             if word in normalized),
            None,
        )
        if status:
            return status
        ignored = {
            "user", "currently", "now", "previously", "actually", "correction",
            "mainly", "primarily", "use", "uses", "using", "for", "the", "a",
        }
        return " ".join(sorted(information_tokens(normalized) - ignored))

    @staticmethod
    def _first_value(text: str, values: tuple[tuple[str, str], ...]) -> str | None:
        for needle, canonical in values:
            if re.search(rf"(?<![a-z0-9]){re.escape(needle)}(?![a-z0-9])", text):
                return canonical
        return None

    @staticmethod
    def _language_scope(text: str) -> str:
        if "machine learning" in text or re.search(r"\bml\b", text):
            return "machine_learning"
        if "data science" in text or re.search(r"\bdata\s+sci", text):
            return "data_science"
        if "interview" in text:
            return "systems_interviews"
        if "systems programming" in text or "for systems" in text:
            return "systems_programming"
        if "embedded" in text:
            return "embedded"
        if "automation" in text:
            return "automation"
        if "web development" in text or "for web" in text:
            return "web_development"
        return "global"

    @staticmethod
    def _device_entity(text: str) -> str | None:
        """Extract a machine/device entity from RAM-related text."""
        for device in ("laptop", "desktop", "server", "workstation", "pc", "computer"):
            if device in text:
                return device
        return None

    @staticmethod
    def _response_context(text: str) -> str | None:
        """Extract the context qualifier from 'concise/detailed answers for X'."""
        match = re.search(
            r"\bfor\s+([a-z]+(?:\s+[a-z]+)??)\s*(?:questions?|tasks?|topics?|work)?\s*[.,!?]?\s*$",
            text,
        )
        if match:
            raw = match.group(1).strip().replace(" ", "_")
            return raw if raw else None
        return None

    @staticmethod
    def _project_scope(text: str) -> tuple[str, str | None]:
        match = re.search(r"\bproject\s+([a-z0-9_-]+)", text)
        if not match:
            return "global", None
        entity = f"project_{match.group(1)}"
        return entity, entity

    @staticmethod
    def _temporal_expression(
        text: str,
    ) -> tuple[TemporalPrecision, str | None, datetime | None]:
        timestamp = re.search(r"\b(20\d{2}-\d{2}-\d{2}T\d{2}:\d{2}(?::\d{2})?(?:Z|[+-]\d{2}:?\d{2}))\b", text)
        if timestamp:
            value = timestamp.group(1).replace("Z", "+00:00")
            return TemporalPrecision.EXACT, timestamp.group(1), datetime.fromisoformat(value)
        date = re.search(r"\b(20\d{2}-\d{2}-\d{2})\b", text)
        if date:
            return (
                TemporalPrecision.DATE,
                date.group(1),
                datetime.strptime(date.group(1), "%Y-%m-%d").replace(tzinfo=timezone.utc),
            )
        month = re.search(rf"\b({_MONTHS})\s+(20\d{{2}})\b", text, re.I)
        if month:
            return TemporalPrecision.MONTH, month.group(0), None
        year = re.search(r"\b(?:in\s+)?(20\d{2})\b", text)
        if year:
            return TemporalPrecision.YEAR, year.group(1), None
        relative = re.search(
            r"\b(now|currently|recently|last (?:week|month|year)|next year|before|previously|later)\b",
            text,
            re.I,
        )
        if relative:
            return TemporalPrecision.RELATIVE, relative.group(0).casefold(), None
        return TemporalPrecision.UNKNOWN, None, None


class TemporalMemoryService:
    """Resolve temporal candidates conservatively and apply one atomic plan."""

    supersession_threshold = 0.70

    def __init__(self, repository: MemoryRepository) -> None:
        self._repository = repository
        self._analyzer = TemporalSlotAnalyzer()

    async def accept(
        self,
        candidate: CandidateMemory,
        *,
        provenance_event_id: UUID | None = None,
    ) -> TemporalResolutionResult:
        memory = Memory(
            content=candidate.content,
            type=candidate.memory_type,
            source_type=candidate.source_type,
            source_uri=candidate.source_uri,
            provenance_event_id=provenance_event_id,
            status=MemoryStatus.CANDIDATE,
            confidence=candidate.confidence,
            importance=candidate.importance,
            tags=candidate.tags,
            observed_at=candidate.observed_at or datetime.now(timezone.utc),
            valid_from=candidate.valid_from,
            valid_to=candidate.valid_to,
            temporal_precision=candidate.temporal_precision,
            temporal_status=candidate.temporal_status,
            temporal_expression=candidate.temporal_hint,
            slot=candidate.slot,
            uncertain=candidate.uncertain or bool(candidate.metadata.get("uncertain")),
            negated=candidate.negated or bool(candidate.metadata.get("negated")),
        )
        return await self.resolve(memory)

    async def resolve(self, candidate: Memory) -> TemporalResolutionResult:
        if candidate.status in {MemoryStatus.DELETED, MemoryStatus.PURGED}:
            raise InvalidTransitionError(
                str(candidate.id), candidate.status.value, MemoryStatus.ACTIVE.value
            )
        prepared = self._analyzer.prepare(candidate)
        assert prepared.slot is not None
        decision = await self.decide(prepared)
        if "late_ingestion" in decision.evidence:
            prepared = prepared.model_copy(update={
                "temporal_status": CandidateTemporalStatus.HISTORICAL
            })
        return await self._repository.apply_temporal_decision(prepared, decision)

    async def decide(self, candidate: Memory) -> TemporalDecision:
        assert candidate.slot is not None
        exact = await self._repository.get_by_hash(candidate.content_hash)
        if exact and exact.status not in {MemoryStatus.DELETED, MemoryStatus.PURGED}:
            return self._decision(
                candidate, TemporalOutcome.DUPLICATE, exact, ["exact_content_hash"], 1.0
            )

        timeline = await self._repository.list_by_slot(candidate.slot.key)
        current = [
            memory for memory in timeline
            if memory.status == MemoryStatus.ACTIVE
            and memory.temporal_status != CandidateTemporalStatus.FUTURE
        ]
        related = self._latest(current)
        if hasattr(self._repository, "latest_active_peer"):
            scoped_peer = await self._repository.latest_active_peer(candidate.slot)
        else:
            # Compatibility for non-SQLite repository implementations.
            all_temporal = await self._repository.list_temporal(limit=500)
            scoped_peer = self._latest([
                memory for memory in all_temporal
                if memory.slot is not None
                and memory.slot.subject == candidate.slot.subject
                and memory.slot.property == candidate.slot.property
                and memory.slot.key != candidate.slot.key
                and memory.status == MemoryStatus.ACTIVE
            ])

        if candidate.temporal_status == CandidateTemporalStatus.HISTORICAL:
            return self._decision(
                candidate, TemporalOutcome.ADD_NEW, related,
                ["historical_claim", "does_not_replace_current"], 0.98,
            )
        if candidate.temporal_status == CandidateTemporalStatus.FUTURE:
            return self._decision(
                candidate,
                TemporalOutcome.COEXIST if related or scoped_peer else TemporalOutcome.ADD_NEW,
                related or scoped_peer,
                ["future_intention", "current_state_preserved"], 0.98,
            )
        if candidate.uncertain:
            return self._decision(
                candidate,
                TemporalOutcome.COEXIST if related or scoped_peer else TemporalOutcome.ADD_NEW,
                related or scoped_peer,
                ["uncertain_claim", "conservative_resolution"], 0.80,
            )
        if related is None:
            if scoped_peer is not None:
                return self._decision(
                    candidate, TemporalOutcome.COEXIST, scoped_peer,
                    ["same_property", "different_scope"], 0.99,
                )
            return self._decision(
                candidate, TemporalOutcome.ADD_NEW, None, ["empty_slot"], 1.0
            )

        candidate_value = self._analyzer.value_for(candidate)
        related_value = self._analyzer.value_for(related)
        if candidate_value == related_value and candidate.negated == related.negated:
            return self._decision(
                candidate, TemporalOutcome.NO_CHANGE, related,
                ["same_slot", "same_normalized_value"], 0.98,
            )
        if self._is_out_of_order(candidate, related):
            historical = candidate.model_copy(update={
                "temporal_status": CandidateTemporalStatus.HISTORICAL
            })
            return self._decision(
                historical, TemporalOutcome.ADD_NEW, related,
                ["effective_time_precedes_current", "late_ingestion"], 0.99,
            )
        if _CORRECTION.search(candidate.content):
            return self._decision(
                candidate, TemporalOutcome.CORRECT, related,
                ["same_slot", "explicit_correction"], 0.99,
            )
        if _CHANGE.search(candidate.content):
            confidence = min(0.99, 0.65 + 0.35 * candidate.confidence)
            if confidence >= self.supersession_threshold:
                return self._decision(
                    candidate, TemporalOutcome.SUPERSEDE, related,
                    ["same_slot", "explicit_change_cue"], confidence,
                )
        return self._decision(
            candidate, TemporalOutcome.CONTRADICT, related,
            ["same_slot", "incompatible_value", "no_transition_evidence"], 0.90,
        )

    async def get_current_state(self, slot: MemorySlot | str) -> list[Memory]:
        key = slot.key if isinstance(slot, MemorySlot) else slot
        timeline = await self._repository.list_by_slot(key)
        return [
            memory for memory in timeline
            if memory.status == MemoryStatus.ACTIVE
            and memory.temporal_status != CandidateTemporalStatus.FUTURE
            and not memory.uncertain
        ]

    async def get_history(self, slot: MemorySlot | str) -> list[Memory]:
        key = slot.key if isinstance(slot, MemorySlot) else slot
        timeline = await self._repository.list_by_slot(key)
        return sorted(timeline, key=self._timeline_order)

    async def get_future(self, slot: MemorySlot | str | None = None) -> list[Memory]:
        if slot is not None:
            values = await self._repository.list_by_slot(
                slot.key if isinstance(slot, MemorySlot) else slot
            )
        else:
            values = await self._repository.list_temporal(limit=500)
        return [
            memory for memory in values
            if memory.temporal_status == CandidateTemporalStatus.FUTURE
        ]

    async def get_previous(self, memory_id: UUID) -> Memory | None:
        memory = await self._repository.get(memory_id)
        if memory is None or memory.supersedes is None:
            return None
        return await self._repository.get(memory.supersedes)

    def _decision(
        self,
        candidate: Memory,
        outcome: TemporalOutcome,
        related: Memory | None,
        evidence: list[str],
        confidence: float,
    ) -> TemporalDecision:
        assert candidate.slot is not None
        changes: list[TemporalChange] = []
        if outcome not in {TemporalOutcome.DUPLICATE, TemporalOutcome.NO_CHANGE}:
            candidate_status = (
                MemoryStatus.CONTRADICTED
                if outcome == TemporalOutcome.CONTRADICT
                else MemoryStatus.HISTORICAL
                if candidate.temporal_status == CandidateTemporalStatus.HISTORICAL
                else MemoryStatus.ACTIVE
            )
            changes.append(TemporalChange(
                memory_id=candidate.id,
                from_status=MemoryStatus.CANDIDATE,
                to_status=candidate_status,
            ))
        if related and outcome in {TemporalOutcome.SUPERSEDE, TemporalOutcome.CORRECT}:
            changes.insert(0, TemporalChange(
                memory_id=related.id,
                from_status=related.status,
                to_status=MemoryStatus.SUPERSEDED,
            ))
        if related and outcome == TemporalOutcome.CONTRADICT:
            changes.insert(0, TemporalChange(
                memory_id=related.id,
                from_status=related.status,
                to_status=MemoryStatus.CONTRADICTED,
            ))
        return TemporalDecision(
            candidate_id=candidate.id,
            slot=candidate.slot,
            outcome=outcome,
            compared_memory_ids=[related.id] if related else [],
            related_memory_id=related.id if related else None,
            evidence=evidence,
            confidence=confidence,
            changes=changes,
        )

    @staticmethod
    def _latest(memories: list[Memory]) -> Memory | None:
        if not memories:
            return None
        return max(memories, key=TemporalMemoryService._timeline_order)

    @staticmethod
    def _timeline_order(memory: Memory) -> tuple[datetime, datetime, str]:
        return (
            memory.valid_from or memory.observed_at,
            memory.observed_at,
            str(memory.id),
        )

    @staticmethod
    def _is_out_of_order(candidate: Memory, current: Memory) -> bool:
        if candidate.valid_from is None:
            return False
        current_effective = current.valid_from or current.observed_at
        return candidate.valid_from < current_effective
