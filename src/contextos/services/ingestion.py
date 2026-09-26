"""Two-stage privacy-gated candidate ingestion for ContextOS Phase 3."""

from __future__ import annotations

from contextos.core.enums import EventType, PrivacyDecision, SecretDetectionMode
from contextos.core.exceptions import IngestionError, SecretDetectedError
from contextos.core.models import IngestRequest, IngestResult, RawEvent
from contextos.core.protocols import (
    EmbeddingService,
    EventRepository,
    LexicalIndex,
    MemoryExtractor,
    MemoryRepository,
    SecretScanner,
    TokenCounter,
    VectorStore,
)
from contextos.services.privacy import PrivacyGate


class IngestionPipeline:
    """Sanitize raw input, extract candidates, and gate candidates again.

    Raw input exists only in the caller/request and transient local variables.
    Only sanitized event content and value-free findings can reach persistence.
    Later-stage constructor dependencies remain for wiring compatibility and are
    deliberately not invoked during candidate extraction.
    """

    def __init__(
        self,
        *,
        secret_scanner: SecretScanner,
        memory_extractor: MemoryExtractor,
        memory_repo: MemoryRepository,
        event_repo: EventRepository,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
        lexical_index: LexicalIndex,
        token_counter: TokenCounter,
        secret_detection_mode: SecretDetectionMode = SecretDetectionMode.STRICT,
    ) -> None:
        self._extractor = memory_extractor
        self._event_repo = event_repo
        self._secret_mode = secret_detection_mode
        self._privacy_gate = PrivacyGate(secret_scanner)

    async def ingest(self, request: IngestRequest) -> IngestResult:
        warnings: list[str] = []

        # skip_secret_scan is retained in the request model for compatibility,
        # but it cannot bypass the Phase 3 persistence boundary.
        gated_input = self._privacy_gate.gate_input(
            request.content,
            source_type=request.source_type,
            source_uri=request.source_uri,
            tags=request.tags,
            source_role=request.source_role,
            mode=self._secret_mode,
        )
        assessment = gated_input.assessment
        secrets_detected = assessment.has_findings
        secrets_redacted = assessment.decision in {
            PrivacyDecision.REDACT,
            PrivacyDecision.QUARANTINE,
        }
        safe_assessment = assessment.model_dump(exclude={"sanitized_text"})

        if assessment.decision == PrivacyDecision.REJECT:
            await self._event_repo.append(RawEvent(
                event_type=EventType.SECRET_DETECTED,
                source_type=gated_input.source_type,
                source_uri=gated_input.source_uri,
                content=None,
                metadata={"privacy_decision": assessment.decision.value},
                privacy_scan_result=safe_assessment,
            ))
            raise SecretDetectedError(
                sorted({finding.category.value for finding in assessment.findings}),
                "Input rejected by the pre-ingest privacy gate.",
            )

        if len(gated_input.content) > 100_000:
            raise IngestionError("Input exceeds maximum length of 100000 characters")

        sanitized_content = gated_input.content
        event = RawEvent(
            event_type=EventType.INGEST,
            source_type=gated_input.source_type,
            source_uri=gated_input.source_uri,
            content=sanitized_content,
            metadata={
                "original_length": len(request.content),
                "processed_length": len(sanitized_content),
                "privacy_decision": assessment.decision.value,
                "scan_bypass_requested": request.skip_secret_scan,
                "source_role": request.source_role.value,
                "source_trust": assessment.source_trust.value,
            },
            privacy_scan_result=safe_assessment,
        )
        await self._event_repo.append(event)

        if assessment.decision == PrivacyDecision.QUARANTINE:
            warnings.append("Input quarantined by the privacy gate; no candidates extracted.")
            return IngestResult(
                event_id=event.id,
                candidates=[],
                privacy_assessment=assessment,
                secrets_detected=secrets_detected,
                secrets_redacted=True,
                warnings=warnings,
            )

        extracted = await self._extractor.extract(
            text=sanitized_content,
            source_type=gated_input.source_type,
            source_uri=gated_input.source_uri,
            suggested_type=request.memory_type,
            tags=gated_input.tags,
            source_role=request.source_role,
            confirmed_user_information=request.confirmed_user_information,
        )
        candidates, blocked = self._privacy_gate.gate_candidates(
            extracted, source_trust=assessment.source_trust
        )
        if secrets_redacted:
            warnings.append("Sensitive values were redacted before extraction.")
        if blocked:
            warnings.append(f"Privacy gate blocked {len(blocked)} candidate(s).")
        if not candidates:
            warnings.append("No safe memory candidates could be extracted from the input.")

        # Acceptance, long-term memory persistence, embedding, and indexing are later phases.
        return IngestResult(
            event_id=event.id,
            candidates=candidates,
            privacy_assessment=assessment,
            blocked_candidate_assessments=blocked,
            secrets_detected=secrets_detected,
            secrets_redacted=secrets_redacted,
            warnings=warnings,
        )
