"""Input scanning and candidate extraction boundary for ContextOS Phase 2."""

from __future__ import annotations

from contextos.core.enums import EventType, SecretDetectionMode
from contextos.core.exceptions import SecretDetectedError
from contextos.core.models import IngestRequest, IngestResult, RawEvent, ScanResult
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


class IngestionPipeline:
    """Scan input, preserve its raw event, and return unaccepted candidates.

    The later-stage constructor dependencies remain accepted for wiring
    compatibility, but Phase 2 deliberately does not invoke them.
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
        self._scanner = secret_scanner
        self._extractor = memory_extractor
        self._event_repo = event_repo
        self._secret_mode = secret_detection_mode

    async def ingest(self, request: IngestRequest) -> IngestResult:
        warnings: list[str] = []
        content = request.content
        secrets_detected = False
        secrets_redacted = False
        scan_result: ScanResult | None = None

        if not request.skip_secret_scan:
            scan_result = self._scanner.scan(content)
            if scan_result.has_secrets:
                secrets_detected = True
                secret_types = [item.value for item in scan_result.secret_types_found]
                if self._secret_mode == SecretDetectionMode.STRICT:
                    await self._event_repo.append(RawEvent(
                        event_type=EventType.SECRET_DETECTED,
                        source_type=request.source_type,
                        source_uri=request.source_uri,
                        metadata={"secret_types": secret_types, "mode": "strict"},
                        privacy_scan_result=scan_result.model_dump(),
                    ))
                    raise SecretDetectedError(
                        secret_types,
                        "Input rejected in strict mode. Use --skip-secret-scan to override, "
                        "or change detection mode to 'redact' or 'warn'.",
                    )
                if self._secret_mode == SecretDetectionMode.REDACT:
                    content, scan_result = self._scanner.redact(content)
                    secrets_redacted = True
                    warnings.append(f"Secrets redacted: {', '.join(secret_types)}.")
                elif self._secret_mode == SecretDetectionMode.WARN:
                    warnings.append(f"Secrets detected: {', '.join(secret_types)}.")

        event = RawEvent(
            event_type=EventType.INGEST,
            source_type=request.source_type,
            source_uri=request.source_uri,
            content=content,
            metadata={
                "original_length": len(request.content),
                "processed_length": len(content),
                "skip_secret_scan": request.skip_secret_scan,
                "source_role": request.source_role.value,
            },
            privacy_scan_result=scan_result.model_dump() if scan_result else None,
        )
        await self._event_repo.append(event)

        candidates = await self._extractor.extract(
            text=content,
            source_type=request.source_type,
            source_uri=request.source_uri,
            suggested_type=request.memory_type,
            tags=request.tags,
            source_role=request.source_role,
            confirmed_user_information=request.confirmed_user_information,
        )
        if not candidates:
            warnings.append("No memory candidates could be extracted from the input.")

        # Validation, acceptance, persistence, embedding, and indexing are later phases.
        return IngestResult(
            event_id=event.id,
            candidates=candidates,
            secrets_detected=secrets_detected,
            secrets_redacted=secrets_redacted,
            warnings=warnings,
        )
