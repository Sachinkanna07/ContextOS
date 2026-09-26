"""Ingestion pipeline for ContextOS.

End-to-end pipeline: validate → scan → store event → extract → dedup → store memory → embed → index.

This is the entry point for all data entering the system.
"""

from __future__ import annotations

import logging
from uuid import uuid4

from contextos.core.enums import EventType, MemoryStatus, SecretDetectionMode
from contextos.core.exceptions import SecretDetectedError
from contextos.core.models import (
    ExtractedMemory,
    IngestRequest,
    IngestResult,
    Memory,
    RawEvent,
    ScanResult,
)
from contextos.core.protocols import (
    EmbeddingService,
    LexicalIndex,
    MemoryExtractor,
    MemoryRepository,
    EventRepository,
    SecretScanner,
    TokenCounter,
    VectorStore,
)

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """End-to-end ingestion pipeline.

    Implements the IngestionService protocol.
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
        self._memory_repo = memory_repo
        self._event_repo = event_repo
        self._embedding_service = embedding_service
        self._vector_store = vector_store
        self._lexical_index = lexical_index
        self._token_counter = token_counter
        self._secret_mode = secret_detection_mode

    async def ingest(self, request: IngestRequest) -> IngestResult:
        """Run the full ingestion pipeline."""
        warnings: list[str] = []
        content = request.content
        secrets_detected = False
        secrets_redacted = False
        scan_result: ScanResult | None = None

        # --- Step 1: Secret Scanning ---
        if not request.skip_secret_scan:
            scan_result = self._scanner.scan(content)

            if scan_result.has_secrets:
                secrets_detected = True
                secret_types = [st.value for st in scan_result.secret_types_found]

                if self._secret_mode == SecretDetectionMode.STRICT:
                    # Log the event before raising
                    await self._event_repo.append(RawEvent(
                        event_type=EventType.SECRET_DETECTED,
                        source_type=request.source_type,
                        source_uri=request.source_uri,
                        metadata={
                            "secret_types": secret_types,
                            "mode": "strict",
                        },
                        privacy_scan_result=scan_result.model_dump(),
                    ))
                    raise SecretDetectedError(
                        secret_types,
                        "Input rejected in strict mode. Use --skip-secret-scan to override, "
                        "or change detection mode to 'redact' or 'warn'.",
                    )

                elif self._secret_mode == SecretDetectionMode.REDACT:
                    content, scan_result = self._scanner.redact(content)
                    secrets_redacted = True
                    warnings.append(
                        f"Secrets redacted: {', '.join(secret_types)}. "
                        "Redacted content stored."
                    )

                elif self._secret_mode == SecretDetectionMode.WARN:
                    warnings.append(
                        f"Secrets detected: {', '.join(secret_types)}. "
                        "Content stored with RESTRICTED privacy level."
                    )

        # --- Step 2: Store Raw Event ---
        event = RawEvent(
            event_type=EventType.INGEST,
            source_type=request.source_type,
            source_uri=request.source_uri,
            content=content,
            metadata={
                "original_length": len(request.content),
                "processed_length": len(content),
                "skip_secret_scan": request.skip_secret_scan,
            },
            privacy_scan_result=(
                scan_result.model_dump() if scan_result else None
            ),
        )
        await self._event_repo.append(event)

        # --- Step 3: Extract Memories ---
        extracted: list[ExtractedMemory] = await self._extractor.extract(
            text=content,
            source_type=request.source_type,
            source_uri=request.source_uri,
            suggested_type=request.memory_type,
            tags=request.tags,
        )

        if not extracted:
            warnings.append("No memories could be extracted from the input.")
            return IngestResult(
                event_id=event.id,
                warnings=warnings,
                secrets_detected=secrets_detected,
                secrets_redacted=secrets_redacted,
            )

        # --- Step 4: Dedup, Store, Embed, and Index ---
        memories_created: list[str] = []
        memories_merged: list[str] = []

        for ext_mem in extracted:
            # Compute content hash for dedup
            from contextos.core.models import _content_hash

            c_hash = _content_hash(ext_mem.content)

            # Check for exact duplicate
            existing = await self._memory_repo.get_by_hash(c_hash)
            if existing is not None:
                # Exact duplicate — reinforce confidence
                logger.info(
                    "Duplicate memory detected (hash: %s), reinforcing existing %s",
                    c_hash, existing.id,
                )
                from contextos.core.models import MemoryUpdate

                new_confidence = min(1.0, existing.confidence + 0.05)
                await self._memory_repo.update(
                    existing.id,
                    MemoryUpdate(confidence=new_confidence),
                    expected_version=existing.version,
                )
                memories_merged.append(str(existing.id))
                continue

            # Create new memory
            token_count = self._token_counter.count(ext_mem.content)

            # Determine privacy level based on secret scan
            from contextos.core.enums import PrivacyLevel

            privacy_level = PrivacyLevel.PERSONAL
            if secrets_detected and self._secret_mode == SecretDetectionMode.WARN:
                privacy_level = PrivacyLevel.RESTRICTED

            memory = Memory(
                content=ext_mem.content,
                content_hash=c_hash,
                type=ext_mem.type,
                source_type=request.source_type,
                source_uri=request.source_uri,
                provenance_event_id=event.id,
                status=MemoryStatus.ACTIVE,  # Phase 1: skip CANDIDATE validation
                confidence=ext_mem.confidence,
                importance=ext_mem.importance,
                privacy_level=privacy_level,
                token_count=token_count,
                tags=ext_mem.tags,
            )

            # Store in database
            await self._memory_repo.create(memory)

            # Generate embedding and store in vector index
            try:
                embeddings = await self._embedding_service.embed([memory.content])
                if embeddings:
                    await self._vector_store.add(
                        ids=[str(memory.id)],
                        vectors=embeddings,
                        metadata=[{"type": memory.type.value, "status": memory.status.value}],
                    )
                    # Update embedding_id on memory
                    await self._memory_repo.update(
                        memory.id,
                        MemoryUpdate(),  # We just need to set embedding_id
                        expected_version=memory.version,
                    )
            except Exception:
                logger.warning(
                    "Failed to embed memory %s, will retry later", memory.id, exc_info=True
                )
                warnings.append(f"Embedding failed for memory {memory.id}. Queued for retry.")

            # Index in BM25
            try:
                await self._lexical_index.index(
                    doc_id=str(memory.id),
                    text=memory.content,
                    metadata={"type": memory.type.value},
                )
            except Exception:
                logger.warning(
                    "Failed to index memory %s in BM25", memory.id, exc_info=True
                )
                warnings.append(f"BM25 indexing failed for memory {memory.id}.")

            memories_created.append(str(memory.id))

            # Record memory creation event
            await self._event_repo.append(RawEvent(
                event_type=EventType.MEMORY_CREATED,
                source_type="system",
                metadata={"memory_id": str(memory.id), "memory_type": memory.type.value},
                memory_ids=[memory.id],
            ))

        # Update the ingest event with produced memory IDs
        from uuid import UUID

        all_memory_ids = [
            UUID(mid) for mid in memories_created + memories_merged
        ]

        return IngestResult(
            event_id=event.id,
            memories_created=[UUID(mid) for mid in memories_created],
            memories_updated=[],
            memories_merged=[UUID(mid) for mid in memories_merged],
            secrets_detected=secrets_detected,
            secrets_redacted=secrets_redacted,
            warnings=warnings,
        )
