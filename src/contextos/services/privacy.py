"""Two-stage local privacy gate for raw input and extracted candidates."""

from __future__ import annotations

import hashlib
import hmac
import json
import re
import secrets
from dataclasses import dataclass
from typing import Any

from contextos.core.enums import (
    PrivacyClassification,
    PrivacyDecision,
    PrivacySeverity,
    SecretDetectionMode,
    SecretType,
    SourceRole,
    SourceTrust,
)
from contextos.core.models import (
    CandidateMemory,
    PrivacyAssessment,
    PrivacyFinding,
    SecretMatch,
)
from contextos.core.protocols import SecretScanner


_CRITICAL_TYPES = {
    SecretType.PRIVATE_KEY,
    SecretType.SSH_PRIVATE_KEY,
    SecretType.PASSWORD,
    SecretType.CONNECTION_STRING,
    SecretType.AUTHORIZATION_HEADER,
    SecretType.SESSION_COOKIE,
    SecretType.ACCESS_TOKEN,
}
_CREDENTIAL_LANGUAGE = re.compile(
    r"\b(?:api key|password|passwd|otp|pin|token|authorization|cookie|session|"
    r"private key|secret key|connection string)\b",
    re.I,
)
_UNTRUSTED_INSTRUCTION = re.compile(
    r"\b(?:ignore (?:all |the )?(?:previous|prior) instructions|system instruction|"
    r"developer instruction|save .{0,40}(?:password|token|secret).{0,20}(?:memory|forever)|"
    r"permanently remember)\b",
    re.I,
)
_FINGERPRINT_KEY = secrets.token_bytes(32)


@dataclass(frozen=True)
class GatedInput:
    assessment: PrivacyAssessment
    content: str
    source_type: str
    source_uri: str | None
    tags: list[str]


def _hash(value: str) -> str:
    return hmac.new(_FINGERPRINT_KEY, value.encode("utf-8"), hashlib.sha256).hexdigest()


def classify_source(source_type: str, source_role: SourceRole) -> SourceTrust:
    """Classify source trust as metadata, never as proof that content is true."""
    normalized = source_type.casefold().replace("-", "_")
    if source_role in {SourceRole.ASSISTANT, SourceRole.SYSTEM}:
        return SourceTrust.MODEL_OUTPUT
    if any(marker in normalized for marker in ("web", "browser", "url")):
        return SourceTrust.EXTERNAL_WEBPAGE
    if any(marker in normalized for marker in ("document", "file", "import")):
        return SourceTrust.IMPORTED_DOCUMENT
    if "tool" in normalized:
        return SourceTrust.TOOL_OUTPUT
    if "connector" in normalized:
        return SourceTrust.LOCAL_TRUSTED_CONNECTOR
    return SourceTrust.DIRECT_USER


class PrivacyGate:
    """Deterministic pre-ingest and post-extraction security policy."""

    def __init__(self, scanner: SecretScanner) -> None:
        self._scanner = scanner

    def gate_input(
        self,
        content: str,
        *,
        source_type: str,
        source_uri: str | None,
        tags: list[str],
        source_role: SourceRole,
        mode: SecretDetectionMode,
    ) -> GatedInput:
        safe_content, findings = self._scan_text(content, "content")
        safe_source_type, source_type_findings = self._scan_text(
            source_type, "source_type"
        )
        findings.extend(source_type_findings)
        safe_source_uri: str | None = None
        if source_uri is not None:
            safe_source_uri, source_uri_findings = self._scan_text(
                source_uri, "source_uri"
            )
            findings.extend(source_uri_findings)
        safe_tags: list[str] = []
        for index, tag in enumerate(tags):
            safe_tag, tag_findings = self._scan_text(tag, f"tags[{index}]")
            safe_tags.append(safe_tag)
            findings.extend(tag_findings)

        trust = classify_source(safe_source_type, source_role)
        untrusted_instruction = (
            trust not in {SourceTrust.DIRECT_USER, SourceTrust.LOCAL_TRUSTED_CONNECTOR}
            and bool(_UNTRUSTED_INSTRUCTION.search(safe_content))
        )
        decision, classification = self._policy(findings, mode)
        raw_envelope = json.dumps(
            [content, source_type, source_uri, tags], ensure_ascii=False, separators=(",", ":")
        )
        assessment = PrivacyAssessment(
            decision=decision,
            classification=classification,
            source_trust=trust,
            findings=findings,
            sanitized_text=safe_content,
            scanned_length=sum(
                len(value) for value in [content, source_type, source_uri or "", *tags]
            ),
            input_hash=_hash(raw_envelope),
            untrusted_instruction_detected=untrusted_instruction,
        )
        return GatedInput(
            assessment=assessment,
            content=safe_content,
            source_type=safe_source_type,
            source_uri=safe_source_uri,
            tags=safe_tags,
        )

    def assess_input(
        self,
        text: str,
        *,
        source_type: str,
        source_role: SourceRole,
        mode: SecretDetectionMode,
    ) -> PrivacyAssessment:
        """Compatibility helper for callers assessing only a content string."""
        return self.gate_input(
            text,
            source_type=source_type,
            source_uri=None,
            tags=[],
            source_role=source_role,
            mode=mode,
        ).assessment

    def assess_candidate(
        self,
        candidate: CandidateMemory,
        *,
        source_trust: SourceTrust,
    ) -> tuple[CandidateMemory | None, PrivacyAssessment]:
        content, findings = self._scan_text(candidate.content, "candidate.content")
        evidence, evidence_findings = self._scan_text(
            candidate.evidence, "candidate.evidence"
        )
        findings.extend(evidence_findings)
        source_type, source_type_findings = self._scan_text(
            candidate.source_type, "candidate.source_type"
        )
        findings.extend(source_type_findings)
        source_uri = None
        if candidate.source_uri is not None:
            source_uri, source_uri_findings = self._scan_text(
                candidate.source_uri, "candidate.source_uri"
            )
            findings.extend(source_uri_findings)
        tags, tag_findings = self._sanitize_structure(candidate.tags, "candidate.tags")
        metadata, metadata_findings = self._sanitize_structure(
            candidate.metadata, "candidate.metadata"
        )
        findings.extend(tag_findings)
        findings.extend(metadata_findings)

        contains_redaction = "[REDACTED:" in content or "[REDACTED:" in evidence
        credential_only = contains_redaction and bool(_CREDENTIAL_LANGUAGE.search(content))
        combined = f"{content}\n{evidence}"
        untrusted_instruction = (
            source_trust not in {
                SourceTrust.DIRECT_USER,
                SourceTrust.LOCAL_TRUSTED_CONNECTOR,
            }
            and bool(_UNTRUSTED_INSTRUCTION.search(combined))
        )
        blocked = credential_only or untrusted_instruction
        assessment = PrivacyAssessment(
            decision=PrivacyDecision.REJECT if blocked else (
                PrivacyDecision.REDACT if findings else PrivacyDecision.ALLOW
            ),
            classification=PrivacyClassification.BLOCKED if blocked else (
                PrivacyClassification.SECRET if findings else PrivacyClassification.SAFE
            ),
            source_trust=source_trust,
            findings=findings,
            sanitized_text=combined,
            scanned_length=len(combined),
            input_hash=_hash(json.dumps(
                candidate.model_dump(mode="json", exclude={"privacy_assessment"}),
                sort_keys=True,
                separators=(",", ":"),
            )),
            untrusted_instruction_detected=untrusted_instruction,
        )
        if blocked:
            return None, assessment
        safe_candidate = candidate.model_copy(update={
            "content": content,
            "evidence": evidence,
            "source_type": source_type,
            "source_uri": source_uri,
            "tags": tags,
            "metadata": metadata,
            "privacy_assessment": assessment,
        })
        return safe_candidate, assessment

    def gate_candidates(
        self,
        candidates: list[CandidateMemory],
        *,
        source_trust: SourceTrust,
    ) -> tuple[list[CandidateMemory], list[PrivacyAssessment]]:
        safe: list[CandidateMemory] = []
        blocked: list[PrivacyAssessment] = []
        for candidate in candidates:
            gated, assessment = self.assess_candidate(candidate, source_trust=source_trust)
            if gated is None:
                blocked.append(assessment)
            else:
                safe.append(gated)
        return safe, blocked

    def _scan_text(self, text: str, location: str) -> tuple[str, list[PrivacyFinding]]:
        scan = self._scanner.scan(text)
        findings = self._safe_findings(text, scan.matches, location)
        return self._redact_from_findings(text, findings), findings

    def _sanitize_structure(
        self, value: Any, location: str
    ) -> tuple[Any, list[PrivacyFinding]]:
        if isinstance(value, str):
            return self._scan_text(value, location)
        if isinstance(value, list):
            safe_list: list[Any] = []
            findings: list[PrivacyFinding] = []
            for index, item in enumerate(value):
                safe_item, item_findings = self._sanitize_structure(
                    item, f"{location}[{index}]"
                )
                safe_list.append(safe_item)
                findings.extend(item_findings)
            return safe_list, findings
        if isinstance(value, dict):
            safe_dict: dict[str, Any] = {}
            findings = []
            for key, item in value.items():
                safe_key, key_findings = self._scan_text(str(key), f"{location}.key")
                safe_item, item_findings = self._sanitize_structure(
                    item, f"{location}.{safe_key}"
                )
                safe_dict[safe_key] = safe_item
                findings.extend(key_findings)
                findings.extend(item_findings)
            return safe_dict, findings
        return value, []

    @staticmethod
    def _policy(
        findings: list[PrivacyFinding], mode: SecretDetectionMode
    ) -> tuple[PrivacyDecision, PrivacyClassification]:
        if not findings:
            return PrivacyDecision.ALLOW, PrivacyClassification.SAFE
        if mode == SecretDetectionMode.STRICT:
            return PrivacyDecision.REJECT, PrivacyClassification.BLOCKED
        if mode == SecretDetectionMode.WARN:
            return PrivacyDecision.QUARANTINE, PrivacyClassification.BLOCKED
        return PrivacyDecision.REDACT, PrivacyClassification.SECRET

    @staticmethod
    def _safe_findings(
        text: str, matches: list[SecretMatch], location: str
    ) -> list[PrivacyFinding]:
        findings: list[PrivacyFinding] = []
        for match in matches:
            value = text[match.start:match.end]
            severity = (
                PrivacySeverity.CRITICAL
                if match.secret_type in _CRITICAL_TYPES
                else PrivacySeverity.HIGH
                if match.secret_type != SecretType.HIGH_ENTROPY
                else PrivacySeverity.MEDIUM
            )
            findings.append(PrivacyFinding(
                category=match.secret_type,
                severity=severity,
                start=match.start,
                end=match.end,
                detector="pattern-secret-scanner",
                location=location,
                confidence=match.confidence,
                safe_preview=f"[REDACTED:{match.secret_type.value}]",
                fingerprint=_hash(value),
            ))
        return findings

    @staticmethod
    def _redact_from_findings(text: str, findings: list[PrivacyFinding]) -> str:
        redacted = text
        for finding in sorted(findings, key=lambda item: item.start, reverse=True):
            redacted = (
                redacted[:finding.start]
                + finding.safe_preview
                + redacted[finding.end:]
            )
        return redacted
