"""Local structured credential detection and overlap-safe redaction."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from contextos.core.enums import SecretType
from contextos.core.models import ScanResult, SecretMatch


@dataclass(frozen=True, slots=True)
class SecretPattern:
    secret_type: SecretType
    pattern: re.Pattern[str]
    confidence: float
    detector: str
    secret_group: int = 0


def _pattern(
    secret_type: SecretType,
    expression: str,
    confidence: float,
    detector: str,
    *,
    flags: int = 0,
    secret_group: int = 0,
) -> SecretPattern:
    return SecretPattern(
        secret_type, re.compile(expression, flags), confidence, detector, secret_group
    )


SECRET_PATTERNS: list[SecretPattern] = [
    _pattern(SecretType.PRIVATE_KEY,
             r"-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----[\s\S]*?"
             r"-----END (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----",
             0.99, "private-key-block", flags=re.MULTILINE),
    _pattern(SecretType.AUTHORIZATION_HEADER,
             r"\bauthorization\s*:\s*(?:bearer|basic)\s+([^\s,;]+)",
             0.99, "authorization-header", flags=re.I, secret_group=1),
    _pattern(SecretType.AWS_ACCESS_KEY, r"(?<![A-Z0-9])AKIA[0-9A-Z]{16}(?![A-Z0-9])",
             0.98, "aws-access-key"),
    _pattern(SecretType.GITHUB_TOKEN, r"(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9_]{36,255}",
             0.98, "github-token"),
    _pattern(SecretType.OPENAI_API_KEY,
             r"sk-[A-Za-z0-9_-]{20,}T3BlbkFJ[A-Za-z0-9_-]{20,}",
             0.99, "openai-legacy-key"),
    _pattern(SecretType.OPENAI_API_KEY, r"sk-proj-[A-Za-z0-9_-]{40,}",
             0.98, "openai-project-key"),
    _pattern(SecretType.ANTHROPIC_API_KEY, r"sk-ant-[A-Za-z0-9_-]{40,}",
             0.98, "anthropic-key"),
    _pattern(SecretType.GOOGLE_API_KEY, r"AIza[0-9A-Za-z_-]{35}",
             0.96, "google-api-key"),
    _pattern(SecretType.SLACK_TOKEN,
             r"xox[baprs]-[0-9a-zA-Z]{10,}(?:-[0-9a-zA-Z]{10,})*",
             0.96, "slack-token"),
    _pattern(SecretType.STRIPE_KEY, r"(?:sk|pk)_(?:test|live)_[0-9a-zA-Z]{24,}",
             0.98, "stripe-key"),
    _pattern(SecretType.JWT,
             r"eyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}",
             0.96, "jwt-token"),
    _pattern(SecretType.CONNECTION_STRING,
             r"(?:mongodb|postgres|postgresql|mysql|redis|amqp)(?:\+[a-z]+)?://"
             r"[^\s:/]+:[^\s@]+@[^\s]+",
             0.96, "credential-connection-string", flags=re.I),
    _pattern(SecretType.CONNECTION_STRING,
             r"(?:https?|ftp)://[^\s:/]+:[^\s@]+@[^\s/]+(?:/[^\s]*)?",
             0.94, "credential-url-userinfo", flags=re.I),
    _pattern(SecretType.BEARER_TOKEN, r"\bbearer\s+([A-Za-z0-9._~+/=-]{12,})",
             0.94, "bearer-token", flags=re.I, secret_group=1),
    _pattern(SecretType.GENERIC_API_KEY,
             r"\b(?:api[_ -]?key|apikey|api[_ -]?secret)\b\s*(?:is|=|:)\s*"
             r"['\"]?([A-Za-z0-9_./+=-]{11,}[A-Za-z0-9_+=-])['\"]?",
             0.92, "credential-assignment", flags=re.I, secret_group=1),
    _pattern(SecretType.ACCESS_TOKEN,
             r"\b(?:access[_ -]?token|refresh[_ -]?token)\b\s*(?:is|=|:)\s*"
             r"['\"]?([A-Za-z0-9_./+=-]{12,})['\"]?",
             0.93, "token-assignment", flags=re.I, secret_group=1),
    _pattern(SecretType.SESSION_COOKIE,
             r"\b(?:session(?:[_ -]?(?:id|token))?|cookie)\b\s*(?:is|=|:)\s*"
             r"['\"]?([A-Za-z0-9_./+=-]{12,})['\"]?",
             0.91, "session-cookie", flags=re.I, secret_group=1),
    _pattern(SecretType.PASSWORD,
             r"\b(?:password|passwd|pwd|pass)\b\s*(?:is|=|:)\s*"
             r"['\"]?([^\s'\"]{6,})['\"]?",
             0.91, "password-assignment", flags=re.I, secret_group=1),
    _pattern(SecretType.OTP,
             r"\b(?:otp|one[ -]?time (?:password|code)|verification code|pin)\b"
             r"\s*(?:is|=|:)\s*([0-9]{4,8})\b",
             0.95, "contextual-otp", flags=re.I, secret_group=1),
    _pattern(SecretType.AWS_SECRET_KEY,
             r"\baws[_ -]?secret(?:[_ -]?access)?[_ -]?key\b\s*(?:is|=|:)\s*"
             r"['\"]?([A-Za-z0-9/+=]{40})['\"]?",
             0.96, "aws-secret-key", flags=re.I, secret_group=1),
]


def _shannon_entropy(value: str) -> float:
    if not value:
        return 0.0
    frequencies: dict[str, int] = {}
    for character in value:
        frequencies[character] = frequencies.get(character, 0) + 1
    length = len(value)
    return -sum(
        (count / length) * math.log2(count / length) for count in frequencies.values()
    )


def _find_high_entropy_strings(
    text: str,
    min_length: int = 20,
    max_length: int = 200,
    entropy_threshold: float = 4.5,
) -> list[SecretMatch]:
    """Optional advisory detector; disabled by default to control false positives."""
    matches: list[SecretMatch] = []
    token_pattern = re.compile(
        r"[A-Za-z0-9+/=_-]{" + str(min_length) + "," + str(max_length) + "}"
    )
    for match in token_pattern.finditer(text):
        entropy = _shannon_entropy(match.group())
        if entropy >= entropy_threshold:
            matches.append(SecretMatch(
                secret_type=SecretType.HIGH_ENTROPY,
                start=match.start(),
                end=match.end(),
                matched_text="[REDACTED:high_entropy]",
                confidence=min(0.89, (entropy - entropy_threshold) / 2.0 + 0.5),
            ))
    return matches


class PatternSecretScanner:
    """Replaceable local scanner implementing the existing SecretScanner protocol."""

    def __init__(
        self,
        *,
        min_confidence: float = 0.5,
        enable_entropy: bool = False,
        entropy_threshold: float = 4.5,
    ) -> None:
        self._min_confidence = min_confidence
        self._enable_entropy = enable_entropy
        self._entropy_threshold = entropy_threshold

    def scan(self, text: str) -> ScanResult:
        if not text:
            return ScanResult(scanned_length=0)
        matches: list[SecretMatch] = []
        for secret_pattern in SECRET_PATTERNS:
            if secret_pattern.confidence < self._min_confidence:
                continue
            for match in secret_pattern.pattern.finditer(text):
                start, end = match.span(secret_pattern.secret_group)
                matches.append(SecretMatch(
                    secret_type=secret_pattern.secret_type,
                    start=start,
                    end=end,
                    matched_text=f"[REDACTED:{secret_pattern.secret_type.value}]",
                    confidence=secret_pattern.confidence,
                ))
        if self._enable_entropy:
            matches.extend(_find_high_entropy_strings(
                text, entropy_threshold=self._entropy_threshold
            ))
        matches = self._deduplicate_overlapping(matches)
        return ScanResult(
            has_secrets=bool(matches), matches=matches, scanned_length=len(text)
        )

    def redact(self, text: str) -> tuple[str, ScanResult]:
        result = self.scan(text)
        redacted = text
        for match in sorted(result.matches, key=lambda item: item.start, reverse=True):
            redacted = (
                redacted[:match.start]
                + f"[REDACTED:{match.secret_type.value}]"
                + redacted[match.end:]
            )
        return redacted, result

    @staticmethod
    def _deduplicate_overlapping(matches: list[SecretMatch]) -> list[SecretMatch]:
        selected: list[SecretMatch] = []
        for current in sorted(matches, key=lambda item: (item.start, -item.confidence, -item.end)):
            if selected and current.start < selected[-1].end:
                previous = selected[-1]
                winner = current if current.confidence > previous.confidence else previous
                selected[-1] = SecretMatch(
                    secret_type=winner.secret_type,
                    start=min(previous.start, current.start),
                    end=max(previous.end, current.end),
                    matched_text=f"[REDACTED:{winner.secret_type.value}]",
                    confidence=max(previous.confidence, current.confidence),
                )
                continue
            selected.append(current)
        return selected
