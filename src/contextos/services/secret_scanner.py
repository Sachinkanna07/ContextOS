"""Pattern-based secret scanner for ContextOS.

Detects credentials, API keys, private keys, passwords, and high-entropy
strings in text. This is the Phase 1 implementation — pattern-based only.

Design decisions:
- Patterns are ordered from most specific to most general.
- Each pattern has a confidence score — API key patterns are high confidence,
  high-entropy detection is lower confidence.
- The scanner never modifies the input. Redaction is a separate method.
- False positives are tracked and tunable via confidence thresholds.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from contextos.core.enums import SecretType
from contextos.core.models import ScanResult, SecretMatch


@dataclass(frozen=True, slots=True)
class SecretPattern:
    """A regex pattern for detecting a specific type of secret."""

    secret_type: SecretType
    pattern: re.Pattern[str]
    confidence: float
    description: str


# ---------------------------------------------------------------------------
# Pattern Definitions
# ---------------------------------------------------------------------------

# Order: most specific first, most general last.
SECRET_PATTERNS: list[SecretPattern] = [
    # AWS
    SecretPattern(
        secret_type=SecretType.AWS_ACCESS_KEY,
        pattern=re.compile(r"(?<![A-Z0-9])AKIA[0-9A-Z]{16}(?![A-Z0-9])"),
        confidence=0.98,
        description="AWS Access Key ID",
    ),
    SecretPattern(
        secret_type=SecretType.AWS_SECRET_KEY,
        pattern=re.compile(r"(?<![A-Za-z0-9/+=])[A-Za-z0-9/+=]{40}(?![A-Za-z0-9/+=])"),
        confidence=0.6,  # Lower confidence — 40-char base64 is common
        description="Potential AWS Secret Access Key",
    ),
    # GitHub
    SecretPattern(
        secret_type=SecretType.GITHUB_TOKEN,
        pattern=re.compile(r"(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9_]{36,255}"),
        confidence=0.97,
        description="GitHub Personal Access Token",
    ),
    # OpenAI
    SecretPattern(
        secret_type=SecretType.OPENAI_API_KEY,
        pattern=re.compile(r"sk-[A-Za-z0-9_-]{20,}T3BlbkFJ[A-Za-z0-9_-]{20,}"),
        confidence=0.99,
        description="OpenAI API Key (legacy format)",
    ),
    SecretPattern(
        secret_type=SecretType.OPENAI_API_KEY,
        pattern=re.compile(r"sk-proj-[A-Za-z0-9_-]{40,}"),
        confidence=0.97,
        description="OpenAI API Key (project format)",
    ),
    # Anthropic
    SecretPattern(
        secret_type=SecretType.ANTHROPIC_API_KEY,
        pattern=re.compile(r"sk-ant-[A-Za-z0-9_-]{40,}"),
        confidence=0.98,
        description="Anthropic API Key",
    ),
    # Google
    SecretPattern(
        secret_type=SecretType.GOOGLE_API_KEY,
        pattern=re.compile(r"AIza[0-9A-Za-z_-]{35}"),
        confidence=0.95,
        description="Google API Key",
    ),
    # Slack
    SecretPattern(
        secret_type=SecretType.SLACK_TOKEN,
        pattern=re.compile(r"xox[baprs]-[0-9a-zA-Z]{10,}(?:-[0-9a-zA-Z]{10,})*"),
        confidence=0.95,
        description="Slack Token",
    ),
    # Stripe
    SecretPattern(
        secret_type=SecretType.STRIPE_KEY,
        pattern=re.compile(r"(?:sk|pk)_(?:test|live)_[0-9a-zA-Z]{24,}"),
        confidence=0.97,
        description="Stripe API Key",
    ),
    # Private keys (PEM format)
    SecretPattern(
        secret_type=SecretType.PRIVATE_KEY,
        pattern=re.compile(
            r"-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----"
            r"[\s\S]*?"
            r"-----END (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----",
            re.MULTILINE,
        ),
        confidence=0.99,
        description="PEM Private Key",
    ),
    # SSH private key (non-PEM indicators)
    SecretPattern(
        secret_type=SecretType.SSH_PRIVATE_KEY,
        pattern=re.compile(r"-----BEGIN OPENSSH PRIVATE KEY-----"),
        confidence=0.99,
        description="OpenSSH Private Key",
    ),
    # JWT (3 base64url segments separated by dots)
    SecretPattern(
        secret_type=SecretType.JWT,
        pattern=re.compile(
            r"eyJ[A-Za-z0-9_-]{10,}\."
            r"eyJ[A-Za-z0-9_-]{10,}\."
            r"[A-Za-z0-9_-]{10,}"
        ),
        confidence=0.90,
        description="JSON Web Token",
    ),
    # Connection strings with passwords
    SecretPattern(
        secret_type=SecretType.CONNECTION_STRING,
        pattern=re.compile(
            r"(?:mongodb|postgres|postgresql|mysql|redis|amqp)"
            r"(?:\+[a-z]+)?://"
            r"[^:]+:[^@]+@",
            re.IGNORECASE,
        ),
        confidence=0.92,
        description="Connection string with embedded credentials",
    ),
    # Generic API key assignments
    SecretPattern(
        secret_type=SecretType.GENERIC_API_KEY,
        pattern=re.compile(
            r"""(?:api[_-]?key|apikey|api[_-]?secret|api[_-]?token)"""
            r"""[\s]*[=:]\s*['\"]?([A-Za-z0-9_\-./+=]{16,})['\"]?""",
            re.IGNORECASE,
        ),
        confidence=0.80,
        description="Generic API key assignment",
    ),
    # Password assignments
    SecretPattern(
        secret_type=SecretType.PASSWORD,
        pattern=re.compile(
            r"""(?:password|passwd|pwd|pass)[\s]*[=:]\s*['\"]?(\S{6,})['\"]?""",
            re.IGNORECASE,
        ),
        confidence=0.75,
        description="Password assignment",
    ),
]


# ---------------------------------------------------------------------------
# Entropy Analysis
# ---------------------------------------------------------------------------


def _shannon_entropy(s: str) -> float:
    """Calculate Shannon entropy of a string in bits per character."""
    if not s:
        return 0.0
    freq: dict[str, int] = {}
    for c in s:
        freq[c] = freq.get(c, 0) + 1
    length = len(s)
    return -sum(
        (count / length) * math.log2(count / length) for count in freq.values()
    )


def _find_high_entropy_strings(
    text: str,
    min_length: int = 20,
    max_length: int = 200,
    entropy_threshold: float = 4.5,
) -> list[SecretMatch]:
    """Find high-entropy substrings that might be secrets.

    Splits text on whitespace and common delimiters, then checks
    each token for high entropy. This catches secrets that don't match
    any specific pattern.
    """
    matches: list[SecretMatch] = []
    # Split on whitespace and common separators, keeping track of positions
    token_pattern = re.compile(r"[A-Za-z0-9+/=_\-]{" + str(min_length) + "," + str(max_length) + "}")

    for m in token_pattern.finditer(text):
        token = m.group()
        entropy = _shannon_entropy(token)
        if entropy >= entropy_threshold:
            matches.append(
                SecretMatch(
                    secret_type=SecretType.HIGH_ENTROPY,
                    start=m.start(),
                    end=m.end(),
                    matched_text=token[:8] + "..." + token[-4:],  # Truncated for safety
                    confidence=min(0.9, (entropy - entropy_threshold) / 2.0 + 0.5),
                )
            )

    return matches


# ---------------------------------------------------------------------------
# Scanner Implementation
# ---------------------------------------------------------------------------


class PatternSecretScanner:
    """Phase 1 secret scanner using regex patterns and entropy analysis.

    Implements the SecretScanner protocol.
    """

    def __init__(
        self,
        *,
        min_confidence: float = 0.5,
        enable_entropy: bool = True,
        entropy_threshold: float = 4.5,
    ) -> None:
        self._min_confidence = min_confidence
        self._enable_entropy = enable_entropy
        self._entropy_threshold = entropy_threshold

    def scan(self, text: str) -> ScanResult:
        """Scan text for secrets. Returns ScanResult with all matches."""
        if not text:
            return ScanResult(scanned_length=0)

        matches: list[SecretMatch] = []

        # Run pattern-based detection
        for sp in SECRET_PATTERNS:
            if sp.confidence < self._min_confidence:
                continue

            for m in sp.pattern.finditer(text):
                matched_text = m.group()
                # Truncate matched text for safety in logs/results
                safe_text = matched_text[:8] + "..." if len(matched_text) > 12 else "***"

                matches.append(
                    SecretMatch(
                        secret_type=sp.secret_type,
                        start=m.start(),
                        end=m.end(),
                        matched_text=safe_text,
                        confidence=sp.confidence,
                    )
                )

        # Run entropy-based detection
        if self._enable_entropy:
            entropy_matches = _find_high_entropy_strings(
                text, entropy_threshold=self._entropy_threshold
            )
            matches.extend(entropy_matches)

        # Deduplicate overlapping matches (keep highest confidence)
        matches = self._deduplicate_overlapping(matches)

        return ScanResult(
            has_secrets=len(matches) > 0,
            matches=matches,
            scanned_length=len(text),
        )

    def redact(self, text: str) -> tuple[str, ScanResult]:
        """Scan and redact secrets from text.

        Returns (redacted_text, scan_result).
        Secrets are replaced with [REDACTED:<type>] placeholders.
        """
        result = self.scan(text)
        if not result.has_secrets:
            return text, result

        # Sort matches by start position, descending, to replace from end
        sorted_matches = sorted(result.matches, key=lambda m: m.start, reverse=True)

        redacted = text
        for match in sorted_matches:
            placeholder = f"[REDACTED:{match.secret_type.value}]"
            redacted = redacted[: match.start] + placeholder + redacted[match.end :]

        return redacted, result

    @staticmethod
    def _deduplicate_overlapping(matches: list[SecretMatch]) -> list[SecretMatch]:
        """Remove overlapping matches, keeping the one with highest confidence."""
        if len(matches) <= 1:
            return matches

        # Sort by start position
        sorted_matches = sorted(matches, key=lambda m: (m.start, -m.confidence))
        result: list[SecretMatch] = [sorted_matches[0]]

        for current in sorted_matches[1:]:
            prev = result[-1]
            if current.start < prev.end:
                # Overlapping — keep the one with higher confidence
                if current.confidence > prev.confidence:
                    result[-1] = current
            else:
                result.append(current)

        return result
