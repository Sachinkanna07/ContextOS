"""Tests for the secret scanner.

These are among the most critical tests in the system. A false negative
means credentials leak to an LLM. We test with real-world secret formats.
"""

from __future__ import annotations

import pytest

from contextos.core.enums import SecretType
from contextos.services.secret_scanner import PatternSecretScanner, _shannon_entropy


class TestShannonEntropy:
    """Tests for entropy calculation."""

    def test_empty_string(self):
        assert _shannon_entropy("") == 0.0

    def test_single_char(self):
        assert _shannon_entropy("aaaaaaa") == 0.0

    def test_high_entropy(self):
        # Random-looking string should have high entropy
        entropy = _shannon_entropy("aB3$xZ9!qW7@mK2#")
        assert entropy > 3.5

    def test_low_entropy(self):
        # Repetitive string should have low entropy
        entropy = _shannon_entropy("abcabcabcabc")
        assert entropy < 2.0


class TestPatternDetection:
    """Test that known secret patterns are detected."""

    @pytest.fixture
    def scanner(self):
        return PatternSecretScanner(min_confidence=0.5, enable_entropy=False)

    def test_aws_access_key(self, scanner):
        text = "My AWS key is AKIAIOSFODNN7EXAMPLE"
        result = scanner.scan(text)
        assert result.has_secrets
        assert SecretType.AWS_ACCESS_KEY in result.secret_types_found

    def test_github_token(self, scanner):
        text = "Token: ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklm"
        result = scanner.scan(text)
        assert result.has_secrets
        assert SecretType.GITHUB_TOKEN in result.secret_types_found

    def test_openai_key_project(self, scanner):
        text = "OPENAI_API_KEY=sk-proj-ABCDEFghijklMNOPqrstUVWXyz1234567890abcdefghij"
        result = scanner.scan(text)
        assert result.has_secrets
        assert SecretType.OPENAI_API_KEY in result.secret_types_found

    def test_anthropic_key(self, scanner):
        text = "key: sk-ant-ABCDEFghijklMNOPqrstUVWXyz1234567890abcdefghij"
        result = scanner.scan(text)
        assert result.has_secrets
        assert SecretType.ANTHROPIC_API_KEY in result.secret_types_found

    def test_private_key_pem(self, scanner):
        text = """-----BEGIN RSA PRIVATE KEY-----
MIIEpAIBAAKCAQEA0Z3VS5JJcds3xfn/ygWyF0PQnGcCKO+POiYY
-----END RSA PRIVATE KEY-----"""
        result = scanner.scan(text)
        assert result.has_secrets
        assert SecretType.PRIVATE_KEY in result.secret_types_found

    def test_connection_string(self, scanner):
        text = "DATABASE_URL=postgresql://admin:secretpass123@db.example.com:5432/mydb"
        result = scanner.scan(text)
        assert result.has_secrets
        assert SecretType.CONNECTION_STRING in result.secret_types_found

    def test_password_assignment(self, scanner):
        text = 'password = "my_super_secret_password_123"'
        result = scanner.scan(text)
        assert result.has_secrets
        assert SecretType.PASSWORD in result.secret_types_found

    def test_jwt(self, scanner):
        text = "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
        result = scanner.scan(text)
        assert result.has_secrets
        assert SecretType.JWT in result.secret_types_found

    def test_google_api_key(self, scanner):
        text = "GOOGLE_API_KEY=AIzaSyDaGmWKa4JsXZ-HjGw7ISLn_3namBGewQe"
        result = scanner.scan(text)
        assert result.has_secrets
        assert SecretType.GOOGLE_API_KEY in result.secret_types_found

    def test_stripe_key(self, scanner):
        text = "stripe_key: " + "sk_test_" + "contextosfixture000000000000000000"
        result = scanner.scan(text)
        assert result.has_secrets
        assert SecretType.STRIPE_KEY in result.secret_types_found

    def test_slack_token(self, scanner):
        text = (
            "SLACK_TOKEN=" + "xoxb-" + "0000000000" + "-0000000000-"
            + "ContextOSFixtureToken"
        )
        result = scanner.scan(text)
        assert result.has_secrets
        assert SecretType.SLACK_TOKEN in result.secret_types_found

    def test_generic_api_key(self, scanner):
        text = 'api_key = "sk_1234567890abcdefghij"'
        result = scanner.scan(text)
        assert result.has_secrets
        assert SecretType.GENERIC_API_KEY in result.secret_types_found


class TestNoFalsePositives:
    """Test that normal text is not flagged."""

    @pytest.fixture
    def scanner(self):
        return PatternSecretScanner(min_confidence=0.7, enable_entropy=False)

    def test_normal_text(self, scanner):
        text = "I prefer Python 3.12 and always use type hints."
        result = scanner.scan(text)
        assert not result.has_secrets

    def test_code_snippet(self, scanner):
        text = "def hello():\n    return 'world'"
        result = scanner.scan(text)
        assert not result.has_secrets

    def test_url_without_credentials(self, scanner):
        text = "Visit https://example.com/docs for more info."
        result = scanner.scan(text)
        assert not result.has_secrets

    def test_technical_discussion(self, scanner):
        text = "The API returns a JSON response with status codes 200 and 404."
        result = scanner.scan(text)
        assert not result.has_secrets


class TestRedaction:
    """Test secret redaction."""

    @pytest.fixture
    def scanner(self):
        return PatternSecretScanner(min_confidence=0.5, enable_entropy=False)

    def test_redact_replaces_secret(self, scanner):
        text = "My key is AKIAIOSFODNN7EXAMPLE and that's it."
        redacted, result = scanner.redact(text)
        assert "AKIAIOSFODNN7EXAMPLE" not in redacted
        assert "[REDACTED:" in redacted
        assert result.has_secrets

    def test_redact_preserves_surrounding_text(self, scanner):
        text = "Hello AKIAIOSFODNN7EXAMPLE world"
        redacted, result = scanner.redact(text)
        assert redacted.startswith("Hello ")
        assert redacted.endswith(" world")

    def test_redact_no_secrets(self, scanner):
        text = "Just normal text here."
        redacted, result = scanner.redact(text)
        assert redacted == text
        assert not result.has_secrets


class TestHighEntropy:
    """Test entropy-based detection."""

    @pytest.fixture
    def scanner(self):
        return PatternSecretScanner(min_confidence=0.3, enable_entropy=True, entropy_threshold=4.5)

    def test_high_entropy_string_detected(self, scanner):
        # This is a random-looking string that should trigger entropy detection
        text = "token: aB3xZ9qW7mK2pL5nR8vD4jF6hG1cE0"
        result = scanner.scan(text)
        # Should detect via entropy or generic pattern
        assert result.has_secrets or len(result.matches) > 0

    def test_normal_words_not_flagged(self, scanner):
        text = "The quick brown fox jumps over the lazy dog."
        result = scanner.scan(text)
        # No high-entropy matches for normal English
        entropy_matches = [
            m for m in result.matches if m.secret_type == SecretType.HIGH_ENTROPY
        ]
        assert len(entropy_matches) == 0


class TestMultipleSecrets:
    """Test detection of multiple secrets in one input."""

    def test_multiple_different_secrets(self):
        scanner = PatternSecretScanner(min_confidence=0.5, enable_entropy=False)
        text = (
            "AWS: AKIAIOSFODNN7EXAMPLE\n"
            "GitHub: ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklm\n"
        )
        result = scanner.scan(text)
        assert result.has_secrets
        found_types = set(result.secret_types_found)
        assert SecretType.AWS_ACCESS_KEY in found_types
        assert SecretType.GITHUB_TOKEN in found_types

    def test_scan_result_count(self):
        scanner = PatternSecretScanner(min_confidence=0.5, enable_entropy=False)
        text = "key1: AKIAIOSFODNN7EXAMPLE key2: AKIAIOSFODNN7EXAMPLF"
        result = scanner.scan(text)
        assert len(result.matches) >= 2
