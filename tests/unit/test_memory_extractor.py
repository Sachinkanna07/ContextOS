"""Tests for the memory extractor."""

from __future__ import annotations

import pytest

from contextos.core.enums import MemoryType
from contextos.services.extraction import RuleBasedMemoryExtractor


@pytest.fixture
def extractor():
    return RuleBasedMemoryExtractor()


class TestBasicExtraction:
    """Test that the extractor produces memories from text."""

    @pytest.mark.asyncio
    async def test_single_sentence(self, extractor):
        memories = await extractor.extract("I prefer Python over JavaScript.")
        assert len(memories) >= 1
        assert any("Python" in m.content for m in memories)

    @pytest.mark.asyncio
    async def test_empty_input(self, extractor):
        assert await extractor.extract("") == []
        assert await extractor.extract("   ") == []

    @pytest.mark.asyncio
    async def test_greeting_skipped(self, extractor):
        memories = await extractor.extract("Hello!")
        assert len(memories) == 0

    @pytest.mark.asyncio
    async def test_ok_skipped(self, extractor):
        memories = await extractor.extract("ok")
        assert len(memories) == 0

    @pytest.mark.asyncio
    async def test_multi_sentence_input(self, extractor):
        text = (
            "I prefer Python 3.12+ and always use type hints. "
            "I work at Acme Corp as a senior engineer. "
            "I'm building a CLI tool called ContextOS."
        )
        memories = await extractor.extract(text)
        assert len(memories) >= 2  # Should extract multiple memories

    @pytest.mark.asyncio
    async def test_paragraph_input(self, extractor):
        text = """
I'm a senior software engineer at Acme Corp. I've been working with
Python for about 8 years now, and I always use type hints.

My current project is a CLI tool called ContextOS. I prefer using
pytest for testing and ruff for linting.
"""
        memories = await extractor.extract(text)
        assert len(memories) >= 2


class TestTypeClassification:
    """Test that memories are classified into correct types."""

    @pytest.mark.asyncio
    async def test_preference_classification(self, extractor):
        memories = await extractor.extract("I prefer dark mode for all my editors.")
        assert len(memories) >= 1
        assert memories[0].type == MemoryType.PREFERENCE

    @pytest.mark.asyncio
    async def test_skill_classification(self, extractor):
        memories = await extractor.extract("I'm proficient in Rust and Go.")
        assert len(memories) >= 1
        assert memories[0].type == MemoryType.SKILL

    @pytest.mark.asyncio
    async def test_fact_classification(self, extractor):
        memories = await extractor.extract("I work at Google as a staff engineer.")
        assert len(memories) >= 1
        assert memories[0].type == MemoryType.FACT

    @pytest.mark.asyncio
    async def test_project_classification(self, extractor):
        memories = await extractor.extract("I'm building a distributed database in Rust.")
        assert len(memories) >= 1
        assert memories[0].type == MemoryType.PROJECT

    @pytest.mark.asyncio
    async def test_opinion_classification(self, extractor):
        memories = await extractor.extract("I think microservices are overused for small teams.")
        assert len(memories) >= 1
        assert memories[0].type == MemoryType.OPINION

    @pytest.mark.asyncio
    async def test_goal_classification(self, extractor):
        memories = await extractor.extract("I want to learn Kubernetes this quarter.")
        assert len(memories) >= 1
        assert memories[0].type == MemoryType.GOAL

    @pytest.mark.asyncio
    async def test_suggested_type_override(self, extractor):
        memories = await extractor.extract(
            "Python is great.", suggested_type=MemoryType.OPINION
        )
        assert len(memories) >= 1
        assert memories[0].type == MemoryType.OPINION


class TestConfidenceScoring:
    """Test that confidence scores reflect linguistic signals."""

    @pytest.mark.asyncio
    async def test_definite_statement_higher_confidence(self, extractor):
        definite = await extractor.extract("I always use vim for editing.")
        uncertain = await extractor.extract("I sometimes use vim for editing.")
        assert len(definite) >= 1
        assert len(uncertain) >= 1
        assert definite[0].confidence > uncertain[0].confidence

    @pytest.mark.asyncio
    async def test_uncertain_language_lower_confidence(self, extractor):
        memories = await extractor.extract("Maybe I should try using Rust sometime.")
        assert len(memories) >= 1
        assert memories[0].confidence < 0.7


class TestTagPropagation:
    """Test that tags are passed through to extracted memories."""

    @pytest.mark.asyncio
    async def test_tags_propagated(self, extractor):
        memories = await extractor.extract(
            "I prefer Python.",
            tags=["coding", "preferences"],
        )
        assert len(memories) >= 1
        assert "coding" in memories[0].tags
        assert "preferences" in memories[0].tags
