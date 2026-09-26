"""Tests for memory lifecycle state transitions."""

from __future__ import annotations

import pytest

from contextos.core.enums import VALID_TRANSITIONS, MemoryStatus


class TestValidTransitions:
    """Test the state transition table is complete and correct."""

    def test_all_states_have_transition_entry(self):
        """Every MemoryStatus must have an entry in VALID_TRANSITIONS."""
        for status in MemoryStatus:
            assert status in VALID_TRANSITIONS, f"Missing transition entry for {status}"

    def test_candidate_can_become_active(self):
        assert MemoryStatus.ACTIVE in VALID_TRANSITIONS[MemoryStatus.CANDIDATE]

    def test_candidate_can_become_merged(self):
        assert MemoryStatus.MERGED in VALID_TRANSITIONS[MemoryStatus.CANDIDATE]

    def test_candidate_can_become_contradicted(self):
        assert MemoryStatus.CONTRADICTED in VALID_TRANSITIONS[MemoryStatus.CANDIDATE]

    def test_candidate_cannot_become_deleted_directly(self):
        assert MemoryStatus.DELETED not in VALID_TRANSITIONS[MemoryStatus.CANDIDATE]

    def test_active_can_become_superseded(self):
        assert MemoryStatus.SUPERSEDED in VALID_TRANSITIONS[MemoryStatus.ACTIVE]

    def test_active_can_become_contradicted(self):
        assert MemoryStatus.CONTRADICTED in VALID_TRANSITIONS[MemoryStatus.ACTIVE]

    def test_active_can_become_expired(self):
        assert MemoryStatus.EXPIRED in VALID_TRANSITIONS[MemoryStatus.ACTIVE]

    def test_active_can_become_deleted(self):
        assert MemoryStatus.DELETED in VALID_TRANSITIONS[MemoryStatus.ACTIVE]

    def test_active_can_self_transition(self):
        """ACTIVE → ACTIVE for reinforcement."""
        assert MemoryStatus.ACTIVE in VALID_TRANSITIONS[MemoryStatus.ACTIVE]

    def test_contradicted_can_resolve_to_active(self):
        assert MemoryStatus.ACTIVE in VALID_TRANSITIONS[MemoryStatus.CONTRADICTED]

    def test_contradicted_can_resolve_to_superseded(self):
        assert MemoryStatus.SUPERSEDED in VALID_TRANSITIONS[MemoryStatus.CONTRADICTED]

    def test_contradicted_can_be_deleted(self):
        assert MemoryStatus.DELETED in VALID_TRANSITIONS[MemoryStatus.CONTRADICTED]

    def test_superseded_can_become_historical(self):
        assert MemoryStatus.HISTORICAL in VALID_TRANSITIONS[MemoryStatus.SUPERSEDED]

    def test_expired_can_become_historical(self):
        assert MemoryStatus.HISTORICAL in VALID_TRANSITIONS[MemoryStatus.EXPIRED]

    def test_historical_can_be_deleted(self):
        assert MemoryStatus.DELETED in VALID_TRANSITIONS[MemoryStatus.HISTORICAL]

    def test_deleted_can_be_purged(self):
        assert MemoryStatus.PURGED in VALID_TRANSITIONS[MemoryStatus.DELETED]

    def test_purged_is_terminal(self):
        assert len(VALID_TRANSITIONS[MemoryStatus.PURGED]) == 0

    def test_merged_is_terminal(self):
        assert len(VALID_TRANSITIONS[MemoryStatus.MERGED]) == 0

    def test_no_transition_to_candidate(self):
        """No state should transition TO candidate — it's the entry state only."""
        for status, targets in VALID_TRANSITIONS.items():
            assert MemoryStatus.CANDIDATE not in targets, (
                f"{status} can transition to CANDIDATE, which should be impossible"
            )


class TestMemoryModel:
    """Test the Memory model itself."""

    def test_content_hash_computed(self):
        from contextos.core.models import Memory
        m = Memory(content="I prefer Python.")
        assert m.content_hash != ""
        assert len(m.content_hash) == 16

    def test_same_content_same_hash(self):
        from contextos.core.models import Memory
        m1 = Memory(content="I prefer Python.")
        m2 = Memory(content="I prefer Python.")
        assert m1.content_hash == m2.content_hash

    def test_normalized_hash(self):
        """Whitespace differences should not affect hash."""
        from contextos.core.models import Memory
        m1 = Memory(content="I prefer Python.")
        m2 = Memory(content="  I   prefer   Python.  ")
        assert m1.content_hash == m2.content_hash

    def test_different_content_different_hash(self):
        from contextos.core.models import Memory
        m1 = Memory(content="I prefer Python.")
        m2 = Memory(content="I prefer JavaScript.")
        assert m1.content_hash != m2.content_hash

    def test_is_retrievable(self):
        from contextos.core.models import Memory
        active = Memory(content="test", status=MemoryStatus.ACTIVE)
        deleted = Memory(content="test", status=MemoryStatus.DELETED)
        assert active.is_retrievable
        assert not deleted.is_retrievable

    def test_is_compilable(self):
        from contextos.core.models import Memory
        active = Memory(content="test", status=MemoryStatus.ACTIVE)
        superseded = Memory(content="test", status=MemoryStatus.SUPERSEDED)
        assert active.is_compilable
        assert not superseded.is_compilable

    def test_confidence_bounds(self):
        from contextos.core.models import Memory
        with pytest.raises(Exception):
            Memory(content="test", confidence=1.5)
        with pytest.raises(Exception):
            Memory(content="test", confidence=-0.1)

    def test_importance_bounds(self):
        from contextos.core.models import Memory
        with pytest.raises(Exception):
            Memory(content="test", importance=2.0)
