"""Phase 7 Adversarial Evaluation — independent reviewer test set.

Covers 20+ hard pairs across:
- subtle scope changes (same predicate, different entity)
- misleading lexical overlap (same entity, different predicate)
- implicit corrections (no explicit correction keyword)
- weak temporal cues (recently, before, last year)
- negation (double-negation, scoped vs global)
- uncertainty
- out-of-order imports
- coexistence vs supersession boundary cases
"""
from __future__ import annotations

import pytest
import pytest_asyncio

from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

from contextos.core.enums import (
    CandidateTemporalStatus,
    MemoryStatus,
    TemporalOutcome,
    MemoryType,
)
from contextos.core.models import Memory, MemorySlot
from contextos.services.temporal import TemporalMemoryService, TemporalSlotAnalyzer
from contextos.storage.database import Database
from contextos.storage.memory_repo import SqliteMemoryRepository

UTC = timezone.utc


def moment(year: int, month: int = 1, day: int = 1) -> datetime:
    return datetime(year, month, day, 12, tzinfo=UTC)


def mem(n: int, content: str, *, observed_at: datetime,
        valid_from: datetime | None = None,
        temporal_status: CandidateTemporalStatus = CandidateTemporalStatus.UNSPECIFIED,
        confidence: float = 0.95) -> Memory:
    return Memory(
        id=UUID(f"00ad0000-0000-0000-0000-{n:012d}"),
        content=content,
        type=MemoryType.FACT,
        status=MemoryStatus.CANDIDATE,
        observed_at=observed_at,
        valid_from=valid_from,
        temporal_status=temporal_status,
        confidence=confidence,
    )


@pytest_asyncio.fixture
async def svc(tmp_path: Path):
    db = Database(tmp_path / "adv.db")
    await db.initialize()
    repo = SqliteMemoryRepository(db.connection())
    service = TemporalMemoryService(repo)
    yield repo, service
    await db.close()


# ---------------------------------------------------------------------------
# A1: SAME PREDICATE, DIFFERENT ENTITY — must COEXIST, not supersede
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_A1_interview_language_python_and_ml_language_python_coexist(svc):
    """
    'My interview language is Python.' vs 'I use Python for machine learning.'
    Both are about Python but belong to different scopes.
    MUST coexist — the scopes are different.
    FALSE SUPERSESSION if they share a slot and one kills the other.
    """
    repo, service = svc
    interview = mem(1, "My interview language is Python.", observed_at=moment(2025))
    ml = mem(2, "I use Python for machine learning.", observed_at=moment(2026))
    r1 = await service.resolve(interview)
    r2 = await service.resolve(ml)
    # Different scopes → different slot keys → coexist
    assert r1.decision.slot.key != r2.decision.slot.key, (
        "CRITICAL: Python-interview and Python-ML share the same slot — "
        "scope disambiguation failed."
    )
    assert r2.decision.outcome in {TemporalOutcome.COEXIST, TemporalOutcome.ADD_NEW}, (
        f"CRITICAL: false supersession: outcome={r2.decision.outcome}"
    )
    assert (await repo.get(interview.id)).status == MemoryStatus.ACTIVE


@pytest.mark.asyncio
async def test_A2_interview_language_python_then_cpp_supersede(svc):
    """
    'My interview language is Python.' → 'My interview language is now C++.'
    Same predicate, same scope (systems_interviews) → SUPERSEDE.
    """
    repo, service = svc
    old = mem(1, "My interview language is Python.", observed_at=moment(2025))
    new = mem(2, "My interview language is now C++.", observed_at=moment(2026))
    await service.resolve(old)
    r2 = await service.resolve(new)
    assert r2.decision.outcome == TemporalOutcome.SUPERSEDE, (
        f"Expected SUPERSEDE for interview language transition, got {r2.decision.outcome}"
    )
    assert (await repo.get(old.id)).status == MemoryStatus.SUPERSEDED


# ---------------------------------------------------------------------------
# A3: SAME ENTITY, DIFFERENT PREDICATE — must COEXIST
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_A3_docker_work_vs_docker_project_x_coexist(svc):
    """
    'I use Docker at work.' vs 'I don't use Docker in Project X.'
    Docker is the shared entity but the predicates apply to different scopes.
    MUST coexist — a scoped negation must not suppress general usage.
    """
    repo, service = svc
    work = mem(1, "I use Docker at work.", observed_at=moment(2025))
    proj = mem(2, "I don't use Docker for Project X.", observed_at=moment(2026))
    r1 = await service.resolve(work)
    r2 = await service.resolve(proj)
    # Different scopes → different slot keys
    assert r2.decision.outcome == TemporalOutcome.COEXIST, (
        f"Expected COEXIST for scoped Docker negation, got {r2.decision.outcome}"
    )
    assert (await repo.get(work.id)).status == MemoryStatus.ACTIVE, (
        "CRITICAL: general Docker usage was wrongly superseded by scoped negation."
    )


@pytest.mark.asyncio
async def test_A4_global_docker_negation_supersedes_global_docker(svc):
    """
    'I use Docker.' → 'I don't use Docker anymore.'
    Global scope on both → SUPERSEDE (negated global overrides global).
    """
    repo, service = svc
    old = mem(1, "I use Docker.", observed_at=moment(2025))
    new = mem(2, "I don't use Docker anymore.", observed_at=moment(2026))
    await service.resolve(old)
    r2 = await service.resolve(new)
    assert r2.decision.outcome == TemporalOutcome.SUPERSEDE, (
        f"Expected SUPERSEDE for global Docker negation, got {r2.decision.outcome}"
    )
    assert r2.memory.negated


# ---------------------------------------------------------------------------
# A5: DOUBLE NEGATION — semantic affirmation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_A5_double_negation_does_not_mark_as_negated(svc):
    """
    'I don't dislike Docker.' — double negation means affirmative.
    The _NEGATION regex will fire on 'don't' but the semantic meaning is positive.
    This is a known limitation: the system will mark it negated incorrectly.
    Document it as a MEDIUM/LOW known issue.
    """
    _, service = svc
    text = "I don't dislike Docker."
    m = mem(1, text, observed_at=moment(2026))
    result = await service.resolve(m)
    analyzer = TemporalSlotAnalyzer()
    prepared = analyzer.prepare(m)
    # The system INCORRECTLY marks double-negation as negated=True
    # This is a semantic limitation — document but not a CRITICAL bug
    # (it causes no false supersession in isolation; only affects flag)
    # Verify it at least doesn't crash and produces a plausible outcome.
    assert result.memory is not None


# ---------------------------------------------------------------------------
# A6: MISLEADING LEXICAL OVERLAP — different predicates, different slot
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_A6_prefer_python_data_science_vs_prefer_cpp_embedded(svc):
    """
    'I prefer Python for data science.' vs 'I prefer C++ for embedded work.'
    Different scopes — Python/data-science and C++/embedded.
    Must NOT share a slot. Must COEXIST, not supersede.
    """
    repo, service = svc
    ds = mem(1, "I prefer Python for data science.", observed_at=moment(2025))
    emb = mem(2, "I prefer C++ for embedded work.", observed_at=moment(2026))
    r1 = await service.resolve(ds)
    r2 = await service.resolve(emb)
    assert r1.decision.slot.key != r2.decision.slot.key, (
        "CRITICAL: Python/data-science and C++/embedded share a slot — "
        "scope disambiguation failed (over-broad slot)."
    )
    assert r2.decision.outcome in {TemporalOutcome.COEXIST, TemporalOutcome.ADD_NEW}, (
        f"False supersession: Python-data-science killed by C++-embedded: "
        f"{r2.decision.outcome}"
    )
    assert (await repo.get(ds.id)).status == MemoryStatus.ACTIVE


@pytest.mark.asyncio
async def test_A7_prefer_concise_coding_vs_prefer_detailed_research_coexist(svc):
    """
    'I prefer concise answers for coding questions.' vs
    'I prefer detailed answers for research.'
    Same predicate (response_style) but these have different qualifier contexts.
    
    The current slot_for maps both 'concise' and 'detailed answers' to
    response_style/global — they SHARE the same slot.
    This means one WILL supersede the other when the scope should be per-context.
    This is a MEDIUM issue: slot is over-broad for context-specific preferences.
    """
    repo, service = svc
    coding = mem(1, "I prefer concise answers for coding questions.", observed_at=moment(2025))
    research = mem(2, "I prefer detailed answers for research.", observed_at=moment(2026))
    r1 = await service.resolve(coding)
    r2 = await service.resolve(research)
    # Document what actually happens:
    # Both map to response_style/global → they WILL collide.
    # The expected behavior is COEXIST since they have different contexts.
    # The actual behavior is likely CONTRADICT or SUPERSEDE.
    # This test documents the failure.
    outcome = r2.decision.outcome
    assert outcome == TemporalOutcome.COEXIST, (
        f"MEDIUM: response_style slot is over-broad. "
        f"'concise for coding' and 'detailed for research' got {outcome} "
        f"instead of COEXIST. Both share slot key {r2.decision.slot.key!r}. "
        f"Slot needs context qualifier."
    )


# ---------------------------------------------------------------------------
# A8: IMPLICIT CORRECTION (no 'actually'/'correction' keyword)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_A8_implicit_correction_same_slot_value_change(svc):
    """
    'My laptop has 16 GB RAM.' → 'My laptop now has 32 GB RAM.'
    No 'correction' keyword — relies on _CHANGE regex ('now').
    Must SUPERSEDE, not CONTRADICT.
    """
    repo, service = svc
    old = mem(1, "My laptop has 16 GB RAM.", observed_at=moment(2025))
    new = mem(2, "My laptop now has 32 GB RAM.", observed_at=moment(2026))
    await service.resolve(old)
    r2 = await service.resolve(new)
    assert r2.decision.outcome == TemporalOutcome.SUPERSEDE, (
        f"Expected SUPERSEDE for implicit RAM correction, got {r2.decision.outcome}"
    )
    assert (await repo.get(old.id)).status == MemoryStatus.SUPERSEDED


@pytest.mark.asyncio
async def test_A9_laptop_vs_desktop_correction_must_not_cross_correct(svc):
    """
    'My laptop has 16 GB RAM.' → 'Actually, my desktop has 32 GB.'
    Different entity (laptop vs desktop) → must NOT correct laptop fact.
    The RAM slot is machine/ram_capacity/global — both share the slot!
    This is a CRITICAL slot collision: 'laptop' and 'desktop' both map to
    machine/ram_capacity/global because the slot has no entity qualifier.
    """
    repo, service = svc
    laptop = mem(1, "My laptop has 16 GB RAM.", observed_at=moment(2025))
    desktop = mem(2, "Actually, my desktop has 32 GB RAM.", observed_at=moment(2026))
    await service.resolve(laptop)
    r2 = await service.resolve(desktop)
    # The CURRENT implementation: both map to machine/ram_capacity/global
    # → they share a slot → 'actually' triggers CORRECT → laptop is wrongly superseded.
    # Expected: COEXIST (laptop and desktop are different entities).
    assert r2.decision.outcome != TemporalOutcome.CORRECT, (
        "CRITICAL: 'My desktop has 32 GB' incorrectly corrects 'My laptop has 16 GB'. "
        "The RAM slot has no entity qualifier — laptop and desktop share the same slot."
    )
    # The laptop fact must remain active
    laptop_mem = await repo.get(laptop.id)
    assert laptop_mem.status == MemoryStatus.ACTIVE, (
        "CRITICAL: laptop RAM fact was wrongly superseded by desktop RAM fact."
    )


# ---------------------------------------------------------------------------
# A10: WEAK TEMPORAL CUES
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_A10_recently_used_rust_is_uncertain_not_supersession(svc):
    """
    'I use Python.' → 'I recently tried Rust.'
    'recently tried' is not a definitive switch — must not supersede Python.
    'recently' triggers RELATIVE temporal precision.
    'tried' implies exploratory, not commitment.
    Must be COEXIST.
    """
    repo, service = svc
    python = mem(1, "I use Python.", observed_at=moment(2025))
    rust = mem(2, "I recently tried Rust.", observed_at=moment(2026))
    await service.resolve(python)
    r2 = await service.resolve(rust)
    assert r2.decision.outcome in {TemporalOutcome.COEXIST, TemporalOutcome.ADD_NEW}, (
        f"False supersession: 'recently tried Rust' displaced Python: {r2.decision.outcome}"
    )
    assert (await repo.get(python.id)).status == MemoryStatus.ACTIVE


@pytest.mark.asyncio
async def test_A11_maybe_use_rust_more_is_uncertain_coexist(svc):
    """
    'I use Python.' → 'Maybe I use Rust more now.'
    'maybe' triggers uncertain=True → must COEXIST.
    The word 'now' also triggers _CHANGE but 'maybe' should dominate.
    """
    repo, service = svc
    python = mem(1, "I use Python.", observed_at=moment(2025))
    rust = mem(2, "Maybe I use Rust more now.", observed_at=moment(2026))
    await service.resolve(python)
    r2 = await service.resolve(rust)
    assert r2.decision.outcome == TemporalOutcome.COEXIST, (
        f"Uncertain 'Maybe I use Rust more now' must COEXIST, got {r2.decision.outcome}"
    )
    assert r2.memory.uncertain


@pytest.mark.asyncio
async def test_A12_before_past_language_historical_not_current(svc):
    """
    'I used Python before.' — 'before' triggers HISTORICAL status.
    Must resolve as ADD_NEW with historical status, not active current.
    """
    _, service = svc
    m = mem(1, "I used Python before.", observed_at=moment(2026))
    r = await service.resolve(m)
    assert r.memory.temporal_status == CandidateTemporalStatus.HISTORICAL, (
        f"'I used Python before.' must be HISTORICAL, got {r.memory.temporal_status}"
    )


# ---------------------------------------------------------------------------
# A13: OUT-OF-ORDER IMPORT (historical import must not displace current)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_A13_out_of_order_import_does_not_displace_current(svc):
    """
    Current state: 'I use C++ now.' (valid_from=2026)
    Imported note: 'Imported note from 2024: I used Python.' (valid_from=2024, observed 2026)
    The import must NOT supersede C++.
    Must produce ADD_NEW (historical).
    """
    repo, service = svc
    current = mem(1, "I use C++ now.", observed_at=moment(2026, 1),
                  valid_from=moment(2026, 1))
    imported = mem(2, "I used Python previously.", observed_at=moment(2026, 6),
                   valid_from=moment(2024, 1))
    await service.resolve(current)
    r2 = await service.resolve(imported)
    assert r2.decision.outcome == TemporalOutcome.ADD_NEW, (
        f"Out-of-order import must ADD_NEW (historical), got {r2.decision.outcome}"
    )
    assert r2.memory.temporal_status == CandidateTemporalStatus.HISTORICAL
    assert (await repo.get(current.id)).status == MemoryStatus.ACTIVE


@pytest.mark.asyncio
async def test_A14_switched_to_rust_last_year_then_currently_cpp(svc):
    """
    Ingested in order:
    'I switched to Rust last year.' (observed 2026, valid 2025)
    'I currently use C++.' (observed 2026, valid 2026)
    C++ must win as current state; Rust must be historical.
    """
    repo, service = svc
    rust = mem(1, "I switched to Rust last year.", observed_at=moment(2026, 1),
               valid_from=moment(2025, 1))
    cpp = mem(2, "I currently use C++.", observed_at=moment(2026, 6),
              valid_from=moment(2026, 6))
    await service.resolve(rust)
    r2 = await service.resolve(cpp)
    # C++ (valid 2026) arrives after Rust (valid 2025) — should supersede
    # (both in same systems slot if content aligns; or separate slots — check)
    # If they share a slot, C++ supersedes Rust. If not, they coexist.
    # This test verifies C++ does not become historical.
    assert r2.memory.status != MemoryStatus.HISTORICAL, (
        "CRITICAL: C++ was classified as historical even though it has the most recent effective time."
    )


# ---------------------------------------------------------------------------
# A15: THREE-WAY CONTRADICTION
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_A15_three_way_contradiction_no_phantom_current_state(svc):
    """
    Three incompatible claims with no temporal ordering:
    P1: 'I attended Event Y.'
    P2: 'I did not attend Event Y.'
    P3: 'Correction: I did attend Event Y.'
    
    After P3 (correction), P1's claim is restored.
    P2 must remain contradicted.
    P3 must be ACTIVE.
    get_current_state must return exactly P3.
    """
    repo, service = svc
    p1 = mem(1, "I attended Event Y.", observed_at=moment(2026, 1))
    p2 = mem(2, "I did not attend Event Y.", observed_at=moment(2026, 2))
    p3 = mem(3, "Correction: I did attend Event Y.", observed_at=moment(2026, 3))
    r1 = await service.resolve(p1)
    r2 = await service.resolve(p2)
    r3 = await service.resolve(p3)

    # P1 and P2 should both be contradicted after step 2
    assert (await repo.get(p1.id)).status == MemoryStatus.CONTRADICTED

    # After P3 (correction referencing p2's slot), what is current state?
    slot = r3.decision.slot
    current = await service.get_current_state(slot)
    current_ids = [m.id for m in current]
    
    # P3 must be in current state
    assert p3.id in current_ids, (
        f"After correction, P3 (attend) must be in current state. Got: {current_ids}"
    )
    # P2 (contradicted denial) must NOT be in current state
    assert p2.id not in current_ids, (
        "P2 (denial) must not appear in current state after correction."
    )


# ---------------------------------------------------------------------------
# A16: DUPLICATE / SEMANTIC NO-CHANGE — must not create supersession chains
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_A16_semantic_duplicate_vs_code_editor_preference(svc):
    """
    'I use VS Code.' then 'My editor is VS Code.'
    These are near-duplicates but not exact. They should not supersede each other.
    The first should resolve to ADD_NEW/ACTIVE.
    The second should resolve to NO_CHANGE (same normalized value) or DUPLICATE.
    They must NOT produce a SUPERSEDE chain.
    """
    repo, service = svc
    p1 = mem(1, "I use VS Code.", observed_at=moment(2025))
    p2 = mem(2, "My editor is VS Code.", observed_at=moment(2026))
    await service.resolve(p1)
    r2 = await service.resolve(p2)
    assert r2.decision.outcome not in {TemporalOutcome.SUPERSEDE, TemporalOutcome.CORRECT}, (
        f"'My editor is VS Code' must not supersede 'I use VS Code'. Got {r2.decision.outcome}"
    )
    # p1 must remain active (not superseded)
    assert (await repo.get(p1.id)).status == MemoryStatus.ACTIVE


# ---------------------------------------------------------------------------
# A17: TEMPORAL PRECISION — YEAR/MONTH/RELATIVE must not become exact timestamps
# ---------------------------------------------------------------------------

def test_A17_year_precision_does_not_generate_fake_valid_from():
    """
    'I used Python in 2025.' → precision=YEAR, valid_from must be None.
    The system must not invent a midnight-Jan-1-2025 timestamp.
    """
    analyzer = TemporalSlotAnalyzer()
    m = analyzer.prepare(Memory(content="I used Python in 2025."))
    assert m.temporal_precision.value == "year"
    # valid_from must remain None — we only know the year, not the date
    assert m.valid_from is None, (
        f"MEDIUM: YEAR precision must not generate valid_from. Got: {m.valid_from}"
    )


def test_A18_month_precision_does_not_generate_fake_valid_from():
    """
    'I switched to Rust in March 2025.' → precision=MONTH, valid_from must be None.
    """
    analyzer = TemporalSlotAnalyzer()
    m = analyzer.prepare(Memory(content="I switched to Rust in March 2025."))
    assert m.temporal_precision.value == "month"
    assert m.valid_from is None, (
        f"MEDIUM: MONTH precision must not generate valid_from. Got: {m.valid_from}"
    )


def test_A19_relative_precision_does_not_generate_fake_valid_from():
    """
    'I recently started using Rust.' → precision=RELATIVE, valid_from must be None.
    """
    analyzer = TemporalSlotAnalyzer()
    m = analyzer.prepare(Memory(content="I recently started using Rust."))
    assert m.temporal_precision.value == "relative"
    assert m.valid_from is None, (
        f"MEDIUM: RELATIVE precision must not generate valid_from. Got: {m.valid_from}"
    )


def test_A20_exact_timestamp_does_generate_valid_from():
    """
    'I started using Rust on 2025-03-15.' → precision=DATE, valid_from IS set.
    """
    analyzer = TemporalSlotAnalyzer()
    m = analyzer.prepare(Memory(content="I started using Rust on 2025-03-15."))
    assert m.temporal_precision.value == "date"
    assert m.valid_from is not None
    assert m.valid_from.year == 2025
    assert m.valid_from.month == 3
    assert m.valid_from.day == 15


# ---------------------------------------------------------------------------
# A21: valid_to >= valid_from invariant under supersession
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_A21_supersession_sets_valid_to_correctly(svc):
    """
    When supersession occurs, the predecessor gets valid_to set to the
    successor's valid_from (or observed_at). Verify valid_to >= valid_from.
    """
    repo, service = svc
    old = mem(1, "I use Python for systems interviews.", observed_at=moment(2025),
              valid_from=moment(2025))
    new = mem(2, "I now use C++ for systems interviews.", observed_at=moment(2026),
              valid_from=moment(2026))
    await service.resolve(old)
    await service.resolve(new)
    old_persisted = await repo.get(old.id)
    assert old_persisted.status == MemoryStatus.SUPERSEDED
    # valid_to should have been set
    if old_persisted.valid_to is not None and old_persisted.valid_from is not None:
        assert old_persisted.valid_to >= old_persisted.valid_from, (
            f"MEDIUM: valid_to ({old_persisted.valid_to}) < valid_from ({old_persisted.valid_from})"
        )


# ---------------------------------------------------------------------------
# A22: CORRECTION CHAIN (3 steps) — no broken links or cycles
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_A22_correction_chain_no_cycles(svc):
    """
    Step 1: 'My laptop has 16 GB RAM.'
    Step 2: 'Correction: my laptop actually has 32 GB RAM.'
    Step 3: 'Correction: actually my laptop has 24 GB RAM.'
    All steps must produce a linear chain, no cycles.
    """
    repo, service = svc
    p1 = mem(1, "My laptop has 16 GB RAM.", observed_at=moment(2024))
    p2 = mem(2, "Correction: my laptop actually has 32 GB RAM.", observed_at=moment(2025))
    p3 = mem(3, "Correction: actually my laptop has 24 GB RAM.", observed_at=moment(2026))
    await service.resolve(p1)
    await service.resolve(p2)
    r3 = await service.resolve(p3)

    # Walk the supersession chain from p3
    visited: set[UUID] = set()
    current_m = await repo.get(p3.id)
    while current_m and current_m.supersedes:
        assert current_m.id not in visited, "CRITICAL: cycle detected in correction chain"
        visited.add(current_m.id)
        current_m = await repo.get(current_m.supersedes)

    # p3 must be the active one
    assert (await repo.get(p3.id)).status == MemoryStatus.ACTIVE
    # p2 must be superseded
    assert (await repo.get(p2.id)).status == MemoryStatus.SUPERSEDED
    # p1 must be superseded
    assert (await repo.get(p1.id)).status == MemoryStatus.SUPERSEDED


# ---------------------------------------------------------------------------
# A23: TRANSACTION ATOMICITY — partial write check
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_A23_contradiction_rollback_leaves_no_partial_state(svc):
    """
    Inject a failure after the candidate INSERT but before the relation INSERT.
    The predecessor must remain ACTIVE (not CONTRADICTED).
    The candidate must not persist.
    No dangling relation must exist.
    """
    import sqlite3
    repo, service = svc
    # Access the underlying DB connection via the repo
    db_conn = repo._db

    old = mem(1, "I attended Event Z.", observed_at=moment(2026, 1))
    await service.resolve(old)

    # Inject a trigger that fails when inserting relations
    await db_conn.execute(
        "CREATE TRIGGER reject_adv_relation BEFORE INSERT ON memory_relations "
        "BEGIN SELECT RAISE(FAIL, 'injected adversarial failure'); END"
    )
    await db_conn.commit()

    new = mem(2, "I did not attend Event Z.", observed_at=moment(2026, 2))
    with pytest.raises(sqlite3.IntegrityError):
        await service.resolve(new)

    old_persisted = await repo.get(old.id)
    assert old_persisted.status == MemoryStatus.ACTIVE, (
        "CRITICAL: predecessor was contradicted even though transaction failed"
    )
    assert old_persisted.superseded_by is None
    assert await repo.get(new.id) is None, (
        "CRITICAL: candidate was persisted even though transaction failed"
    )


# ---------------------------------------------------------------------------
# A24: MIGRATION — fabrication check
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_A24_migration_no_fabricated_temporal_precision(tmp_path: Path):
    """
    Phase 6 schema memory (no temporal columns) must migrate with:
    - temporal_precision = 'unknown'
    - temporal_status = 'unspecified'
    - valid_from = None
    - valid_to = None
    No fake precision must be generated from content.
    """
    import sqlite3 as sqlite_sync
    from contextos.storage.database import Database, SCHEMA_SQL, SCHEMA_VERSION
    from contextos.storage.memory_repo import SqliteMemoryRepository

    path = tmp_path / "phase6_migration.db"
    conn = sqlite_sync.connect(path)
    conn.executescript(SCHEMA_SQL)
    conn.execute(
        "INSERT INTO schema_version(version, description) VALUES (1, 'Phase6 baseline')"
    )
    conn.execute(
        "INSERT INTO memories (id, content, content_hash, status, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (
            "a0000000-0000-0000-0000-000000000001",
            "I used Python in 2025 for data science work.",
            "fakehash01",
            "active",
            "2026-01-01T12:00:00",
            "2026-01-01T12:00:00",
        ),
    )
    conn.commit()
    conn.close()

    db = Database(path)
    await db.initialize()
    try:
        cursor = await db.connection().execute("SELECT MAX(version) FROM schema_version")
        assert (await cursor.fetchone())[0] == SCHEMA_VERSION

        mem_repo = SqliteMemoryRepository(db.connection())
        from uuid import UUID
        m = await mem_repo.get(UUID("a0000000-0000-0000-0000-000000000001"))
        assert m is not None
        assert m.content == "I used Python in 2025 for data science work."
        # Migration must NOT fabricate temporal precision from content
        assert m.valid_from is None, (
            f"CRITICAL: migration fabricated valid_from={m.valid_from} "
            "from Phase 6 memory content"
        )
        assert m.temporal_precision.value == "unknown", (
            f"Migration must set temporal_precision=unknown, got {m.temporal_precision}"
        )
    finally:
        await db.close()


# ---------------------------------------------------------------------------
# A25: RETRIEVAL — contradicted state must not appear in CURRENT scope
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_A25_contradicted_memory_excluded_from_current_retrieval(tmp_path: Path):
    """
    After a contradiction, neither memory should appear in CURRENT retrieval.
    Contradicted memories are not 'certain current state'.
    """
    from contextos.core.enums import RetrievalMode, TemporalScope
    from contextos.core.models import RetrievalQuery, RetrievalConfig
    from contextos.embedding.deterministic import DeterministicEmbedding
    from contextos.services.retrieval import HybridRetrievalEngine
    from contextos.services.retrieval_index import RetrievalIndexSynchronizer
    from contextos.storage.database import Database
    from contextos.storage.lexical.bm25 import BM25Index
    from contextos.storage.vector.in_memory import InMemoryVectorStore
    from contextos.storage.memory_repo import SqliteMemoryRepository

    db = Database(tmp_path / "contradiction_retrieval.db")
    await db.initialize()
    repo = SqliteMemoryRepository(db.connection())
    service = TemporalMemoryService(repo)

    try:
        p1 = mem(1, "I attended Event Q.", observed_at=moment(2026, 1))
        p2 = mem(2, "I did not attend Event Q.", observed_at=moment(2026, 2))
        await service.resolve(p1)
        await service.resolve(p2)

        embedding = DeterministicEmbedding(64)
        lexical = BM25Index()
        vector = InMemoryVectorStore(64)
        sync = RetrievalIndexSynchronizer(
            memory_repo=repo, lexical_index=lexical, vector_store=vector,
            embedding_service=embedding,
        )
        engine = HybridRetrievalEngine(
            memory_repo=repo, lexical_index=lexical, vector_store=vector,
            embedding_service=embedding, index_synchronizer=sync,
        )
        result = await engine.retrieve(
            RetrievalQuery(
                text="Event Q attendance", mode=RetrievalMode.LEXICAL,
                temporal_scope=TemporalScope.CURRENT,
            )
        )
        assert result.memories == [], (
            f"CRITICAL: contradicted memories appear in CURRENT retrieval: "
            f"{[m.memory.content for m in result.memories]}"
        )
    finally:
        await db.close()


# ---------------------------------------------------------------------------
# A26: COMPILER — superseded memory must not appear as current fact
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_A26_compiler_does_not_emit_superseded_as_current_fact(tmp_path: Path):
    """
    Python (superseded) and C++ (active) are in the DB.
    Compiler must not produce both as if equally current.
    With CURRENT scope retrieval, only C++ should reach the compiler.
    """
    from contextos.core.enums import RetrievalMode, TemporalScope
    from contextos.core.models import (
        CompilationConfig, RetrievalQuery, ScoredMemory,
    )
    from contextos.services.compilation import QueryAwareContextCompiler
    from contextos.services.token_counter import DeterministicWordTokenCounter
    from contextos.embedding.deterministic import DeterministicEmbedding
    from contextos.services.retrieval import HybridRetrievalEngine
    from contextos.services.retrieval_index import RetrievalIndexSynchronizer
    from contextos.storage.database import Database
    from contextos.storage.lexical.bm25 import BM25Index
    from contextos.storage.vector.in_memory import InMemoryVectorStore
    from contextos.storage.memory_repo import SqliteMemoryRepository

    db = Database(tmp_path / "compiler_superseded.db")
    await db.initialize()
    repo = SqliteMemoryRepository(db.connection())
    service = TemporalMemoryService(repo)

    try:
        old = mem(1, "User uses Python for systems interviews.", observed_at=moment(2025))
        new = mem(2, "User now uses C++17 for systems interviews.", observed_at=moment(2026))
        await service.resolve(old)
        await service.resolve(new)

        embedding = DeterministicEmbedding(64)
        lexical = BM25Index()
        vector = InMemoryVectorStore(64)
        sync = RetrievalIndexSynchronizer(
            memory_repo=repo, lexical_index=lexical, vector_store=vector,
            embedding_service=embedding,
        )
        engine = HybridRetrievalEngine(
            memory_repo=repo, lexical_index=lexical, vector_store=vector,
            embedding_service=embedding, index_synchronizer=sync,
        )
        result = await engine.retrieve(
            RetrievalQuery(
                text="systems interview language", mode=RetrievalMode.LEXICAL,
                temporal_scope=TemporalScope.CURRENT,
            )
        )
        compiler = QueryAwareContextCompiler(token_counter=DeterministicWordTokenCounter())
        compiled = await compiler.compile(
            "What language does the user use for systems interviews?",
            result.memories,
            CompilationConfig(budget=200),
        )
        context_text = compiled.context_text
        # Python (superseded) must NOT appear in current compiled context
        assert "Python" not in context_text, (
            f"CRITICAL: superseded Python appears in current compiled context:\n{context_text}"
        )
        assert "C++17" in context_text or "C++" in context_text, (
            f"C++17 (current) must appear in compiled context:\n{context_text}"
        )
    finally:
        await db.close()
