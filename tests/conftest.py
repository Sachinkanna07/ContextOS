"""Shared test fixtures for ContextOS."""

from __future__ import annotations

import asyncio
from pathlib import Path
from uuid import uuid4

import pytest
import pytest_asyncio

from contextos.core.enums import MemoryStatus, MemoryType, PrivacyLevel
from contextos.core.models import Memory
from contextos.storage.database import Database


@pytest.fixture(scope="session")
def event_loop():
    """Create a session-scoped event loop."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture
async def db(tmp_path: Path) -> Database:
    """Create a temporary database for testing."""
    database = Database(tmp_path / "test.db")
    await database.initialize()
    yield database
    await database.close()


@pytest_asyncio.fixture
async def db_conn(db: Database):
    """Get a database connection."""
    return db.connection()


@pytest.fixture
def sample_memory() -> Memory:
    """A sample memory for testing."""
    return Memory(
        id=uuid4(),
        content="I prefer Python 3.12+ and always use type hints.",
        type=MemoryType.PREFERENCE,
        source_type="cli_input",
        status=MemoryStatus.ACTIVE,
        confidence=0.9,
        importance=0.7,
        privacy_level=PrivacyLevel.PERSONAL,
        token_count=15,
        tags=["python", "coding"],
    )


@pytest.fixture
def sample_memories() -> list[Memory]:
    """A list of sample memories for testing."""
    return [
        Memory(
            content="I prefer Python 3.12+ and always use type hints.",
            type=MemoryType.PREFERENCE,
            status=MemoryStatus.ACTIVE,
            confidence=0.9,
            importance=0.7,
            token_count=15,
            tags=["python"],
        ),
        Memory(
            content="I work at Acme Corp as a senior engineer.",
            type=MemoryType.FACT,
            status=MemoryStatus.ACTIVE,
            confidence=0.95,
            importance=0.8,
            token_count=12,
        ),
        Memory(
            content="I use pytest for testing and ruff for linting.",
            type=MemoryType.PREFERENCE,
            status=MemoryStatus.ACTIVE,
            confidence=0.85,
            importance=0.6,
            token_count=13,
        ),
        Memory(
            content="I'm building a CLI tool called ContextOS.",
            type=MemoryType.PROJECT,
            status=MemoryStatus.ACTIVE,
            confidence=0.9,
            importance=0.9,
            token_count=11,
        ),
        Memory(
            content="I think ORMs are overengineered for most use cases.",
            type=MemoryType.OPINION,
            status=MemoryStatus.ACTIVE,
            confidence=0.7,
            importance=0.4,
            token_count=13,
        ),
    ]
