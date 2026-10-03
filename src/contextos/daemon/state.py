"""Durable startup identity and crash-released locks for the local daemon.

Lock order is lifecycle, then registration. The child only takes registration,
so it can publish recoverable identity while its starter waits for readiness.
"""

from __future__ import annotations

import errno
import os
import sys
import tempfile
import threading
import time
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field

from contextos.core.exceptions import DaemonLockTimeoutError

if TYPE_CHECKING:
    from collections.abc import Iterator

_thread_locks = {
    "contextos.lock": threading.Lock(),
    "contextos.registration.lock": threading.Lock(),
}


class StartupState(BaseModel):
    """One generation; creation times prevent stored PIDs authorizing PID reuse."""

    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    startup_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    phase: Literal["pending", "registered"] = "pending"
    starter_pid: int = Field(gt=0)
    starter_created: float = Field(gt=0)
    deadline: float = Field(gt=0)
    boot_time: float = Field(gt=0)
    host: Literal["127.0.0.1", "localhost", "::1"]
    port: int = Field(gt=0, le=65535)
    root_pid: int | None = Field(default=None, gt=0)
    root_created: float | None = Field(default=None, gt=0)
    pid: int | None = Field(default=None, gt=0)
    created: float | None = Field(default=None, gt=0)


def regular_path(path: Path) -> None:
    """Fail closed on redirected state/lock paths, including Windows junctions."""
    if path.is_symlink() or path.is_junction():
        raise RuntimeError(f"Refusing redirected daemon state path: {path.name}")
    if path.exists() and not path.is_file():
        raise RuntimeError(f"Daemon state path is not a regular file: {path.name}")


def atomic_write(path: Path, content: str) -> None:
    """Exclusive same-directory temporary file, flush, then atomic replacement."""
    regular_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".tmp.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        regular_path(path)
        os.replace(temporary, path)
    finally:
        with suppress(FileNotFoundError):
            Path(temporary).unlink()


@contextmanager
def file_lock(path: Path, timeout: float) -> Iterator[None]:
    """Persistent inode, kernel ownership, bounded monotonic acquisition."""
    regular_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout
    thread_lock = _thread_locks[path.name]
    if not thread_lock.acquire(timeout=max(0, deadline - time.monotonic())):
        raise DaemonLockTimeoutError("Timed out waiting for in-process lifecycle lock")
    try:
        # O_NOFOLLOW adds race protection where the OS supports it. State is
        # stored in the user's private local data directory on all platforms.
        flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(path, flags, 0o600)
        with os.fdopen(fd, "r+b") as stream:
            acquired = False
            try:
                while True:
                    try:
                        stream.seek(0)
                        if sys.platform == "win32":
                            import msvcrt

                            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                        else:
                            import fcntl

                            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                        acquired = True
                        break
                    except OSError as exc:
                        if exc.errno not in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                            raise
                        if time.monotonic() >= deadline:
                            raise DaemonLockTimeoutError(
                                f"Timed out waiting for ContextOS lifecycle lock after {timeout:g}s"
                            ) from exc
                        time.sleep(min(0.02, max(0, deadline - time.monotonic())))
                yield
            finally:
                if acquired:
                    with suppress(OSError):
                        stream.seek(0)
                        if sys.platform == "win32":
                            import msvcrt

                            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
                        else:
                            import fcntl

                            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
    finally:
        thread_lock.release()
