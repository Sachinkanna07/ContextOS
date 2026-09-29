"""Daemon process manager for ContextOS.

Handles starting, stopping, and status-checking the background daemon process.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import sys
from pathlib import Path

import psutil
import uvicorn

from contextos.config.settings import Settings, load_settings

logger = logging.getLogger(__name__)


def _pid_file(settings: Settings) -> Path:
    return settings.daemon.data_dir / "contextos.pid"


def _lock_file(settings: Settings) -> Path:
    return settings.daemon.data_dir / "contextos.lock"


def is_running(settings: Settings | None = None) -> tuple[bool, int | None]:
    """Check if the daemon is running. Returns (running, pid)."""
    if settings is None:
        settings = load_settings()

    pid_path = _pid_file(settings)
    if not pid_path.exists():
        return False, None

    try:
        pid = int(pid_path.read_text().strip())
        if psutil.pid_exists(pid):
            proc = psutil.Process(pid)
            # A reused PID must never authorize stopping an unrelated Python process.
            args = proc.cmdline()
            is_contextos = any(
                args[index:index + 2] == ["-m", "contextos"]
                for index in range(len(args) - 1)
            )
            if proc.is_running() and is_contextos:
                return True, pid
        # Stale PID file
        pid_path.unlink(missing_ok=True)
        return False, None
    except (ValueError, psutil.Error):
        pid_path.unlink(missing_ok=True)
        return False, None


def start_daemon(settings: Settings | None = None, foreground: bool = False) -> None:
    """Start the ContextOS daemon."""
    if settings is None:
        settings = load_settings()

    running, pid = is_running(settings)
    if running:
        from contextos.core.exceptions import DaemonAlreadyRunningError
        raise DaemonAlreadyRunningError(pid)  # type: ignore

    # Ensure data directory exists
    settings.daemon.data_dir.mkdir(parents=True, exist_ok=True)

    if foreground:
        _run_server(settings)
    else:
        _spawn_background(settings)


def stop_daemon(settings: Settings | None = None) -> None:
    """Stop the ContextOS daemon."""
    if settings is None:
        settings = load_settings()

    running, pid = is_running(settings)
    if not running or pid is None:
        from contextos.core.exceptions import DaemonNotRunningError
        raise DaemonNotRunningError()

    try:
        proc = psutil.Process(pid)
        proc.terminate()
        proc.wait(timeout=10)
    except psutil.TimeoutExpired:
        proc.kill()
    except psutil.NoSuchProcess:
        pass

    _pid_file(settings).unlink(missing_ok=True)
    _lock_file(settings).unlink(missing_ok=True)


def _run_server(settings: Settings) -> None:
    """Run the API server (blocking)."""
    asyncio.run(_async_run_server(settings))


async def _async_run_server(settings: Settings) -> None:
    """Async server startup."""
    from contextos.daemon.wiring import wire_services
    from contextos.api.server import create_app, set_services

    # Wire services
    services = await wire_services(settings)
    set_services(services)

    # Write PID file
    pid_path = _pid_file(settings)
    pid_path.write_text(str(os.getpid()))

    # Create and run app
    app = create_app()

    config = uvicorn.Config(
        app,
        host=settings.daemon.host,
        port=settings.daemon.port,
        log_level=settings.daemon.log_level,
        access_log=False,
    )
    server = uvicorn.Server(config)

    try:
        await server.serve()
    finally:
        # Cleanup
        db = services.get("database")
        if db:
            await db.close()
        pid_path.unlink(missing_ok=True)
        _lock_file(settings).unlink(missing_ok=True)


def _spawn_background(settings: Settings) -> None:
    """Spawn the daemon as a background process."""
    import subprocess

    cmd = [
        sys.executable, "-m", "contextos",
        "start", "--foreground",
    ]

    # Platform-specific background process creation
    if sys.platform == "win32":
        # Windows: CREATE_NEW_PROCESS_GROUP + DETACHED_PROCESS
        CREATE_NEW_PROCESS_GROUP = 0x00000200
        DETACHED_PROCESS = 0x00000008
        subprocess.Popen(
            cmd,
            creationflags=CREATE_NEW_PROCESS_GROUP | DETACHED_PROCESS,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
        )
    else:
        # Unix: use start_new_session
        subprocess.Popen(
            cmd,
            start_new_session=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
        )
