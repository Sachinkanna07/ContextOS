"""Serialized daemon lifecycle with recoverable registration and OS identity checks."""

from __future__ import annotations

import asyncio
import os
import sys
import threading
import time
from contextlib import contextmanager, suppress
from typing import TYPE_CHECKING
from uuid import uuid4

import httpx
import psutil
import uvicorn
from pydantic import ValidationError

from contextos.config.settings import Settings, load_settings
from contextos.core.exceptions import DaemonAlreadyRunningError, DaemonNotRunningError
from contextos.daemon.state import StartupState, atomic_write, file_lock, regular_path

if TYPE_CHECKING:
    import subprocess
    from collections.abc import Iterable, Iterator
    from pathlib import Path


def _pid_file(settings: Settings) -> Path:
    return settings.daemon.data_dir / "contextos.pid"


def _lock_file(settings: Settings) -> Path:
    return settings.daemon.data_dir / "contextos.lock"


def _state_file(settings: Settings) -> Path:
    return settings.daemon.data_dir / "contextos.state.json"


@contextmanager
def lifecycle_lock(settings: Settings, timeout: float | None = None) -> Iterator[None]:
    """Serialize start, stop and recovery; never unlink the kernel lock file."""
    duration = settings.daemon.lock_timeout if timeout is None else timeout
    with file_lock(_lock_file(settings), duration):
        yield


@contextmanager
def _registration_lock(settings: Settings) -> Iterator[None]:
    with file_lock(
        settings.daemon.data_dir / "contextos.registration.lock",
        settings.daemon.lock_timeout,
    ):
        yield


def _read_state(settings: Settings) -> StartupState | None:
    path = _state_file(settings)
    regular_path(path)
    try:
        return StartupState.model_validate_json(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except ValidationError as exc:
        raise RuntimeError("Invalid daemon startup state; refusing unsafe recovery") from exc


def _write_state(settings: Settings, state: StartupState) -> None:
    atomic_write(_state_file(settings), state.model_dump_json() + "\n")


def _write_pid_file_atomic(settings: Settings, pid: int) -> None:
    if type(pid) is not int or pid <= 0:
        raise ValueError("Daemon PID must be a positive integer")
    atomic_write(_pid_file(settings), f"{pid}\n")


def _safe_delete_pid_file(settings: Settings, expected_pids: int | Iterable[int]) -> bool:
    """Caller holds registration lock (or lifecycle lock for legacy-only state)."""
    expected = {expected_pids} if isinstance(expected_pids, int) else set(expected_pids)
    path = _pid_file(settings)
    regular_path(path)
    try:
        if int(path.read_text(encoding="utf-8").strip()) in expected:
            path.unlink()
            return True
    except (ValueError, FileNotFoundError):
        pass
    return False


def _delete_generation(settings: Settings, startup_id: str, owned_pids: Iterable[int]) -> None:
    """Compare generation and PIDs under registration lock, preserving replacements."""
    with _registration_lock(settings):
        state = _read_state(settings)
        if state is not None and state.startup_id == startup_id:
            _state_file(settings).unlink()
            _safe_delete_pid_file(settings, owned_pids)


def _verified_process(pid: int, created: float | None = None) -> psutil.Process | None:
    """Retain PID/creation identity and require the daemon's exact module command."""
    if type(pid) is not int or pid <= 0:
        return None
    try:
        proc = psutil.Process(pid)
        args = proc.cmdline()
        is_daemon = args[1:] == ["-m", "contextos", "start", "--foreground"]
        if created is not None and proc.create_time() != created:
            return None
        if proc.is_running() and is_daemon:
            return proc
    except psutil.Error:
        pass
    return None


def _running_state(settings: Settings) -> tuple[bool, int | None]:
    """Recover identity from authoritative state, with RC1 PID-file compatibility."""
    with _registration_lock(settings):
        state = _read_state(settings)
        if state is not None:
            if state.pid is not None and state.created is not None:
                proc = _verified_process(state.pid, state.created)
                if proc is not None:
                    return True, state.pid
                _state_file(settings).unlink()
                _safe_delete_pid_file(settings, state.pid)
            elif (
                abs(state.boot_time - psutil.boot_time()) < 2
                and time.monotonic()
                < state.deadline
                <= time.monotonic() + settings.daemon.readiness_timeout
            ):
                return False, None
            else:
                _state_file(settings).unlink()
        path = _pid_file(settings)
        regular_path(path)
        try:
            content = path.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            return False, None
        try:
            pid = int(content)
        except ValueError:
            pid = None
        if pid is not None and _verified_process(pid) is not None:
            return True, pid
        if path.read_text(encoding="utf-8").strip() == content:
            path.unlink()
        return False, None


def _recover_registration(settings: Settings) -> tuple[bool, int | None]:
    """Allow a detached child to claim its intent after its starter crashes.

    Expiry invalidation and child claiming use the same registration lock. A late
    child exits before initializing services rather than replacing a newer state.
    A pending intent is never treated as permission to spawn another daemon.
    """
    recovery_deadline = time.monotonic() + settings.daemon.readiness_timeout
    while True:
        running, pid = _running_state(settings)
        if running:
            return running, pid
        with _registration_lock(settings):
            state = _read_state(settings)
            if state is None:
                return False, None
            if time.monotonic() >= recovery_deadline and state.phase == "pending":
                _state_file(settings).unlink()
                return False, None
        time.sleep(min(0.05, max(0, state.deadline - time.monotonic())))


def is_running(settings: Settings | None = None) -> tuple[bool, int | None]:
    settings = settings or load_settings()
    with lifecycle_lock(settings):
        return _recover_registration(settings)


def start_daemon(settings: Settings | None = None, foreground: bool = False) -> None:
    settings = settings or load_settings()
    settings.daemon.data_dir.mkdir(parents=True, exist_ok=True)
    if foreground:
        serialized = os.environ.get("CONTEXTOS_DAEMON_SETTINGS")
        if serialized:
            settings = Settings.model_validate_json(serialized)
        _run_server(settings)
        return
    with lifecycle_lock(settings):
        running, pid = _recover_registration(settings)
        if running and pid is not None:
            wait_until_ready(settings, expected_pid=pid)
            raise DaemonAlreadyRunningError(pid)
        _spawn_background(settings)


def _child_processes(pid: int) -> list[psutil.Process]:
    try:
        return psutil.Process(pid).children(recursive=True)
    except psutil.Error:
        return []


def _spawn_identity(pid: int) -> psutil.Process:
    return psutil.Process(pid)


def _terminate_tree(processes: Iterable[psutil.Process], timeout: float = 2) -> None:
    retained = list(dict.fromkeys(processes))
    for proc in reversed(retained):
        with suppress(psutil.NoSuchProcess):
            proc.terminate()
    _, alive = psutil.wait_procs(retained, timeout=timeout)
    for proc in alive:
        with suppress(psutil.NoSuchProcess):
            proc.kill()
    _, survivors = psutil.wait_procs(alive, timeout=2)
    if survivors:
        raise RuntimeError("Could not stop owned daemon process tree")


def stop_daemon(settings: Settings | None = None) -> None:
    settings = settings or load_settings()
    with lifecycle_lock(settings):
        running, pid = _recover_registration(settings)
        if not running or pid is None:
            raise DaemonNotRunningError()
        with _registration_lock(settings):
            state = _read_state(settings)
            proc = _verified_process(pid, state.created if state else None)
            if proc is None:
                raise DaemonNotRunningError()
            tree = [proc, *proc.children(recursive=True)]
            if state and state.root_pid and state.root_created:
                root = _verified_process(state.root_pid, state.root_created)
                if root and root.pid != pid and any(p == root for p in proc.parents()):
                    tree = [root, *root.children(recursive=True)]
        _terminate_tree(tree, timeout=10)
        if state:
            _delete_generation(settings, state.startup_id, [p.pid for p in tree])
        else:
            with _registration_lock(settings):
                _safe_delete_pid_file(settings, [p.pid for p in tree])


def _new_intent(settings: Settings) -> StartupState:
    starter = psutil.Process()
    return StartupState(
        startup_id=uuid4().hex,
        starter_pid=starter.pid,
        starter_created=starter.create_time(),
        deadline=time.monotonic() + settings.daemon.readiness_timeout,
        boot_time=psutil.boot_time(),
        host=settings.daemon.host,
        port=settings.daemon.port,
    )


def _claim_startup(settings: Settings) -> str:
    """Child publishes durable identity before it can bind or advertise health."""
    token = os.environ.get("CONTEXTOS_DAEMON_STARTUP_ID")
    if not token:
        with lifecycle_lock(settings):
            running, pid = _recover_registration(settings)
            if running and pid is not None:
                raise DaemonAlreadyRunningError(pid)
            state = _new_intent(settings)
            with _registration_lock(settings):
                _write_state(settings, state)
        token = state.startup_id
    current = psutil.Process()
    with _registration_lock(settings):
        state = _read_state(settings)
        if (
            state is None
            or state.startup_id != token
            or state.phase != "pending"
            or time.monotonic() >= state.deadline
            or abs(state.boot_time - psutil.boot_time()) >= 2
            or state.host != settings.daemon.host
            or state.port != settings.daemon.port
        ):
            raise RuntimeError("Daemon startup handoff expired or was replaced")
        if state.root_pid and state.root_created:
            root = _verified_process(state.root_pid, state.root_created)
            if root is None or (root != current and root not in current.parents()):
                raise RuntimeError("Daemon startup root identity does not match")
        state = state.model_copy(
            update={
                "phase": "registered",
                "pid": current.pid,
                "created": current.create_time(),
            }
        )
        _write_state(settings, state)
        _write_pid_file_atomic(settings, current.pid)
    return token


def _run_server(settings: Settings) -> None:
    token = _claim_startup(settings)
    state = _read_state(settings)
    if state is None or state.startup_id != token:
        raise RuntimeError("Daemon registration disappeared before initialization")
    startup_finished = threading.Event()

    def enforce_deadline() -> None:
        if not startup_finished.wait(max(0, state.deadline - time.monotonic())):
            # Own process only. Even synchronous initialization cannot leave an
            # indefinitely unhealthy detached daemon after starter death.
            os._exit(1)

    threading.Thread(target=enforce_deadline, daemon=True).start()
    try:
        asyncio.run(_async_run_server(settings, startup_finished))
    finally:
        startup_finished.set()
        _delete_generation(settings, token, [os.getpid()])


async def _async_run_server(settings: Settings, startup_finished: threading.Event) -> None:
    from contextos.api.server import create_app, set_services
    from contextos.daemon.wiring import wire_services

    services = await wire_services(settings)
    set_services(services)

    class RegisteredServer(uvicorn.Server):
        async def startup(self, sockets=None):
            await super().startup(sockets=sockets)
            if self.started:
                startup_finished.set()

    try:
        server = RegisteredServer(
            uvicorn.Config(
                create_app(),
                host=settings.daemon.host,
                port=settings.daemon.port,
                log_level=settings.daemon.log_level,
                access_log=False,
            )
        )
        await server.serve()
    finally:
        await services["database"].close()


def _spawn_background(settings: Settings) -> None:
    """Starter owns serialization; child independently owns durable publication."""
    import subprocess

    state = _new_intent(settings)
    with _registration_lock(settings):
        _write_state(settings, state)
    process = None
    root = None
    children: list[psutil.Process] = []
    try:
        kwargs = (
            {"creationflags": 0x08000200}
            if sys.platform == "win32"
            else {"start_new_session": True}
        )
        process = subprocess.Popen(
            [sys.executable, "-m", "contextos", "start", "--foreground"],
            env={
                **os.environ,
                "CONTEXTOS_DAEMON_SETTINGS": settings.model_dump_json(),
                "CONTEXTOS_DAEMON_STARTUP_ID": state.startup_id,
            },
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            **kwargs,
        )
        root = _spawn_identity(process.pid)
        with _registration_lock(settings):
            current = _read_state(settings)
            if current is None or current.startup_id != state.startup_id:
                raise RuntimeError("Daemon startup exited before becoming ready")
            _write_state(
                settings,
                current.model_copy(
                    update={
                        "root_pid": root.pid,
                        "root_created": root.create_time(),
                    }
                ),
            )
        wait_until_ready(
            settings,
            process=process,
            expected_pid=process.pid,
            startup_id=state.startup_id,
            descendants=children,
        )
    except BaseException:
        owned = {process.pid} if process is not None else set()
        if process is not None:
            if root is not None and root.is_running():
                with suppress(psutil.NoSuchProcess):
                    children.extend(root.children(recursive=True))
            with _registration_lock(settings):
                current = _read_state(settings)
                if current and current.startup_id == state.startup_id and current.pid:
                    child = _verified_process(current.pid, current.created)
                    if child is not None:
                        children.append(child)
            owned.update(child.pid for child in children)
            _terminate_tree(children)
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2)
        _delete_generation(settings, state.startup_id, owned)
        raise


def _owns_listener(proc: psutil.Process, settings: Settings) -> bool:
    """Fail closed unless the retained process owns this exact loopback listener."""
    addresses = (
        {"127.0.0.1"}
        if settings.daemon.host == "localhost"
        else {
            settings.daemon.host,
        }
    )
    try:
        if not proc.is_running():
            return False
        connections = proc.net_connections(kind="tcp")
        owns = any(
            conn.status == psutil.CONN_LISTEN
            and conn.laddr
            and conn.laddr.port == settings.daemon.port
            and conn.laddr.ip in addresses
            for conn in connections
        )
        return owns and proc.is_running()
    except psutil.Error:
        return False


def wait_until_ready(
    settings: Settings,
    *,
    expected_pid: int,
    process: subprocess.Popen | None = None,
    timeout: float | None = None,
    startup_id: str | None = None,
    descendants: list[psutil.Process] | None = None,
) -> int:
    """Bounded polling; HTTP PID, birth identity, lineage and listener must agree."""
    if type(expected_pid) is not int or expected_pid <= 0:
        raise ValueError("wait_until_ready requires a positive expected_pid")
    if process is not None and expected_pid != process.pid:
        raise ValueError("expected_pid must match the spawned process")
    owner = _verified_process(expected_pid)
    if owner is None:
        raise RuntimeError("Expected PID is not a live ContextOS daemon")
    timeout = settings.daemon.readiness_timeout if timeout is None else timeout
    deadline = time.monotonic() + timeout
    host = "127.0.0.1" if settings.daemon.host == "localhost" else settings.daemon.host
    host = f"[{host}]" if ":" in host else host
    with httpx.Client(trust_env=False) as client:
        while time.monotonic() < deadline:
            if (process is not None and process.poll() is not None) or not owner.is_running():
                raise RuntimeError("ContextOS daemon exited before becoming ready")
            candidates = {expected_pid: owner}
            if process is not None:
                try:
                    children = owner.children(recursive=True)
                except psutil.NoSuchProcess as exc:
                    raise RuntimeError("ContextOS daemon exited before becoming ready") from exc
                candidates.update({child.pid: child for child in children})
                if descendants is not None:
                    descendants.extend(child for child in children if child not in descendants)
            try:
                response = client.get(
                    f"http://{host}:{settings.daemon.port}/api/v1/status",
                    timeout=min(0.5, max(0.001, deadline - time.monotonic())),
                )
                data = response.json() if response.status_code == 200 else None
                pid = data.get("pid") if isinstance(data, dict) else None
                if isinstance(data, dict) and data.get("daemon_running") is True:
                    responder = candidates.get(pid) if type(pid) is int and pid > 0 else None
                    if responder is not None and _owns_listener(responder, settings):
                        with _registration_lock(settings):
                            state = _read_state(settings)
                            registered = startup_id is None or (
                                state is not None
                                and state.startup_id == startup_id
                                and state.phase == "registered"
                                and state.pid == pid
                                and state.created == responder.create_time()
                            )
                        if registered and owner.is_running() and responder.is_running():
                            return pid
            except (httpx.HTTPError, ValueError, psutil.NoSuchProcess):
                pass
            time.sleep(min(0.05, max(0, deadline - time.monotonic())))
    raise RuntimeError(f"ContextOS daemon did not become ready within {timeout:g} seconds")
