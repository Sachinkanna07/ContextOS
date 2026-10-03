"""Deterministic parent-death boundaries and OS listener attribution regressions."""

from __future__ import annotations

import json
import multiprocessing
import os
import socket
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import Mock

import psutil
import pytest
from typer.testing import CliRunner

from contextos.cli.app import app
from contextos.config.settings import Settings
from contextos.core.exceptions import DaemonAlreadyRunningError
from contextos.daemon import manager


def _settings(tmp_path):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    return Settings(daemon={"data_dir": tmp_path, "port": port})


@contextmanager
def _responder(payload):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield server.server_port
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)
        assert not worker.is_alive()

_CHILD_GATE = """
import os, time
from pathlib import Path
if os.environ.get("CONTEXTOS_DAEMON_STARTUP_ID"):
    from contextos.daemon import manager
    mode = os.environ["RC2_GATE_MODE"]
    marker, release = Path(os.environ["RC2_MARKER"]), Path(os.environ["RC2_RELEASE"])
    def gate():
        marker.write_text(str(os.getpid()))
        deadline = time.monotonic() + 45
        while not release.exists():
            if time.monotonic() >= deadline:
                raise RuntimeError("test gate was not released")
            time.sleep(.02)
    if mode == "before":
        original = manager._claim_startup
        def claim(settings):
            gate()
            return original(settings)
        manager._claim_startup = claim
    elif mode == "publication":
        original = manager._write_pid_file_atomic
        def publish(settings, pid):
            gate()
            return original(settings, pid)
        manager._write_pid_file_atomic = publish
    elif mode == "stalled":
        async def stalled(*args):
            import asyncio
            marker.write_text(str(os.getpid()))
            await asyncio.Event().wait()
        manager._async_run_server = stalled
"""


def _starter(serialized: str, mode: str, hooks: str, marker: str, release: str) -> None:
    os.environ.update(
        {
            "RC2_GATE_MODE": mode,
            "RC2_MARKER": marker,
            "RC2_RELEASE": release,
            "PYTHONPATH": hooks + os.pathsep + os.environ.get("PYTHONPATH", ""),
        }
    )
    if mode == "ready":
        original = manager.wait_until_ready

        def ready(*args, **kwargs):
            pid = original(*args, **kwargs)
            Path(marker).write_text(str(pid))
            while not Path(release).exists():
                time.sleep(0.02)
            return pid

        manager.wait_until_ready = ready
    manager.start_daemon(Settings.model_validate_json(serialized))


def _await_marker(marker: Path, starter, timeout: float = 40) -> int:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if marker.exists():
            return int(marker.read_text())
        assert starter.is_alive(), f"starter exited {starter.exitcode} before boundary"
        time.sleep(0.02)
    pytest.fail("Deterministic child boundary was not reached")


@pytest.mark.parametrize("mode", ["before", "publication", "ready"])
def test_killed_starter_leaves_registered_manageable_child(tmp_path, monkeypatch, mode):
    settings = _settings(tmp_path / "data")
    hooks = tmp_path / "hooks"
    hooks.mkdir()
    (hooks / "sitecustomize.py").write_text(_CHILD_GATE)
    marker, release = tmp_path / "boundary", tmp_path / "release"
    ctx = multiprocessing.get_context("spawn")
    starter = ctx.Process(
        target=_starter,
        args=(
            settings.model_dump_json(),
            mode,
            str(hooks),
            str(marker),
            str(release),
        ),
    )
    tracked = []
    starter.start()
    try:
        child_pid = _await_marker(marker, starter)
        child = psutil.Process(child_pid)
        tracked = [child, *child.children(recursive=True)]
        starter.terminate()
        starter.join(timeout=5)
        assert not starter.is_alive()
        release.touch()
        with pytest.raises(DaemonAlreadyRunningError) as already:
            manager.start_daemon(settings)
        assert already.value.pid == child_pid
        state = manager._read_state(settings)
        assert state.phase == "registered" and state.pid == child_pid
        assert state.created == child.create_time()
        assert manager._owns_listener(child, settings)
        monkeypatch.setattr("contextos.config.settings.load_settings", lambda: settings)
        doctor = CliRunner().invoke(app, ["doctor", "--json"])
        assert doctor.exit_code == 0 and '"overall": true' in doctor.output, doctor.output
        manager.stop_daemon(settings)
        _, alive = psutil.wait_procs(tracked, timeout=5)
        assert not alive
        assert not manager._state_file(settings).exists()
        assert not manager._pid_file(settings).exists()
        with manager.lifecycle_lock(settings, timeout=0.2):
            pass
    finally:
        release.touch()
        if starter.is_alive():
            starter.terminate()
        starter.join(timeout=5)
        if manager.is_running(settings)[0]:
            manager.stop_daemon(settings)
        for proc in tracked:
            if proc.is_running():
                proc.kill()


def test_stalled_child_exits_on_own_deadline_after_starter_death(tmp_path):
    settings = _settings(tmp_path / "data")
    settings.daemon.readiness_timeout = 20
    settings.daemon.lock_timeout = 30
    hooks = tmp_path / "hooks"
    hooks.mkdir()
    (hooks / "sitecustomize.py").write_text(_CHILD_GATE)
    marker, release = tmp_path / "boundary", tmp_path / "release"
    ctx = multiprocessing.get_context("spawn")
    starter = ctx.Process(
        target=_starter,
        args=(
            settings.model_dump_json(),
            "stalled",
            str(hooks),
            str(marker),
            str(release),
        ),
    )
    starter.start()
    child = None
    try:
        child = psutil.Process(_await_marker(marker, starter))
        starter.terminate()
        starter.join(timeout=5)
        child.wait(timeout=25)
        assert not child.is_running()
        assert manager.is_running(settings) == (False, None)
        assert not manager._state_file(settings).exists()
        assert not manager._pid_file(settings).exists()
    finally:
        if starter.is_alive():
            starter.terminate()
        starter.join(timeout=5)
        if child and child.is_running():
            child.kill()


def test_live_daemon_pid_echo_from_unrelated_listener_is_rejected(tmp_path):
    settings = _settings(tmp_path)
    manager.start_daemon(settings)
    _, pid = manager.is_running(settings)
    port = settings.daemon.port
    try:
        with _responder({"daemon_running": True, "pid": pid}) as impostor:
            settings.daemon.port = impostor
            assert not manager._owns_listener(psutil.Process(pid), settings)
            with pytest.raises(RuntimeError, match="did not become ready"):
                manager.wait_until_ready(settings, expected_pid=pid, timeout=0.2)
            assert manager._pid_file(settings).read_text().strip() == str(pid)
    finally:
        settings.daemon.port = port
        manager.stop_daemon(settings)


def test_python_c_command_cannot_spoof_daemon_module_argv():
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import time; time.sleep(60)",
            "-m",
            "contextos",
            "start",
            "--foreground",
        ],
        creationflags=0x08000000 if sys.platform == "win32" else 0,
    )
    try:
        assert manager._verified_process(process.pid) is None
    finally:
        process.terminate()
        process.wait(timeout=5)


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1"])
def test_listener_requires_exact_address_port_and_retained_identity(host):
    settings = Settings(daemon={"host": host, "port": 12345})
    proc = Mock()
    address = "127.0.0.1" if host == "localhost" else host
    conn = Mock(status=psutil.CONN_LISTEN, laddr=Mock(ip=address, port=12345))
    proc.net_connections.return_value = [conn]
    assert manager._owns_listener(proc, settings)
    conn.laddr.port = 54321
    assert not manager._owns_listener(proc, settings)
    conn.laddr.port = 12345
    conn.laddr.ip = "0.0.0.0"
    assert not manager._owns_listener(proc, settings)
    conn.laddr.ip = address
    proc.is_running.side_effect = [True, False]
    assert not manager._owns_listener(proc, settings)


def test_previous_boot_pending_intent_is_invalidated_without_wait(tmp_path):
    settings = _settings(tmp_path)
    state = manager._new_intent(settings).model_copy(
        update={
            "boot_time": psutil.boot_time() - 100,
            "deadline": time.monotonic() + 864000,
        }
    )
    with manager._registration_lock(settings):
        manager._write_state(settings, state)
    before = time.monotonic()
    assert manager.is_running(settings) == (False, None)
    assert time.monotonic() - before < 2
    assert not manager._state_file(settings).exists()


def test_cleanup_preserves_new_generation_even_when_pid_matches(tmp_path):
    settings = _settings(tmp_path)
    old, replacement = manager._new_intent(settings), manager._new_intent(settings)
    with manager._registration_lock(settings):
        manager._write_state(settings, replacement)
        manager._write_pid_file_atomic(settings, os.getpid())
    manager._delete_generation(settings, old.startup_id, [os.getpid()])
    assert manager._read_state(settings).startup_id == replacement.startup_id
    assert manager._pid_file(settings).exists()


def test_expired_child_cannot_claim_replacement_generation(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    old, replacement = manager._new_intent(settings), manager._new_intent(settings)
    with manager._registration_lock(settings):
        manager._write_state(settings, replacement)
    monkeypatch.setenv("CONTEXTOS_DAEMON_STARTUP_ID", old.startup_id)
    with pytest.raises(RuntimeError, match="expired or was replaced"):
        manager._claim_startup(settings)
    assert manager._read_state(settings).startup_id == replacement.startup_id
    assert not manager._pid_file(settings).exists()
