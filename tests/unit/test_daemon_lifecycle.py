"""Final RC2 regressions for recorded ownership and serialized lifecycle state."""

from __future__ import annotations

import json
import multiprocessing
import os
import socket
import subprocess
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
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


@pytest.mark.parametrize(
    "payload",
    [
        {"daemon_running": True, "pid": 0},
        {"daemon_running": True, "pid": os.getpid()},  # different, live process
        {"daemon_running": True, "pid": -1},
        {"daemon_running": True, "pid": "malformed"},
        {"daemon_running": True},
        {"daemon_running": True, "pid": True},
        {"daemon_running": True, "pid": [123]},
        ["malformed status object"],
    ],
)
def test_existing_daemon_rejects_unrelated_endpoint(tmp_path, monkeypatch, payload):
    settings = _settings(tmp_path)
    manager.start_daemon(settings)
    running, pid = manager.is_running(settings)
    assert running and pid > 0
    original_port = settings.daemon.port
    original_wait = manager.wait_until_ready

    def bounded_wait(*args, **kwargs):
        kwargs["timeout"] = 0.15
        return original_wait(*args, **kwargs)

    monkeypatch.setattr(manager, "wait_until_ready", bounded_wait)
    monkeypatch.setattr("contextos.config.settings.load_settings", lambda: settings)
    try:
        with _responder(payload) as unrelated_port:
            settings.daemon.port = unrelated_port
            result = CliRunner().invoke(app, ["start"])
            assert result.exit_code == 1, result.output
            assert "already running" not in result.output
            assert "[OK]" not in result.output
            assert manager._pid_file(settings).read_text().strip() == str(pid)
            assert psutil.pid_exists(pid)
    finally:
        settings.daemon.port = original_port
        manager.stop_daemon(settings)


@pytest.mark.parametrize("pid", [0, -1, None, "123", True])
def test_readiness_requires_positive_expected_pid(pid):
    with pytest.raises(ValueError, match="positive expected_pid"):
        manager.wait_until_ready(Settings(), expected_pid=pid)


def test_readiness_requires_explicit_expected_pid():
    with pytest.raises(TypeError, match="expected_pid"):
        manager.wait_until_ready(Settings(), process=Mock(pid=123))


def test_readiness_rejects_unrelated_process_even_with_matching_health(tmp_path):
    with _responder({"daemon_running": True, "pid": os.getpid()}) as port:
        settings = Settings(daemon={"data_dir": tmp_path, "port": port})
        with pytest.raises(RuntimeError, match="not a live ContextOS"):
            manager.wait_until_ready(settings, expected_pid=os.getpid())


@pytest.mark.parametrize("content", ["0", "-12", "", "malformed", "999999999"])
def test_invalid_and_stale_pid_state_is_removed_under_lock(tmp_path, content):
    settings = _settings(tmp_path)
    manager._pid_file(settings).write_text(content)
    assert manager.is_running(settings) == (False, None)
    assert not manager._pid_file(settings).exists()


def _concurrent_starter(serialized, barrier, owner_entered, release, results):
    """Both processes reach the lock boundary before either can start."""
    settings = Settings.model_validate_json(serialized)
    original_lock = manager.lifecycle_lock
    original_spawn = manager._spawn_background
    original_popen = subprocess.Popen

    @contextmanager
    def gated_lock(*args, **kwargs):
        barrier.wait(timeout=30)
        with original_lock(*args, **kwargs):
            yield

    def gated_spawn(*args, **kwargs):
        owner_entered.set()
        assert release.wait(timeout=30)
        return original_spawn(*args, **kwargs)

    def recorded_popen(*args, **kwargs):
        process = original_popen(*args, **kwargs)
        results.put(("spawned", process.pid))
        return process

    manager.lifecycle_lock = gated_lock
    manager._spawn_background = gated_spawn
    subprocess.Popen = recorded_popen
    try:
        manager.start_daemon(settings)
        results.put(("started", None))
    except DaemonAlreadyRunningError as exc:
        results.put(("already", exc.pid))
    except BaseException as exc:
        results.put(("failed", repr(exc)))


def test_concurrent_start_preserves_survivor_and_stop(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    ctx = multiprocessing.get_context("spawn")
    barrier = ctx.Barrier(3)
    owner_entered, release = ctx.Event(), ctx.Event()
    results = ctx.Queue()
    workers = [
        ctx.Process(
            target=_concurrent_starter,
            args=(
                settings.model_dump_json(),
                barrier,
                owner_entered,
                release,
                results,
            ),
        )
        for _ in range(2)
    ]
    spawned = []
    tracked = []
    try:
        for worker in workers:
            worker.start()
        barrier.wait(timeout=30)
        assert owner_entered.wait(timeout=30)
        # The other invocation has reached acquisition while this owner is paused.
        release.set()
        for worker in workers:
            worker.join(timeout=30)
            assert worker.exitcode == 0
        messages = [results.get(timeout=5) for _ in range(3)]
        assert sorted(kind for kind, _ in messages) == ["already", "spawned", "started"]
        spawned = [pid for kind, pid in messages if kind == "spawned"]
        assert len(spawned) == 1
        running, pid = manager.is_running(settings)
        assert running and pid > 0
        assert manager._pid_file(settings).read_text().strip() == str(pid)
        root = psutil.Process(spawned[0])
        tracked = [root, *root.children(recursive=True)]
        assert pid in {proc.pid for proc in tracked}
        assert [value for kind, value in messages if kind == "already"] == [pid]
        monkeypatch.setattr("contextos.config.settings.load_settings", lambda: settings)
        runner = CliRunner()
        doctor = runner.invoke(app, ["doctor", "--json"])
        assert doctor.exit_code == 0 and '"overall": true' in doctor.output
        stop = runner.invoke(app, ["stop"])
        assert stop.exit_code == 0, stop.output
        assert not manager._pid_file(settings).exists()
        _, alive = psutil.wait_procs(tracked, timeout=5)
        assert not alive, [proc.pid for proc in alive]
        # Persistent lock is reusable after stop, not stale ownership.
        assert manager._lock_file(settings).exists()
        with manager.lifecycle_lock(settings, timeout=0.2):
            pass
    finally:
        release.set()
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
            worker.join(timeout=5)
        if manager.is_running(settings)[0]:
            manager.stop_daemon(settings)
        for proc in tracked:
            if proc.is_running():
                proc.kill()
        results.close()


def test_failed_starter_preserves_another_legitimate_daemons_state(tmp_path, monkeypatch):
    survivor_settings = _settings(tmp_path / "survivor")
    manager.start_daemon(survivor_settings)
    _, survivor_pid = manager.is_running(survivor_settings)
    settings = _settings(tmp_path / "attempt")
    settings.daemon.data_dir.mkdir()
    failed = Mock(pid=123)
    failed.poll.return_value = None
    child = Mock(pid=456)
    monkeypatch.setattr(subprocess, "Popen", Mock(return_value=failed))
    root = Mock(pid=123)
    root.create_time.return_value = 1.0
    root.children.return_value = [child]
    monkeypatch.setattr(manager, "_spawn_identity", lambda pid: root)
    monkeypatch.setattr(manager, "_child_processes", lambda pid: [child])
    monkeypatch.setattr(psutil, "wait_procs", lambda children, timeout: (children, []))

    def replaced_state(*args, **kwargs):
        manager._write_pid_file_atomic(settings, survivor_pid)
        raise RuntimeError("startup failed after state replacement")

    monkeypatch.setattr(manager, "wait_until_ready", replaced_state)
    try:
        with pytest.raises(RuntimeError, match="state replacement"):
            manager.start_daemon(settings)
        assert manager._pid_file(settings).read_text().strip() == str(survivor_pid)
        assert manager.is_running(survivor_settings) == (True, survivor_pid)
        failed.terminate.assert_called_once()
        child.terminate.assert_called_once()
    finally:
        manager.stop_daemon(survivor_settings)


def _lock_holder(serialized, acquired, release):
    settings = Settings.model_validate_json(serialized)
    with manager.lifecycle_lock(settings):
        acquired.set()
        assert release.wait(timeout=30)


def test_interprocess_lock_timeout_is_nonzero_and_does_not_mutate_state(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    manager._pid_file(settings).write_text("123\n")
    ctx = multiprocessing.get_context("spawn")
    acquired, release = ctx.Event(), ctx.Event()
    holder = ctx.Process(target=_lock_holder, args=(settings.model_dump_json(), acquired, release))
    original_lock = manager.lifecycle_lock
    monkeypatch.setattr(
        manager,
        "lifecycle_lock",
        lambda settings: original_lock(settings, timeout=0.1),
    )
    monkeypatch.setattr("contextos.config.settings.load_settings", lambda: settings)
    holder.start()
    try:
        assert acquired.wait(timeout=30)
        before = time.monotonic()
        result = CliRunner().invoke(app, ["start"])
        assert time.monotonic() - before < 5
        assert result.exit_code == 1, result.output
        assert "Timed out" in result.output and "lifecycle lock" in result.output
        assert manager._pid_file(settings).read_text() == "123\n"
    finally:
        release.set()
        holder.join(timeout=5)
        if holder.is_alive():
            holder.terminate()
            holder.join(timeout=5)
    assert holder.exitcode == 0


def test_crashed_lock_owner_does_not_brick_lifecycle(tmp_path):
    settings = _settings(tmp_path)
    ctx = multiprocessing.get_context("spawn")
    acquired, release = ctx.Event(), ctx.Event()
    holder = ctx.Process(target=_lock_holder, args=(settings.model_dump_json(), acquired, release))
    holder.start()
    try:
        assert acquired.wait(timeout=30)
        holder.terminate()
        holder.join(timeout=5)
        assert not holder.is_alive()
        with manager.lifecycle_lock(settings, timeout=0.2):
            assert manager._lock_file(settings).exists()
    finally:
        if holder.is_alive():
            holder.terminate()
            holder.join(timeout=5)


def test_bind_failure_cleans_only_owned_tree(tmp_path):
    settings = _settings(tmp_path)
    with socket.socket() as busy:
        busy.bind((settings.daemon.host, settings.daemon.port))
        busy.listen()
        with pytest.raises(RuntimeError, match=r"exited|did not become ready"):
            manager.start_daemon(settings)
    assert not manager._pid_file(settings).exists()
    assert manager.is_running(settings) == (False, None)


@pytest.mark.parametrize("content", ["", "malformed", "0", "456"])
def test_cleanup_never_deletes_unowned_or_invalid_state(tmp_path, content):
    settings = _settings(tmp_path)
    manager._pid_file(settings).write_text(content)
    with manager.lifecycle_lock(settings):
        assert manager._safe_delete_pid_file(settings, 123) is False
    assert manager._pid_file(settings).read_text() == content


def test_failed_atomic_publication_preserves_previous_state(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    manager._pid_file(settings).write_text("456\n")

    def fail_replace(*args):
        raise OSError("replace failed")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        manager._write_pid_file_atomic(settings, 123)
    assert manager._pid_file(settings).read_text() == "456\n"
    assert not list(tmp_path.glob("*.tmp.*"))
