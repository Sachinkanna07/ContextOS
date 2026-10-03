"""RC2 startup readiness and owned-child cleanup regressions."""

import socket
import subprocess
import sys
from unittest.mock import Mock

import httpx
import pytest
from typer.testing import CliRunner

from contextos.cli.app import app
from contextos.config.settings import Settings
from contextos.daemon import manager


def test_start_immediate_doctor_and_already_running(tmp_path, monkeypatch):
    settings = Settings()
    settings.daemon.data_dir = tmp_path
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        settings.daemon.port = sock.getsockname()[1]
    monkeypatch.setattr("contextos.config.settings.load_settings", lambda: settings)
    runner = CliRunner()
    try:
        for _ in range(2):
            result = runner.invoke(app, ["start"])
            assert result.exit_code == 0, result.output
            doctor = runner.invoke(app, ["doctor", "--json"])
            assert doctor.exit_code == 0, doctor.output
            assert '"overall": true' in doctor.output
        assert "already running" in result.output
    finally:
        if manager.is_running(settings)[0]:
            manager.stop_daemon(settings)
    assert not manager._pid_file(settings).exists()


def test_readiness_retries_until_owned_child_responds(monkeypatch):
    process = Mock(pid=123)
    process.poll.return_value = None
    responses = [
        httpx.ConnectError("starting"),
        httpx.Response(200, json={"daemon_running": True, "pid": 999}),
        httpx.Response(200, json={"daemon_running": True, "pid": 456}),
    ]
    child = Mock(pid=456)
    owner = Mock(pid=123)
    owner.children.return_value = [child]
    monkeypatch.setattr(manager, "_verified_process", lambda pid: owner)
    monkeypatch.setattr(manager, "_owns_listener", lambda proc, settings: proc is child)
    client = Mock()
    client.get.side_effect = responses
    monkeypatch.setattr(httpx.Client, "__enter__", lambda self: client)
    assert manager.wait_until_ready(Settings(), process=process, expected_pid=123) == 456
    assert client.get.call_count == 3


@pytest.mark.parametrize("kill", [False, True])
def test_timeout_cleans_owned_child(tmp_path, monkeypatch, kill):
    settings = Settings()
    settings.daemon.data_dir = tmp_path
    process = Mock(pid=123)
    process.poll.return_value = None
    if kill:
        process.wait.side_effect = [subprocess.TimeoutExpired("daemon", 2), None]
    manager._pid_file(settings).write_text("456")
    child = Mock(pid=456)
    monkeypatch.setattr(manager, "_child_processes", lambda pid: [child])
    monkeypatch.setattr(manager.psutil, "wait_procs", lambda children, timeout: (children, []))
    manager._lock_file(settings).touch()
    spawn = Mock(return_value=process)
    monkeypatch.setattr(subprocess, "Popen", spawn)
    root = Mock(pid=123)
    root.create_time.return_value = 1.0
    root.children.return_value = [child]
    monkeypatch.setattr(manager, "_spawn_identity", lambda pid: root)

    def fail(*args, **kwargs):
        raise RuntimeError("did not become ready")

    monkeypatch.setattr(manager, "wait_until_ready", fail)
    with pytest.raises(RuntimeError, match="did not become ready"):
        manager._spawn_background(settings)
    if sys.platform == "win32":
        assert spawn.call_args.kwargs["creationflags"] == 0x08000200
    child.terminate.assert_called_once()
    process.terminate.assert_called_once()
    assert process.kill.called == kill
    assert not manager._pid_file(settings).exists()
    # The lock inode must remain stable for current holders and queued starters.
    assert manager._lock_file(settings).exists()


def test_readiness_timeout_and_early_exit(monkeypatch):
    monkeypatch.setattr(manager, "_verified_process", lambda pid: Mock())
    with pytest.raises(RuntimeError, match="within"):
        manager.wait_until_ready(Settings(), expected_pid=123, timeout=0)
    process = Mock(pid=123)
    process.poll.return_value = 1
    with pytest.raises(RuntimeError, match="exited"):
        manager.wait_until_ready(Settings(), process=process, expected_pid=123)


def test_cli_failure_has_nonzero_exit(monkeypatch):
    monkeypatch.setattr(manager, "is_running", lambda settings: (False, None))

    def fail(*args, **kwargs):
        raise RuntimeError("did not become ready")

    monkeypatch.setattr(manager, "start_daemon", fail)
    result = CliRunner().invoke(app, ["start"])
    assert result.exit_code == 1
    assert "did not become ready" in result.output
    assert "[OK]" not in result.output
