"""The Windows process mechanisms (design sections 4 and 4.1; WP19): singleton, detached start, kill-on-close
groups and priority.

Every child is a trivial Python process (``_child.py``) that exits by itself within seconds. The tests wait
only on processes they started, directly or through such a child, and never signal any other process.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil
import pytest

from narration.contracts import codes
from narration.contracts.errors import NarrationError
from narration.platform import get_platform

from ._support import WINDOWS_ONLY, heartbeat_stopped, is_beating, read_json, run_child, started, wait_for_file

if sys.platform != "win32":
    raise pytest.skip.Exception(WINDOWS_ONLY, allow_module_level=True)

from narration.platform import _windows

pytestmark = pytest.mark.timeout(120)

NO_SUCH_PID = 0xFFFFFFFC
"""A pid no process has (Windows pids are small multiples of 4)."""


@pytest.fixture
def store(tmp_path: Path) -> Path:
    root = tmp_path / "store"
    root.mkdir()
    return root


# ---------------------------------------------------------------- the singleton (section 4)
def test_singleton_refuses_another_process_while_held_s4(store: Path, tmp_path: Path) -> None:
    platform = get_platform()
    out = tmp_path / "second.json"
    with platform.singleton(store) as acquired:
        assert acquired
        run_child("singleton", store, out)
        assert read_json(out) == {"acquired": False}
    run_child("singleton", store, out)
    assert read_json(out) == {"acquired": True}


def test_singleton_refuses_a_second_holder_in_the_same_process_s4(store: Path) -> None:
    platform = get_platform()
    with platform.singleton(store) as first, platform.singleton(store) as second:
        assert first
        assert not second
    seen: list[bool] = []

    def other_thread() -> None:
        with platform.singleton(store) as acquired:
            seen.append(acquired)

    with platform.singleton(store) as first:
        thread = threading.Thread(target=other_thread)
        thread.start()
        thread.join(timeout=30)
    assert first
    assert seen == [False]
    with platform.singleton(store) as again:
        assert again


def test_singleton_may_be_released_from_another_thread_s4(store: Path) -> None:
    platform = get_platform()
    held = platform.singleton(store)
    assert held.__enter__()
    thread = threading.Thread(target=held.__exit__, args=(None, None, None))
    thread.start()
    thread.join(timeout=30)
    with platform.singleton(store) as acquired:
        assert acquired


def test_a_daemon_that_dies_frees_the_singleton_s4(store: Path, tmp_path: Path) -> None:
    platform = get_platform()
    marker, go = tmp_path / "held.json", tmp_path / "go"
    with started("hold", store, marker, go) as holder:
        assert wait_for_file(marker)
        assert read_json(marker) == {"acquired": True}
        with platform.singleton(store) as acquired:
            assert not acquired
        go.touch()
        assert holder.wait(timeout=30) == 0
    with platform.singleton(store) as acquired:
        assert acquired


def test_a_mutex_abandoned_by_a_dead_holder_is_acquired_s4(store: Path, tmp_path: Path) -> None:
    name = _windows.singleton_name(store)
    marker, go = tmp_path / "held.json", tmp_path / "go"
    with started("hold", store, marker, go) as holder:
        assert wait_for_file(marker)
        # A second handle keeps the mutex object alive across the holder's death, so it is abandoned, not gone.
        extra = _windows._CreateMutexW(None, False, name)
        assert extra
        try:
            go.touch()
            assert holder.wait(timeout=30) == 0
            mutex = _windows._MutexHolder(name)
            assert mutex.acquire()
            assert mutex.abandoned
            mutex.release()
        finally:
            _windows._CloseHandle(extra)


def test_singleton_name_is_one_per_store_folder_s4(store: Path, tmp_path: Path) -> None:
    name = _windows.singleton_name(store)
    assert name.startswith("Global\\narration-mcp.daemon.")
    digest = name.removeprefix("Global\\narration-mcp.daemon.")
    assert len(digest) == 64
    assert set(digest) <= set("0123456789abcdef")
    for spelling in (
        Path(str(store).upper()),
        Path(str(store).replace("\\", "/")),
        store / "x" / "..",
        store / ".",
    ):
        assert _windows.singleton_name(spelling) == name, spelling
    other = tmp_path / "other"
    other.mkdir()
    assert _windows.singleton_name(other) != name


def test_singleton_name_sees_through_a_junction_s4(store: Path, tmp_path: Path) -> None:
    import _winapi

    link = tmp_path / "link"
    _winapi.CreateJunction(str(store), str(link))
    try:
        assert _windows.singleton_name(link) == _windows.singleton_name(store)
    finally:
        os.rmdir(link)


# ---------------------------------------------------------------- detached start (section 4.1)
def test_detached_start_uses_the_flags_and_handles_of_s4_1(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    class FakePopen:
        def __init__(self, argv: list[str], **kwargs: object) -> None:
            seen.update(kwargs, argv=argv)
            self.pid = 4242

    monkeypatch.setattr(_windows, "_DetachedPopen", FakePopen)
    pid = get_platform().spawn_detached(("python", "-m", "narration.daemon"), cwd=tmp_path, env={"A": "1"})
    assert pid == 4242
    assert seen == {
        "argv": ["python", "-m", "narration.daemon"],
        "cwd": tmp_path,
        "env": {"A": "1"},
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "close_fds": True,
        "creationflags": 0x01000000 | 0x00000008 | 0x00000200,
    }
    assert _windows.DETACHED_CREATION_FLAGS == (
        subprocess.CREATE_BREAKAWAY_FROM_JOB | subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    )


def test_breakaway_refused_is_daemon_unavailable_s4_1(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(argv: list[str], **kwargs: object) -> None:
        raise OSError(None, "Access is denied", None, 5)

    monkeypatch.setattr(_windows, "_DetachedPopen", refuse)
    with pytest.raises(NarrationError) as info:
        get_platform().spawn_detached(["python"], cwd=tmp_path, env={})
    error = info.value
    assert error.code == codes.DAEMON_UNAVAILABLE
    assert error.retryable is True
    assert error.retry_after_s == _windows.DAEMON_RETRY_AFTER_S
    assert error.hint == codes.error_code(codes.DAEMON_UNAVAILABLE).hint
    assert "narration-admin daemon start" in error.hint
    assert error.details is not None
    assert error.details["reason"] == "breakaway_refused"
    assert error.details["winerror"] == 5
    assert isinstance(error.__cause__, PermissionError)


def test_other_start_failures_are_not_daemon_unavailable_s4_1(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(argv: list[str], **kwargs: object) -> None:
        raise OSError(None, "The system cannot find the file specified", None, 2)

    monkeypatch.setattr(_windows, "_DetachedPopen", missing)
    with pytest.raises(FileNotFoundError):
        get_platform().spawn_detached(["no-such-program"], cwd=tmp_path, env={})
    with pytest.raises(ValueError, match="argv is empty"):
        get_platform().spawn_detached([], cwd=tmp_path, env={})


def test_a_job_that_forbids_breakaway_gives_daemon_unavailable_s4_1(tmp_path: Path) -> None:
    out = tmp_path / "refused.json"
    run_child("refused", out)
    result = read_json(out)
    assert "spawned_pid" not in result, "a daemon was started although the job forbids breakaway"
    assert result["code"] == codes.DAEMON_UNAVAILABLE
    assert result["retryable"] is True
    assert result["retry_after_s"] == _windows.DAEMON_RETRY_AFTER_S
    assert result["details"] == {
        "reason": "breakaway_refused",
        "winerror": 5,
        "in_job": True,
        "job_allows_breakaway": False,
    }


def _client(tmp_path: Path, how: str) -> tuple[dict[str, object], Path, Path]:
    """Run a client stand-in in a kill-on-close job; it starts a daemon stand-in and exits at once."""
    marker, go, out, cwd = tmp_path / "daemon.json", tmp_path / "go", tmp_path / "client.json", tmp_path / "cwd"
    cwd.mkdir()
    run_child("client", marker, go, out, cwd, "mark-4-1", how)
    time.sleep(0.5)  # the client has exited; give Windows a moment to finish closing its job
    go.touch()
    return read_json(out), marker, cwd


def test_a_detached_daemon_survives_its_client_s4_1(tmp_path: Path) -> None:
    client, marker, cwd = _client(tmp_path, "platform")
    if "refused" in client:
        pytest.skip("this test runner's own Job Object forbids breakaway, so no detached start can succeed here")
    assert wait_for_file(marker), "the daemon died with its client's job"
    seen = read_json(marker)
    # The pid returned is argv[0]'s: under a venv that is the launcher, whose child is the interpreter.
    assert client["daemon_pid"] in (seen["pid"], seen["ppid"])
    assert os.path.normcase(seen["cwd"]) == os.path.normcase(str(cwd))
    assert seen["mark"] == "mark-4-1"
    assert seen["stdin"] == ""


def test_without_breakaway_the_daemon_dies_with_its_client_control_s4_1(tmp_path: Path) -> None:
    # The control for the test above: the same client, started without CREATE_BREAKAWAY_FROM_JOB.
    client, marker, _ = _client(tmp_path, "plain")
    assert "daemon_pid" in client
    assert not wait_for_file(marker, timeout_s=8.0)


# ---------------------------------------------------------------- kill-on-close groups (sections 4 and 4.1)
def test_closing_the_group_kills_its_processes_s4_1(tmp_path: Path) -> None:
    marker = tmp_path / "worker.json"
    with started("sleep", marker, 15) as worker:
        with get_platform().kill_on_close_group() as add:
            add(worker.pid)
            assert wait_for_file(marker)
            assert is_beating(marker)
        assert heartbeat_stopped(marker), "the worker's interpreter still runs after its group closed"
        worker.wait(timeout=10)


def test_workers_die_when_their_daemon_dies_s4(tmp_path: Path) -> None:
    worker_marker, out, go = tmp_path / "worker.json", tmp_path / "supervisor.json", tmp_path / "go"
    with started("sleep", worker_marker, 15) as worker:
        assert wait_for_file(worker_marker)
        with started("supervisor", worker.pid, out, go) as supervisor:
            assert wait_for_file(out)
            assert is_beating(worker_marker)
            go.touch()
            assert supervisor.wait(timeout=30) == 0
        assert heartbeat_stopped(worker_marker), "the worker's interpreter outlived its daemon"
        worker.wait(timeout=10)


def test_a_closed_group_refuses_new_processes_s4_1() -> None:
    with get_platform().kill_on_close_group() as add:
        pass
    with pytest.raises(ValueError, match="closed"):
        add(NO_SUCH_PID)


def test_adding_a_missing_process_raises_os_error_s4_1() -> None:
    with get_platform().kill_on_close_group() as add, pytest.raises(OSError):
        add(NO_SUCH_PID)


# ---------------------------------------------------------------- priority (section 4.1)
def test_workers_run_at_below_normal_priority_s4_1(tmp_path: Path) -> None:
    marker = tmp_path / "worker.json"
    with started("sleep", marker, 15) as worker:
        assert wait_for_file(marker)
        get_platform().set_below_normal_priority(worker.pid)
        # worker.pid may be a venv launcher; the interpreter doing the work (its child) must be set too.
        interpreter = read_json(marker)["pid"]
        for pid in {worker.pid, interpreter}:
            assert psutil.Process(pid).nice() == psutil.BELOW_NORMAL_PRIORITY_CLASS, pid


def test_priority_of_a_missing_process_raises_os_error_s4_1() -> None:
    with pytest.raises(OSError):
        get_platform().set_below_normal_priority(NO_SUCH_PID)
