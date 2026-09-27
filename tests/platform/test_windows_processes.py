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

from ._support import (
    HOST_FORBIDS_BREAKAWAY,
    WINDOWS_ONLY,
    child_argv,
    heartbeat_stopped,
    host_lets_a_child_leave_every_job,
    is_beating,
    read_json,
    run_child,
    run_in_job,
    start_in_job,
    started,
    wait_for_file,
)

if sys.platform != "win32":
    raise pytest.skip.Exception(WINDOWS_ONLY, allow_module_level=True)

from narration.platform import _windows

pytestmark = pytest.mark.timeout(120)

NO_SUCH_PID = 0xFFFFFFFC
"""A pid no process has (Windows pids are small multiples of 4)."""

SUSPENDED_DETACHED = 0x01000000 | 0x00000008 | 0x00000200 | 0x00000004
"""``CREATE_BREAKAWAY_FROM_JOB | DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_SUSPENDED``."""

PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
SYNCHRONIZE = 0x00100000


class FakeSuspended:
    """A ``Popen`` stand-in for ``spawn_detached``'s child: records what was asked, and whether it was ended."""

    def __init__(self, argv: list[str], **kwargs: object) -> None:
        self.argv, self.kwargs = argv, kwargs
        self.pid = 4242
        self.killed = False
        self.returncode: int | None = None

    def kill(self) -> None:
        self.killed = True
        self.returncode = 1

    def wait(self, timeout: float | None = None) -> int:
        assert self.killed, "waited on a child that was not ended"
        return 1


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
    made: list[FakeSuspended] = []
    resumed: list[int] = []
    monkeypatch.setattr(
        _windows, "_DetachedPopen", lambda argv, **kwargs: made.append(FakeSuspended(argv, **kwargs)) or made[-1]
    )
    monkeypatch.setattr(_windows, "_in_job", lambda pid, job: False)
    monkeypatch.setattr(_windows, "_resume", resumed.append)
    pid = get_platform().spawn_detached(("python", "-m", "narration.daemon"), cwd=tmp_path, env={"A": "1"})
    assert pid == 4242
    (child,) = made
    assert child.argv == ["python", "-m", "narration.daemon"]
    assert child.kwargs == {
        "cwd": tmp_path,
        "env": {"A": "1"},
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "close_fds": True,
        "creationflags": SUSPENDED_DETACHED,
    }
    assert _windows.DETACHED_CREATION_FLAGS == (
        subprocess.CREATE_BREAKAWAY_FROM_JOB | subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    )
    assert resumed == [4242], "a daemon in no job is let run"
    assert not child.killed


def test_a_daemon_left_in_a_job_is_ended_before_it_runs_s4_1(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    made: list[FakeSuspended] = []
    monkeypatch.setattr(
        _windows, "_DetachedPopen", lambda argv, **kwargs: made.append(FakeSuspended(argv, **kwargs)) or made[-1]
    )
    monkeypatch.setattr(_windows, "_in_job", lambda pid, job: pid == 4242 and job is None)
    monkeypatch.setattr(_windows, "_resume", lambda pid: pytest.fail("a daemon still in a job was let run"))
    with pytest.raises(NarrationError) as info:
        get_platform().spawn_detached(["python"], cwd=tmp_path, env={})
    error = info.value
    assert error.code == codes.DAEMON_UNAVAILABLE
    assert error.retryable is True
    assert error.retry_after_s == _windows.DAEMON_RETRY_AFTER_S
    assert "narration-admin daemon start" in error.hint
    assert error.details is not None and error.details["reason"] == _windows.LEFT_IN_JOB
    assert set(error.details) == {"reason", "in_job", "job_allows_breakaway"}
    (child,) = made
    assert child.killed, "the suspended daemon was ended, so nothing ran inside the job"


def test_a_job_check_that_fails_ends_the_daemon_s4_1(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    made: list[FakeSuspended] = []

    def cannot_tell(pid: int, job: object) -> bool:
        raise OSError(None, "The parameter is incorrect", None, 87)

    monkeypatch.setattr(
        _windows, "_DetachedPopen", lambda argv, **kwargs: made.append(FakeSuspended(argv, **kwargs)) or made[-1]
    )
    monkeypatch.setattr(_windows, "_in_job", cannot_tell)
    monkeypatch.setattr(_windows, "_resume", lambda pid: pytest.fail("a daemon of unknown membership was let run"))
    with pytest.raises(NarrationError) as info:
        get_platform().spawn_detached(["python"], cwd=tmp_path, env={})
    error = info.value
    assert error.code == codes.DAEMON_UNAVAILABLE and error.retryable is True
    assert error.details is not None
    assert (error.details["reason"], error.details["winerror"]) == (_windows.JOB_CHECK_FAILED, 87)
    assert isinstance(error.__cause__, OSError)
    assert made[0].killed


@pytest.mark.parametrize("where", ["_in_job", "_resume"])
def test_an_interrupt_during_the_check_or_the_resume_ends_the_child_s4_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, where: str
) -> None:
    # Ctrl+C in narration-admin's main thread, say: whatever interrupts the check or the resume, no suspended
    # child is left behind, unreaped and never running.
    made: list[FakeSuspended] = []

    def interrupt(*args: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(
        _windows, "_DetachedPopen", lambda argv, **kwargs: made.append(FakeSuspended(argv, **kwargs)) or made[-1]
    )
    monkeypatch.setattr(_windows, "_in_job", interrupt if where == "_in_job" else (lambda pid, job: False))
    monkeypatch.setattr(_windows, "_resume", interrupt if where == "_resume" else (lambda pid: None))
    with pytest.raises(KeyboardInterrupt):
        get_platform().spawn_detached(["python"], cwd=tmp_path, env={})
    assert made[0].killed, f"the suspended child survived an interrupt in {where}"


def test_a_daemon_in_no_job_is_resumed_and_runs_s4_1(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # The success path with a real process, whatever this host's jobs: the membership question is answered
    # "no job" here, so the real suspended child must be resumed, run, and exit with its own code. (On a host
    # whose jobs forbid breakaway the child sits in the host's job meanwhile, as every test process does.)
    import _winapi

    monkeypatch.setattr(_windows, "_in_job", lambda pid, job: False)
    marker, go = tmp_path / "ran.json", tmp_path / "go"
    pid = get_platform().spawn_detached(child_argv("exit", marker, 7, go), cwd=tmp_path, env=dict(os.environ))
    assert wait_for_file(marker), "the suspended child was never resumed"
    seen = read_json(marker)
    assert pid in (seen["pid"], seen["ppid"])
    # The child waits for `go`, so it is alive here and this handle is its own: its exit code is read from it.
    handle = _winapi.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION | SYNCHRONIZE, False, pid)
    try:
        go.touch()
        assert _winapi.WaitForSingleObject(handle, 30_000) == _winapi.WAIT_OBJECT_0, "the child did not exit"
        assert _winapi.GetExitCodeProcess(handle) == 7, "the child exits with its own code"
    finally:
        _winapi.CloseHandle(handle)


def test_a_daemon_that_cannot_be_resumed_is_ended_s4_1(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    made: list[FakeSuspended] = []

    def cannot_resume(pid: int) -> None:
        raise OSError(f"cannot resume process {pid}")

    monkeypatch.setattr(
        _windows, "_DetachedPopen", lambda argv, **kwargs: made.append(FakeSuspended(argv, **kwargs)) or made[-1]
    )
    monkeypatch.setattr(_windows, "_in_job", lambda pid, job: False)
    monkeypatch.setattr(_windows, "_resume", cannot_resume)
    with pytest.raises(OSError, match="cannot resume process 4242"):
        get_platform().spawn_detached(["python"], cwd=tmp_path, env={})
    assert made[0].killed, "a daemon that stays suspended is not left behind"


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
        "reason": _windows.BREAKAWAY_REFUSED,
        "winerror": 5,
        "in_job": True,
        "job_allows_breakaway": False,
    }


@pytest.mark.parametrize("inner", ["silent", "breakaway"])
def test_a_daemon_that_would_stay_in_a_clients_job_is_refused_s4_1(tmp_path: Path, inner: str) -> None:
    # The topology the lead saw a daemon die in (spike k, KNOW): a client's kill-on-close job that forbids
    # breakaway (the MCP Python SDK's) around a job that allows it (the venv launcher's, silently). Windows
    # then accepts CREATE_BREAKAWAY_FROM_JOB, takes the child out of the inner job only, and leaves it in the
    # client's; before the fix it ran there and died with the client.
    out = tmp_path / "nested.json"
    client_job = _windows._JobObject(kill_on_close=True)  # pyright: ignore[reportPrivateUsage]
    try:
        run_in_job(client_job, "nested", out, inner)
        result = read_json(out)
    finally:
        client_job.close()
    assert "spawned_pid" not in result, "a daemon was let run inside the client's job"
    assert result["code"] == codes.DAEMON_UNAVAILABLE
    assert result["retryable"] is True
    assert result["retry_after_s"] == _windows.DAEMON_RETRY_AFTER_S
    assert result["details"] == {
        "reason": _windows.LEFT_IN_JOB,
        "in_job": True,
        "job_allows_breakaway": True,  # the innermost job's answer, which is why CreateProcess did not refuse
    }


def assert_refused_by_the_host(client: dict[str, object], marker: Path) -> None:
    """On a host whose own jobs forbid breakaway (``HOST_FORBIDS_BREAKAWAY``), a client stand-in's detached
    start is refused with ``left_in_job`` (its own job allows breakaway, so the refusal comes from the host's),
    and no daemon ever runs: the marker a daemon would write never appears."""
    refused = client.get("refused")
    assert isinstance(refused, dict), f"{HOST_FORBIDS_BREAKAWAY}; yet a daemon was let run: {client}"
    assert refused["code"] == codes.DAEMON_UNAVAILABLE
    assert refused["details"]["reason"] == _windows.LEFT_IN_JOB, refused
    assert refused["retry_after_s"] == _windows.DAEMON_RETRY_AFTER_S
    assert not wait_for_file(marker, timeout_s=3.0), "a daemon ran although its start was refused"


def test_a_daemon_leaves_nested_jobs_that_allow_breakaway_s4_1(tmp_path: Path) -> None:
    # libuv's job (Node.js's, measured; Claude Code's, we believe: kill-on-close, breakaway and silent
    # breakaway) around the venv launcher's: the daemon must end up in no job, and outlive both the client
    # and the client's job. On a host whose own jobs forbid breakaway it must be refused instead (the rule
    # is "in no Job Object at all"), and nothing must run.
    marker, go, out, cwd = tmp_path / "daemon.json", tmp_path / "go", tmp_path / "client.json", tmp_path / "cwd"
    cwd.mkdir()
    client_job = _windows._JobObject(kill_on_close=True, allow_breakaway=True, silent_breakaway=True)  # pyright: ignore[reportPrivateUsage]
    try:
        run_in_job(client_job, "client", marker, go, out, cwd, "mark-k", "nested")
        client = read_json(out)
        if host_lets_a_child_leave_every_job():
            assert "refused" not in client, client
            assert client["daemon_in_any_job"] is False, "the daemon's first process left every job"
            assert client_job.contains(client["daemon_pid"]) is False
    finally:
        client_job.close()  # the client's job closes, as when the client exits
    time.sleep(0.5)
    go.touch()
    if not host_lets_a_child_leave_every_job():
        assert_refused_by_the_host(client, marker)
        return
    assert wait_for_file(marker), "the daemon died with its client's job"
    seen = read_json(marker)
    assert client["daemon_pid"] in (seen["pid"], seen["ppid"])
    assert seen["mark"] == "mark-k"


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
    if not host_lets_a_child_leave_every_job():
        assert_refused_by_the_host(client, marker)
        return
    assert "refused" not in client, client
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


def test_a_job_says_which_processes_are_in_it_s4_1(tmp_path: Path) -> None:
    marker = tmp_path / "worker.json"
    job = _windows._JobObject(kill_on_close=True)  # pyright: ignore[reportPrivateUsage]
    try:
        worker = start_in_job(job, child_argv("sleep", marker, 15))
        try:
            assert wait_for_file(marker)
            interpreter = read_json(marker)["pid"]
            assert job.contains(worker.pid) is True
            assert job.contains(interpreter) is True, "a launcher's child is born into its jobs"
            assert job.contains(os.getpid()) is False
            with pytest.raises(OSError):
                job.contains(NO_SUCH_PID)
        finally:
            worker.kill()
            worker.wait(timeout=30)
    finally:
        job.close()
    with pytest.raises(ValueError, match="closed"):
        job.contains(os.getpid())


def test_a_silent_breakaway_job_keeps_the_members_children_out_s4_1(tmp_path: Path) -> None:
    # Like a venv launcher's own job, or libuv's: a member's child is born outside it, whether it asks or not.
    marker = tmp_path / "worker.json"
    job = _windows._JobObject(kill_on_close=True, silent_breakaway=True)  # pyright: ignore[reportPrivateUsage]
    try:
        worker = start_in_job(job, child_argv("sleep", marker, 15))
        try:
            assert wait_for_file(marker)
            interpreter = read_json(marker)["pid"]
            assert job.contains(worker.pid) is True
            if interpreter == worker.pid:
                pytest.skip("sys.executable is not a venv launcher, so the child started no process of its own")
            assert job.contains(interpreter) is False, "a child of a member breaks away silently"
        finally:
            worker.kill()
            worker.wait(timeout=30)
    finally:
        job.close()


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
