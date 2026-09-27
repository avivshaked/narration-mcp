"""``narration-admin daemon start | stop [--now] | status`` (design sections 4.1 and 7.1).

The daemon here is a status planted in the store: this test process stands in for it (its pid, a start
time after this process was created), so ``running_daemon`` says it runs. ``start`` goes through the
platform stand-in, which records the detached command instead of starting it. One Windows test runs a real
daemon in the foreground, as the test's own child, and ends it.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import psutil
import pytest

from narration.admin import daemon as admin_daemon
from narration.admin.cli import EXIT_FAILED, EXIT_OK
from narration.config import load_config
from narration.contracts.errors import NarrationError, UnsupportedPlatform
from narration.contracts.models import DaemonStatus, GpuStatus, WorkerInfo
from narration.platform.testing import StandInPlatform
from narration.store import NarrationStore
from narration.store.store import utc_iso

from .conftest import AdminRun, write_config

pytestmark = pytest.mark.timeout(120)


@pytest.fixture
def store(config_path: Path, platform: StandInPlatform) -> Iterator[NarrationStore]:
    config = load_config(config_path)
    with NarrationStore(config.server.store_root, platform) as s:
        yield s


def status(state: str = "idle", *, pid: int | None = None, alive: bool = True) -> DaemonStatus:
    """A status as a daemon writes it; with ``alive``, one ``running_daemon`` believes (this process)."""
    me = psutil.Process()
    return DaemonStatus(
        state=state,  # pyright: ignore[reportArgumentType]
        pid=me.pid if pid is None else pid,
        started_at=utc_iso(me.create_time() + (1.0 if alive else -3600.0)),
        workers=(WorkerInfo(role="qwen3", pid=me.pid),),
        current_job=None,
        gpu=GpuStatus(name="A GPU", total_mb=24000, free_mb=20000, in_use=True, holder="qwen", unload_in_s=42.0),
        est_drain_s=None,
        updated_at=utc_iso(time.time()),
    )


def answer_when_posted(
    store: NarrationStore, result: dict[str, Any] | None, *, then: DaemonStatus | None = None
) -> threading.Thread:
    """A thread that plays the daemon: it waits for a stop, answers it with ``result`` (None: does not answer)
    and then writes ``then``."""

    def run() -> None:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            pending = store.pending_commands()
            if pending:
                if result is not None:
                    store.complete_command(pending[0].command_id, result)
                if then is not None:
                    store.put_daemon_status(then)
                return
            time.sleep(0.02)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread


# ---------------------------------------------------------------- status
def test_status_with_no_store_says_no_daemon_runs_s4_1(admin: AdminRun, config_path: Path) -> None:
    ran = admin("--config", str(config_path), "daemon", "status")
    assert ran.code == EXIT_FAILED and "No daemon runs" in ran.out and "daemon start" in ran.out


def test_status_shows_the_running_daemon_and_what_it_holds_s4_1(
    admin: AdminRun, config_path: Path, store: NarrationStore
) -> None:
    store.put_daemon_status(status("busy"))
    ran = admin("--config", str(config_path), "daemon", "status")
    assert ran.code == EXIT_OK
    assert f"pid {os.getpid()}, busy" in ran.out
    assert "worker: qwen3" in ran.out and "loaded: qwen" in ran.out and "20000 of 24000 MB free" in ran.out


def test_status_as_json_says_whether_it_runs_s4_1(admin: AdminRun, config_path: Path, store: NarrationStore) -> None:
    store.put_daemon_status(status("idle"))
    ran = admin("--config", str(config_path), "daemon", "status", "--json")
    data = json.loads(ran.out)
    assert ran.code == EXIT_OK and data["running"] is True and data["status"]["state"] == "idle"
    store.put_daemon_status(status("stopped"))
    ran = admin("--config", str(config_path), "daemon", "status", "--json")
    assert ran.code == EXIT_FAILED and json.loads(ran.out)["running"] is False


@pytest.mark.parametrize(
    ("planted", "said"),
    [
        (status("stopped"), "The last one said 'stopped'"),
        (status("busy", alive=False), "went without saying it stopped"),
    ],
)
def test_status_of_a_daemon_that_is_gone_says_so_s4_1(
    admin: AdminRun, config_path: Path, store: NarrationStore, planted: DaemonStatus, said: str
) -> None:
    store.put_daemon_status(planted)
    ran = admin("--config", str(config_path), "daemon", "status")
    assert ran.code == EXIT_FAILED and said in ran.out


def test_status_of_a_torn_status_file_says_it_cannot_be_read_s4_1(
    admin: AdminRun, config_path: Path, store: NarrationStore
) -> None:
    path = store.layout.daemon_json_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"state": "idle", "pid"', encoding="utf-8")
    ran = admin("--config", str(config_path), "daemon", "status")
    assert ran.code == EXIT_FAILED and "cannot be read as a status" in ran.out


# ---------------------------------------------------------------- stop
def test_stop_posts_nothing_when_no_daemon_runs_s4_1(admin: AdminRun, config_path: Path, store: NarrationStore) -> None:
    """A daemon honours only stops posted after its launch, so one posted now would stop nothing and be
    answered ``stopped: false`` by the next daemon: none is posted (WP30's rule for WP37)."""
    store.put_daemon_status(status("busy", alive=False))
    ran = admin("--config", str(config_path), "daemon", "stop")
    assert ran.code == EXIT_OK and "no stop was posted" in ran.out
    assert store.pending_commands() == ()
    assert admin("--config", str(config_path), "daemon", "stop", "--now").code == EXIT_OK
    assert store.pending_commands() == ()


def test_stop_with_no_store_posts_nothing_and_creates_no_store_s4_1(admin: AdminRun, config_path: Path) -> None:
    ran = admin("--config", str(config_path), "daemon", "stop")
    assert ran.code == EXIT_OK and "nothing to stop" in ran.out
    assert not (config_path.parent / "store").exists()


@pytest.mark.parametrize(("flag", "kind"), [((), "stop"), (("--now",), "stop_now")])
def test_stop_posts_the_stop_to_a_running_daemon_s4_1(
    admin: AdminRun, config_path: Path, store: NarrationStore, flag: tuple[str, ...], kind: str
) -> None:
    store.put_daemon_status(status("busy"))
    ran = admin("--config", str(config_path), "daemon", "stop", *flag, "--wait", "0")
    assert ran.code == EXIT_OK and "Asked the daemon" in ran.out
    assert [c.kind for c in store.pending_commands()] == [kind]


def test_stop_waits_for_the_daemons_answer_s4_1(admin: AdminRun, config_path: Path, store: NarrationStore) -> None:
    store.put_daemon_status(status("busy"))
    player = answer_when_posted(store, {"stopped": True, "requeued": ["j1"]}, then=status("stopped"))
    ran = admin("--config", str(config_path), "daemon", "stop", "--wait", "10")
    player.join(timeout=10)
    assert ran.code == EXIT_OK
    assert "The daemon stopped. 1 job(s) went back to the queue" in ran.out


def test_a_stop_the_daemon_answers_stopped_false_is_reported_s4_1(
    admin: AdminRun, config_path: Path, store: NarrationStore
) -> None:
    store.put_daemon_status(status("idle"))
    player = answer_when_posted(store, {"stopped": False, "reason": "it was posted before this daemon was launched"})
    ran = admin("--config", str(config_path), "daemon", "stop", "--wait", "10")
    player.join(timeout=10)
    assert ran.code == EXIT_FAILED and "did not take the stop as its own" in ran.err


def test_a_daemon_that_exits_before_reading_the_stop_is_reported_s4_1(
    admin: AdminRun, config_path: Path, store: NarrationStore
) -> None:
    store.put_daemon_status(status("idle"))
    player = answer_when_posted(store, None, then=status("stopped"))
    ran = admin("--config", str(config_path), "daemon", "stop", "--wait", "10")
    player.join(timeout=10)
    assert ran.code == EXIT_OK and "exited before it read the stop" in ran.out


def test_a_stop_not_done_within_the_wait_says_what_to_do_s4_1(
    admin: AdminRun, config_path: Path, store: NarrationStore
) -> None:
    store.put_daemon_status(status("busy"))
    ran = admin("--config", str(config_path), "daemon", "stop", "--wait", "0.3")
    assert ran.code == EXIT_FAILED and "has not stopped within 0.3 s" in ran.err and "--now" in ran.err


# ---------------------------------------------------------------- start
def test_start_starts_a_detached_daemon_with_its_identity_marker_s4_1(
    admin: AdminRun, config_path: Path, platform: StandInPlatform
) -> None:
    ran = admin("--config", str(config_path), "daemon", "start", "--wait", "0")
    assert ran.code == EXIT_OK and "launcher pid 4242" in ran.out
    ((argv, cwd, _env),) = platform.spawned
    store_root = str(config_path.parent / "store")
    assert argv[1:5] == ["-P", "-m", "narration.daemon", "--store"]
    assert os.path.normcase(argv[5]) == os.path.normcase(store_root)
    assert argv[6:8] == ["--config", str(config_path.resolve())]
    assert "--launched-at" in argv
    assert os.path.normcase(str(cwd)) == os.path.normcase(store_root)


def test_start_waits_for_the_daemon_to_serve_and_says_when_it_does_not_s4_1(admin: AdminRun, config_path: Path) -> None:
    ran = admin("--config", str(config_path), "daemon", "start", "--wait", "0.3")
    assert ran.code == EXIT_FAILED and "has not said it serves within 0.3 s" in ran.err and "daemon.log" in ran.err


def test_start_leaves_a_serving_daemon_alone_s4_1(
    admin: AdminRun, config_path: Path, store: NarrationStore, platform: StandInPlatform
) -> None:
    store.put_daemon_status(status("idle"))
    ran = admin("--config", str(config_path), "daemon", "start")
    assert ran.code == EXIT_OK and "A daemon already serves" in ran.out
    assert platform.spawned == []


def test_start_refused_breakaway_offers_the_foreground_s4_1(
    admin: AdminRun, config_path: Path, platform: StandInPlatform
) -> None:
    platform.refuse_spawn = NarrationError("DAEMON_UNAVAILABLE", "breakaway from the job was refused")
    ran = admin("--config", str(config_path), "daemon", "start")
    assert ran.code == EXIT_FAILED
    assert "breakaway from the job was refused" in ran.err and "daemon start --foreground" in ran.err


@pytest.mark.parametrize(
    ("reason", "said"),
    [
        ("breakaway_refused", "forbids breakaway"),
        ("left_in_job", "forbids breakaway"),
        ("job_check_failed", "could not confirm"),
        (None, "forbids breakaway"),
    ],
)
def test_start_refused_names_only_the_cause_windows_established_s4_1(
    admin: AdminRun, config_path: Path, platform: StandInPlatform, reason: str | None, said: str
) -> None:
    details = None if reason is None else {"reason": reason}
    platform.refuse_spawn = NarrationError("DAEMON_UNAVAILABLE", "no detached start", details=details)
    ran = admin("--config", str(config_path), "daemon", "start")
    assert ran.code == EXIT_FAILED
    assert said in ran.err, ran.err
    assert "no Job Object at all" in ran.err, "the rule is stated"
    assert "daemon start --foreground" in ran.err, "the way out is offered in every case"
    assert ("could not confirm" in ran.err) is (reason == "job_check_failed"), "an unknown cause is not asserted"


def test_start_says_when_the_daemon_exits_for_want_of_work_s4_1(admin: AdminRun, config_path: Path) -> None:
    ran = admin("--config", str(config_path), "daemon", "start", "--wait", "0")
    assert ran.code == EXIT_OK
    assert "idle_exit_min" in ran.out and "15 min" in ran.out, ran.out  # the default [daemon] idle_exit_min


def test_start_on_an_unsupported_os_says_so_s4_1(admin: AdminRun, config_path: Path, platform: StandInPlatform) -> None:
    platform.refuse_spawn = UnsupportedPlatform("spawn_detached", "plan9")
    ran = admin("--config", str(config_path), "daemon", "start")
    assert ran.code == EXIT_FAILED and "runs on Windows only" in ran.err


def test_the_foreground_daemon_has_the_detached_ones_command_line_s4_1(
    config_path: Path, store: NarrationStore
) -> None:
    argv = admin_daemon.foreground_argv(store, config_path, python=Path("py"), launched_at=12.5)
    assert argv == [
        "py",
        "-P",
        "-m",
        "narration.daemon",
        "--store",
        str(store.root),
        "--config",
        str(config_path),
        "--launched-at",
        "12.5",
    ]


@pytest.mark.skipif(sys.platform != "win32", reason="the daemon runs on Windows only in v1")
def test_start_in_the_foreground_runs_a_real_daemon_until_it_exits_s4_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real daemon (the null runner, no work) run in the foreground exits when idle (``idle_exit_min = 0``).
    It is this test's own child, and is killed through its handle if it is still running at the end."""
    import io

    from narration.admin.__main__ import main

    config_path = write_config(tmp_path / "service", "[daemon]\nidle_exit_min = 0\n")
    started: list[subprocess.Popen[Any]] = []
    real_popen = subprocess.Popen

    def recording_popen(*args: Any, **kwargs: Any) -> subprocess.Popen[Any]:
        process = real_popen(*args, **kwargs)
        started.append(process)
        return process

    monkeypatch.setattr(subprocess, "Popen", recording_popen)
    out, err = io.StringIO(), io.StringIO()
    codes: list[int] = []
    runner = threading.Thread(
        target=lambda: codes.append(
            main(["--config", str(config_path), "daemon", "start", "--foreground"], out=out, err=err, environ={})
        ),
        daemon=True,
    )
    try:
        runner.start()
        runner.join(timeout=90)
        assert codes == [EXIT_OK], (out.getvalue(), err.getvalue())
        assert "Running the daemon" in out.getvalue()
        assert len(started) == 1 and started[0].returncode == 0
    finally:
        for process in started:
            if process.poll() is None:
                process.kill()  # our own child: we hold its handle, so its pid cannot have been reused
                process.wait(timeout=30)
