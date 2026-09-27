"""The daemon as a real process on Windows (sections 4, 4.1): the named mutex, the kill-on-close Job Object,
``stop_now``, and a detached start that outlives its session.

Every daemon here is started by the test (as its child, or through a session stand-in it starts) and is
stopped by the test in ``finally``, through the store and, if that fails, by killing it. A test never kills
or reads a process by a bare pid (AGENTS.md hard rule 7): it acts only through ``psutil.Process`` objects
captured while their identity is proven (psutil then refuses a pid that another process has taken since):

- a daemon started as the test's child: from the child's own tree, while the test holds the child's handle
  (an unwaited ``Popen``), so its pid cannot have been reused;
- a detached daemon: ``owned.capture_daemon`` (created before the status it wrote, child of the launcher
  the session got back);
- workers: the captured daemon's descendants (psutil checks each child is younger than its parent).
"""

from __future__ import annotations

import contextlib
import json
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import psutil
import pytest

from narration.contracts import codes
from narration.contracts.models import DaemonStatus
from narration.daemon import start
from narration.daemon.sweep import read_status
from narration.store import NarrationStore
from narration.store.store import parse_iso

from .conftest import fake_env, make_job, wait_until
from .owned import OwnedDaemon, capture_daemon, stop_detached

if sys.platform != "win32":
    raise pytest.skip.Exception(
        "the daemon's process mechanisms are implemented for Windows only in v1 (plan.md Q2)", allow_module_level=True
    )

from narration.platform import _windows, get_platform
from tests.platform._support import HOST_FORBIDS_BREAKAWAY, host_lets_a_child_leave_every_job, start_in_job

pytestmark = pytest.mark.timeout(180)

SESSION = Path(__file__).with_name("_session.py")
FAKE = ["--fake-workers", "--runner", "narration.daemon.testing:FakeWorkerRunner"]
NULL_RUNNER = ["--runner", "narration.daemon.seam:NullRunner"]
TEMP_MARKS = (".tmp-", ".staging-", ".trash-")


@pytest.fixture
def service(tmp_path: Path) -> Path:
    root = tmp_path / "service"
    for name in ("qwen3tts", "qa"):
        (root / "workers" / name).mkdir(parents=True)
    (root / "narration.toml").write_text("[server]\nstore_root = 'store'\nmodels_root = 'models'\n", encoding="utf-8")
    return root


@pytest.fixture
def real_store(service: Path) -> Iterator[NarrationStore]:
    with NarrationStore(service / "store", get_platform()) as store:
        yield store


def daemon_argv(service: Path, *extra: str) -> list[str]:
    return start.daemon_argv(
        service / "store", service / "narration.toml", python=Path(sys.executable), extra=["--poll-s", "0.05", *extra]
    )


@contextlib.contextmanager
def daemon_child(
    service: Path, *extra: str, env: dict[str, str] | None = None, safe_path: bool = True
) -> Iterator[subprocess.Popen[bytes]]:
    """A daemon started as this test's child (by hand, without ``-P``, when ``safe_path`` is False); on the
    way out, stopped (``stop_now``), else killed."""
    argv = daemon_argv(service, *extra)
    process = subprocess.Popen(
        argv if safe_path else [a for a in argv if a != "-P"],
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        yield process
    finally:
        if process.poll() is None:
            with NarrationStore(service / "store", get_platform()) as store:
                store.post_command("stop_now")
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                tree = popen_tree(process)
                for member in tree:
                    with contextlib.suppress(psutil.NoSuchProcess):
                        member.kill()
                psutil.wait_procs(tree, timeout=15)
                process.wait(timeout=30)


def popen_tree(process: subprocess.Popen[bytes]) -> list[psutil.Process]:
    """A child of this test and its descendants. Call it only while the child is not yet waited for: the
    test then holds its handle, so its pid still names it."""
    try:
        root = psutil.Process(process.pid)
        return [root, *root.children(recursive=True)]
    except psutil.NoSuchProcess:
        return []


def daemon_in(process: subprocess.Popen[bytes], pid: int | None) -> psutil.Process:
    """The daemon interpreter ``pid`` inside this test's child ``process`` (a venv launcher runs it as its
    child)."""
    found = [p for p in popen_tree(process) if p.pid == pid]
    assert found, f"pid {pid} is not in the daemon this test started"
    return found[0]


def still_running(processes: list[psutil.Process]) -> list[psutil.Process]:
    """The ones still running (``is_running`` is False for a pid another process has taken since)."""
    running: list[psutil.Process] = []
    for process in processes:
        try:
            if process.is_running() and process.status() != psutil.STATUS_ZOMBIE:
                running.append(process)
        except psutil.NoSuchProcess:
            continue
    return running


def wait_status(
    store: NarrationStore, predicate: Callable[[DaemonStatus], bool], what: str, timeout_s: float = 60.0
) -> DaemonStatus:
    found: list[DaemonStatus] = []

    def check() -> bool:
        status = read_status(store)
        if status is not None and predicate(status):
            found.append(status)
            return True
        return False

    wait_until(check, timeout_s, what)
    return found[0]


def rendered(store: NarrationStore) -> int:
    renders = store.root / "renders"
    return sum(1 for shard in renders.iterdir() for _ in shard.iterdir()) if renders.is_dir() else 0


def temp_leftovers(root: Path) -> list[Path]:
    return [p for p in root.rglob("*") if any(p.name.startswith(mark) for mark in TEMP_MARKS)]


# ---------------------------------------------------------------- the singleton (section 4)
def test_a_second_daemon_for_the_same_store_does_not_start_s4(service: Path, real_store: NarrationStore) -> None:
    with daemon_child(service, "--idle-exit-s", "120") as first:
        serving = wait_status(real_store, lambda s: s.state == "idle" and s.pid is not None, "the first daemon")
        assert serving.pid in {p.pid for p in popen_tree(first)}, "the daemon records its own pid (section 4.1)"
        second = subprocess.run(
            daemon_argv(service, "--idle-exit-s", "120"), stdin=subprocess.DEVNULL, capture_output=True, timeout=60
        )
        assert second.returncode == 0, "it exits quietly"
        still = read_status(real_store)
        assert still is not None and (still.state, still.pid) == ("idle", serving.pid)
        assert first.poll() is None
        posted = real_store.post_command("stop")
        assert first.wait(timeout=30) == 0
        done = real_store.wait_for_command(posted.command_id, timeout_s=0)
        assert done is not None and done.result == {"stopped": True, "requeued": []}


@pytest.mark.parametrize("safe_path", [True, False])
def test_a_daemon_started_without_safe_path_warns_in_its_log_s17(
    service: Path, real_store: NarrationStore, safe_path: bool
) -> None:
    with daemon_child(service, safe_path=safe_path) as daemon:
        wait_status(real_store, lambda s: s.state == "idle" and s.pid is not None, "the daemon")
        real_store.post_command("stop")
        assert daemon.wait(timeout=30) == 0
    text = (real_store.layout.logs_dir() / "daemon.log").read_text(encoding="utf-8")
    assert ("started without -P" in text) is not safe_path


# ---------------------------------------------------------------- kill-on-close and stop_now (sections 4, 4.1)
def test_workers_die_with_the_daemon_even_when_it_is_killed_s4(
    service: Path, real_store: NarrationStore, tmp_path: Path
) -> None:
    job = make_job(real_store, "The first line.", "The second line.", "The third line.")
    spec = {"faults": [{"kind": "delay", "seconds": 60, "when": {"text_contains": "second"}}]}
    with daemon_child(service, *FAKE, "--idle-exit-s", "120", env=fake_env(tmp_path, spec)) as child:
        wait_until(lambda: (j := real_store.get_job(job.job_id)) is not None and j.progress.segments_done == 1, 60)
        busy = wait_status(real_store, lambda s: s.state == "busy" and bool(s.workers), "a worker in flight")
        daemon = daemon_in(child, busy.pid)
        workers = daemon.children(recursive=True)
        assert {w.pid for w in busy.workers} <= {p.pid for p in workers}, "the workers are the daemon's own"
        daemon.kill()  # the daemon dies hard: no finally, no shutdown
        wait_until(lambda: not still_running(workers), 20, "the workers to die with the daemon")
        child.wait(timeout=30)
    left = real_store.get_job(job.job_id)
    assert left is not None and left.status == "running", "a killed daemon gives nothing back itself"
    # The next daemon sweeps up: the job goes back to the queue. It runs no job engine (NULL_RUNNER), so what the
    # sweep did is what is left; the default runner would take the job again at once.
    with daemon_child(service, *NULL_RUNNER, "--idle-exit-s", "0.5") as nxt:
        assert nxt.wait(timeout=60) == 0
    after = real_store.get_job(job.job_id)
    assert after is not None and after.status == "queued"


def test_the_daemon_drives_the_job_engine_by_default_s4(service: Path, real_store: NarrationStore) -> None:
    # No --runner: the entry point loads the job engine (WP31). This installation has no QA models, so the
    # engine fails the job it takes with BACKEND_NOT_INSTALLED rather than render takes it cannot check.
    job = make_job(real_store, "The first line.")
    with daemon_child(service, "--idle-exit-s", "0.5") as child:
        assert child.wait(timeout=60) == 0
    after = real_store.get_job(job.job_id)
    assert after is not None and after.status == "failed" and after.error is not None
    assert after.error.code == codes.BACKEND_NOT_INSTALLED and after.error.hint
    assert rendered(real_store) == 0


def test_stop_now_ends_the_workers_and_leaves_no_partial_file_s4_1(
    service: Path, real_store: NarrationStore, tmp_path: Path
) -> None:
    job = make_job(real_store, "The first line.", "The second line.", "The third line.")
    spec = {"faults": [{"kind": "delay", "seconds": 60, "when": {"text_contains": "second"}}]}
    with daemon_child(service, *FAKE, "--idle-exit-s", "120", env=fake_env(tmp_path, spec)) as child:
        wait_until(lambda: (j := real_store.get_job(job.job_id)) is not None and j.progress.segments_done == 1, 60)
        busy = wait_status(real_store, lambda s: s.state == "busy" and bool(s.workers), "a worker in flight")
        workers = daemon_in(child, busy.pid).children(recursive=True)
        assert {w.pid for w in busy.workers} <= {p.pid for p in workers}
        posted = real_store.post_command("stop_now")
        started = time.monotonic()
        assert child.wait(timeout=30) == 0
        assert time.monotonic() - started < 15.0, "stop_now does not wait out the 60 s segment"
    assert still_running(workers) == []
    after = real_store.get_job(job.job_id)
    assert after is not None and (after.status, after.progress.segments_done) == ("queued", 1)
    assert rendered(real_store) == 1
    assert temp_leftovers(real_store.root) == []
    assert not (real_store.root / "scratch" / "daemon-fake" / job.job_id).exists()
    report = real_store.verify()
    assert (report["missing"], report["mismatched"], report["errors"]) == ([], [], [])
    done = real_store.wait_for_command(posted.command_id, timeout_s=0)
    assert done is not None and done.result == {"stopped": True, "requeued": [job.job_id]}
    final = read_status(real_store)
    assert final is not None and final.state == "stopped"


# ---------------------------------------------------------------- detached start (section 4.1; spike g)
def test_the_launch_time_decides_which_stops_a_real_daemon_honours_s4_1(
    service: Path, real_store: NarrationStore
) -> None:
    earlier = real_store.post_command("stop")
    launched = parse_iso(earlier.requested_at) - 5.0  # a launcher that started the daemon before the stop
    with daemon_child(service, "--launched-at", repr(launched)) as honouring:
        assert honouring.wait(timeout=60) == 0, "the stop was asked after its launch"
    done = real_store.wait_for_command(earlier.command_id, timeout_s=0)
    assert done is not None and done.result == {"stopped": True, "requeued": []}
    old = real_store.post_command("stop")
    time.sleep(0.05)
    with daemon_child(service) as serving:  # launched after the stop, as it began to run
        answered = real_store.wait_for_command(old.command_id, timeout_s=60)
        assert answered is not None and answered.result is not None and answered.result["stopped"] is False
        assert serving.poll() is None, "it keeps serving"


def test_a_detached_daemon_outlives_its_session_s4_1(service: Path, real_store: NarrationStore, tmp_path: Path) -> None:
    out = tmp_path / "session.json"
    test_began = time.time()
    session = subprocess.run(
        [
            sys.executable,
            str(SESSION),
            str(service / "store"),
            str(service / "narration.toml"),
            str(out),
            "job",  # the session runs in a kill-on-close Job Object that allows breakaway
            *FAKE,
            "--idle-exit-s",
            "120",
            "--poll-s",
            "0.05",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        timeout=60,
    )
    assert session.returncode == 0, session.stderr
    result = json.loads(out.read_text(encoding="utf-8"))
    if not host_lets_a_child_leave_every_job():
        # The rule is "in no Job Object at all": on such a host the start is refused, and no daemon runs.
        assert result.get("code") == "DAEMON_UNAVAILABLE", f"{HOST_FORBIDS_BREAKAWAY}; yet: {result}"
        assert result["details"]["reason"] == _windows.LEFT_IN_JOB, result
        time.sleep(1.0)
        assert read_status(real_store) is None, "the daemon was ended before it wrote anything"
        assert not (real_store.layout.logs_dir() / "daemon.log").exists(), "it never ran"
        return
    assert "code" not in result, result
    launcher = int(result["spawned_pid"])
    owned: OwnedDaemon | None = None
    try:
        status = wait_status(real_store, lambda s: s.state == "idle" and s.pid is not None, "the detached daemon")
        owned = capture_daemon(status, launcher)
        assert owned is not None and owned.daemon is not None, "the daemon is the one the session started"
        daemon = owned.daemon
        assert daemon.create_time() >= test_began - 1.0
        # It serves after its session and the session's job are gone: a command and a job, through the store.
        done = real_store.wait_for_command(real_store.post_command("release_gpu").command_id, timeout_s=30)
        assert done is not None and done.result == {"released": False, "holder_before": None, "busy_job": None}
        job = make_job(real_store, "Spoken after the session ended.")
        wait_until(lambda: (j := real_store.get_job(job.job_id)) is not None and j.status == "completed", 60)
        # No console for the daemon (KNOW, spike g): it runs as pythonw.exe, and neither it nor its launcher has
        # a conhost.exe child. (Its workers do: CREATE_NO_WINDOW makes a console without a window.)
        assert daemon.name().lower() == "pythonw.exe"
        consoles = [q.pid for root in owned.roots() for q in root.children() if q.name().lower() == "conhost.exe"]
        assert consoles == [], "neither the daemon nor its launcher has a console of its own"
    finally:
        stop_detached(real_store, owned)


def test_a_session_whose_job_forbids_breakaway_starts_no_daemon_s4_1(
    service: Path, real_store: NarrationStore, tmp_path: Path
) -> None:
    out = tmp_path / "session.json"
    session = subprocess.run(
        [
            sys.executable,
            str(SESSION),
            str(service / "store"),
            str(service / "narration.toml"),
            str(out),
            "no-breakaway",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        timeout=60,
    )
    assert session.returncode == 0, session.stderr
    result = json.loads(out.read_text(encoding="utf-8"))
    assert result["code"] == "DAEMON_UNAVAILABLE"
    assert result["details"]["reason"] == _windows.BREAKAWAY_REFUSED
    assert result["retry_after_s"] == 60.0
    time.sleep(1.0)
    assert read_status(real_store) is None, "no daemon was started, detached or not"


def test_a_session_in_a_clients_job_around_a_launchers_starts_no_daemon_s4_1(
    service: Path, real_store: NarrationStore, tmp_path: Path
) -> None:
    # The lead's case (spike k, KNOW): the MCP Python SDK's kill-on-close job, which forbids breakaway, around
    # the venv launcher's job, which allows it silently. Windows accepts the breakaway from the inner job and
    # leaves the daemon in the client's, where it died with the client. Now it is ended before it runs.
    out = tmp_path / "session.json"
    client_job = _windows._JobObject(kill_on_close=True)  # pyright: ignore[reportPrivateUsage]
    try:
        session = start_in_job(
            client_job,
            [sys.executable, str(SESSION), str(service / "store"), str(service / "narration.toml"), str(out), "nested"],
        )
        assert session.wait(timeout=60) == 0
        assert out.exists(), "the session wrote its result"
        result = json.loads(out.read_text(encoding="utf-8"))
    finally:
        client_job.close()
    assert "spawned_pid" not in result, "a daemon was let run inside the client's job"
    assert result["code"] == "DAEMON_UNAVAILABLE"
    assert result["details"]["reason"] == _windows.LEFT_IN_JOB
    assert result["retry_after_s"] == 60.0
    time.sleep(1.0)
    assert read_status(real_store) is None, "the daemon was ended before it wrote anything"
    assert not (real_store.layout.logs_dir() / "daemon.log").exists(), "it never ran"
