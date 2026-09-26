"""The daemon as a real process on Windows (sections 4, 4.1): the named mutex, the kill-on-close Job Object,
``stop_now``, and a detached start that outlives its session.

Every daemon here is started by the test (as its child, or through a session stand-in it starts) and is
stopped by the test in ``finally``, through the store and, if that fails, by killing it: only after checking
that the process is the one the test started (its parent is the launcher the test or its session got back,
and it was created after the test began). Worker pids come from the daemon's own ``run/daemon.json`` and
are checked to be descendants of that daemon before anything reads them.
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

from narration.contracts.models import DaemonStatus
from narration.daemon.sweep import read_status
from narration.store import NarrationStore

from .conftest import fake_env, make_job, wait_until

if sys.platform != "win32":
    raise pytest.skip.Exception(
        "the daemon's process mechanisms are implemented for Windows only in v1 (plan.md Q2)", allow_module_level=True
    )

from narration.platform import get_platform

pytestmark = pytest.mark.timeout(180)

SESSION = Path(__file__).with_name("_session.py")
FAKE = ["--fake-workers", "--runner", "narration.daemon.testing:FakeWorkerRunner"]
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
    return [
        sys.executable,
        "-m",
        "narration.daemon",
        "--store",
        str(service / "store"),
        "--config",
        str(service / "narration.toml"),
        "--poll-s",
        "0.05",
        *extra,
    ]


@contextlib.contextmanager
def daemon_child(service: Path, *extra: str, env: dict[str, str] | None = None) -> Iterator[subprocess.Popen[bytes]]:
    """A daemon started as this test's child; on the way out, stopped (``stop_now``), else killed."""
    process = subprocess.Popen(
        daemon_argv(service, *extra),
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
                kill_tree(process.pid)
                process.wait(timeout=30)


def kill_tree(pid: int) -> None:
    """Kill a process this test started, and its descendants."""
    try:
        root = psutil.Process(pid)
        tree = [root, *root.children(recursive=True)]
    except psutil.NoSuchProcess:
        return
    for member in tree:
        with contextlib.suppress(psutil.NoSuchProcess):
            member.kill()
    psutil.wait_procs(tree, timeout=15)


def descendants(pid: int) -> set[int]:
    try:
        return {p.pid for p in psutil.Process(pid).children(recursive=True)}
    except psutil.NoSuchProcess:
        return set()


def alive(pid: int) -> bool:
    try:
        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


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
        assert serving.pid in {first.pid} | descendants(first.pid), "the daemon records its own pid (section 4.1)"
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


# ---------------------------------------------------------------- kill-on-close and stop_now (sections 4, 4.1)
def test_workers_die_with_the_daemon_even_when_it_is_killed_s4(
    service: Path, real_store: NarrationStore, tmp_path: Path
) -> None:
    job = make_job(real_store, "The first line.", "The second line.", "The third line.")
    spec = {"faults": [{"kind": "delay", "seconds": 60, "when": {"text_contains": "second"}}]}
    with daemon_child(service, *FAKE, "--idle-exit-s", "120", env=fake_env(tmp_path, spec)) as child:
        wait_until(lambda: (j := real_store.get_job(job.job_id)) is not None and j.progress.segments_done == 1, 60)
        busy = wait_status(real_store, lambda s: s.state == "busy" and bool(s.workers), "a worker in flight")
        assert busy.pid is not None and busy.pid in descendants(child.pid)
        workers = {w.pid for w in busy.workers}
        assert workers <= descendants(busy.pid), "the workers are the daemon's own children"
        worker_tree = set().union(*({pid} | descendants(pid) for pid in workers))
        psutil.Process(busy.pid).kill()  # the daemon dies hard: no finally, no shutdown
        wait_until(lambda: not any(alive(pid) for pid in worker_tree), 20, "the workers to die with the daemon")
        child.wait(timeout=30)
    left = real_store.get_job(job.job_id)
    assert left is not None and left.status == "running", "a killed daemon gives nothing back itself"
    # The next daemon sweeps up: the job goes back to the queue.
    with daemon_child(service, "--idle-exit-s", "0.5") as nxt:
        assert nxt.wait(timeout=60) == 0
    after = real_store.get_job(job.job_id)
    assert after is not None and after.status == "queued"


def test_stop_now_ends_the_workers_and_leaves_no_partial_file_s4_1(
    service: Path, real_store: NarrationStore, tmp_path: Path
) -> None:
    job = make_job(real_store, "The first line.", "The second line.", "The third line.")
    spec = {"faults": [{"kind": "delay", "seconds": 60, "when": {"text_contains": "second"}}]}
    with daemon_child(service, *FAKE, "--idle-exit-s", "120", env=fake_env(tmp_path, spec)) as child:
        wait_until(lambda: (j := real_store.get_job(job.job_id)) is not None and j.progress.segments_done == 1, 60)
        busy = wait_status(real_store, lambda s: s.state == "busy" and bool(s.workers), "a worker in flight")
        assert busy.pid is not None
        workers = {w.pid for w in busy.workers}
        assert workers <= descendants(busy.pid)
        worker_tree = set().union(*({pid} | descendants(pid) for pid in workers))
        posted = real_store.post_command("stop_now")
        started = time.monotonic()
        assert child.wait(timeout=30) == 0
        assert time.monotonic() - started < 15.0, "stop_now does not wait out the 60 s segment"
    assert not any(alive(pid) for pid in worker_tree)
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
    if "code" in result:
        pytest.skip(f"this test runner's own Job Object forbids breakaway ({result})")
    launcher = int(result["spawned_pid"])
    status: DaemonStatus | None = None
    try:
        status = wait_status(real_store, lambda s: s.state == "idle" and s.pid is not None, "the detached daemon")
        assert status.pid is not None
        daemon = psutil.Process(status.pid)
        assert daemon.create_time() >= test_began - 1.0
        assert status.pid == launcher or daemon.ppid() == launcher, "the daemon the session started"
        # It serves after its session and the session's job are gone: a command and a job, through the store.
        done = real_store.wait_for_command(real_store.post_command("release_gpu").command_id, timeout_s=30)
        assert done is not None and done.result == {"released": False, "holder_before": None, "busy_job": None}
        job = make_job(real_store, "Spoken after the session ended.")
        wait_until(lambda: (j := real_store.get_job(job.job_id)) is not None and j.status == "completed", 60)
        # No console for the daemon (KNOW, spike g): it runs as pythonw.exe, and neither it nor its launcher has
        # a conhost.exe child. (Its workers do: CREATE_NO_WINDOW makes a console without a window.)
        assert daemon.name().lower() == "pythonw.exe"
        own = {launcher, status.pid}
        everything = psutil.Process(launcher).children(recursive=True) if status.pid != launcher else []
        consoles = [p.pid for p in [*everything, *daemon.children(recursive=True)] if p.name().lower() == "conhost.exe"]
        assert [pid for pid in consoles if psutil.Process(pid).ppid() in own] == []
    finally:
        _stop_detached(real_store, status, launcher, test_began)


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
    assert result["details"]["reason"] == "breakaway_refused"
    assert result["retry_after_s"] == 60.0
    time.sleep(1.0)
    assert read_status(real_store) is None, "no daemon was started, detached or not"


def _stop_detached(store: NarrationStore, status: DaemonStatus | None, launcher: int, began: float) -> None:
    """Stop the detached daemon this test's session started: a ``stop`` through the store, and if it does not
    exit, a kill of that process (checked to be the one the session started) and its tree."""
    posted = store.post_command("stop")
    store.wait_for_command(posted.command_id, timeout_s=30)
    pid = status.pid if status is not None and status.pid is not None else launcher
    deadline = time.monotonic() + 30
    while alive(pid) and time.monotonic() < deadline:
        time.sleep(0.1)
    if not alive(pid):
        return
    process = psutil.Process(pid)
    if process.create_time() >= began - 1.0 and (pid == launcher or process.ppid() == launcher):
        kill_tree(pid)
    with contextlib.suppress(psutil.NoSuchProcess):
        if alive(launcher) and psutil.Process(launcher).create_time() >= began - 1.0:
            kill_tree(launcher)
