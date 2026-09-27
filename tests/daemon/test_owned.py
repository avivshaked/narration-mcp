"""What a process test may kill (AGENTS.md hard rule 7): only a daemon it can prove it started, and only
through the ``psutil.Process`` objects it captured then (``owned``; WP30 review, finding 1).

These run everywhere: the processes are this test's own children, which it ends in ``finally``.
"""

from __future__ import annotations

import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager

import psutil
import pytest

from narration.contracts.models import DaemonStatus, GpuStatus
from narration.store import NarrationStore
from narration.store.store import utc_iso

from .owned import capture_daemon, reattach, stop_detached

pytestmark = pytest.mark.timeout(120)

SLEEPER = "import time; time.sleep(120)"
# A launcher stand-in: starts the "daemon" (a sleeper), prints its own pid and the sleeper's, and waits.
LAUNCHER = (
    "import subprocess, sys\n"
    f"child = subprocess.Popen([sys.executable, '-c', {SLEEPER!r}])\n"
    "print(__import__('os').getpid(), child.pid, flush=True)\n"
    "child.wait()\n"
)


CLOCK_CATCH_UP_S = 5.0
"""How long ``a_time_after_creation`` waits for the wall clock to reach a creation time (a tick is 15.625 ms)."""


def a_time_after_creation(pid: int, *, wall: Callable[[], float] = time.time) -> float:
    """A reading of ``wall`` no earlier than the creation of process ``pid``: a time a status could honestly
    give as ``started_at``, as a real daemon's is (it reads its clock after its interpreter has started).

    Reading ``time.time()`` just after the child exists is not enough, and made these tests flaky on Windows.
    There, Python 3.12's ``time.time`` is ``GetSystemTimeAsFileTime``, which moves once per timer tick (15.625
    ms unless some program asks for finer), while psutil's ``create_time`` is exact to well under a
    millisecond. So a reading taken within a tick of the creation can name an earlier moment, and
    ``capture_daemon`` rightly refuses a status that says the daemon ran before it was created. This waits
    until the clock has passed the creation time.
    """
    created = psutil.Process(pid).create_time()
    deadline = time.monotonic() + CLOCK_CATCH_UP_S
    while (now := wall()) < created:
        if time.monotonic() >= deadline:
            raise AssertionError(f"the wall clock did not reach process {pid}'s creation in {CLOCK_CATCH_UP_S} s")
        time.sleep(0.001)
    return now


def status_of(pid: int, started_at: float) -> DaemonStatus:
    return DaemonStatus(
        state="idle",
        pid=pid,
        started_at=utc_iso(started_at),
        workers=(),
        current_job=None,
        gpu=GpuStatus(name=None, total_mb=None, free_mb=None, in_use=False, holder=None, unload_in_s=None),
        est_drain_s=None,
        updated_at=utc_iso(time.time()),
    )


@contextmanager
def child(code: str) -> Iterator[subprocess.Popen[str]]:
    """A Python child of this test, ended (with its tree) on the way out."""
    process = subprocess.Popen(
        [sys.executable, "-c", code], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, text=True
    )
    try:
        yield process
    finally:
        if process.poll() is None:
            # Safe by pid: this test still holds the child's handle (not waited), so its pid cannot be reused.
            try:
                root = psutil.Process(process.pid)
                tree = [root, *root.children(recursive=True)]
            except psutil.NoSuchProcess:
                tree = []
            for member in tree:
                try:
                    member.kill()
                except psutil.NoSuchProcess:
                    continue
            psutil.wait_procs(tree, timeout=15)
        process.wait(timeout=30)
        if process.stdout is not None:
            process.stdout.close()


def test_a_process_that_only_has_the_daemons_pid_is_never_captured(store: NarrationStore) -> None:
    """The review's case: the status names a pid that, by now, belongs to a process created after the daemon
    wrote its status (here a fresh child of this test, whose parent is the "launcher" pid given). The old
    rule (created after the test began, parent is the launcher) accepted it; it must be refused, and a stop
    must then kill nothing."""
    before = time.time() - 1.0
    with child(SLEEPER) as victim:
        stale = status_of(victim.pid, started_at=before)  # the "daemon" wrote this before the victim existed
        assert capture_daemon(stale, launcher_pid=psutil.Process().pid) is None
        stop_detached(store, None, wait_s=0.2)
        assert victim.poll() is None, "nothing that is not provably ours is killed"


def test_the_daemon_that_wrote_its_status_is_captured_with_its_launcher_and_killed_through_them() -> None:
    with child(LAUNCHER) as launcher:
        assert launcher.stdout is not None
        launcher_pid, daemon_pid = map(int, launcher.stdout.readline().split())
        status = status_of(daemon_pid, started_at=a_time_after_creation(daemon_pid))
        owned = capture_daemon(status, launcher_pid=launcher_pid)
        assert owned is not None and owned.daemon is not None and owned.launcher is not None
        assert (owned.daemon.pid, owned.launcher.pid) == (daemon_pid, launcher_pid)
        # A wrong launcher: this test process, which is alive and is neither the daemon nor its parent. (Not
        # ``launcher.pid + 1``: where pids are sequential, that is the daemon itself, which a capture accepts as
        # its own launcher.) The same status was just accepted, so only the launcher can be why it is refused.
        assert capture_daemon(status, launcher_pid=psutil.Process().pid) is None
        owned.kill()
        assert not owned.running()
        assert launcher.wait(timeout=30) is not None


def test_a_status_written_in_the_millisecond_the_daemon_started_proves_it() -> None:
    # started_at is cut to the millisecond, so it may name a moment just before the process was created.
    with child(LAUNCHER) as launcher:
        assert launcher.stdout is not None
        launcher_pid, daemon_pid = map(int, launcher.stdout.readline().split())
        created = psutil.Process(daemon_pid).create_time()
        owned = capture_daemon(status_of(daemon_pid, started_at=created), launcher_pid=launcher_pid)
        assert owned is not None and owned.daemon is not None and owned.daemon.pid == daemon_pid
        later = capture_daemon(status_of(daemon_pid, started_at=created - 0.002), launcher_pid=launcher_pid)
        assert later is None, "a process created after the status's millisecond is not the daemon"
        owned.kill()


def test_a_status_time_from_a_clock_a_tick_behind_the_creation_waits_for_it() -> None:
    """The Windows CI flake, pinned with a clock that lags: its first reading is a tick before the daemon's
    creation, which ``capture_daemon`` refuses; ``a_time_after_creation`` takes the first reading that is not."""
    with child(LAUNCHER) as launcher:
        assert launcher.stdout is not None
        launcher_pid, daemon_pid = map(int, launcher.stdout.readline().split())
        created = psutil.Process(daemon_pid).create_time()
        tick_behind = created - 0.015625
        assert capture_daemon(status_of(daemon_pid, started_at=tick_behind), launcher_pid=launcher_pid) is None
        readings = iter([tick_behind, created - 0.0005, created + 0.0002])
        started_at = a_time_after_creation(daemon_pid, wall=lambda: next(readings))
        assert started_at == created + 0.0002
        owned = capture_daemon(status_of(daemon_pid, started_at=started_at), launcher_pid=launcher_pid)
        assert owned is not None and owned.daemon is not None and owned.daemon.pid == daemon_pid
        owned.kill()


def test_a_recorded_identity_is_reattached_only_on_an_exact_creation_time() -> None:
    with child(LAUNCHER) as launcher:
        assert launcher.stdout is not None
        launcher_pid, daemon_pid = map(int, launcher.stdout.readline().split())
        owned = capture_daemon(status_of(daemon_pid, started_at=a_time_after_creation(daemon_pid)), launcher_pid)
        assert owned is not None
        identity = owned.identity()
        again = reattach(identity)
        assert again is not None and again.daemon is not None and again.daemon.pid == daemon_pid
        moved = {**identity, "daemon_created": float(identity["daemon_created"] or 0) + 1.0, "launcher_pid": None}
        assert reattach(moved) is None, "another creation time is another process"
        owned.kill()
        assert reattach(identity) is None, "gone"
