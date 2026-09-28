"""The daemon's control loop (sections 4, 4.1, 7.6): a ``Daemon`` on a thread, the platform stand-in, fake workers.

These run on every OS. The real Windows mechanisms (the named mutex, the Job Object, detached start) are
exercised with the real entry point in ``test_process.py``.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from narration.backend.service import operator_stop
from narration.contracts.models import DaemonStatus, JobRecord
from narration.contracts.names import STOP_ERROR, STOP_IDLE, STOP_INTERRUPTED, STOP_OPERATOR
from narration.daemon.seam import NullRunner, RunnerHost, ShutdownReason, return_job
from narration.daemon.service import EXIT_ERROR, EXIT_OK, STALE_STOP_REASON, Daemon
from narration.daemon.start import ensure_daemon
from narration.daemon.supervisor import WorkerSupervisor
from narration.daemon.sweep import read_status
from narration.daemon.testing import SCRATCH_DIR, FakeWorkerRunner
from narration.platform.testing import StandInPlatform
from narration.store import NarrationStore
from narration.store.store import parse_iso, utc_iso

from .conftest import (
    UNREADABLE_STATUSES,
    WAIT_S,
    DaemonFactory,
    make_job,
    our_children,
    plant_status,
    running,
    wait_until,
)
from .test_seam_and_sweep import NO_SUCH_PID, status

pytestmark = pytest.mark.timeout(120)

TEMP_MARKS = (".tmp-", ".staging-", ".trash-")


# ---------------------------------------------------------------- scripted runners
class LoadOnce:
    """Loads the qwen group once (as a job would), then finds no work."""

    def __init__(self) -> None:
        self.loaded = False

    def step(self, host: RunnerHost) -> bool:
        if self.loaded:
            return False
        host.workers.load("qwen", {"device": "cpu"}, gpu=True, timeout_s=30)
        self.loaded = True
        return True

    def has_work(self, host: RunnerHost) -> bool:
        return not self.loaded

    def shutdown(self, host: RunnerHost, reason: ShutdownReason) -> None:
        return None


class HoldJob:
    """Claims a job, loads the qwen group, and holds the job (one short step at a time) until stopped."""

    def __init__(self) -> None:
        self.job: JobRecord | None = None
        self.reasons: list[ShutdownReason] = []

    def step(self, host: RunnerHost) -> bool:
        if self.job is None:
            self.job = host.store.claim_next_job(host.holder)
            if self.job is None:
                return False
            host.job_started(self.job)
            host.job_phase("loading_model")
            host.workers.load("qwen", {"device": "cpu"}, gpu=True, timeout_s=30)
            host.job_phase("rendering")
            return True
        host.sleep(0.05)
        return True

    def has_work(self, host: RunnerHost) -> bool:
        return self.job is not None or any(job.status == "queued" for job in host.store.queued_jobs())

    def shutdown(self, host: RunnerHost, reason: ShutdownReason) -> None:
        self.reasons.append(reason)
        if self.job is not None:
            return_job(host.store, self.job.job_id, reason=reason)
            host.job_finished()


class KeepsJob(HoldJob):
    """Like ``HoldJob``, but its shutdown forgets to give the job back (the daemon's safety net must)."""

    def shutdown(self, host: RunnerHost, reason: ShutdownReason) -> None:
        self.reasons.append(reason)


class RaisesTwice:
    def __init__(self) -> None:
        self.calls = 0

    def step(self, host: RunnerHost) -> bool:
        self.calls += 1
        if self.calls <= 2:
            raise RuntimeError("a bug in the runner")
        return False

    def has_work(self, host: RunnerHost) -> bool:
        return False

    def shutdown(self, host: RunnerHost, reason: ShutdownReason) -> None:
        return None


class LooksAgain(FakeWorkerRunner):
    """The fake runner, with hooks around the daemon's last look before an idle exit (``has_work``)."""

    def __init__(self, *, before: threading.Event | None = None, after: threading.Event | None = None) -> None:
        super().__init__()
        self.before = before
        """Set by the test when the look may be taken (a job may have been queued meanwhile)."""
        self.after = after
        """Set by the test when the daemon may go on once it has looked."""
        self.looked = threading.Event()
        self.states_seen: list[str | None] = []
        self.answers: list[bool] = []

    def has_work(self, host: RunnerHost) -> bool:
        seen = read_status(host.store)
        self.states_seen.append(seen.state if seen is not None else None)
        if self.before is not None:
            assert self.before.wait(WAIT_S), "the test never let the look be taken"
        answer = super().has_work(host)
        self.answers.append(answer)
        self.looked.set()
        if self.after is not None:
            assert self.after.wait(WAIT_S), "the test never let the daemon go on"
        return answer


class LooksBadly(NullRunner):
    """A runner whose ``has_work`` fails once (a bug), then finds no work."""

    def __init__(self) -> None:
        self.calls = 0

    def has_work(self, host: RunnerHost) -> bool:
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("a bug in has_work")
        return False


def rendered(store: NarrationStore) -> int:
    renders = store.root / "renders"
    return sum(1 for shard in renders.iterdir() for _ in shard.iterdir()) if renders.is_dir() else 0


def temp_leftovers(root: Path) -> list[Path]:
    return [p for p in root.rglob("*") if any(p.name.startswith(mark) for mark in TEMP_MARKS)]


def job_status(store: NarrationStore, job: JobRecord) -> Any:
    found = store.get_job(job.job_id)
    assert found is not None
    return found


# ---------------------------------------------------------------- start, singleton, idle exit (sections 4, 4.1)
def test_the_daemon_records_its_own_pid_and_exits_when_idle_s4(
    run_daemon: DaemonFactory, store: NarrationStore
) -> None:
    started = time.monotonic()
    daemon = run_daemon(NullRunner(), idle_exit_s=1.0)
    serving = daemon.wait_serving()
    assert serving.pid == os.getpid(), "the daemon's own pid, not a launcher's"
    assert serving.state == "idle" and serving.started_at is not None
    assert daemon.join() == EXIT_OK
    assert time.monotonic() - started >= 1.0
    final = store.get_daemon_status()
    assert final is not None and (final.state, final.pid) == ("stopped", None)


def test_a_second_daemon_for_the_same_store_exits_quietly_s4(
    run_daemon: DaemonFactory, store: NarrationStore, platform: StandInPlatform
) -> None:
    serving = status("idle", NO_SUCH_PID, utc_iso(time.time()))
    with platform.hold(store.root):
        store.put_daemon_status(serving)
        started = time.monotonic()
        daemon = run_daemon(NullRunner(), takeover_wait_s=20.0)
        assert daemon.join(10) == EXIT_OK
        assert time.monotonic() - started < 5.0, "a holder that serves is not waited for"
    assert store.get_daemon_status() == serving, "it wrote nothing"


def test_a_daemon_takes_over_from_one_that_is_stopping_s4(
    run_daemon: DaemonFactory, store: NarrationStore, platform: StandInPlatform
) -> None:
    with platform.hold(store.root):
        store.put_daemon_status(status("stopping", NO_SUCH_PID, utc_iso(time.time())))
        daemon = run_daemon(NullRunner(), takeover_wait_s=20.0)
        time.sleep(0.5)
        assert daemon.thread.is_alive(), "it waits while the other says it is stopping"
    serving = daemon.wait_serving()
    assert serving.state == "idle"
    assert daemon.command("stop").result == {"stopped": True, "requeued": []}
    assert daemon.join() == EXIT_OK


def test_the_sweep_runs_on_start_s4_1(run_daemon: DaemonFactory, store: NarrationStore) -> None:
    left = make_job(store, "Left running.", status="running")
    store.put_daemon_status(status("busy", NO_SUCH_PID, utc_iso(time.time() - 60)))
    daemon = run_daemon(NullRunner())
    daemon.wait_serving()
    assert job_status(store, left).status == "queued"
    daemon.command("stop")
    assert daemon.join() == EXIT_OK


@pytest.mark.parametrize("content", UNREADABLE_STATUSES)
def test_an_unreadable_status_is_swept_as_a_daemon_that_died_s4_1(
    run_daemon: DaemonFactory, store: NarrationStore, content: bytes
) -> None:
    left = make_job(store, "Left running.", status="running")
    plant_status(store, content)
    daemon = run_daemon(NullRunner())
    daemon.wait_serving()
    assert job_status(store, left).status == "queued"
    daemon.command("stop")
    assert daemon.join() == EXIT_OK


@pytest.mark.parametrize("left", ["stopped", "no status", "unreadable"])
def test_a_waiting_daemon_takes_over_from_a_holder_that_has_not_said_it_serves_s4(
    run_daemon: DaemonFactory, store: NarrationStore, platform: StandInPlatform, left: str
) -> None:
    # An exiting daemon writes "stopped" before it releases the singleton; a missing or torn status says
    # nothing either way. None of these is a daemon that serves, so the new one waits and takes over.
    with platform.hold(store.root):
        if left == "stopped":
            store.put_daemon_status(status("stopped", None, None))
        elif left == "unreadable":
            plant_status(store, b"")
        daemon = run_daemon(NullRunner(), takeover_wait_s=20.0)
        time.sleep(0.5)
        assert daemon.thread.is_alive(), "it waits for the holder to go"
    assert daemon.wait_serving().state == "idle", "it took over"
    daemon.command("stop")
    assert daemon.join() == EXIT_OK


def test_a_waiting_daemon_gives_up_at_takeover_wait_s_and_writes_nothing_s4(
    run_daemon: DaemonFactory, store: NarrationStore, platform: StandInPlatform
) -> None:
    planted = plant_status(store, b"")
    with platform.hold(store.root):
        started = time.monotonic()
        daemon = run_daemon(NullRunner(), takeover_wait_s=0.5)
        assert daemon.join(10) == EXIT_OK
        assert 0.5 <= time.monotonic() - started < 5.0, "it waits takeover_wait_s, and no longer"
    assert planted.read_bytes() == b"", "it wrote nothing"


def test_a_stop_posted_before_the_daemon_was_launched_is_answered_and_not_honoured_s4_1(
    run_daemon: DaemonFactory, store: NarrationStore
) -> None:
    old = store.post_command("stop")  # asked when no daemon ran
    time.sleep(0.02)
    job = make_job(store, "Queued after the old stop.")
    daemon = run_daemon(FakeWorkerRunner(), idle_exit_s=60.0)
    done = store.wait_for_command(old.command_id, timeout_s=WAIT_S)
    assert done is not None and done.result == {"stopped": False, "reason": STALE_STOP_REASON}
    wait_until(lambda: job_status(store, job).status == "completed", what="the daemon to keep serving")
    assert daemon.thread.is_alive()
    assert daemon.command("stop").result == {"stopped": True, "requeued": []}, "a stop asked of it is honoured"
    assert daemon.join() == EXIT_OK


def test_a_stop_posted_after_launch_stops_the_daemon_before_any_work_s4_1(
    run_daemon: DaemonFactory, store: NarrationStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = make_job(store, "Never taken.")
    written: list[str] = []
    put = store.put_daemon_status

    def record(status: DaemonStatus) -> None:
        written.append(status.state)
        put(status)

    monkeypatch.setattr(store, "put_daemon_status", record)
    runner = HoldJob()
    daemon = run_daemon(runner, start=False)  # launched, not yet running
    time.sleep(0.02)
    posted = store.post_command("stop")
    daemon.start()
    assert daemon.join() == EXIT_OK
    done = store.wait_for_command(posted.command_id, timeout_s=0)
    assert done is not None and done.result == {"stopped": True, "requeued": []}
    assert runner.job is None and job_status(store, job).status == "queued", "no step was taken"
    assert written and "idle" not in written and written[-1] == "stopped", "it never said it serves"


def test_a_stop_posted_while_waiting_for_the_singleton_is_honoured_s4_1(
    run_daemon: DaemonFactory, store: NarrationStore, platform: StandInPlatform
) -> None:
    # A stop is for the service, not for one daemon process: B, which holds the singleton when it reads the
    # stop, completes it by stopping, though the stop was posted before B took the singleton.
    with platform.hold(store.root):
        store.put_daemon_status(status("stopping", NO_SUCH_PID, utc_iso(time.time())))
        daemon = run_daemon(NullRunner(), takeover_wait_s=20.0)
        time.sleep(0.3)
        assert daemon.thread.is_alive(), "it waits while the holder says it is stopping"
        posted = store.post_command("stop")
    assert daemon.join() == EXIT_OK
    done = store.wait_for_command(posted.command_id, timeout_s=0)
    assert done is not None and done.result == {"stopped": True, "requeued": []}
    final = read_status(store)
    assert final is not None and final.state == "stopped"


def test_a_stop_the_exiting_daemon_answered_stops_the_one_waiting_to_take_over_s4_1(
    run_daemon: DaemonFactory, store: NarrationStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    written: list[tuple[str, str | None]] = []
    put = store.put_daemon_status

    def record(status: DaemonStatus) -> None:
        written.append((status.state, status.started_at))
        put(status)

    monkeypatch.setattr(store, "put_daemon_status", record)
    go_on = threading.Event()
    first = LooksAgain(after=go_on)
    a = run_daemon(first, idle_exit_s=0.3)
    assert first.looked.wait(WAIT_S) and first.answers == [False]  # A is exiting: it said stopping
    job = make_job(store, "Queued while A exits.")
    second = HoldJob()
    b = run_daemon(second, takeover_wait_s=20.0)
    time.sleep(0.3)
    assert b.thread.is_alive(), "B waits for A to go"
    posted = store.post_command("stop")  # an operator's stop, after B's launch
    wait_until(lambda: a.daemon.stopping == "segment", what="A to read the stop")
    go_on.set()
    assert a.join() == EXIT_OK
    answered = store.wait_for_command(posted.command_id, timeout_s=0)
    assert answered is not None and answered.result == {"stopped": True, "requeued": []}, "A answered it"
    assert b.join() == EXIT_OK, "B stops too"
    assert second.job is None and job_status(store, job).status == "queued", "B claimed no job"
    b_started = b.daemon.started_at
    assert b_started is not None, "B took over"
    assert ("idle", b_started) not in written, "B never said it serves"
    assert ("stopping", b_started) in written and written[-1][0] == "stopped", "B stopped as a stop does"
    final = store.get_daemon_status()
    assert final is not None and (final.stop_reason, final.started_at) == (STOP_OPERATOR, b_started), "B's own"
    found = operator_stop(store, job_status(store, job))
    assert found is not None and found.command_id == posted.command_id, "the job waits for the next start"


def test_a_stop_answered_stopped_false_does_not_stop_a_waiting_daemon_s4_1(
    run_daemon: DaemonFactory, store: NarrationStore
) -> None:
    # Lead's ruling: only an answer of stopped: true holds for the service. A stop that was stale for the
    # daemon that read it, and answered stopped: false, does not stop the one that takes over.
    posted = store.post_command("stop")
    store.complete_command(posted.command_id, {"stopped": False, "reason": STALE_STOP_REASON})
    job = make_job(store, "Run by the daemon that takes over.")
    daemon = run_daemon(FakeWorkerRunner(), launched_at=parse_iso(posted.requested_at) - 0.5)
    wait_until(lambda: job_status(store, job).status == "completed", what="the daemon to serve")
    daemon.command("stop")
    assert daemon.join() == EXIT_OK


def test_an_answered_stop_from_the_future_does_not_stop_new_daemons_s4_1(
    run_daemon: DaemonFactory, store: NarrationStore, caplog: pytest.LogCaptureFixture
) -> None:
    # The wall clock stepped back an hour after a stop was posted and answered: its stamp is in the future,
    # so it is not trusted. Before the fix, every new daemon stopped at once until the clock caught up.
    posted = store.post_command("stop")
    store.complete_command(posted.command_id, {"stopped": True, "requeued": []})

    def an_hour_back() -> float:
        return time.time() - 3600.0

    with caplog.at_level(logging.INFO, logger="narration.daemon.service"):
        for _ in range(2):
            daemon = run_daemon(NullRunner(), wall=an_hour_back)
            assert daemon.wait_serving().state == "idle", "it serves"
            daemon.command("stop")  # posted after its launch, as the store's clock sees it; pending: honoured
            assert daemon.join() == EXIT_OK
    assert any("later than now" in record.getMessage() for record in caplog.records)


def test_an_answered_stop_posted_before_the_launch_does_not_stop_the_daemon_s4_1(
    run_daemon: DaemonFactory, store: NarrationStore
) -> None:
    old = store.post_command("stop")
    store.complete_command(old.command_id, {"stopped": True, "requeued": []})  # answered by a daemon since gone
    time.sleep(0.02)
    daemon = run_daemon(NullRunner())
    assert daemon.wait_serving().state == "idle"
    daemon.command("stop")
    assert daemon.join() == EXIT_OK


@pytest.mark.parametrize(("after_stamp_s", "honoured"), [(0.0005, True), (0.0015, False)])
def test_an_answered_stop_is_judged_by_its_millisecond_against_the_launch_s4_1(
    run_daemon: DaemonFactory, store: NarrationStore, after_stamp_s: float, honoured: bool
) -> None:
    posted = store.post_command("stop_now")
    store.complete_command(posted.command_id, {"stopped": True, "requeued": []})
    runner = HoldJob()
    job = make_job(store, "Queued before the launch.")
    daemon = run_daemon(runner, launched_at=parse_iso(posted.requested_at) + after_stamp_s)
    if honoured:
        assert daemon.join() == EXIT_OK
        assert runner.job is None and job_status(store, job).status == "queued", "no work was done"
    else:
        daemon.wait_status(lambda s: s.state == "busy", "the daemon to take the job")
        daemon.command("stop")
        assert daemon.join() == EXIT_OK


@pytest.mark.parametrize(("after_stamp_s", "honoured"), [(0.0005, True), (0.0015, False), (-0.5, True)])
def test_a_stop_is_judged_by_its_millisecond_against_the_launch_s4_1(
    run_daemon: DaemonFactory, store: NarrationStore, after_stamp_s: float, honoured: bool
) -> None:
    # requested_at is cut to the millisecond: a daemon launched within the stop's millisecond may have been
    # launched before the stop, so the stop is for it; one launched after that millisecond is not.
    posted = store.post_command("stop")
    daemon = run_daemon(NullRunner(), launched_at=parse_iso(posted.requested_at) + after_stamp_s)
    done = store.wait_for_command(posted.command_id, timeout_s=WAIT_S)
    assert done is not None and done.result is not None and done.result["stopped"] is honoured
    if not honoured:
        daemon.wait_serving()
        daemon.command("stop")
    assert daemon.join() == EXIT_OK


def test_a_daemon_gives_up_on_a_holder_that_stays_stopping_s4(
    run_daemon: DaemonFactory, store: NarrationStore, platform: StandInPlatform
) -> None:
    with platform.hold(store.root):
        planted = status("stopping", NO_SUCH_PID, utc_iso(time.time()))
        store.put_daemon_status(planted)
        started = time.monotonic()
        daemon = run_daemon(NullRunner(), takeover_wait_s=0.5)
        assert daemon.join(10) == EXIT_OK
        assert time.monotonic() - started < 5.0, "it waits takeover_wait_s at most"
        assert read_status(store) == planted, "it wrote nothing"


# ---------------------------------------------------------------- the idle exit strands no job (seam, "The idle exit")
def test_a_job_queued_as_the_daemon_turns_to_exit_is_run_by_it_s4(
    run_daemon: DaemonFactory, store: NarrationStore
) -> None:
    arrived = threading.Event()
    runner = LooksAgain(before=arrived)
    daemon = run_daemon(runner, idle_exit_s=0.3)
    daemon.wait_status(lambda s: s.state == "stopping", "the daemon to turn to exit")
    job = make_job(store, "Queued as the daemon turned to exit.")  # a front-end saw "stopping" too late
    arrived.set()
    wait_until(lambda: job_status(store, job).status == "completed", what="the daemon to run the job")
    assert daemon.join() == EXIT_OK
    assert runner.states_seen[0] == "stopping", "it says stopping before it looks"
    assert runner.answers[0] is True, "it found the job and stayed"


def test_a_job_queued_after_the_last_look_is_run_by_the_next_daemon_s4(
    run_daemon: DaemonFactory, store: NarrationStore
) -> None:
    go_on = threading.Event()
    first = LooksAgain(after=go_on)
    a = run_daemon(first, idle_exit_s=0.3)
    assert first.looked.wait(WAIT_S) and first.answers == [False]
    seen = read_status(store)
    assert seen is not None and seen.state == "stopping", "a front-end sees the daemon on its way out"
    job = make_job(store, "Queued after the last look.")
    b = run_daemon(FakeWorkerRunner(), takeover_wait_s=20.0, idle_exit_s=0.5)
    time.sleep(0.3)
    assert b.thread.is_alive() and b.daemon.started_at is None, "B waits for A to go"
    go_on.set()
    assert a.join() == EXIT_OK
    wait_until(lambda: job_status(store, job).status == "completed", what="B to run the job")
    assert b.join() == EXIT_OK


def test_a_front_end_that_commits_then_reads_the_status_strands_no_job_s4(
    run_daemon: DaemonFactory, store: NarrationStore, tmp_path: Path
) -> None:
    # seam, "The front-end's order": the daemon writes stopping, then looks; the front-end commits, then
    # reads. After a look that found nothing, a committed job's front-end sees stopping and starts a daemon.
    go_on = threading.Event()
    first = LooksAgain(after=go_on)
    a = run_daemon(first, idle_exit_s=0.3)
    assert first.looked.wait(WAIT_S) and first.answers == [False]
    make_job(store, "Committed before the front-end looks.")
    front_end = StandInPlatform()
    result = ensure_daemon(store, tmp_path / "narration.toml", platform=front_end)
    go_on.set()
    assert result.started and len(front_end.spawned) == 1, "it saw stopping, and started a daemon"
    assert a.join() == EXIT_OK


def test_a_daemon_waiting_to_take_over_gives_up_when_the_holder_stays_s4(
    run_daemon: DaemonFactory, store: NarrationStore
) -> None:
    b_waits = threading.Event()
    first = LooksAgain(before=b_waits)
    a = run_daemon(first, idle_exit_s=0.3)
    a_serving = a.wait_status(lambda s: s.state == "stopping", "A to turn to exit")
    job = make_job(store, "Queued as A turned to exit.")
    b = run_daemon(NullRunner(), takeover_wait_s=20.0)
    time.sleep(0.3)
    assert b.thread.is_alive(), "B waits while A says it is stopping"
    b_waits.set()
    started = time.monotonic()
    assert b.join(10) == EXIT_OK
    assert time.monotonic() - started < 5.0, "B gives up once A says idle or busy again"
    assert b.daemon.started_at is None, "B wrote nothing"
    wait_until(lambda: job_status(store, job).status == "completed", what="A to run the job")
    assert a.join() == EXIT_OK
    assert first.answers[0] is True
    assert a.daemon.started_at == a_serving.started_at


def test_a_has_work_that_fails_keeps_the_daemon_serving_s4(
    run_daemon: DaemonFactory, caplog: pytest.LogCaptureFixture
) -> None:
    runner = LooksBadly()
    daemon = run_daemon(runner, idle_exit_s=0.3)
    assert daemon.join() == EXIT_OK
    assert runner.calls == 2, "it stayed after the failed look, and exited after the next"
    assert any("has_work failed" in record.getMessage() for record in caplog.records)


# ---------------------------------------------------------------- release_gpu (section 7.6)
def test_release_gpu_unloads_an_idle_model_at_once_s7_6(run_daemon: DaemonFactory) -> None:
    daemon = run_daemon(LoadOnce())
    loaded = daemon.wait_status(lambda s: s.gpu.in_use and bool(s.workers), "the model to load")
    assert loaded.gpu.holder == "qwen"
    workers = our_children(w.pid for w in loaded.workers)
    done = daemon.command("release_gpu")
    assert done.result == {"released": True, "holder_before": "qwen", "busy_job": None}
    after = daemon.wait_status(lambda s: not s.gpu.in_use, "the release to show")
    assert after.workers == () and after.gpu.holder is None
    wait_until(lambda: not any(running(w) for w in workers), 15, "the worker to exit")


def test_release_gpu_with_nothing_loaded_releases_nothing_s7_6(run_daemon: DaemonFactory) -> None:
    daemon = run_daemon(NullRunner())
    daemon.wait_serving()
    assert daemon.command("release_gpu").result == {"released": False, "holder_before": None, "busy_job": None}


def test_release_gpu_while_a_job_runs_changes_nothing_and_names_it_s7_6(
    run_daemon: DaemonFactory, store: NarrationStore
) -> None:
    job = make_job(store, "A job in flight.")
    daemon = run_daemon(HoldJob())
    busy = daemon.wait_status(lambda s: s.state == "busy" and s.gpu.in_use, "the job to hold the GPU")
    assert busy.current_job is not None and busy.current_job.job_id == job.job_id
    done = daemon.command("release_gpu")
    assert done.result == {"released": False, "holder_before": "qwen", "busy_job": job.job_id}
    still = daemon.status()
    assert still is not None and still.gpu.in_use and still.workers
    assert all(running(w) for w in our_children(w.pid for w in still.workers))


# ---------------------------------------------------------------- idle unload (section 4)
def test_an_idle_model_is_unloaded_after_idle_unload_s_s4(run_daemon: DaemonFactory) -> None:
    daemon = run_daemon(LoadOnce(), idle_unload_s=0.6, idle_exit_s=60.0)
    countdown = daemon.wait_status(lambda s: s.gpu.in_use and s.gpu.unload_in_s is not None, "the countdown")
    assert countdown.gpu.unload_in_s is not None and 0.0 <= countdown.gpu.unload_in_s <= 0.6
    after = daemon.wait_status(lambda s: not s.gpu.in_use and not s.workers, "the idle unload")
    assert after.state == "idle" and after.gpu.unload_in_s is None
    assert daemon.thread.is_alive(), "unloading is not exiting"


def test_no_idle_unload_while_a_job_holds_the_gpu_s4(run_daemon: DaemonFactory, store: NarrationStore) -> None:
    make_job(store, "A long job.")
    daemon = run_daemon(HoldJob(), idle_unload_s=0.2, idle_exit_s=0.2)
    daemon.wait_status(lambda s: s.state == "busy" and s.gpu.in_use, "the job to hold the GPU")
    time.sleep(1.0)
    still = daemon.status()
    assert still is not None and still.state == "busy" and still.gpu.in_use
    assert daemon.thread.is_alive(), "a daemon with a job in hand never idles out"


# ---------------------------------------------------------------- stop and stop_now (section 4.1)
def test_stop_finishes_the_in_flight_segment_then_exits_s4_1(run_daemon: DaemonFactory, store: NarrationStore) -> None:
    job = make_job(store, "The first line.", "The second line.", "The third line.")
    spec = {"faults": [{"kind": "delay", "seconds": 1.5, "when": {"text_contains": "second"}}]}
    daemon = run_daemon(FakeWorkerRunner(), spec=spec)
    wait_until(lambda: job_status(store, job).progress.segments_done == 1, what="the first segment")
    time.sleep(0.2)  # the second segment's render is in flight
    posted = store.post_command("stop")
    assert daemon.join() == EXIT_OK
    after = job_status(store, job)
    assert after.status == "queued", "given back, to resume on the next start"
    assert after.progress.segments_done == 2, "the in-flight segment was finished"
    assert rendered(store) == 2
    done = store.wait_for_command(posted.command_id, timeout_s=0)
    assert done is not None and done.result == {"stopped": True, "requeued": [job.job_id]}


def test_stop_now_requeues_the_segment_and_leaves_no_partial_file_s4_1(
    run_daemon: DaemonFactory, store: NarrationStore
) -> None:
    job = make_job(store, "The first line.", "The second line.", "The third line.")
    spec = {"faults": [{"kind": "delay", "seconds": 8, "when": {"text_contains": "second"}}]}
    daemon = run_daemon(FakeWorkerRunner(), spec=spec)
    wait_until(lambda: job_status(store, job).progress.segments_done == 1, what="the first segment")
    busy = daemon.wait_status(lambda s: bool(s.workers), "a worker")
    workers = our_children(w.pid for w in busy.workers)
    time.sleep(0.3)  # the second segment's render is in flight
    posted = store.post_command("stop_now")
    stopped_at = time.monotonic()
    assert daemon.join() == EXIT_OK
    assert time.monotonic() - stopped_at < 6.0, "stop_now does not wait for the segment"
    after = job_status(store, job)
    assert (after.status, after.progress.segments_done) == ("queued", 1)
    assert rendered(store) == 1, "the abandoned segment was not published"
    assert not (store.root / "scratch" / SCRATCH_DIR / job.job_id).exists(), "its scratch files are gone"
    assert temp_leftovers(store.root) == []
    report = store.verify()
    assert (report["missing"], report["mismatched"], report["errors"]) == ([], [], [])
    assert not any(running(w) for w in workers), "the workers were killed"
    done = store.wait_for_command(posted.command_id, timeout_s=0)
    assert done is not None and done.result == {"stopped": True, "requeued": [job.job_id]}


def test_a_requeued_job_resumes_from_the_cache_on_the_next_start_s4_1(
    run_daemon: DaemonFactory, store: NarrationStore
) -> None:
    job = make_job(store, "The first line.", "The second line.")
    spec = {"faults": [{"kind": "delay", "seconds": 8, "when": {"text_contains": "second"}}]}
    first = run_daemon(FakeWorkerRunner(), spec=spec)
    wait_until(lambda: job_status(store, job).progress.segments_done == 1, what="the first segment")
    time.sleep(0.3)
    store.post_command("stop_now")
    assert first.join() == EXIT_OK
    second = run_daemon(FakeWorkerRunner(), idle_exit_s=0.5)
    assert second.join() == EXIT_OK
    assert job_status(store, job).status == "completed"
    assert rendered(store) == 2


def test_stop_now_escalates_a_stop_in_progress_s4_1(run_daemon: DaemonFactory, store: NarrationStore) -> None:
    job = make_job(store, "The first line.", "The second line.")
    spec = {"faults": [{"kind": "delay", "seconds": 10, "when": {"text_contains": "second"}}]}
    daemon = run_daemon(FakeWorkerRunner(), spec=spec)
    wait_until(lambda: job_status(store, job).progress.segments_done == 1, what="the first segment")
    time.sleep(0.3)
    store.post_command("stop")
    daemon.wait_status(lambda s: s.state == "stopping", "stopping")
    started = time.monotonic()
    store.post_command("stop_now")
    assert daemon.join() == EXIT_OK
    assert time.monotonic() - started < 6.0
    assert job_status(store, job).status == "queued"


def test_the_daemon_gives_back_a_job_the_runner_kept_s4_1(run_daemon: DaemonFactory, store: NarrationStore) -> None:
    job = make_job(store, "Kept.")
    runner = KeepsJob()
    daemon = run_daemon(runner)
    daemon.wait_status(lambda s: s.state == "busy", "busy")
    done = daemon.command("stop")
    assert daemon.join() == EXIT_OK
    assert runner.reasons == ["segment"]
    assert job_status(store, job).status == "queued"
    assert done.result == {"stopped": True, "requeued": [job.job_id]}


def test_the_runner_is_shut_down_once_with_the_reason_s4_1(run_daemon: DaemonFactory, store: NarrationStore) -> None:
    make_job(store, "Held.")
    runner = HoldJob()
    daemon = run_daemon(runner)
    daemon.wait_status(lambda s: s.state == "busy", "busy")
    daemon.command("stop_now")
    assert daemon.join() == EXIT_OK
    assert runner.reasons == ["now"]


def test_an_idle_exit_shuts_the_runner_down_as_idle_s4(run_daemon: DaemonFactory) -> None:
    runner = HoldJob()
    daemon = run_daemon(runner, idle_exit_s=0.3)
    assert daemon.join() == EXIT_OK
    assert runner.reasons == ["idle"]


# ---------------------------------------------------------------- workers and failures
def test_a_worker_that_crashes_mid_job_is_started_again_appA(run_daemon: DaemonFactory, store: NarrationStore) -> None:
    job = make_job(store, "The first line.", "The second line.")
    spec = {"faults": [{"kind": "crash", "times": 1, "when": {"text_contains": "second"}}]}
    daemon = run_daemon(FakeWorkerRunner(), spec=spec, idle_exit_s=0.5)
    assert daemon.join() == EXIT_OK
    assert job_status(store, job).status == "completed"
    assert rendered(store) == 2


def test_a_worker_that_cannot_start_fails_the_job_once_s14(run_daemon: DaemonFactory, store: NarrationStore) -> None:
    job = make_job(store, "One line.")
    daemon = run_daemon(FakeWorkerRunner(), broken_workers=True, idle_exit_s=0.5)
    assert daemon.join() == EXIT_OK
    failed = job_status(store, job)
    assert failed.status == "failed"
    assert failed.error is not None and failed.error.code == "BACKEND_NOT_INSTALLED"


def test_a_step_that_raises_is_logged_and_the_loop_goes_on(
    run_daemon: DaemonFactory, caplog: pytest.LogCaptureFixture
) -> None:
    runner = RaisesTwice()
    daemon = run_daemon(runner, idle_exit_s=0.3, poll_s=0.02)
    assert daemon.join() == EXIT_OK
    assert runner.calls >= 3
    assert any("a bug" in record.getMessage() for record in caplog.records)


def test_every_command_is_completed_with_a_result_s4_1(run_daemon: DaemonFactory, store: NarrationStore) -> None:
    daemon = run_daemon(NullRunner())
    daemon.wait_serving()
    release = store.post_command("release_gpu")
    stop = store.post_command("stop")
    assert daemon.join() == EXIT_OK
    for posted in (release, stop):
        done = store.wait_for_command(posted.command_id, timeout_s=0)
        assert done is not None and done.done_at is not None and done.result is not None
    assert store.pending_commands() == ()


# ---------------------------------------------------------------- what each exit leaves for a queued job (WP36, L1)
# The front-end starts a daemon for a queued job under a stopped status unless an operator's stop left it there
# (narration.backend.service.operator_stop). These pin what each real exit leaves in the store.


def test_an_operators_stop_is_found_for_a_job_queued_before_it_s4_1(
    run_daemon: DaemonFactory, store: NarrationStore
) -> None:
    job = make_job(store, "Queued before the stop.")
    daemon = run_daemon(NullRunner())
    daemon.wait_serving()
    stop = daemon.command("stop")
    assert daemon.join() == EXIT_OK
    final = store.get_daemon_status()
    assert final is not None and (final.state, final.stop_reason) == ("stopped", STOP_OPERATOR)
    assert final.started_at == daemon.daemon.started_at, "the start time is kept (contracts 1.6.11)"
    found = operator_stop(store, job_status(store, job))
    assert found is not None and found.command_id == stop.command_id


def test_a_daemon_whose_control_loop_failed_leaves_no_operators_stop_s4_1(
    run_daemon: DaemonFactory, store: NarrationStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = make_job(store, "Queued before the failure.")

    def fails(self: Daemon) -> None:
        raise RuntimeError("a bug in the control loop")

    monkeypatch.setattr(Daemon, "_control_loop", fails)
    daemon = run_daemon(NullRunner())
    assert daemon.join() == EXIT_ERROR
    final = store.get_daemon_status()
    assert final is not None and final.state == "stopped", "its finally still wrote stopped"
    assert (final.stop_reason, final.started_at) == (STOP_ERROR, daemon.daemon.started_at)
    assert operator_stop(store, job_status(store, job)) is None, "so get_job starts a daemon for the job"


def test_a_daemon_whose_supervisor_failed_leaves_no_operators_stop_s4_1(
    run_daemon: DaemonFactory, store: NarrationStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = make_job(store, "Queued before the failure.")

    def fails(self: WorkerSupervisor) -> WorkerSupervisor:
        raise RuntimeError("the worker supervisor could not open")

    monkeypatch.setattr(WorkerSupervisor, "__enter__", fails)
    daemon = run_daemon(NullRunner())
    with pytest.raises(RuntimeError, match="could not open"):
        daemon.join()
    final = store.get_daemon_status()
    assert final is not None and final.state == "stopped", "its finally still wrote stopped"
    assert (final.stop_reason, final.started_at) == (STOP_ERROR, daemon.daemon.started_at)
    assert operator_stop(store, job_status(store, job)) is None, "so get_job starts a daemon for the job"


def test_an_idle_exit_leaves_no_operators_stop_s4(run_daemon: DaemonFactory, store: NarrationStore) -> None:
    job = make_job(store, "Queued as the daemon went idle.")  # the null runner never sees it: as if queued late
    daemon = run_daemon(NullRunner(), idle_exit_s=0.3)
    assert daemon.join() == EXIT_OK
    final = store.get_daemon_status()
    assert final is not None and final.state == "stopped"
    assert (final.stop_reason, final.started_at) == (STOP_IDLE, daemon.daemon.started_at)
    assert operator_stop(store, job_status(store, job)) is None, "so get_job starts a daemon for the job"


def test_an_interrupted_daemon_says_so_and_leaves_no_operators_stop_s4_1(
    run_daemon: DaemonFactory, store: NarrationStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = make_job(store, "Queued before the interrupt.")

    def interrupted(self: Daemon) -> None:
        raise KeyboardInterrupt  # Ctrl+C in the terminal of a --foreground daemon

    monkeypatch.setattr(Daemon, "_control_loop", interrupted)
    daemon = run_daemon(NullRunner())
    assert daemon.join() == EXIT_OK
    final = store.get_daemon_status()
    assert final is not None and (final.state, final.stop_reason) == ("stopped", STOP_INTERRUPTED)
    assert operator_stop(store, job_status(store, job)) is None, "so get_job starts a daemon for the job"


def test_a_runner_thread_that_failed_is_an_error_s4_1(
    run_daemon: DaemonFactory, store: NarrationStore, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def fails(self: Daemon, host: object) -> bool:
        raise RuntimeError("a bug between steps")

    monkeypatch.setattr(Daemon, "_idle", fails)
    with caplog.at_level(logging.ERROR, logger="narration.daemon.service"):
        daemon = run_daemon(NullRunner())
        assert daemon.join() == EXIT_OK, "the control loop saw the runner end, with no stop asked"
    final = store.get_daemon_status()
    assert final is not None and (final.state, final.stop_reason) == ("stopped", STOP_ERROR)
    assert "the job runner's thread failed" in caplog.text, "in the daemon's log, not only on a NUL stderr"


def test_a_stop_answered_on_the_way_out_of_a_failure_is_the_operators_s4_1(
    run_daemon: DaemonFactory, store: NarrationStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The operator's stop was posted as the control loop failed: the exit answers it stopped: true, so the
    # jobs queued before it wait for the next start (the log has the failure).
    job = make_job(store, "Queued before the stop.")
    posted: list[str] = []

    def fails(self: Daemon) -> None:
        posted.append(store.post_command("stop").command_id)
        raise RuntimeError("a bug in the control loop")

    monkeypatch.setattr(Daemon, "_control_loop", fails)
    daemon = run_daemon(NullRunner())
    assert daemon.join() == EXIT_ERROR
    final = store.get_daemon_status()
    assert final is not None and (final.state, final.stop_reason) == ("stopped", STOP_OPERATOR)
    found = operator_stop(store, job_status(store, job))
    assert found is not None and found.command_id == posted[0]
