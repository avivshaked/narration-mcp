"""The seam's job hand-back (sections 4.1, 8) and the sweep a daemon makes on start (section 4.1)."""

from __future__ import annotations

import os
import time
from types import SimpleNamespace
from typing import Any, cast

import psutil
import pytest

from narration.config import Config
from narration.contracts.models import DaemonStatus, GpuStatus
from narration.contracts.names import DaemonState
from narration.daemon.seam import JobRunner, NullRunner, RunnerHost, WorkerPool, return_job
from narration.daemon.service import Daemon, _Host, posted_after_launch  # pyright: ignore[reportPrivateUsage]
from narration.daemon.settings import DaemonSettings
from narration.daemon.supervisor import WorkerSupervisor
from narration.daemon.sweep import (
    READ_ATTEMPTS,
    READ_PAUSE_S,
    StatusUnreadable,
    daemon_alive,
    read_status,
    sweep,
)
from narration.daemon.testing import SCRATCH_DIR, VOICE_SEED, FakeWorkerRunner
from narration.store import NarrationStore
from narration.store.layout import StorePathError
from narration.store.store import utc_iso

from .conftest import UNREADABLE_STATUSES, make_job, plant_status
from .standin import StandInPlatform

NO_SUCH_PID = 0xFFFFFFFC
"""A pid no process has (Windows pids are multiples of 4; Linux pids stay far below this)."""


def status(state: DaemonState, pid: int | None, started_at: str | None) -> DaemonStatus:
    return DaemonStatus(
        state=state,
        pid=pid,
        started_at=started_at,
        workers=(),
        current_job=None,
        gpu=GpuStatus(name=None, total_mb=None, free_mb=None, in_use=False, holder=None, unload_in_s=None),
        est_drain_s=None,
        updated_at=utc_iso(time.time()),
    )


# ---------------------------------------------------------------- the seam
def test_the_shipped_runners_and_the_supervisor_keep_the_seam(
    config: Config, store: NarrationStore, platform: StandInPlatform
) -> None:
    assert isinstance(NullRunner(), JobRunner)
    assert isinstance(FakeWorkerRunner(), JobRunner)
    supervisor = WorkerSupervisor(config, platform)
    assert isinstance(supervisor, WorkerPool)
    daemon = Daemon(
        settings=DaemonSettings(store_root=config.server.store_root),
        config=config,
        store=store,
        platform=platform,
        runner=NullRunner(),
    )
    assert isinstance(_Host(daemon, supervisor), RunnerHost)


def test_the_host_gives_the_runner_the_daemons_platform_read_only(
    config: Config, store: NarrationStore, platform: StandInPlatform
) -> None:
    daemon = Daemon(
        settings=DaemonSettings(store_root=config.server.store_root),
        config=config,
        store=store,
        platform=platform,
        runner=NullRunner(),
    )
    host = _Host(daemon, WorkerSupervisor(config, platform))
    assert host.platform is platform
    with pytest.raises(AttributeError):
        host.platform = StandInPlatform()  # pyright: ignore[reportAttributeAccessIssue]


def test_the_seam_exports_the_residency_error_the_pool_raises() -> None:
    from narration.daemon import seam, supervisor

    assert seam.ResidencyError is supervisor.ResidencyError
    assert "ResidencyError" in seam.__all__


def test_the_shipped_runners_look_for_work_without_side_effects_s4(store: NarrationStore) -> None:
    host = SimpleNamespace(store=store)
    assert NullRunner().has_work(cast(RunnerHost, host)) is False
    runner = FakeWorkerRunner()
    assert runner.has_work(cast(RunnerHost, host)) is False
    running = make_job(store, "Run by another.", status="running")
    assert runner.has_work(cast(RunnerHost, host)) is False, "only a job step would claim counts"
    job = make_job(store, "Waiting.")
    assert runner.has_work(cast(RunnerHost, host)) is True
    assert runner.has_work(cast(RunnerHost, host)) is True
    for record in (job, running):
        after = store.get_job(record.job_id)
        assert after is not None and after == record, "it claimed and changed nothing"


class _Client:
    """A worker client that records its requests."""

    def __init__(self, pid: int) -> None:
        self.pid = pid
        self.requests: list[str] = []

    def request(self, op: str, params: Any, *, timeout_s: float) -> dict[str, Any]:
        self.requests.append(op)
        return {}


class _Pool:
    """A pool with one loaded group, whose worker the test swaps."""

    def __init__(self, client: _Client) -> None:
        self.current = client

    def client(self, group: str, **_: Any) -> _Client:
        return self.current

    def loaded(self) -> tuple[str, ...]:
        return ("qwen",)

    def load(self, group: str, params: Any, *, gpu: bool, timeout_s: float) -> dict[str, Any]:
        return {}


def test_worker_state_is_kept_per_worker_instance_not_per_pid(config: Config, store: NarrationStore) -> None:
    # seam: a prepared voice belongs to the worker instance; a new worker may be given an old worker's pid.
    clip = store.scratch_path(SCRATCH_DIR, f"voice-{VOICE_SEED}.wav")
    clip.parent.mkdir(parents=True, exist_ok=True)
    clip.write_bytes(b"a clip made earlier")
    first = _Client(pid=4242)
    pool = _Pool(first)
    host = cast(RunnerHost, SimpleNamespace(workers=pool, config=config, store=store, job_phase=lambda _phase: None))
    runner = FakeWorkerRunner()
    runner._ready(host)  # pyright: ignore[reportPrivateUsage]
    runner._ready(host)  # pyright: ignore[reportPrivateUsage]
    assert first.requests == ["prepare_voice"], "prepared once while that worker lives"
    pool.current = second = _Client(pid=4242)
    runner._ready(host)  # pyright: ignore[reportPrivateUsage]
    assert second.requests == ["prepare_voice"], "a new worker with the same pid is prepared again"


def test_a_running_job_goes_back_to_the_queue_s4_1(store: NarrationStore) -> None:
    job = make_job(store, "One.", "Two.", status="running")
    store.update_job(job.job_id, phase="rendering")
    back = return_job(store, job.job_id, reason="the daemon stopped")
    assert back is not None
    assert (back.status, back.phase) == ("queued", None)
    assert back.message is not None and "the daemon stopped" in back.message
    assert back.request == job.request


def test_a_cancelling_job_is_finished_as_cancelled_s8(store: NarrationStore) -> None:
    job = make_job(store, "One.", status="cancelling")
    back = return_job(store, job.job_id, reason="the daemon stopped")
    assert back is not None and back.status == "cancelled"


def test_a_job_in_any_other_status_is_left_alone_s4_1(store: NarrationStore) -> None:
    for status_ in ("queued", "completed", "failed", "cancelled"):
        job = make_job(store, f"Job {status_}.", status=status_)
        assert return_job(store, job.job_id, reason="x") is None
        after = store.get_job(job.job_id)
        assert after is not None and after.status == status_
    assert return_job(store, "job_01ABCDEFGHJKMNPQRSTVWXYZ00", reason="x") is None


# ---------------------------------------------------------------- daemon_alive (the pid check)
def test_a_live_daemon_is_its_pid_created_before_it_started_s4_1() -> None:
    assert daemon_alive(status("idle", os.getpid(), utc_iso(time.time())))


def test_a_pid_given_to_a_later_process_is_not_the_daemon_s4_1() -> None:
    created = psutil.Process(os.getpid()).create_time()
    assert not daemon_alive(status("busy", os.getpid(), utc_iso(created - 3600.0)))


def test_a_gone_or_stopped_daemon_is_not_alive_s4_1() -> None:
    now = utc_iso(time.time())
    assert not daemon_alive(None)
    assert not daemon_alive(status("idle", NO_SUCH_PID, now))
    assert not daemon_alive(status("stopped", os.getpid(), now))
    assert not daemon_alive(status("idle", None, now))


# ---------------------------------------------------------------- the sweep (section 4.1)
def test_the_sweep_requeues_what_a_dead_daemon_left_running_s4_1(store: NarrationStore) -> None:
    running = make_job(store, "Running.", status="running")
    cancelling = make_job(store, "Cancelling.", status="cancelling")
    queued = make_job(store, "Queued.")
    done = make_job(store, "Done.", status="completed")
    previous = status("busy", NO_SUCH_PID, utc_iso(time.time() - 60))
    report = sweep(store, previous, started_at=utc_iso(time.time()))
    assert report.previous == "died"
    assert report.previous_pid == NO_SUCH_PID
    assert report.requeued == (running.job_id,)
    assert report.cancelled == (cancelling.job_id,)
    statuses = {j: store.get_job(j) for j in (running.job_id, cancelling.job_id, queued.job_id, done.job_id)}
    assert {j: r.status for j, r in statuses.items() if r is not None} == {
        running.job_id: "queued",
        cancelling.job_id: "cancelled",
        queued.job_id: "queued",
        done.job_id: "completed",
    }


@pytest.mark.parametrize("content", UNREADABLE_STATUSES)
def test_an_unreadable_status_file_is_reported_as_unreadable_s4_1(store: NarrationStore, content: bytes) -> None:
    plant_status(store, content)
    with pytest.raises(StatusUnreadable):
        read_status(store, sleep=lambda _s: None)


def test_the_sweep_takes_an_unreadable_status_for_a_daemon_that_died_s4_1(store: NarrationStore) -> None:
    left = make_job(store, "Left running.", status="running")
    report = sweep(store, None, started_at=utc_iso(time.time()), previous_unreadable=True)
    assert (report.previous, report.previous_pid, report.requeued) == ("died", None, (left.job_id,))


def test_the_sweep_tells_a_clean_stop_and_no_daemon_at_all_s4_1(store: NarrationStore) -> None:
    assert sweep(store, None, started_at=utc_iso(time.time())).previous == "none"
    assert sweep(store, status("stopped", None, None), started_at=utc_iso(time.time())).previous == "clean"


def test_the_sweep_waits_for_a_daemon_still_exiting_s4_1(store: NarrationStore) -> None:
    calls: list[int] = []

    def alive(_: DaemonStatus | None) -> bool:
        calls.append(1)
        return len(calls) < 4  # still alive for the first looks, then gone

    report = sweep(
        store,
        status("stopping", 1234, utc_iso(time.time())),
        started_at=utc_iso(time.time()),
        alive=alive,
        sleep=lambda _: None,
    )
    assert report.previous == "exiting"
    assert len(calls) >= 4


@pytest.mark.parametrize(
    ("requested_at", "launched_at", "after"),
    [
        ("1970-01-01T00:16:40.002Z", 1000.0005, True),  # a later millisecond
        ("1970-01-01T00:16:40.000Z", 1000.0005, True),  # the launch's own millisecond: may be after it
        ("1970-01-01T00:16:39.999Z", 1000.0005, False),  # the millisecond before
        ("1970-01-01T00:16:39.998Z", 1000.0, False),
        ("1970-01-01T00:16:40.000Z", 1000.0, True),
    ],
)
def test_a_stop_counts_as_posted_after_launch_to_its_millisecond_s4_1(
    requested_at: str, launched_at: float, after: bool
) -> None:
    assert posted_after_launch(requested_at, launched_at) is after


def test_the_sweep_leaves_every_command_to_the_daemon_s4_1(store: NarrationStore) -> None:
    # A stop is for the service, not for one daemon process (lead's decision): none is discarded as stale.
    posted = {store.post_command(kind).command_id for kind in ("stop", "stop_now", "release_gpu")}
    time.sleep(0.01)
    sweep(store, None, started_at=utc_iso(time.time() + 1.0))
    assert {c.command_id for c in store.pending_commands()} == posted


def test_the_sweep_needs_no_process_but_the_recorded_pid(store: NarrationStore) -> None:
    # A regression guard for the rule "never inspect a process that is not provably ours": the sweep reads
    # only the recorded pid's existence and creation time, through daemon_alive, which it takes as a parameter.
    seen: list[int | None] = []

    def alive(previous: DaemonStatus | None) -> bool:
        seen.append(previous.pid if previous else None)
        return False

    sweep(store, status("idle", 555, utc_iso(time.time())), started_at=utc_iso(time.time()), alive=alive)
    assert seen == [555]


# ---------------------------------------------------------------- reading run/daemon.json while it is replaced
class FlakyStatusStore:
    """A store whose ``get_daemon_status`` fails the first times it is called."""

    def __init__(self, *errors: BaseException) -> None:
        self.errors = list(errors)
        self.calls = 0

    def get_daemon_status(self) -> DaemonStatus | None:
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return status("idle", os.getpid(), utc_iso(time.time()))


def test_a_status_read_during_a_rename_is_tried_again_s4_1() -> None:
    flaky = FlakyStatusStore(PermissionError(13, "Permission denied"), StorePathError("a prefixed realpath"))
    pauses: list[float] = []
    got = read_status(flaky, sleep=pauses.append)  # pyright: ignore[reportArgumentType]
    assert got is not None and got.state == "idle"
    assert flaky.calls == 3
    assert pauses == [READ_PAUSE_S, READ_PAUSE_S]


def test_a_status_read_that_keeps_failing_is_raised_s4_1() -> None:
    flaky = FlakyStatusStore(*(PermissionError(13, "Permission denied") for _ in range(READ_ATTEMPTS)))
    with pytest.raises(PermissionError):
        read_status(flaky, sleep=lambda _s: None)  # pyright: ignore[reportArgumentType]
    assert flaky.calls == READ_ATTEMPTS


def test_other_status_read_errors_are_raised_at_once_s4_1() -> None:
    flaky = FlakyStatusStore(IsADirectoryError(21, "a folder where the file should be"))
    with pytest.raises(IsADirectoryError):
        read_status(flaky, sleep=lambda _s: None)  # pyright: ignore[reportArgumentType]
    assert flaky.calls == 1


def test_a_status_that_is_not_one_is_unreadable_at_once_s4_1() -> None:
    flaky = FlakyStatusStore(ValueError("Expecting value: line 1 column 1 (char 0)"))
    with pytest.raises(StatusUnreadable, match="Expecting value"):
        read_status(flaky, sleep=lambda _s: None)  # pyright: ignore[reportArgumentType]
    assert flaky.calls == 1
