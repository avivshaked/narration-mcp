"""The daemon's control loop (sections 4, 4.1, 7.6): a ``Daemon`` on a thread, the platform stand-in, fake workers.

These run on every OS. The real Windows mechanisms (the named mutex, the Job Object, detached start) are
exercised with the real entry point in ``test_process.py``.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import psutil
import pytest

from narration.contracts.models import JobRecord
from narration.daemon.seam import NullRunner, RunnerHost, ShutdownReason, return_job
from narration.daemon.service import EXIT_OK
from narration.daemon.testing import SCRATCH_DIR, FakeWorkerRunner
from narration.store import NarrationStore
from narration.store.store import utc_iso

from .conftest import UNREADABLE_STATUSES, DaemonFactory, make_job, plant_status, wait_until
from .standin import StandInPlatform
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

    def shutdown(self, host: RunnerHost, reason: ShutdownReason) -> None:
        return None


def rendered(store: NarrationStore) -> int:
    renders = store.root / "renders"
    return sum(1 for shard in renders.iterdir() for _ in shard.iterdir()) if renders.is_dir() else 0


def temp_leftovers(root: Path) -> list[Path]:
    return [p for p in root.rglob("*") if any(p.name.startswith(mark) for mark in TEMP_MARKS)]


def job_status(store: NarrationStore, job: JobRecord) -> Any:
    found = store.get_job(job.job_id)
    assert found is not None
    return found


def alive(pid: int) -> bool:
    try:
        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


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
    with platform.hold(store.root):
        daemon = run_daemon(NullRunner(), takeover_wait_s=0.5)
        assert daemon.join(10) == EXIT_OK
    assert store.get_daemon_status() is None, "it wrote nothing"


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


def test_an_unreadable_status_of_the_holder_is_not_a_daemon_stopping_s4(
    run_daemon: DaemonFactory, store: NarrationStore, platform: StandInPlatform
) -> None:
    planted = plant_status(store, b"")
    with platform.hold(store.root):
        started = time.monotonic()
        daemon = run_daemon(NullRunner(), takeover_wait_s=20.0)
        assert daemon.join(10) == EXIT_OK
        assert time.monotonic() - started < 5.0, "it does not wait for a daemon it cannot see stopping"
    assert planted.read_bytes() == b"", "it wrote nothing"


def test_a_stale_stop_does_not_stop_a_new_daemon_s4_1(run_daemon: DaemonFactory, store: NarrationStore) -> None:
    stale = store.post_command("stop")
    time.sleep(0.02)
    daemon = run_daemon(NullRunner())
    daemon.wait_serving()
    done = store.wait_for_command(stale.command_id, timeout_s=10)
    assert done is not None and done.result is not None and done.result["stopped"] is False
    time.sleep(0.3)
    assert daemon.thread.is_alive()
    daemon.command("stop")
    assert daemon.join() == EXIT_OK


# ---------------------------------------------------------------- release_gpu (section 7.6)
def test_release_gpu_unloads_an_idle_model_at_once_s7_6(run_daemon: DaemonFactory) -> None:
    daemon = run_daemon(LoadOnce())
    loaded = daemon.wait_status(lambda s: s.gpu.in_use and bool(s.workers), "the model to load")
    assert loaded.gpu.holder == "qwen"
    worker_pids = [w.pid for w in loaded.workers]
    done = daemon.command("release_gpu")
    assert done.result == {"released": True, "holder_before": "qwen", "busy_job": None}
    after = daemon.wait_status(lambda s: not s.gpu.in_use, "the release to show")
    assert after.workers == () and after.gpu.holder is None
    wait_until(lambda: not any(alive(pid) for pid in worker_pids), 15, "the worker to exit")


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
    assert all(alive(w.pid) for w in still.workers)


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
    worker_pids = [w.pid for w in busy.workers]
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
    assert not any(alive(pid) for pid in worker_pids), "the workers were killed"
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
