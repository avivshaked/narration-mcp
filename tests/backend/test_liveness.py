"""An active job whose daemon has gone (design sections 4, 4.1, 7.4 and 14; plan.md DC-2).

A job stays ``queued``, ``running`` or ``cancelling`` in the store after its daemon has gone (a crash, a machine
restart, a daemon killed with its client), and only a daemon moves it on. So ``get_job`` and ``cancel_job`` ask
the launcher for a daemon when none runs, and ``get_job`` says so in its reply, or answers ``DAEMON_UNAVAILABLE``
with what to do when none can be started. The fake launcher starts nothing; where a test needs the new daemon's
start-up, it runs the daemon's own sweep (``narration.daemon.sweep``) at ``ensure``.

A queued job that an operator's stop left in the queue starts no daemon (``operator_stop``); a queued job
under any other ``stopped`` status does. While a launched daemon is still starting (``run/launch.json``,
``narration.daemon.start``), no other is asked for; once its process has gone without writing a status, the
start failed, and get_job says so rather than launching again at once. Those tests run the real
``DetachedLauncher`` and ``start_detached`` over the stand-in platform, which records each launch and runs a
real process that sleeps in place of the daemon (``LaunchingPlatform``).

Every text is invented for these tests.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import subprocess
import sys
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import anyio
import psutil
import pytest
from jsonschema import Draft202012Validator

from narration.backend.launch import DAEMON_RETRY_S, DetachedLauncher, unavailable
from narration.backend.service import (
    DAEMON_START_GRACE_S,
    STOP_TO_STOPPED_S,
    NarrationBackend,
    Revival,
    operator_stop,
)
from narration.config import DaemonConfig
from narration.contracts import codes
from narration.contracts.errors import NarrationError, UnsupportedPlatform
from narration.contracts.models import DaemonCommand, JobRecord
from narration.contracts.names import (
    STOP_ERROR,
    STOP_IDLE,
    STOP_INTERRUPTED,
    STOP_OPERATOR,
    DaemonCommandKind,
    StopReason,
)
from narration.contracts.schemas import TOOLS_BY_NAME
from narration.daemon import start as daemon_start
from narration.daemon.settings import DaemonSettings
from narration.daemon.sweep import launch_alive, sweep
from narration.platform.testing import StandInPlatform
from narration.store.store import parse_iso, utc_iso
from tests.jobs.support import LAMPS

from .conftest import Service
from .support import FakeMeasurements, daemon_status, make_backend, pins_for


def valid(tool: str, structured: dict[str, Any]) -> dict[str, Any]:
    validator = Draft202012Validator(TOOLS_BY_NAME[tool].output_schema)
    failures = [f"{'/'.join(map(str, e.absolute_path))}: {e.message}" for e in validator.iter_errors(structured)]
    assert failures == []
    return structured


def get_job(backend: NarrationBackend, args: dict[str, Any]) -> dict[str, Any]:
    async def main() -> dict[str, Any]:
        return await backend.get_job(args, None)

    return valid("get_job", anyio.run(main))


def refused(fn: Callable[[], Any]) -> NarrationError:
    with pytest.raises(NarrationError) as caught:
        fn()
    return caught.value


def left_running(service: Service, backend: NarrationBackend | None = None) -> str:
    """A job a daemon took and then left: ``running`` in the store, and no daemon runs (the fake launcher's
    status is None)."""
    job_id = (backend or service.backend).submit_job_sync(service.request(LAMPS))["job_id"]
    assert service.world.store.claim_job(job_id, "a-daemon") is not None
    service.launcher.ensured.clear()
    return job_id


def start_up_sweep(service: Service) -> Callable[[], None]:
    """What a new daemon does first about the one before it, which died busy: the daemon's own sweep."""

    def run() -> None:
        dead = daemon_status(state="busy")
        sweep(service.world.store, dead, started_at=utc_iso(time.time()), alive=lambda _: False)

    return run


# ======================================================================== get_job with no daemon (4.1, 7.4)


def test_get_job_asks_for_a_daemon_for_a_job_left_running_and_says_so_s4_1(service: Service) -> None:
    job_id = left_running(service)
    service.launcher.on_ensure = start_up_sweep(service)
    out = get_job(service.backend, {"job_id": job_id, "wait_s": 5})
    assert len(service.launcher.ensured) == 1, "one daemon asked for"
    assert out["status"] == "queued", "the new daemon's sweep put the job back on the queue"
    assert out["message"].startswith("queued again: ")
    assert "No daemon was running this job, so get_job asked for one to start" in out["message"]
    assert "daemon state" not in out["message"], "the job has moved on: 'stopped' would be stale"
    assert service.world.job(job_id).status == "queued"


def test_get_job_reports_the_request_before_the_new_daemon_has_run_s7_4(service: Service) -> None:
    job_id = left_running(service)
    out = get_job(service.backend, {"job_id": job_id})
    assert len(service.launcher.ensured) == 1
    assert out["status"] == "running"
    message = out["message"]
    assert "No daemon was running this job (daemon state: stopped), so get_job asked for one to start" in message
    assert "back on the queue" in message
    assert service.world.job(job_id).message == "queued", "the note is the reply's; the job's record is unchanged"


def test_get_job_asks_for_a_daemon_for_a_queued_job_with_none_after_the_grace_s4(service: Service) -> None:
    job_id = service.backend.submit_job_sync(service.request(LAMPS))["job_id"]
    service.launcher.ensured.clear()
    assert service.world.store.get_daemon_status() is None, "no daemon has run: no operator stopped one"
    out = get_job(service.backend, {"job_id": job_id})
    assert service.launcher.ensured == [], "the daemon its submission asked for may still be starting up"
    assert out["message"] == "queued"
    service.backend.clock = lambda: time.time() + DAEMON_START_GRACE_S + 1
    out = get_job(service.backend, {"job_id": job_id})
    assert len(service.launcher.ensured) == 1
    assert out["status"] == "queued"
    assert "No daemon was serving the queue (daemon state: stopped)" in out["message"]


def test_a_job_stamped_from_a_clock_stepped_back_counts_as_just_written_s4(service: Service) -> None:
    job_id = service.backend.submit_job_sync(service.request(LAMPS))["job_id"]
    service.launcher.ensured.clear()
    service.backend.clock = lambda: time.time() - DAEMON_START_GRACE_S / 2
    get_job(service.backend, {"job_id": job_id})
    assert service.launcher.ensured == [], "a stamp a little in the future is within the grace"
    service.backend.clock = lambda: time.time() - DAEMON_START_GRACE_S - 5
    get_job(service.backend, {"job_id": job_id})
    assert len(service.launcher.ensured) == 1, "one far in the future is not trusted to hold the job"


def test_the_start_counts_against_wait_s_s7_4(service: Service, monkeypatch: pytest.MonkeyPatch) -> None:
    job_id = left_running(service)
    started: list[float] = []
    reads_after_start: list[str] = []
    read = service.backend.store.get_job

    def counting(job_id: str) -> JobRecord | None:
        if started:
            reads_after_start.append(job_id)
        return read(job_id)

    def slow_start() -> None:
        time.sleep(0.6)  # longer than wait_s
        started.append(time.monotonic())

    monkeypatch.setattr(service.backend.store, "get_job", counting)
    service.backend.poll_s = 0.05
    service.launcher.on_ensure = slow_start
    out = get_job(service.backend, {"job_id": job_id, "wait_s": 0.5})
    assert out["status"] == "running"
    # Counted, not timed: with the start outside wait_s, about ten polls would follow it; inside, none does.
    assert len(reads_after_start) <= 1, "only the read after the start: the wait ends at wait_s, the start included"


def test_a_daemon_that_cannot_start_is_daemon_unavailable_with_its_hint_dc2(service: Service) -> None:
    job_id = left_running(service)
    service.launcher.error = unavailable("breakaway from the client's job was refused", details={"reason": "test"})
    error = refused(lambda: get_job(service.backend, {"job_id": job_id, "wait_s": 5}))
    assert (error.code, error.retryable, error.retry_after_s) == (codes.DAEMON_UNAVAILABLE, True, DAEMON_RETRY_S)
    assert job_id in error.message and "running" in error.message
    assert "breakaway from the client's job was refused" in error.message
    assert error.hint is not None
    assert "narration-admin daemon start" in error.hint and "get_job" in error.hint
    assert error.details == {"reason": "test", "job_id": job_id, "job_status": "running", "daemon_state": "stopped"}
    assert service.world.job(job_id).status == "running", "the job is kept"


def test_a_failed_start_without_a_wait_gets_one_on_the_way_out_dc2(service: Service) -> None:
    job_id = left_running(service)
    service.launcher.error = NarrationError(codes.DAEMON_UNAVAILABLE, "no start")
    error = refused(lambda: get_job(service.backend, {"job_id": job_id}))
    assert (error.code, error.retryable) == (codes.DAEMON_UNAVAILABLE, True)
    assert error.retry_after_s == DAEMON_RETRY_S


def test_a_daemon_that_cannot_run_here_keeps_its_own_hint_s14(service: Service) -> None:
    job_id = left_running(service)
    service.launcher.error = UnsupportedPlatform("spawn_detached", "plan9")
    error = refused(lambda: get_job(service.backend, {"job_id": job_id}))
    assert (error.code, error.retryable, error.retry_after_s) == (codes.DAEMON_UNAVAILABLE, False, None)
    assert error.hint == UnsupportedPlatform.HINT
    assert error.details is not None and error.details["job_status"] == "running"


@pytest.mark.parametrize("state", ["idle", "busy", "stopping"])
def test_a_running_daemon_is_not_asked_for_again_s4_1(service: Service, state: str) -> None:
    job_id = left_running(service)
    service.launcher.status = daemon_status(state=state, holder="qwen", job_id=job_id)
    out = get_job(service.backend, {"job_id": job_id, "wait_s": 0.2})
    assert service.launcher.ensured == []
    assert out["status"] == "running"
    assert out["message"] == "queued", "the job's own message, with no note"


def test_a_finished_job_starts_no_daemon_s7_4(service: Service) -> None:
    job_id = service.backend.submit_job_sync(service.request(LAMPS))["job_id"]
    service.world.run()
    service.launcher.ensured.clear()
    out = get_job(service.backend, {"job_id": job_id})
    assert out["status"] == "completed"
    assert service.launcher.ensured == []


def test_with_autostart_off_the_reply_says_who_must_start_the_daemon_s16(
    service: Service, caplog: pytest.LogCaptureFixture
) -> None:
    config = dataclasses.replace(service.world.config, daemon=DaemonConfig(autostart=False))
    backend, _, launcher = make_backend(dataclasses.replace(service.world, config=config))
    job_id = left_running(service)
    with caplog.at_level(logging.INFO, logger="narration.backend.service"):
        out = get_job(backend, {"job_id": job_id})
    assert len(launcher.ensured) == 1, "the launcher decides; with autostart off it starts nothing"
    assert "[daemon] autostart is off" in out["message"]
    assert "narration-admin daemon start" in out["message"]
    assert "asked the launcher for one" not in caplog.text, "no start is claimed in the log"


# ======================================================================== cancel_job with no daemon (7.6, 8)


def test_cancelling_a_job_left_running_asks_for_a_daemon_to_finish_the_cancel_s8(service: Service) -> None:
    job_id = left_running(service)
    service.launcher.on_ensure = start_up_sweep(service)
    out = valid("cancel_job", service.backend.cancel_job_sync({"job_id": job_id, "reason": "changed"}))
    assert out == {"status": "cancelling", "completed": False}
    assert len(service.launcher.ensured) == 1
    assert service.world.job(job_id).status == "cancelled", "the new daemon's sweep finished the cancel"


def test_a_cancel_is_kept_when_no_daemon_can_start_s8(service: Service) -> None:
    job_id = left_running(service)
    service.launcher.error = unavailable("breakaway from the client's job was refused")
    out = valid("cancel_job", service.backend.cancel_job_sync({"job_id": job_id}))
    assert out == {"status": "cancelling", "completed": False}
    assert service.world.job(job_id).status == "cancelling"
    error = refused(lambda: get_job(service.backend, {"job_id": job_id}))
    assert error.code == codes.DAEMON_UNAVAILABLE
    assert error.details is not None and error.details["job_status"] == "cancelling"
    assert error.hint is not None and "finishes it" in error.hint


def test_cancelling_with_a_daemon_running_asks_for_none_s8(service: Service) -> None:
    job_id = left_running(service)
    service.launcher.status = daemon_status(state="busy", holder="qwen", job_id=job_id)
    out = valid("cancel_job", service.backend.cancel_job_sync({"job_id": job_id}))
    assert out == {"status": "cancelling", "completed": False}
    assert service.launcher.ensured == []


# ======================================================================== one launch per start (4.1)


class LaunchingPlatform(StandInPlatform):
    """The stand-in platform as ``start_detached``'s: each launch is recorded, and runs a real process that sleeps
    in place of the daemon, still starting. A test ends them (``end_launched``) for a daemon that exited before it
    served; the fixture ends them all."""

    def __init__(self) -> None:
        super().__init__()
        self.launched: list[subprocess.Popen[bytes]] = []

    def spawn_detached(self, argv: Sequence[str], *, cwd: Path, env: Mapping[str, str]) -> int:
        super().spawn_detached(argv, cwd=cwd, env=env)  # records the command, or raises refuse_spawn
        sleeper = [sys.executable, "-c", "import time; time.sleep(120)"]
        process = subprocess.Popen(
            sleeper, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        self.launched.append(process)
        return process.pid

    def end_launched(self) -> None:
        for process in self.launched:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=30)


@pytest.fixture
def platform(monkeypatch: pytest.MonkeyPatch) -> Iterator[LaunchingPlatform]:
    launching = LaunchingPlatform()
    monkeypatch.setattr("narration.platform.get_platform", lambda: launching)
    try:
        yield launching
    finally:
        launching.end_launched()


@pytest.fixture
def launching(service: Service, platform: LaunchingPlatform, tmp_path: Path) -> NarrationBackend:
    """The backend over the real ``DetachedLauncher`` (so ``ensure_daemon`` and ``start_detached``)."""
    config_file = tmp_path / "narration.toml"
    config_file.write_text("", encoding="utf-8")
    found = pins_for(service.world.config.server.models_root)
    return NarrationBackend(
        service.world.config,
        service.world.store,
        service.platform,
        launcher=DetachedLauncher(config_file),
        pins=lambda: found,
        measurements=FakeMeasurements(service.world.store),
        poll_s=0.02,
    )


DEAD_PID = 2**31 - 7
"""A pid no process has (Windows pids are multiples of 4)."""


def dead_status(service: Service, started_at: float) -> None:
    """A daemon status that says ``busy`` under a pid that is gone: the daemon died."""
    status = dataclasses.replace(daemon_status(state="busy"), pid=DEAD_PID, started_at=utc_iso(started_at))
    service.world.store.put_daemon_status(status)


def died_after_start(service: Service) -> None:
    """The daemon launched last wrote its status, then died: a status newer than the launch, and its pid gone.
    A launch after this one is stamped later than that status."""
    launch = daemon_start.read_launch(service.world.store.root)
    dead_status(service, (launch.launched_at if launch is not None else time.time()) + 0.002)
    time.sleep(0.02)


def launched_running(service: Service, backend: NarrationBackend, platform: LaunchingPlatform) -> str:
    """A job submitted through ``backend`` (which launched a daemon), taken by that daemon, which then died:
    ``running`` with no daemon, and no launch still starting. The platform's record of launches is cleared."""
    job_id = backend.submit_job_sync(service.request(LAMPS))["job_id"]
    assert service.world.store.claim_job(job_id, "a-daemon") is not None
    died_after_start(service)
    platform.spawned.clear()
    return job_id


def launched_ago(service: Service, seconds: float) -> None:
    """Record the last launch, under the same pid, as made ``seconds`` ago."""
    launch = daemon_start.read_launch(service.world.store.root)
    pid = launch.pid if launch is not None else DEAD_PID
    daemon_start.record_launch(service.world.store.root, pid=pid, launched_at=time.time() - seconds)


def test_a_launch_is_recorded_for_the_next_launcher_s4_1(
    service: Service, platform: LaunchingPlatform, launching: NarrationBackend
) -> None:
    job_id = launched_running(service, launching, platform)
    before = time.time()
    get_job(launching, {"job_id": job_id})
    assert len(platform.spawned) == 1
    launch = daemon_start.read_launch(service.world.store.root)
    assert launch is not None and launch.pid == platform.launched[-1].pid
    assert before - 0.01 <= launch.launched_at <= time.time()
    assert service.world.store.layout.launch_json_path() == daemon_start.launch_path(service.world.store.root)


def test_repeated_polls_during_a_start_launch_one_daemon_s4_1(
    service: Service, platform: LaunchingPlatform, launching: NarrationBackend
) -> None:
    job_id = launched_running(service, launching, platform)
    first = get_job(launching, {"job_id": job_id})
    assert "asked for one to start" in first["message"]
    for _ in range(3):
        again = get_job(launching, {"job_id": job_id, "wait_s": 0.05})
        assert "is still starting, so get_job asked for no other" in again["message"]
    assert len(platform.spawned) == 1, "one launch while it starts"


def test_cancel_then_get_job_launch_one_daemon_s4_1(
    service: Service, platform: LaunchingPlatform, launching: NarrationBackend
) -> None:
    job_id = launched_running(service, launching, platform)
    assert launching.cancel_job_sync({"job_id": job_id}) == {"status": "cancelling", "completed": False}
    out = get_job(launching, {"job_id": job_id})
    assert out["status"] == "cancelling"
    assert "still starting" in out["message"]
    assert len(platform.spawned) == 1


def test_a_stuck_start_costs_one_launch_per_window_s4_1(
    service: Service, platform: LaunchingPlatform, launching: NarrationBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(daemon_start, "launch_alive", lambda pid, launched_at: True)  # stuck: its process runs
    job_id = launched_running(service, launching, platform)
    get_job(launching, {"job_id": job_id})
    dead_status(service, time.time() - 10 * daemon_start.START_WINDOW_S)  # no status since the launches below
    launched_ago(service, daemon_start.START_WINDOW_S / 2)
    get_job(launching, {"job_id": job_id})
    assert len(platform.spawned) == 1, "within the window, the start is left to finish"
    launched_ago(service, daemon_start.START_WINDOW_S + 1)
    get_job(launching, {"job_id": job_id})
    get_job(launching, {"job_id": job_id})
    assert len(platform.spawned) == 2, "past it, one more launch, and that one's window starts"


def test_a_daemon_that_started_then_died_is_replaced_at_once_s4_1(
    service: Service, platform: LaunchingPlatform, launching: NarrationBackend
) -> None:
    job_id = launched_running(service, launching, platform)
    get_job(launching, {"job_id": job_id})
    died_after_start(service)
    get_job(launching, {"job_id": job_id})
    assert len(platform.spawned) == 2


def test_a_refused_start_holds_back_no_later_launch_s4_1(
    service: Service, platform: LaunchingPlatform, launching: NarrationBackend
) -> None:
    job_id = left_running(service)  # submitted through the fake launcher: no launch recorded
    platform.refuse_spawn = NarrationError(codes.DAEMON_UNAVAILABLE, "left in a job", retry_after_s=60.0)
    error = refused(lambda: get_job(launching, {"job_id": job_id}))
    assert error.code == codes.DAEMON_UNAVAILABLE
    assert daemon_start.read_launch(service.world.store.root) is None, "the platform let no daemon run"
    platform.refuse_spawn = None
    get_job(launching, {"job_id": job_id})
    assert len(platform.spawned) == 1, "the next poll launches at once"


def test_a_launched_daemon_that_exits_before_it_serves_is_a_failed_start_s4_1(
    service: Service, platform: LaunchingPlatform, launching: NarrationBackend
) -> None:
    job_id = launched_running(service, launching, platform)
    get_job(launching, {"job_id": job_id})  # launches one
    platform.end_launched()  # it exits before it writes a status (an import or configuration error, say)
    error = refused(lambda: get_job(launching, {"job_id": job_id}))
    assert (error.code, error.retryable) == (codes.DAEMON_UNAVAILABLE, True)
    assert error.retry_after_s is not None and error.retry_after_s >= DAEMON_RETRY_S
    assert job_id in error.message and "exited before it served" in error.message
    assert error.details is not None
    assert error.details["log"] == str(service.world.store.root / "logs" / "daemon.log"), "the store's own log"
    assert error.details["launched_pid"] == platform.launched[-1].pid
    assert error.hint is not None and "narration-admin daemon start" in error.hint and "details.log" in error.hint
    assert len(platform.spawned) == 1, "no other launched at once: it would most likely fail the same way"
    launched_ago(service, daemon_start.START_WINDOW_S + 1)
    get_job(launching, {"job_id": job_id})
    assert len(platform.spawned) == 2, "past the window, a call launches one again"


def test_an_older_daemon_that_failed_through_its_finally_is_still_a_failed_start_s4_1(
    service: Service, platform: LaunchingPlatform, launching: NarrationBackend
) -> None:
    job_id = launched_running(service, launching, platform)
    get_job(launching, {"job_id": job_id})
    time.sleep(0.01)
    # An older daemon took the singleton, failed, and its finally wrote stopped with no start time: nothing says a
    # start since the launch, so the rule before contracts 1.6.11 holds.
    stopped_status(service, reason=LEGACY)
    platform.end_launched()
    error = refused(lambda: get_job(launching, {"job_id": job_id}))
    assert error.code == codes.DAEMON_UNAVAILABLE and error.details is not None and "log" in error.details
    assert len(platform.spawned) == 1


@pytest.mark.parametrize("reason", [STOP_ERROR, STOP_IDLE, STOP_INTERRUPTED, STOP_OPERATOR])
def test_a_launched_daemon_that_started_then_stopped_in_the_window_is_not_a_failed_start_s4_1(
    service: Service, platform: LaunchingPlatform, launching: NarrationBackend, reason: StopReason
) -> None:
    job_id = launched_running(service, launching, platform)
    get_job(launching, {"job_id": job_id})  # launches one
    launch = daemon_start.read_launch(service.world.store.root)
    assert launch is not None
    time.sleep(0.01)
    # It took the singleton, served, then stopped, well inside the start window: its stopped status keeps its start.
    stopped_status(service, reason=reason, started_at=launch.launched_at + 0.002)
    platform.end_launched()
    assert daemon_start.check_launch(service.world.store) is None, "the launch started"
    out = get_job(launching, {"job_id": job_id})  # a running job: no operator's stop holds it
    assert "so get_job asked for one to start" in out["message"]
    assert "exited before it served" not in out["message"]
    assert len(platform.spawned) == 2, "a daemon is asked for at once"


def test_a_stopped_status_while_the_launched_daemon_runs_is_still_a_start_s4_1(
    service: Service, platform: LaunchingPlatform, launching: NarrationBackend
) -> None:
    job_id = launched_running(service, launching, platform)
    get_job(launching, {"job_id": job_id})
    time.sleep(0.01)
    launch = daemon_start.read_launch(service.world.store.root)
    assert launch is not None
    # The exiting daemon's, written while the launched one waits for the singleton: it started before the launch.
    stopped_status(service, reason=STOP_IDLE, started_at=launch.launched_at - 60)
    out = get_job(launching, {"job_id": job_id})
    assert "is still starting, so get_job asked for no other" in out["message"]
    assert len(platform.spawned) == 1


def test_a_launch_is_alive_only_under_its_own_process_s4_1() -> None:
    created = psutil.Process().create_time()
    assert launch_alive(os.getpid(), created)
    assert launch_alive(os.getpid(), created + 1.0), "created a moment before the time was taken"
    assert not launch_alive(os.getpid(), created + 60), "a process older than the launch is not the one launched"
    assert not launch_alive(os.getpid(), created - 60), "one created long after the launch reused its pid"
    assert not launch_alive(DEAD_PID, time.time())
    assert not launch_alive(0, time.time())


def test_the_start_window_covers_a_takeover_wait_s4_1(tmp_path: Path) -> None:
    takeover = DaemonSettings(store_root=tmp_path).takeover_wait_s
    window = daemon_start.START_WINDOW_S
    assert window >= takeover + 20, "a launched daemon may wait that long for the singleton"


def test_check_launch_reads_the_record_the_status_and_the_process_s4_1(
    service: Service, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = service.world.store
    alive = [True]
    monkeypatch.setattr(daemon_start, "launch_alive", lambda pid, launched_at: alive[0])

    def state(at: float) -> str | None:
        check = daemon_start.check_launch(store, now=at)
        return check.state if check is not None else None

    assert daemon_start.check_launch(store) is None, "no launch recorded"
    now = time.time()
    daemon_start.record_launch(store.root, pid=11, launched_at=now)
    assert state(now + 1) == "starting" and daemon_start.launch_in_progress(store, now=now + 1) is not None
    assert state(now - 10) == "starting", "a clock stepped back a little"
    assert state(now - daemon_start.START_WINDOW_S - 1) is None
    assert state(now + daemon_start.START_WINDOW_S) is None
    alive[0] = False
    assert state(now + 1) == "failed", "its process has gone, and it wrote no status"
    assert daemon_start.launch_in_progress(store, now=now + 1) is None, "a failed start is not in progress"
    store.put_daemon_status(dataclasses.replace(daemon_status(), started_at=utc_iso(now + 0.5)))
    assert state(now + 1) is None, "it has started: whether it still runs is running_daemon's to say"
    stopped = dataclasses.replace(daemon_status(state="stopped"), pid=None, updated_at=utc_iso(now + 2))
    store.put_daemon_status(dataclasses.replace(stopped, started_at=utc_iso(now + 0.5), stop_reason=STOP_ERROR))
    assert state(now + 3) is None, "it started, then stopped: a start, running or not"
    store.put_daemon_status(dataclasses.replace(stopped, started_at=utc_iso(now), stop_reason=STOP_IDLE))
    assert state(now + 3) is None, "a start in the launch's own instant counts"
    store.put_daemon_status(dataclasses.replace(stopped, started_at=utc_iso(now - 60), stop_reason=STOP_IDLE))
    assert state(now + 3) == "failed", "a stopped status from a daemon started before the launch, its process gone"
    store.put_daemon_status(dataclasses.replace(stopped, started_at=None))
    assert state(now + 3) == "failed", "an older daemon's stopped status keeps no start time"
    alive[0] = True
    assert state(now + 3) == "starting", "the exiting daemon's stopped, while the launched one waits"
    daemon_start.launch_path(store.root).write_text("{not json", encoding="utf-8")
    assert daemon_start.read_launch(store.root) is None, "a torn record is no launch"


# ======================================================================== a queued job after a stop (4.1)


def queued(service: Service) -> str:
    """A job submitted and still queued; its submission's ``ensure`` is forgotten. Store times are cut to the
    millisecond, so the helper waits a little: whatever a test does next is stamped later."""
    job_id = service.backend.submit_job_sync(service.request(LAMPS))["job_id"]
    service.launcher.ensured.clear()
    time.sleep(0.01)
    return job_id


def answered(service: Service, command: DaemonCommand) -> DaemonCommand:
    """The daemon answers a stop it stops for, as ``narration.daemon.service`` does."""
    time.sleep(0.01)
    return service.world.store.complete_command(command.command_id, {"stopped": True, "requeued": []})


LEGACY: None = None
"""A ``stop_reason`` of None: the status an older daemon (before contracts 1.6.11) wrote, with no reason and, once
``stopped``, no start time."""


def stopped_status(
    service: Service,
    *,
    reason: StopReason | None,
    after: DaemonCommand | None = None,
    later_s: float = 0.01,
    started_at: float | None = None,
) -> None:
    """``run/daemon.json`` as a daemon's exit leaves it: ``stopped`` for ``reason``, with no pid, written ``later_s``
    after ``after`` was answered (or now). It keeps the daemon's start time (``started_at``, by default a second
    before it stopped), except under ``LEGACY``, which keeps none."""
    at = parse_iso(after.done_at) + later_s if after is not None and after.done_at is not None else time.time()
    start = None if reason is None else utc_iso(at - 1.0 if started_at is None else started_at)
    status = dataclasses.replace(
        daemon_status(state="stopped"), pid=None, started_at=start, updated_at=utc_iso(at), stop_reason=reason
    )
    service.world.store.put_daemon_status(status)


def past_the_grace(service: Service) -> None:
    service.backend.clock = lambda: time.time() + DAEMON_START_GRACE_S + 1


@pytest.mark.parametrize(
    ("kind", "later_s", "reason"),
    [
        ("stop", 0.01, STOP_OPERATOR),
        ("stop_now", 0.01, STOP_OPERATOR),
        ("stop", 5.0, STOP_OPERATOR),
        ("stop", STOP_TO_STOPPED_S + 60, STOP_OPERATOR),
        ("stop", 0.01, LEGACY),
        ("stop", 5.0, LEGACY),
    ],
    ids=[
        "stop",
        "stop_now",
        "honoured by the daemon that took over",
        "a slow takeover: no timing rule under a reason",
        "an older daemon's status",
        "an older daemon's status, the daemon that took over",
    ],
)
def test_a_job_queued_before_an_operators_stop_waits_for_the_next_start_s4_1(
    service: Service, kind: DaemonCommandKind, later_s: float, reason: StopReason | None
) -> None:
    job_id = queued(service)
    stop = answered(service, service.world.store.post_command(kind))
    stopped_status(service, after=stop, later_s=later_s, reason=reason)
    past_the_grace(service)
    out = get_job(service.backend, {"job_id": job_id})
    assert service.launcher.ensured == [], "the operator stopped the service; get_job starts none"
    assert out["status"] == "queued"
    message = out["message"]
    assert "A stop was posted after this job was queued (narration-admin daemon stop, or" in message
    assert "narration-admin daemon start" in message and "submit_job" in message, "how the job resumes"
    found = operator_stop(service.world.store, service.world.job(job_id))
    assert found is not None and found.command_id == stop.command_id


def test_an_operators_stop_starts_nothing_with_autostart_off_either_s4_1(service: Service) -> None:
    config = dataclasses.replace(service.world.config, daemon=DaemonConfig(autostart=False))
    backend, _, launcher = make_backend(dataclasses.replace(service.world, config=config))
    job_id = queued(service)
    stopped_status(service, after=answered(service, service.world.store.post_command("stop")), reason=STOP_OPERATOR)
    backend.clock = lambda: time.time() + DAEMON_START_GRACE_S + 1
    out = get_job(backend, {"job_id": job_id})
    assert launcher.ensured == []
    assert "narration-admin daemon stop" in out["message"] and "autostart is off" not in out["message"]


def test_a_job_queued_while_a_stop_finished_its_segment_starts_a_daemon_s4_1(service: Service) -> None:
    # (c): the stop was posted, the job was submitted while the old daemon finished its in-flight segment (the
    # daemon its submission launched gave up waiting for the singleton), then the old daemon answered and stopped.
    stop = service.world.store.post_command("stop")
    time.sleep(0.01)
    job_id = queued(service)
    stopped_status(service, after=answered(service, stop), reason=STOP_OPERATOR)
    past_the_grace(service)
    out = get_job(service.backend, {"job_id": job_id})
    assert len(service.launcher.ensured) == 1, "a stop asked before the job was queued is not its stop"
    assert "so get_job asked for one to start" in out["message"]


@pytest.mark.parametrize(
    ("why", "reason"),
    [
        ("its control loop or worker supervisor failed", STOP_ERROR),
        ("it exited idle as the job was queued", STOP_IDLE),
        ("it was interrupted in its terminal", STOP_INTERRUPTED),
        ("an older daemon stopped with no stop answered", LEGACY),
    ],
)
def test_a_daemon_that_stopped_with_no_operators_stop_is_replaced_s4_1(
    service: Service, why: str, reason: StopReason | None
) -> None:
    # (a), (b) and (d): each exit writes stopped (tests/daemon/test_daemon.py pins that), and none answers a stop.
    job_id = queued(service)
    stopped_status(service, reason=reason)
    past_the_grace(service)
    out = get_job(service.backend, {"job_id": job_id})
    assert len(service.launcher.ensured) == 1, why
    assert "so get_job asked for one to start" in out["message"]


def test_a_job_whose_submission_could_not_start_a_daemon_gets_one_later_s4_1(service: Service) -> None:
    # (e): the submission's start failed (the caller was told), and the daemon that was exiting then stopped.
    service.launcher.error = NarrationError(codes.DAEMON_UNAVAILABLE, "no start", retryable=True)
    error = refused(lambda: service.backend.submit_job_sync(service.request(LAMPS)))
    assert error.details is not None
    job_id = str(error.details["job_id"])
    assert service.world.job(job_id).status == "queued"
    service.launcher.error = None
    service.launcher.ensured.clear()
    time.sleep(0.01)
    stopped_status(service, reason=STOP_IDLE)
    past_the_grace(service)
    get_job(service.backend, {"job_id": job_id})
    assert len(service.launcher.ensured) == 1


def test_a_daemon_that_crashed_is_replaced_for_a_queued_job_s4_1(service: Service) -> None:
    job_id = queued(service)
    dead_status(service, time.time())  # busy, and its pid is gone: no finally ran
    past_the_grace(service)
    get_job(service.backend, {"job_id": job_id})
    assert len(service.launcher.ensured) == 1


@pytest.mark.parametrize(
    ("reason", "later_s"),
    [
        (STOP_ERROR, 0.01),
        (STOP_IDLE, 0.01),
        (STOP_INTERRUPTED, 5.0),
        (LEGACY, STOP_TO_STOPPED_S + 60),
    ],
    ids=["failed at once", "idle at once", "interrupted", "an older daemon's status, past the 30 s"],
)
def test_a_daemon_that_served_after_the_stop_then_stopped_for_another_reason_is_replaced_s4_1(
    service: Service, reason: StopReason | None, later_s: float
) -> None:
    job_id = queued(service)
    stop = answered(service, service.world.store.post_command("stop"))
    # A later daemon started, served, and stopped for something else. Under a stop reason that decides it however
    # soon after the stop's answer the status was written; under an older daemon's status, only the timing rule.
    stopped_status(service, after=stop, later_s=later_s, reason=reason)
    past_the_grace(service)
    get_job(service.backend, {"job_id": job_id})
    assert len(service.launcher.ensured) == 1, "the operator's stop was lifted by the start that followed it"


def test_an_older_daemons_status_keeps_the_30_s_rule_s4_1(service: Service) -> None:
    store = service.world.store
    job_id = queued(service)
    stop = answered(service, store.post_command("stop"))
    stopped_status(service, after=stop, later_s=STOP_TO_STOPPED_S - 1, reason=LEGACY)
    assert operator_stop(store, service.world.job(job_id)) is not None, "answered within 30 s of stopped"
    stopped_status(service, after=stop, later_s=STOP_TO_STOPPED_S + 1, reason=LEGACY)
    assert operator_stop(store, service.world.job(job_id)) is None, "answered longer before: a later daemon's stop"


def test_only_a_stop_answered_stopped_true_is_the_operators_s4_1(service: Service) -> None:
    store = service.world.store
    job_id = queued(service)
    stale = store.post_command("stop")
    time.sleep(0.01)
    done = store.complete_command(stale.command_id, {"stopped": False, "reason": "stale"})
    stopped_status(service, after=done, reason=STOP_OPERATOR)
    assert operator_stop(store, service.world.job(job_id)) is None, "a stop answered stopped: false"
    release = store.post_command("release_gpu")
    time.sleep(0.01)
    stopped_status(service, after=store.complete_command(release.command_id, {"released": False}), reason=STOP_OPERATOR)
    assert operator_stop(store, service.world.job(job_id)) is None, "not a stop"


def test_no_stopped_status_means_no_operators_stop_s4_1(service: Service) -> None:
    store = service.world.store
    job_id = queued(service)
    stop = answered(service, store.post_command("stop"))
    assert operator_stop(store, service.world.job(job_id)) is None, "no status at all"
    store.put_daemon_status(dataclasses.replace(daemon_status(state="stopping"), updated_at=stop.done_at or ""))
    assert operator_stop(store, service.world.job(job_id)) is None, "a daemon that has not stopped yet"
    (store.root / "run" / "daemon.json").write_text("{torn", encoding="utf-8")
    assert operator_stop(store, service.world.job(job_id)) is None, "an unreadable status"


def test_the_operators_stop_note_is_dropped_once_a_daemon_moved_the_job_on_s4_1() -> None:
    revival = Revival(action="stopped", status="queued")
    assert "stays in the queue" in revival.note(status_now="queued")
    assert "a daemon has started since" in revival.note(status_now="running")


def test_a_stop_in_the_jobs_own_millisecond_counts_as_after_it_s4_1(service: Service) -> None:
    store = service.world.store
    job_id = queued(service)
    stop = answered(service, store.post_command("stop"))
    stopped_status(service, after=stop, reason=STOP_OPERATOR)
    job = service.world.job(job_id)
    same = dataclasses.replace(job, created_at=stop.requested_at)
    assert operator_stop(store, same) is not None, "store times are cut to the millisecond, as posted_after_launch"
    later = dataclasses.replace(job, created_at=utc_iso(parse_iso(stop.requested_at) + 0.002))
    assert operator_stop(store, later) is None, "a job queued after the stop belongs to the next start"
