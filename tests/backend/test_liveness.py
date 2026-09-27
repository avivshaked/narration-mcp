"""An active job whose daemon has gone (design sections 4, 4.1, 7.4 and 14; plan.md DC-2).

A job stays ``queued``, ``running`` or ``cancelling`` in the store after its daemon has gone (a crash, a machine
restart, a daemon killed with its client), and only a daemon moves it on. So ``get_job`` and ``cancel_job`` ask
the launcher for a daemon when none runs, and ``get_job`` says so in its reply, or answers ``DAEMON_UNAVAILABLE``
with what to do when none can be started. The fake launcher starts nothing; where a test needs the new daemon's
start-up, it runs the daemon's own sweep (``narration.daemon.sweep``) at ``ensure``.

While a launched daemon is still starting (``run/launch.json``, ``narration.daemon.start``), no other is asked
for. Those tests run the real ``DetachedLauncher`` and ``start_detached`` over the stand-in platform, which
records each launch and starts nothing.

Every text is invented for these tests.
"""

from __future__ import annotations

import dataclasses
import logging
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import anyio
import pytest
from jsonschema import Draft202012Validator

from narration.backend.launch import DAEMON_RETRY_S, DetachedLauncher, unavailable
from narration.backend.service import DAEMON_START_GRACE_S, NarrationBackend
from narration.config import DaemonConfig
from narration.contracts import codes
from narration.contracts.errors import NarrationError, UnsupportedPlatform
from narration.contracts.models import JobRecord
from narration.contracts.schemas import TOOLS_BY_NAME
from narration.daemon import start as daemon_start
from narration.daemon.settings import DaemonSettings
from narration.daemon.sweep import sweep
from narration.platform.testing import StandInPlatform
from narration.store.store import utc_iso
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


@pytest.fixture
def platform(monkeypatch: pytest.MonkeyPatch) -> Iterator[StandInPlatform]:
    """The stand-in platform as ``start_detached``'s: each launch is recorded, and nothing is started."""
    stand_in = StandInPlatform()
    monkeypatch.setattr("narration.platform.get_platform", lambda: stand_in)
    yield stand_in


@pytest.fixture
def launching(service: Service, platform: StandInPlatform, tmp_path: Path) -> NarrationBackend:
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


def launched_running(service: Service, backend: NarrationBackend, platform: StandInPlatform) -> str:
    """A job submitted through ``backend`` (which launched a daemon), taken by that daemon, which then died:
    ``running`` with no daemon, and no launch still starting. The platform's record of launches is cleared."""
    job_id = backend.submit_job_sync(service.request(LAMPS))["job_id"]
    assert service.world.store.claim_job(job_id, "a-daemon") is not None
    died_after_start(service)
    platform.spawned.clear()
    return job_id


def launched_ago(service: Service, seconds: float) -> None:
    """Record the last launch as made ``seconds`` ago."""
    daemon_start.record_launch(service.world.store.root, pid=4242, launched_at=time.time() - seconds)


def test_a_launch_is_recorded_for_the_next_launcher_s4_1(
    service: Service, platform: StandInPlatform, launching: NarrationBackend
) -> None:
    job_id = launched_running(service, launching, platform)
    before = time.time()
    get_job(launching, {"job_id": job_id})
    assert len(platform.spawned) == 1
    launch = daemon_start.read_launch(service.world.store.root)
    assert launch is not None and launch.pid == platform.spawn_pid
    assert before - 0.01 <= launch.launched_at <= time.time()


def test_repeated_polls_during_a_start_launch_one_daemon_s4_1(
    service: Service, platform: StandInPlatform, launching: NarrationBackend
) -> None:
    job_id = launched_running(service, launching, platform)
    first = get_job(launching, {"job_id": job_id})
    assert "asked for one to start" in first["message"]
    for _ in range(3):
        again = get_job(launching, {"job_id": job_id, "wait_s": 0.05})
        assert "is still starting, so get_job asked for no other" in again["message"]
    assert len(platform.spawned) == 1, "one launch while it starts"


def test_cancel_then_get_job_launch_one_daemon_s4_1(
    service: Service, platform: StandInPlatform, launching: NarrationBackend
) -> None:
    job_id = launched_running(service, launching, platform)
    assert launching.cancel_job_sync({"job_id": job_id}) == {"status": "cancelling", "completed": False}
    out = get_job(launching, {"job_id": job_id})
    assert out["status"] == "cancelling"
    assert "still starting" in out["message"]
    assert len(platform.spawned) == 1


def test_a_stuck_start_costs_one_launch_per_window_s4_1(
    service: Service, platform: StandInPlatform, launching: NarrationBackend
) -> None:
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
    service: Service, platform: StandInPlatform, launching: NarrationBackend
) -> None:
    job_id = launched_running(service, launching, platform)
    get_job(launching, {"job_id": job_id})
    died_after_start(service)
    get_job(launching, {"job_id": job_id})
    assert len(platform.spawned) == 2


def test_the_start_window_covers_a_takeover_wait_s4_1(tmp_path: Path) -> None:
    takeover = DaemonSettings(store_root=tmp_path).takeover_wait_s
    window = daemon_start.START_WINDOW_S
    assert window >= takeover + 20, "a launched daemon may wait that long for the singleton"


def test_launch_in_progress_reads_the_record_and_the_status_s4_1(service: Service) -> None:
    store = service.world.store
    assert daemon_start.launch_in_progress(store) is None, "no launch recorded"
    now = time.time()
    daemon_start.record_launch(store.root, pid=11, launched_at=now)
    assert daemon_start.launch_in_progress(store, now=now + 1) is not None
    assert daemon_start.launch_in_progress(store, now=now - 10) is not None, "a clock stepped back a little"
    assert daemon_start.launch_in_progress(store, now=now - daemon_start.START_WINDOW_S - 1) is None
    assert daemon_start.launch_in_progress(store, now=now + daemon_start.START_WINDOW_S) is None
    store.put_daemon_status(dataclasses.replace(daemon_status(), started_at=utc_iso(now + 0.5)))
    assert daemon_start.launch_in_progress(store, now=now + 1) is None, "it has started"
    daemon_start.launch_path(store.root).write_text("{not json", encoding="utf-8")
    assert daemon_start.read_launch(store.root) is None, "a torn record is no launch"
