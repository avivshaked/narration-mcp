"""An active job whose daemon has gone (design sections 4, 4.1, 7.4 and 14; plan.md DC-2).

A job stays ``queued``, ``running`` or ``cancelling`` in the store after its daemon has gone (a crash, a machine
restart, a daemon killed with its client), and only a daemon moves it on. So ``get_job`` and ``cancel_job`` ask
the launcher for a daemon when none runs, and ``get_job`` says so in its reply, or answers ``DAEMON_UNAVAILABLE``
with what to do when none can be started. The fake launcher starts nothing; where a test needs the new daemon's
start-up, it runs the daemon's own sweep (``narration.daemon.sweep``) at ``ensure``.

Every text is invented for these tests.
"""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Callable
from typing import Any

import anyio
import pytest
from jsonschema import Draft202012Validator

from narration.backend.launch import DAEMON_RETRY_S, unavailable
from narration.backend.service import DAEMON_START_GRACE_S, NarrationBackend
from narration.config import DaemonConfig
from narration.contracts import codes
from narration.contracts.errors import NarrationError, UnsupportedPlatform
from narration.contracts.schemas import TOOLS_BY_NAME
from narration.daemon.sweep import sweep
from narration.store.store import utc_iso
from tests.jobs.support import LAMPS

from .conftest import Service
from .support import daemon_status, make_backend


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


def left_running(service: Service) -> str:
    """A job a daemon took and then left: ``running`` in the store, and no daemon runs (the fake launcher's
    status is None)."""
    job_id = service.backend.submit_job_sync(service.request(LAMPS))["job_id"]
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


def test_get_job_starts_a_daemon_for_a_job_left_running_and_says_so_s4_1(service: Service) -> None:
    job_id = left_running(service)
    service.launcher.on_ensure = start_up_sweep(service)
    out = get_job(service.backend, {"job_id": job_id, "wait_s": 5})
    assert len(service.launcher.ensured) == 1, "one daemon asked for"
    assert out["status"] == "queued", "the new daemon's sweep put the job back on the queue"
    assert out["message"].startswith("queued again: ")
    assert "daemon state: stopped" in out["message"]
    assert "get_job started one" in out["message"]
    assert service.world.job(job_id).status == "queued"


def test_get_job_reports_the_start_before_the_new_daemon_has_run_s7_4(service: Service) -> None:
    job_id = left_running(service)
    out = get_job(service.backend, {"job_id": job_id})
    assert len(service.launcher.ensured) == 1
    assert out["status"] == "running"
    assert "No daemon was running this job (daemon state: stopped), so get_job started one" in out["message"]
    assert "back on the queue" in out["message"]
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


def test_the_start_counts_against_wait_s_s7_4(service: Service) -> None:
    job_id = left_running(service)
    service.launcher.on_ensure = lambda: time.sleep(1.0)
    started = time.monotonic()
    out = get_job(service.backend, {"job_id": job_id, "wait_s": 1.0})
    assert out["status"] == "running"
    assert time.monotonic() - started < 1.6, "the wait ends at wait_s, the start included"


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


def test_with_autostart_off_the_reply_says_who_must_start_the_daemon_s16(service: Service) -> None:
    config = dataclasses.replace(service.world.config, daemon=DaemonConfig(autostart=False))
    backend, _, launcher = make_backend(dataclasses.replace(service.world, config=config))
    job_id = left_running(service)
    out = get_job(backend, {"job_id": job_id})
    assert len(launcher.ensured) == 1, "the launcher decides; with autostart off it starts nothing"
    assert "[daemon] autostart is off" in out["message"]
    assert "narration-admin daemon start" in out["message"]


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
