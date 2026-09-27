"""The claim loop behind the daemon's seam: ``has_work``, the runner the daemon loads by name, and workers the
daemon stops between jobs (plan.md WP30 and WP31; design sections 4 and 4.1)."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from narration.contracts import codes
from narration.contracts.errors import NarrationError
from narration.contracts.models import JobRecord
from narration.jobs import default_runner, stages
from narration.jobs.handlers import Registry
from narration.jobs.host import RunnerHost
from narration.jobs.runner import EngineRunner
from narration.jobs.state import Outcome

from .conftest import World
from .support import KETTLE, LAMPS, drive


def test_has_work_is_the_steps_test_and_claims_nothing_s4_1(world: World) -> None:
    assert not world.runner.has_work(world.host)
    job = world.submit(LAMPS)
    assert world.runner.has_work(world.host)
    assert world.runner.has_work(world.host)
    assert world.job(job.job_id).status == "queued"  # asking claimed nothing
    assert world.host.started == []

    world.runner.step(world.host)
    assert world.job(job.job_id).status == "running"
    assert world.runner.has_work(world.host)  # a job held is work
    world.run()
    assert world.job(job.job_id).status == "completed"
    assert not world.runner.has_work(world.host)


def test_has_work_ignores_jobs_that_are_not_queued_s4_1(world: World) -> None:
    job = world.submit(LAMPS)
    assert world.store.claim_job(job.job_id, "another-daemon") is not None
    assert not world.runner.has_work(world.host)


def test_shutdown_stops_the_thread_that_renews_the_engines_leases_s4(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(stages, "LEASE_RENEW_S", 0.02)
    world.faults({"kind": "delay", "op": "synthesize", "seconds": 0.2})
    job = world.submit(LAMPS)
    world.run()
    assert world.job(job.job_id).status == "completed"
    assert world.engine.core.leases.running  # it waits, idle, for the next job's leases

    world.runner.shutdown(world.host, "idle")
    assert not world.engine.core.leases.running


def test_the_daemons_default_runner_fails_jobs_it_cannot_score_s4(world: World) -> None:
    runner = default_runner()
    assert isinstance(runner, EngineRunner)
    first, second = world.submit(LAMPS), world.submit(KETTLE)
    assert runner.has_work(world.host)

    drive(runner, world.host)

    for job_id in (first.job_id, second.job_id):
        failed = world.job(job_id)
        assert failed.status == "failed" and failed.error is not None
        assert failed.error.code == codes.BACKEND_NOT_INSTALLED and failed.error.hint
    assert world.pool.starts == 0  # nothing was rendered that could not be checked
    assert world.host.finished == 2
    with pytest.raises(RuntimeError):
        _ = runner.registry


@pytest.mark.parametrize("how", ["stop", "unload"])
def test_workers_the_daemon_let_go_between_jobs_are_loaded_and_prepared_again_s4(world: World, how: str) -> None:
    world.submit(LAMPS)
    world.run()
    for group in ("qwen", "qa"):
        if how == "stop":
            world.pool.stop(group)  # release_gpu, or the idle unload stopping the processes
        else:
            world.pool.unload(group, timeout_s=10.0)
    job = world.submit(KETTLE)
    world.run()

    assert world.job(job.job_id).status == "completed"
    assert world.pool.loads == ["qwen", "qa", "qwen", "qa"]
    assert world.pool.calls[("qwen", "prepare_voice")] == 2  # the voice is prepared again in the new load
    assert world.pool.refused == 0


def test_a_second_gpu_group_is_never_loaded_over_the_first_s4(world: World) -> None:
    world.submit(LAMPS, KETTLE, takes=2)
    world.faults({"kind": "token_cap", "when": {"text_contains": "kettle"}})  # retakes: Qwen, QA, Qwen, QA...
    world.run()
    assert world.pool.refused == 0
    assert world.pool.loads == ["qwen", "qa", "qwen", "qa", "qwen", "qa"]


@dataclass
class _Design:
    """A job of another kind, as a later work package's handler keeps it."""

    job_id: str
    pieces: int = 0


class _DesignHandler:
    """A handler for ``design`` jobs that completes each in two pieces (WP33's place in the registry)."""

    def __init__(self) -> None:
        self.opened: list[str] = []

    def open(self, host: RunnerHost, job: JobRecord) -> _Design:
        self.opened.append(job.job_id)
        return _Design(job.job_id)

    def advance(self, host: RunnerHost, run: _Design) -> Outcome:
        run.pieces += 1
        if run.pieces < 2:
            return "worked"
        host.store.update_job(run.job_id, expect_status="running", status="completed", result={"design_id": "d"})
        return "finished"

    def save(self, host: RunnerHost, run: _Design) -> bool:
        return host.store.update_job(run.job_id, expect_status="running", message=f"{run.pieces}") is not None

    def cancel(self, host: RunnerHost, run: _Design) -> JobRecord | None:
        return host.store.update_job(run.job_id, expect_status="cancelling", status="cancelled")

    def fail(self, host: RunnerHost, job: JobRecord, error: NarrationError, run: _Design | None) -> JobRecord | None:
        return host.store.update_job(job.job_id, expect_status="running", status="failed", error=error.error)

    def release(self, run: _Design) -> None:
        return None

    def remaining_audio_s(self, run: _Design) -> float:
        return 0.0


def test_jobs_are_dispatched_by_kind_to_the_handler_registered_for_it_s8(world: World) -> None:
    design = _DesignHandler()
    base = Registry.of(world.engine)
    registry = Registry({**base.handlers, "design": design}, base.residency, base.throughput)
    runner = EngineRunner(registry)
    made = world.submit_body({"description": "a calm voice", "takes": 1}, kind="design")
    spoken = world.submit(LAMPS)

    drive(runner, world.host)

    assert design.opened == [made.job_id]
    done = world.job(made.job_id)
    assert done.status == "completed" and done.result == {"design_id": "d"}
    assert world.job(spoken.job_id).status == "completed"  # generate still goes to the job engine
    assert world.pool.texts() == [LAMPS]
