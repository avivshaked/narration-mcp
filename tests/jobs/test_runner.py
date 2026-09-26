"""The claim loop behind the daemon's seam: ``has_work``, the runner the daemon loads by name, and workers the
daemon stops between jobs (plan.md WP30 and WP31; design sections 4 and 4.1)."""

from __future__ import annotations

import pytest

from narration.contracts import codes
from narration.jobs import default_runner
from narration.jobs.runner import EngineRunner

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
        _ = runner.engine


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
