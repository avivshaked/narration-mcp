"""A ``measure`` job's life in the daemon (design sections 4, 4.1 and 8): given back and taken again, giving way
to an interactive job, cancelled, and stopped by a take the engine could not make. Every take made is kept in
the cache, so a measurement taken up again never renders a take twice."""

from __future__ import annotations

from collections import Counter

from narration.contracts import codes
from narration.measure import lookup_measurement, voice_hash_of
from narration.measure.handler import RETRY_AFTER_S
from tests.jobs.support import ENGINE_ID, VOICE_TRANSCRIPT

from .support import MeasureWorld

LANTERN = "A lantern swings above the ferry ramp while the tide slides out."


def renders(world: MeasureWorld) -> Counter[tuple[str, int]]:
    """How often each (engine text, seed) was sent to the Qwen worker."""
    return Counter((str(p["engine_text"]), int(p["seed"])) for _, op, p in world.pool.requests if op == "synthesize")


def synthesized(world: MeasureWorld) -> int:
    return world.pool.calls[("qwen", "synthesize")]


def is_measured(world: MeasureWorld) -> bool:
    vh = voice_hash_of(clip_sha256=world.clip_sha256, transcript=VOICE_TRANSCRIPT, config=world.config)
    return lookup_measurement(world.store, voice_hash=vh, engine_profile_id=ENGINE_ID) is not None


def test_a_measurement_given_back_resumes_from_the_cache_s4_1(world: MeasureWorld) -> None:
    job = world.measure()
    world.step_until(lambda: synthesized(world) == 3)
    world.host.stop_mode = "now"
    world.runner.shutdown(world.host, "now")
    back = world.job(job.job_id)
    assert back.status == "queued" and back.phase is None

    world.restart()
    world.run()
    done = world.job(job.job_id)
    assert done.status == "completed", done.error
    assert set(renders(world).values()) == {1}  # no take rendered twice
    assert world.host.started == [job.job_id, job.job_id]
    assert is_measured(world)


def test_a_measurement_gives_way_to_an_interactive_job_and_resumes_s4(world: MeasureWorld) -> None:
    """The interactive job here is a generation with the voice being measured, which is refused; what matters
    is that the measurement lets go between two pieces of work and finishes from the cache afterwards."""
    job = world.measure()
    world.step_until(lambda: synthesized(world) == 3)
    urgent = world.generate(LANTERN, priority="interactive")
    world.run()
    assert world.host.started == [job.job_id, urgent.job_id, job.job_id]
    refused = world.job(urgent.job_id)
    assert refused.error is not None and refused.error.code == codes.VOICE_NOT_MEASURED
    done = world.job(job.job_id)
    assert done.status == "completed", done.error
    assert set(renders(world).values()) == {1}


def test_a_cancelled_measurement_publishes_nothing_and_keeps_its_takes_s8(world: MeasureWorld) -> None:
    job = world.measure()
    world.step_until(lambda: synthesized(world) == 3)
    assert world.store.update_job(job.job_id, expect_status="running", status="cancelling") is not None
    world.run()
    cancelled = world.job(job.job_id)
    assert cancelled.status == "cancelled"
    assert not is_measured(world)

    again = world.measure()
    world.run()
    assert world.job(again.job_id).status == "completed"
    assert set(renders(world).values()) == {1}  # the three renders were kept
    assert is_measured(world)


def test_a_take_the_engine_cannot_make_stops_the_measurement_retryably_s3_2(world: MeasureWorld) -> None:
    """A worker that crashes twice on one take: a measurement short of a take is not the measurement, so the job
    fails with a retryable ``INTERNAL``, and the identical request finishes from what was made."""
    world.faults({"kind": "crash", "op": "synthesize", "when": {"text_contains": "summit"}, "times": 2})
    job = world.measure()
    world.run()
    failed = world.job(job.job_id)
    assert failed.status == "failed"
    assert failed.error is not None and failed.error.code == codes.INTERNAL
    assert failed.error.retryable and failed.error.retry_after_s == RETRY_AFTER_S
    assert failed.error.details is not None
    assert failed.error.details["segment_id"] == "ladder-150"
    assert failed.error.details["flag"] == codes.WORKER_CRASHED
    assert not is_measured(world)

    world.faults()
    again = world.measure()
    world.run()
    assert world.job(again.job_id).status == "completed", world.job(again.job_id).error
    counts = renders(world)
    crashed = [pair for pair in counts if "summit" in pair[0] and counts[pair] > 1]
    assert len(crashed) == 1 and counts[crashed[0]] == 3  # two crashes, then made once
    assert all(n == 1 for pair, n in counts.items() if pair not in crashed)


def test_running_out_of_gpu_memory_twice_is_gpu_unavailable_s4(world: MeasureWorld) -> None:
    world.faults({"kind": "gpu_oom", "op": "synthesize", "when": {"text_contains": "Tarnholm"}, "times": 2})
    job = world.measure()
    world.run()
    failed = world.job(job.job_id)
    assert failed.status == "failed"
    assert failed.error is not None and failed.error.code == codes.GPU_UNAVAILABLE
    assert failed.error.retryable and failed.error.retry_after_s is not None
    assert failed.error.details is not None
    assert (failed.error.details["segment_id"], failed.error.details["flag"]) == ("ladder-080", codes.GPU_OOM)
    assert not is_measured(world)


def test_the_handler_stops_the_lease_thread_it_shares_when_closed_s4_1(world: MeasureWorld) -> None:
    """``Registry.close`` (the runner's shutdown) closes the measure handler too: it shares the job engine's
    ``LeaseKeeper``, and closing it more than once is harmless."""
    world.measure()
    world.run()
    leases = world.engine.core.leases
    assert leases.running  # started at the first lease, idle since
    world.handler.close()
    assert not leases.running
    world.runner.registry.close()  # the engine and the handler again
    assert not leases.running
