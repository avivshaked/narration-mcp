"""WP31's acceptance tests (plan.md): a shared paragraph is rendered once, a failing take is retaken up to
``max_retakes`` and never again on resubmission, and the same request twice completes from the cache.

Every worker is the fake role; every text is invented for these tests.
"""

from __future__ import annotations

import threading
from collections import Counter
from pathlib import Path

import pytest

from narration import keys
from narration.contracts import codes
from narration.contracts.models import JobRecord

from .conftest import World, make_world
from .support import ENGINE_HASH, KETTLE, LAMPS, ORCHARD, voice_hash


def _take_ids(job: JobRecord) -> dict[str, list[str | None]]:
    return {item.segment_id: [a.take_id for a in item.attempts] for item in job.items}


def _seed(world: World, text: str, attempt: int) -> int:
    return keys.seed(voice_hash=voice_hash(world.clip_sha256), engine_text=text, attempt=attempt)


# ======================================================================== a shared paragraph is rendered once


def test_overlapping_jobs_render_a_shared_paragraph_once_s4(world: World) -> None:
    first = world.submit(LAMPS, KETTLE)
    second = world.submit(KETTLE, ORCHARD, ids=["intro", "p09"])  # the segment id is in no key (section 10.3)

    world.run()

    assert Counter(world.pool.texts()) == Counter({LAMPS: 1, KETTLE: 1, ORCHARD: 1})
    one, two = world.job(first.job_id), world.job(second.job_id)
    assert one.status == two.status == "completed"
    shared_one = next(i for i in one.items if i.segment_id == "p02").attempts[0]
    shared_two = next(i for i in two.items if i.segment_id == "intro").attempts[0]
    assert shared_one.render_key == shared_two.render_key
    assert shared_one.take_id == shared_two.take_id
    assert shared_one.analysis_id == shared_two.analysis_id  # the analysis key has no segment id either
    assert (shared_one.fresh, shared_two.fresh) == (True, False)


def _run_to_the_end(world: World, job: JobRecord, errors: list[BaseException]) -> None:
    try:
        run = world.engine.open(world.host, job)
        for _ in range(2000):
            if world.engine.advance(world.host, run) == "finished":
                return
        state = [(a.segment, a.attempt, a.stage) for seg in run.segments for a in seg.attempts()]
        raise AssertionError(f"job {job.job_id} did not finish: {state}, {run.message}")
    except BaseException as exc:  # handed to the test thread
        errors.append(exc)


def test_two_holders_share_a_paragraph_through_its_lease_s4(
    world: World, tmp_path: Path, anchor: tuple[float, ...]
) -> None:
    other = make_world(tmp_path, anchor, holder="second-holder")
    try:
        # The shared paragraph takes a second to render, so the other holder finds its lease in flight.
        world.faults({"kind": "delay", "op": "synthesize", "seconds": 1.0, "when": {"text_contains": "kettle"}})
        world.host.real_sleep_s = other.host.real_sleep_s = 0.02  # a wait takes some real time here
        first = world.submit(KETTLE, LAMPS)
        second = world.submit(KETTLE, ORCHARD, ids=["intro", "p09"])
        mine = world.store.claim_job(first.job_id, world.host.holder)
        theirs = other.store.claim_job(second.job_id, other.host.holder)
        assert mine is not None and theirs is not None

        errors: list[BaseException] = []
        threads = [
            threading.Thread(target=_run_to_the_end, args=(world, mine, errors)),
            threading.Thread(target=_run_to_the_end, args=(other, theirs, errors)),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=120)
        assert not errors, errors
        assert not any(t.is_alive() for t in threads)

        rendered = Counter(world.pool.texts()) + Counter(other.pool.texts())
        assert rendered == Counter({KETTLE: 1, LAMPS: 1, ORCHARD: 1})
        one, two = world.job(first.job_id), world.job(second.job_id)
        assert one.status == two.status == "completed"
        a = next(i for i in one.items if i.segment_id == "p01").attempts[0]
        b = next(i for i in two.items if i.segment_id == "intro").attempts[0]
        assert a.take_id == b.take_id
        assert sorted([a.fresh, b.fresh]) == [False, True]
    finally:
        other.pool.close()
        other.store.close()


def test_work_another_holder_is_making_is_waited_for_never_made_twice_s4(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    key = keys.render_key(
        engine_profile_hash=ENGINE_HASH,
        voice_hash=voice_hash(world.clip_sha256),
        engine_text=LAMPS,
        seed=_seed(world, LAMPS, 0),
    )
    status, lease = world.store.claim(key, "another-process", ttl_s=600)
    assert status == "claimed" and lease is not None
    waited: list[str] = []

    def recording_wait_for(key: str, *, timeout_s: float) -> bool:
        waited.append(key)
        return False

    monkeypatch.setattr(world.store, "wait_for", recording_wait_for)
    job = world.submit(LAMPS, KETTLE)

    for _ in range(25):
        world.runner.step(world.host)
    assert world.pool.texts() == [KETTLE]  # the paragraph in flight elsewhere is not rendered here
    assert world.job(job.job_id).status == "running"
    # Every wait inside a step goes through host.sleep, which a stop ends at once; the store's own wait
    # (real time, deaf to a stop) is never used.
    assert waited == []
    assert world.host.sleeps.count(world.engine.parts.defer_s) >= 5

    lease.release()  # the other holder gave up without a result: now it is this job's to make
    world.run()
    assert world.pool.texts() == [KETTLE, LAMPS]
    assert world.job(job.job_id).status == "completed"


def test_work_in_flight_elsewhere_costs_no_model_load_s4(world: World) -> None:
    key = keys.render_key(
        engine_profile_hash=ENGINE_HASH,
        voice_hash=voice_hash(world.clip_sha256),
        engine_text=LAMPS,
        seed=_seed(world, LAMPS, 0),
    )
    status, lease = world.store.claim(key, "another-process", ttl_s=600)
    assert status == "claimed" and lease is not None
    job = world.submit(LAMPS)
    for _ in range(10):
        world.runner.step(world.host)
    assert world.pool.loads == []  # the key is claimed before its model group is loaded
    assert world.job(job.job_id).status == "running"

    lease.release()
    world.run()
    assert world.job(job.job_id).status == "completed"
    assert world.pool.loads == ["qwen", "qa"]


# ======================================================================== retakes, and never again on resubmission


def test_failing_take_is_retaken_up_to_max_retakes_and_never_again_on_resubmission_s8(world: World) -> None:
    world.faults({"kind": "token_cap", "when": {"text_contains": "lamplighter"}})  # every take of it is cut
    first = world.submit(LAMPS, KETTLE, max_retakes=2)

    world.run()

    seeds = [p["seed"] for _, op, p in world.pool.requests if op == "synthesize" and p["engine_text"] == LAMPS]
    assert seeds == [_seed(world, LAMPS, n) for n in (0, 1, 2)]  # attempts 0, 1, 2: one take, two retakes
    job = world.job(first.job_id)
    assert job.status == "completed" and job.outcome == "needs_attention" and job.round == 2
    lamps = next(i for i in job.items if i.segment_id == "p01")
    assert lamps.state == "failed_qa" and lamps.retakes_used == 2 and lamps.takes_ok == 0
    assert [(a.attempt, a.round, a.verdict) for a in lamps.attempts] == [(0, 0, "fail"), (1, 1, "fail"), (2, 2, "fail")]
    assert [[f.code for f in a.flags] for a in lamps.attempts] == [[], [codes.RETAKEN], [codes.RETAKEN]]
    for attempt in lamps.attempts:
        assert attempt.analysis_id is not None
        analysis = world.store.get_analysis_by_id(attempt.analysis_id)
        assert analysis is not None
        assert codes.TOKEN_CAP_HIT in {f.code for f in analysis.qa.flags}
    kettle = next(i for i in job.items if i.segment_id == "p02")
    assert kettle.state == "passed" and [a.attempt for a in kettle.attempts] == [0]

    renders, scored = world.pool.calls[("qwen", "synthesize")], world.pool.calls[("qa", "transcribe")]
    again = world.submit(LAMPS, KETTLE, max_retakes=2)
    world.run()

    assert world.pool.calls[("qwen", "synthesize")] == renders  # nothing is retaken again
    assert world.pool.calls[("qa", "transcribe")] == scored
    redo = world.job(again.job_id)
    assert redo.status == "completed" and redo.outcome == "needs_attention"
    assert _take_ids(redo) == _take_ids(job)
    assert not any(a.fresh for item in redo.items for a in item.attempts)


def test_a_retake_on_the_next_attempt_that_passes_is_suggested_s8(world: World) -> None:
    world.faults({"kind": "token_cap", "when": {"text_contains": "lamplighter", "seed": _seed(world, LAMPS, 0)}})
    job = world.submit(LAMPS)

    world.run()

    done = world.job(job.job_id)
    item = done.items[0]
    assert [(a.attempt, a.verdict) for a in item.attempts] == [(0, "fail"), (1, "pass")]
    assert item.state == "passed" and item.takes_ok == 1 and item.retakes_used == 1
    assert done.outcome == "all_passed"
    assert done.result is not None
    suggestion = done.result["suggestions"][0]
    assert suggestion["take_id"] == item.attempts[1].take_id
    assert suggestion["suggestion"]["tier"] == 1


def test_no_retake_when_max_retakes_is_zero_s8(world: World) -> None:
    world.faults({"kind": "token_cap", "when": {"text_contains": "lamplighter"}})
    job = world.submit(LAMPS, max_retakes=0)
    world.run()
    item = world.job(job.job_id).items[0]
    assert [a.attempt for a in item.attempts] == [0]
    assert item.state == "failed_qa" and item.retakes_used == 0
    assert world.pool.texts() == [LAMPS]


def test_max_retakes_is_per_take_slot_s8(world: World) -> None:
    world.faults({"kind": "token_cap", "when": {"text_contains": "lamplighter"}})
    job = world.submit(LAMPS, takes=2, max_retakes=2)
    world.run()
    done = world.job(job.job_id)
    item = done.items[0]
    # Two slots, each retaken twice, on the next attempt numbers in slot order, a round at a time.
    assert [(a.attempt, a.round) for a in item.attempts] == [(0, 0), (2, 1), (4, 2), (1, 0), (3, 1), (5, 2)]
    assert item.retakes_used == 4 and done.round == 2
    assert world.pool.calls[("qwen", "synthesize")] == 6


def test_retakes_of_named_attempts_follow_every_attempt_so_far_s8(world: World) -> None:
    world.faults({"kind": "token_cap", "when": {"text_contains": "lamplighter"}})
    body = world.request(LAMPS, max_retakes=1)
    body["segments"][0]["attempts"] = [5, 2]
    job = world.submit_body(body)
    world.run()
    item = world.job(job.job_id).items[0]
    assert [(a.attempt, a.round) for a in item.attempts] == [(5, 0), (6, 1), (2, 0), (7, 1)]


# ======================================================================== the same request twice, from the cache


def test_the_same_request_twice_completes_from_the_cache_s4(world: World) -> None:
    first = world.submit(LAMPS, KETTLE, takes=2)
    world.run()
    before = Counter(world.pool.calls)
    loads = list(world.pool.loads)
    assert before[("qwen", "synthesize")] == 4 and before[("qa", "transcribe")] == 4

    second = world.submit(LAMPS, KETTLE, takes=2)
    world.run()

    assert world.pool.calls == before  # no render, no post-processing request, no scoring, no load
    assert world.pool.loads == loads
    one, two = world.job(first.job_id), world.job(second.job_id)
    assert two.status == "completed" and two.outcome == one.outcome == "all_passed"
    assert _take_ids(two) == _take_ids(one)
    assert [a.analysis_id for i in two.items for a in i.attempts] == [
        a.analysis_id for i in one.items for a in i.attempts
    ]
    assert all(a.fresh for i in one.items for a in i.attempts)
    assert not any(a.fresh for i in two.items for a in i.attempts)
    assert two.result == one.result  # the same suggestions and the same consistency report


def test_engine_instances_share_nothing_but_the_store_s0_2(world: World) -> None:
    """Stateless: a new engine (a restarted daemon) answers the same request from the store alone."""
    first = world.submit(LAMPS)
    world.run()
    world.new_engine()
    renders = world.pool.calls[("qwen", "synthesize")]
    second = world.submit(LAMPS)
    world.run()
    assert world.pool.calls[("qwen", "synthesize")] == renders
    assert _take_ids(world.job(second.job_id)) == _take_ids(world.job(first.job_id))
