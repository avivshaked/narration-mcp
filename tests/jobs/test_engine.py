"""The job engine and its claim loop with the fake workers: what a job does when things go wrong, and what
it tells the daemon and the caller (design sections 3.2, 4, 7.4, 8, 10.1, 11.1).

Every text is invented for these tests.
"""

from __future__ import annotations

import hashlib
from collections import Counter

import pytest

from narration.contracts import codes
from narration.contracts.models import Progress
from narration.jobs.admission import GPU_RECHECK_S, GPU_UNAVAILABLE_RETRY_S
from narration.jobs.engine import OOM_WAIT_S
from narration.jobs.pins import call_cap

from .conftest import World
from .support import GENERATION, KETTLE, LAMPS, LANTERN, ORCHARD, QWEN_VRAM_MB, FixedProbe, write_clip


def _step_until(world: World, predicate: object, *, limit: int = 200) -> None:
    assert callable(predicate)
    for _ in range(limit):
        if predicate():
            return
        world.runner.step(world.host)
    raise AssertionError("the condition never held")


# ======================================================================== section 3.2: over-long segments


def test_an_over_long_segment_is_rendered_and_warned_about_never_refused_s3_2(world: World) -> None:
    world.measure(max_segment_chars=40)
    job = world.submit(LAMPS, "Short and plain.")
    world.run()
    done = world.job(job.job_id)
    assert done.status == "completed"
    assert world.pool.texts() == [LAMPS, "Short and plain."]
    long_one, short_one = done.items
    warned = [f for f in long_one.flags if f.code == codes.SEGMENT_TOO_LONG]
    assert len(warned) == 1 and warned[0].severity == "warn" and warned[0].segment_id == "p01"
    assert not any(f.code == codes.SEGMENT_TOO_LONG for f in short_one.flags)
    assert long_one.attempts[0].verdict is not None  # scored like any other


# ======================================================================== section 4: execution problems


def test_out_of_memory_unloads_waits_and_retries_once_s4(world: World) -> None:
    world.faults({"kind": "gpu_oom", "op": "synthesize", "times": 1})
    job = world.submit(LAMPS)
    world.run()
    done = world.job(job.job_id)
    assert done.status == "completed" and done.items[0].state == "passed"
    assert OOM_WAIT_S in world.host.sleeps
    assert "qwen" in world.pool.unloads
    assert world.pool.calls[("qwen", "synthesize")] == 2
    assert not done.items[0].flags


def test_out_of_memory_after_the_retry_fails_that_take_and_the_job_goes_on_s4(world: World) -> None:
    world.faults({"kind": "gpu_oom", "op": "synthesize", "when": {"text_contains": "lamplighter"}, "times": 2})
    job = world.submit(LAMPS, KETTLE)
    world.run()
    done = world.job(job.job_id)
    assert done.status == "completed" and done.outcome == "needs_attention"
    lamps, kettle = done.items
    assert lamps.state == "error"
    (flag,) = [f for f in lamps.flags if f.code == codes.GPU_OOM]
    assert flag.severity == "error" and flag.segment_id == "p01" and flag.retake_trigger is False
    assert lamps.attempts[0].take_id is None and lamps.retakes_used == 0  # an execution error is no retake
    assert kettle.state == "passed"


def test_a_worker_crash_is_retried_once_in_a_new_worker_s4(world: World) -> None:
    world.faults({"kind": "crash", "op": "synthesize", "times": 1})
    job = world.submit(LAMPS)
    world.run()
    done = world.job(job.job_id)
    assert done.status == "completed" and done.items[0].state == "passed"
    assert world.pool.starts == 3  # qwen, qwen again after the crash, qa


def test_a_worker_that_crashes_twice_fails_the_take_with_worker_crashed_s4(world: World) -> None:
    world.faults({"kind": "crash", "op": "transcribe", "times": 2})
    job = world.submit(LAMPS)
    world.run()
    item = world.job(job.job_id).items[0]
    assert item.state == "error"
    assert [f.code for f in item.flags] == [codes.WORKER_CRASHED]
    assert item.attempts[0].take_id is not None and item.attempts[0].analysis_id is None


def test_an_alignment_error_is_a_failed_take_and_retaken_s11_2(world: World) -> None:
    world.faults({"kind": "alignment_error", "times": 1})
    job = world.submit(LAMPS)
    world.run()
    item = world.job(job.job_id).items[0]
    assert [(a.attempt, a.verdict) for a in item.attempts] == [(0, "fail"), (1, "pass")]
    first = world.store.get_analysis_by_id(item.attempts[0].analysis_id or "")
    assert first is not None
    assert codes.ALIGNMENT_ERROR in {f.code for f in first.qa.flags}
    assert all(c.start_s is None and c.end_s is None for c in first.alignment.cues)  # never interpolated


# ======================================================================== section 4: the scheduler


def test_a_scoring_only_job_loads_only_the_qa_group_s4(world: World) -> None:
    world.submit(LAMPS)
    world.run()
    calls, loads = Counter(world.pool.calls), list(world.pool.loads)
    hint = {"term": "lamplighter", "asr_aliases": ["lamp lighter"]}  # a new analysis key, the same render
    job = world.submit_body(world.request(LAMPS, hints=[hint]), kind="analyse")
    world.run()
    done = world.job(job.job_id)
    assert done.status == "completed"
    new = world.pool.calls - calls
    assert set(new) == {("qa", "transcribe"), ("qa", "embed"), ("qa", "align")}
    assert world.pool.loads == loads  # the QA group was still resident: no load at all
    assert done.items[0].attempts[0].fresh is False


def test_one_model_group_at_a_time_and_each_loaded_once_a_round_s4(world: World) -> None:
    world.submit(LAMPS, KETTLE, ORCHARD, takes=2)
    world.run()
    assert world.pool.loads == ["qwen", "qa"]
    assert world.pool.unloads == ["qwen"]


def test_an_interactive_job_goes_before_a_queued_batch_job_s4(world: World) -> None:
    batch = world.submit(LAMPS)
    interactive = world.submit(KETTLE, priority="interactive")
    world.run()
    assert world.host.started == [interactive.job_id, batch.job_id]


def test_a_batch_job_gives_way_to_an_interactive_one_and_resumes_from_the_cache_s4(world: World) -> None:
    batch = world.submit(LAMPS, KETTLE, ORCHARD)
    _step_until(world, lambda: world.pool.calls[("qwen", "synthesize")] == 2)
    interactive = world.submit(LANTERN, priority="interactive")
    world.run()

    assert world.host.started == [batch.job_id, interactive.job_id, batch.job_id]
    assert Counter(world.pool.texts()) == Counter({LAMPS: 1, KETTLE: 1, ORCHARD: 1, LANTERN: 1})
    first, second = world.job(batch.job_id), world.job(interactive.job_id)
    assert first.status == second.status == "completed"
    assert first.updated_at >= second.updated_at
    assert all(a.fresh for item in first.items for a in item.attempts)  # rendered by it, before and after


def test_waiting_for_free_vram_is_reported_and_ends_with_gpu_unavailable_s4(world: World) -> None:
    probe = FixedProbe(free_mb=QWEN_VRAM_MB)  # short of the need plus the 1 GB margin
    world.new_engine(probe=probe)
    job = world.submit(LAMPS)
    for _ in range(4):
        world.runner.step(world.host)
    waiting = world.job(job.job_id)
    assert waiting.status == "running" and waiting.phase == "waiting_for_gpu"
    assert world.host.sleeps == [GPU_RECHECK_S] * 3
    assert world.host.gpu_facts[-1].waiting_since is not None
    assert world.host.gpu_facts[-1].free_mb == QWEN_VRAM_MB and world.host.gpu_facts[-1].need_mb["qwen"] == QWEN_VRAM_MB
    assert world.pool.loads == []

    world.run()
    failed = world.job(job.job_id)
    assert failed.status == "failed" and failed.error is not None
    assert failed.error.code == codes.GPU_UNAVAILABLE and failed.error.retryable
    assert failed.error.retry_after_s == GPU_UNAVAILABLE_RETRY_S
    assert sum(world.host.sleeps) == pytest.approx(60 * world.config.gpu.wait_timeout_min)
    assert world.host.gpu_facts[-1].waiting_since is None


def test_the_vram_wait_ends_when_memory_frees_s4(world: World) -> None:
    probe = FixedProbe(free_mb=1000)
    world.new_engine(probe=probe)
    job = world.submit(LAMPS)
    for _ in range(3):
        world.runner.step(world.host)
    probe.free_mb = 20_000
    world.run()
    assert world.job(job.job_id).status == "completed"


# ======================================================================== sections 4.1 and 8: cancel and stop


def test_cancel_keeps_what_was_made_and_skips_the_rest_s8(world: World) -> None:
    job = world.submit(LAMPS, KETTLE, ORCHARD)
    _step_until(world, lambda: world.pool.calls[("qwen", "synthesize")] == 2)
    assert world.store.update_job(job.job_id, expect_status="running", status="cancelling") is not None
    world.run()

    done = world.job(job.job_id)
    assert done.status == "cancelled"
    assert [i.state for i in done.items] == ["skipped"] * 3
    assert all(codes.CANCELLED in {f.code for f in i.flags} for i in done.items)
    assert [a.render_id is not None for i in done.items for a in i.attempts] == [True, True, False]

    again = world.submit(LAMPS, KETTLE, ORCHARD)
    world.run()
    assert world.job(again.job_id).status == "completed"
    assert Counter(world.pool.texts()) == Counter({LAMPS: 1, KETTLE: 1, ORCHARD: 1})  # the renders were kept


def test_stop_now_gives_the_job_back_and_the_next_daemon_finishes_it_from_the_cache_s4_1(world: World) -> None:
    job = world.submit(LAMPS, KETTLE)
    _step_until(world, lambda: world.pool.calls[("qwen", "synthesize")] == 1)
    world.host.stop_mode = "now"
    world.runner.shutdown(world.host, "now")
    back = world.job(job.job_id)
    assert back.status == "queued" and back.phase is None
    assert back.items[0].attempts[0].fresh  # what it made is remembered as its own

    world.host.stop_mode = None
    world.new_engine()
    world.run()
    done = world.job(job.job_id)
    assert done.status == "completed"
    assert world.pool.texts() == [LAMPS, KETTLE]
    assert all(a.fresh for i in done.items for a in i.attempts)
    assert world.host.started == [job.job_id, job.job_id]


# ======================================================================== job-level failures


def test_a_voice_with_no_measurement_fails_the_job_s3_2(world: World) -> None:
    other = world.root / "voice" / "other.wav"
    other.write_bytes(world.clip.read_bytes() + b"\x00\x00")
    body = world.request(LAMPS)
    body["voice"]["path"] = str(other)
    body["voice"]["sha256"] = hashlib.sha256(other.read_bytes()).hexdigest()
    job = world.submit_body(body)
    world.run()
    failed = world.job(job.job_id)
    assert failed.status == "failed" and failed.error is not None and failed.error.code == codes.VOICE_NOT_MEASURED
    assert world.pool.starts == 0


def test_a_clip_that_changed_since_submit_is_voice_file_mismatch_s17(world: World) -> None:
    job = world.submit(LAMPS)
    write_clip(world.clip)
    world.clip.write_bytes(world.clip.read_bytes()[:-2] + b"\x01\x01")
    world.run()
    failed = world.job(job.job_id)
    assert failed.status == "failed" and failed.error is not None
    assert failed.error.code == codes.VOICE_FILE_MISMATCH


def test_an_engine_other_than_the_one_expected_is_engine_changed_s10_1(world: World) -> None:
    job = world.submit_body(world.request(LAMPS, expect_engine_profile="sha256:" + "0" * 64))
    world.run()
    failed = world.job(job.job_id)
    assert failed.status == "failed" and failed.error is not None and failed.error.code == codes.ENGINE_CHANGED


def test_no_pinned_engine_is_backend_not_installed_s4(world: World, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(world.store, "current_engine_profile", lambda kind: None)
    job = world.submit(LAMPS)
    world.run()
    failed = world.job(job.job_id)
    assert failed.status == "failed" and failed.error is not None
    assert failed.error.code == codes.BACKEND_NOT_INSTALLED and failed.error.hint


def test_a_kind_this_engine_does_not_run_fails_with_a_hint_s8(world: World) -> None:
    job = world.submit_body({"description": "a calm voice", "takes": 1}, kind="design")
    world.run()
    failed = world.job(job.job_id)
    assert failed.status == "failed" and failed.error is not None
    assert failed.error.code == codes.INTERNAL and not failed.error.retryable and failed.error.hint
    assert world.host.finished == 1


# ======================================================================== what the job record shows


def test_progress_grows_with_retakes_and_never_goes_back_s7_4(world: World) -> None:
    world.faults({"kind": "token_cap", "when": {"text_contains": "lamplighter"}})
    job = world.submit(LAMPS, max_retakes=2)
    seen: list[Progress] = []
    while world.runner.step(world.host):
        seen.append(world.job(job.job_id).progress)
    dones = [p.done_s for p in seen]
    totals = [p.total_s for p in seen]
    assert dones == sorted(dones)
    assert totals == sorted(totals) and totals[-1] == pytest.approx(3 * totals[0])
    assert seen[-1].fraction == 1.0 and seen[-1].segments_done == seen[-1].segments_total == 1
    phases = {p for p in world.host.phases if p is not None}
    assert {"loading_model", "rendering", "postprocessing", "scoring", "retaking", "suggesting"} <= phases


def test_the_consistency_report_is_kept_with_the_job_and_changes_no_verdict_s11_1(world: World) -> None:
    job = world.submit(LAMPS, KETTLE, ORCHARD)
    world.run()
    done = world.job(job.job_id)
    assert done.result is not None
    report = done.result["consistency"]
    assert report["min"] is not None and report["median"] is not None and report["outliers"] == []
    assert [s["take_id"] for s in done.result["suggestions"]] == [i.attempts[0].take_id for i in done.items]
    assert all(a.verdict == "pass" for i in done.items for a in i.attempts)


def test_every_audio_changing_setting_is_passed_explicitly_s10_1(world: World) -> None:
    world.submit(LAMPS)
    world.run()
    profile = world.store.current_engine_profile("base")
    assert profile is not None
    (load,) = [p for g, op, p in world.pool.requests if (g, op) == ("qwen", "load")]
    assert load["settings"] == {"non_streaming_mode": False, "generation": GENERATION}
    assert load["determinism"] == {
        "tf32": False,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "deterministic_algorithms": "warn_only",
    }
    assert load["dtype"] == "bfloat16" and load["attn_implementation"] == "sdpa"
    (synth,) = [p for g, op, p in world.pool.requests if (g, op) == ("qwen", "synthesize")]
    assert synth["max_new_tokens"] == call_cap(profile, LAMPS) < GENERATION["max_new_tokens"]
    assert synth["language"] == "English"


def test_a_cached_analysis_holds_nothing_the_request_alone_gave_s10_2(world: World) -> None:
    job = world.submit(LAMPS)
    world.run()
    attempt = world.job(job.job_id).items[0].attempts[0]
    analysis = world.store.get_analysis_by_id(attempt.analysis_id or "")
    assert analysis is not None
    assert all(c.received == c.spoken for c in analysis.text.cues)
    assert all(f.segment_id is None for f in analysis.qa.flags)
    assert analysis.embedding is not None and len(analysis.embedding) == len(world.anchor)
