"""The ``design`` job (design sections 3.1, 3.5, 3.6, 10.1, 10.3, 11.1, 14 and 17.4) on the fake workers.

Every description and design text is invented for these tests, or is the service's own design text.
"""

from __future__ import annotations

import dataclasses
import unicodedata
from pathlib import Path

import pytest

from narration.contracts import codes, names
from narration.contracts.models import Candidate
from narration.contracts.names import CanaryStatus
from narration.design import description_sha256, design_seed
from narration.jobs.pins import call_cap, qwen_load_payload
from narration.jobs.voice import require_synthetic

from .support import DESIGN_ENGINE_ID, DESIGN_HASH, MARSH, NEGATED, WARM, DesignWorld, FixedGuard, make_world, sha256_of


def designed(world: DesignWorld, job_id: str) -> tuple[Candidate, ...]:
    """The candidates of a design job that completed."""
    job = world.job(job_id)
    assert job.status == "completed", (job.status, job.error)
    return world.store.get_design(str(job.request["design_id"]))


# ======================================================================== the candidates (section 3.1)


def test_a_design_publishes_its_candidates_with_clip_transcript_and_seed_s3_1(world: DesignWorld) -> None:
    job = world.design(takes=2)
    world.run()
    candidates = designed(world, job.job_id)
    assert [c.index for c in candidates] == [0, 1]
    spoken = world.config.voice_design.design_text
    digest = description_sha256(WARM)
    for c in candidates:
        clip = Path(c.clip.path)
        assert clip.is_file() and sha256_of(clip) == c.clip.sha256
        assert clip.parent == world.store.design_dir(c.design_id, c.index)
        assert c.transcript == spoken
        assert c.transcript_check is not None and c.transcript_check.ok
        assert (c.description, c.description_sha256, c.design_text) == (WARM, digest, spoken)
        assert c.seed == design_seed(description_sha256=digest, design_text=spoken, index=c.index)
        assert (c.engine_profile.id, c.engine_profile.hash) == (DESIGN_ENGINE_ID, DESIGN_HASH)
        assert c.lint.policy == "warn" and c.lint.findings == ()
    assert candidates[0].seed != candidates[1].seed
    assert candidates[0].clip.sha256 != candidates[1].clip.sha256
    finished = world.job(job.job_id)
    assert finished.outcome == "all_passed"
    assert finished.progress.fraction == 1.0 and finished.progress.segments_done == 2
    assert finished.result is not None and finished.result["design_id"] == job.request["design_id"]
    assert [c["flags"] for c in finished.result["candidates"]] == [[], []]


def test_the_transcript_is_the_design_texts_spoken_form_s3_1_s9_1(world: DesignWorld) -> None:
    sent = unicodedata.normalize("NFD", "Café  lights hum\n across the  quiet square.")
    job = world.design(takes=1, design_text=sent)
    world.run()
    (candidate,) = designed(world, job.job_id)
    assert candidate.transcript == "Café lights hum across the quiet square."
    assert unicodedata.is_normalized("NFC", candidate.transcript)
    assert candidate.design_text == sent
    (request,) = world.requests("design")
    assert request["design_text"] == candidate.transcript


def test_a_negated_description_is_designed_and_linted_s3_5(world: DesignWorld) -> None:
    job = world.design(description=NEGATED, takes=1)
    world.run()
    (candidate,) = designed(world, job.job_id)
    assert [f.phrase for f in candidate.lint.findings] == ["not theatrical", "never rushed", "no rasp"]
    assert candidate.description == NEGATED
    (request,) = world.requests("design")
    assert request["description"] == NEGATED, "the description is sent as it was written"


def test_every_audio_changing_setting_is_passed_explicitly_s10_1(world: DesignWorld) -> None:
    job = world.design(takes=2, design_text=MARSH)
    world.run()
    profile = world.store.current_engine_profile("design")
    assert profile is not None
    loads = [p for g, o, p in world.pool.requests if g == "qwen" and o == "load"]
    assert loads == [qwen_load_payload(profile, world.config.gpu.device)]
    assert loads[0]["settings"]["non_streaming_mode"] is True
    for request in world.requests("design"):
        assert request["max_new_tokens"] == call_cap(profile, MARSH)
        assert request["language"] == names.LANGUAGE
    assert designed(world, job.job_id)


def test_a_design_loads_voice_design_once_then_the_qa_group_once_s4(world: DesignWorld) -> None:
    job = world.design(takes=3)
    world.run()
    assert len(designed(world, job.job_id)) == 3
    assert world.pool.loads == ["qwen", "qa"]
    ops = [o for _, o, _ in world.pool.requests if o in ("design", "transcribe", "profile")]
    assert ops == ["design"] * 3 + ["transcribe", "profile"] * 3


def test_the_same_design_designs_the_same_voices_s10_3(world: DesignWorld) -> None:
    first = world.design(takes=2, name="one")
    world.run()
    again = world.design(takes=2, name="another name")
    world.run()
    one, two = designed(world, first.job_id), designed(world, again.job_id)
    assert one[0].design_id != two[0].design_id
    assert [(c.seed, c.clip.sha256) for c in one] == [(c.seed, c.clip.sha256) for c in two]


# ======================================================================== provenance (section 17.4)


def test_every_candidate_is_on_the_provenance_list_s17_4(world: DesignWorld) -> None:
    job = world.design(takes=2)
    world.run()
    for candidate in designed(world, job.job_id):
        assert world.store.is_provenance(candidate.clip.sha256)
        require_synthetic(world.store, (), candidate.clip.sha256)  # no allowlist edit: VOICE_NOT_SYNTHETIC unraised
    lines = (world.store.root / "provenance.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2 and all(job.request["design_id"] in line for line in lines)


def test_measure_voice_accepts_a_designed_candidate_s3_2_s17_4(world: DesignWorld) -> None:
    """A candidate goes straight from design_voice to measure_voice: the synthetic-voices rule accepts it, and
    its transcript passes measure_voice's check."""
    job = world.design(takes=1)
    world.run()
    (candidate,) = designed(world, job.job_id)
    measure = world.measure(candidate.clip.path, candidate.clip.sha256, candidate.transcript)
    world.run(max_steps=4000)
    done = world.job(measure.job_id)
    assert done.status == "completed", done.error
    assert done.result is not None and done.result["max_segment_chars"] is not None


# ======================================================================== the profile (section 3.6)


def test_each_candidate_carries_its_profile_with_a_speaking_rate_s3_6(world: DesignWorld) -> None:
    job = world.design(takes=1)
    world.run()
    (candidate,) = designed(world, job.job_id)
    profile = candidate.profile
    assert profile is not None
    assert profile.audio_sha256 == candidate.clip.sha256
    assert profile.profile_version == names.PROFILE_VERSION
    assert profile.measurements.speaking_rate_wpm is not None and profile.measurements.speaking_rate_wpm > 0
    assert Path(profile.pictures.spectrogram).is_file() and Path(profile.pictures.pitch).is_file()


def test_the_profile_kept_for_the_bytes_has_no_speaking_rate_s3_6_s10_2(world: DesignWorld) -> None:
    """``profile_voice`` of a candidate's bytes answers from the cache, and its answer follows from its request
    alone: no transcript was sent, so no speaking rate, whether or not the clip was designed here."""
    job = world.design(takes=1)
    world.run()
    (candidate,) = designed(world, job.job_id)
    kept = world.store.get_profile(candidate.clip.sha256, names.PROFILE_VERSION)
    assert kept is not None and candidate.profile is not None
    assert kept.measurements.speaking_rate_wpm is None
    assert kept.measurements == dataclasses.replace(candidate.profile.measurements, speaking_rate_wpm=None)
    assert kept.pictures == candidate.profile.pictures
    profiled = world.profile(Path(candidate.clip.path), candidate.clip.sha256)
    world.run()
    assert world.job(profiled.job_id).result == {"audio_sha256": candidate.clip.sha256, "profile_version": "profile-1"}


# ====================================================================== flags and the outcome (sections 10.1, 11.1, 14)


@pytest.mark.parametrize(
    ("status", "tier", "flagged"),
    [
        ("similarity_pass", "bit_exact", True),
        ("similarity_pass", "similar", False),
        ("hash_match", "bit_exact", False),
    ],
)
def test_a_canary_similarity_pass_is_shown_as_canary_mismatch_info_s10_1(
    tmp_path: Path, status: CanaryStatus, tier: str, flagged: bool
) -> None:
    guard = FixedGuard(status)
    world = make_world(tmp_path, guard=guard, tier=tier)  # pyright: ignore[reportArgumentType]
    try:
        job = world.design(takes=2)
        world.run()
        assert guard.profiles == [DESIGN_ENGINE_ID], "the VoiceDesign load was checked by the canary gate"
        done = world.job(job.job_id)
        assert done.status == "completed" and done.outcome == "all_passed"
        assert done.result is not None
        for entry in done.result["candidates"]:
            codes_ = [(f["code"], f["severity"]) for f in entry["flags"]]
            assert codes_ == ([(codes.CANARY_MISMATCH, "info")] if flagged else [])
    finally:
        world.close()


def test_a_candidate_that_does_not_say_its_text_is_flagged_and_still_published_s11_1(world: DesignWorld) -> None:
    spoken = world.config.voice_design.design_text
    world.faults(
        {"kind": "say", "op": "design", "text": "Ducks paddle slowly around the old stone fountain.", "times": 1}
    )
    job = world.design(takes=2)
    world.run()
    candidates = designed(world, job.job_id)
    assert len(candidates) == 2
    assert candidates[0].transcript_check is not None and not candidates[0].transcript_check.ok
    assert candidates[0].transcript == spoken, "the transcript is the design text, never what was heard"
    assert candidates[1].transcript_check is not None and candidates[1].transcript_check.ok
    assert all(world.store.is_provenance(c.clip.sha256) for c in candidates)
    done = world.job(job.job_id)
    assert done.outcome == "needs_attention"
    assert done.result is not None
    first, second = done.result["candidates"]
    assert [f["code"] for f in first["flags"]] == [codes.WER_HIGH]
    assert first["flags"][0]["severity"] == "fail" and "REF_TEXT_MISMATCH" in first["flags"][0]["message"]
    assert (first["transcript_ok"], second["transcript_ok"], second["flags"]) == (False, True, [])


def test_a_design_that_reaches_its_cap_is_flagged_token_cap_hit_s11_1(world: DesignWorld) -> None:
    world.faults({"kind": "token_cap", "op": "design", "times": 1})
    job = world.design(takes=1)
    world.run()
    done = world.job(job.job_id)
    assert done.status == "completed" and done.outcome == "needs_attention"
    assert done.result is not None
    assert codes.TOKEN_CAP_HIT in [f["code"] for f in done.result["candidates"][0]["flags"]]


# ====================================================================== failures and the job's life (sections 4, 8, 14)


def test_no_pinned_voice_design_engine_fails_the_job_with_a_hint_s14(tmp_path: Path) -> None:
    world = make_world(tmp_path, pin_design=False)
    try:
        job = world.design(takes=1)
        world.run()
        done = world.job(job.job_id)
        assert done.status == "failed" and done.error is not None
        assert (done.error.code, done.error.retryable) == (codes.BACKEND_NOT_INSTALLED, False)
        assert done.error.hint and "engine pin" in done.error.hint
    finally:
        world.close()


def test_a_crashed_design_is_tried_once_more_s4(world: DesignWorld) -> None:
    world.faults({"kind": "crash", "op": "design", "times": 1})
    job = world.design(takes=1)
    world.run()
    assert len(designed(world, job.job_id)) == 1
    assert len(world.requests("design")) == 2


def test_out_of_gpu_memory_twice_fails_the_job_retryably_s4_s14(world: DesignWorld) -> None:
    world.faults({"kind": "gpu_oom", "op": "design"})
    job = world.design(takes=1)
    world.run()
    done = world.job(job.job_id)
    assert done.status == "failed" and done.error is not None
    assert (done.error.code, done.error.retryable) == (codes.GPU_UNAVAILABLE, True)
    assert done.error.retry_after_s is not None and done.error.hint
    assert len(world.requests("design")) == 2
    assert world.store.get_design(str(job.request["design_id"])) == ()


def test_a_worker_refusal_fails_the_job_retryably_with_a_hint_s14(world: DesignWorld) -> None:
    world.faults({"kind": "error", "op": "transcribe", "code": "RENDER_FAILED", "message": "planted"})
    job = world.design(takes=1)
    world.run()
    done = world.job(job.job_id)
    assert done.status == "failed" and done.error is not None
    assert (done.error.code, done.error.retryable) == (codes.INTERNAL, True)
    assert done.error.retry_after_s is not None and done.error.hint
    assert not list(world.store.root.joinpath("scratch", "jobs").glob(f"{job.job_id}/*")), "scratch removed"


def test_a_cancelled_design_keeps_what_it_published_s8(world: DesignWorld) -> None:
    job = world.design(takes=3)
    world.step_until(lambda: len(world.store.get_design(str(job.request["design_id"]))) == 1)
    world.store.update_job(job.job_id, expect_status="running", status="cancelling")
    world.run()
    done = world.job(job.job_id)
    assert done.status == "cancelled"
    kept = world.store.get_design(str(job.request["design_id"]))
    assert [c.index for c in kept] == [0]
    assert world.store.is_provenance(kept[0].clip.sha256)
    assert done.result is not None and [c["index"] for c in done.result["candidates"]] == [0]


def test_a_design_given_back_resumes_with_what_it_published_s4_1(world: DesignWorld) -> None:
    world.host.stop_mode = None
    job = world.design(takes=2)
    world.step_until(lambda: len(world.store.get_design(str(job.request["design_id"]))) == 1)
    world.host.stop_mode = "segment"
    world.runner.shutdown(world.host, "segment")
    assert world.job(job.job_id).status == "queued"
    designs_before = len(world.requests("design"))
    world.restart()
    world.run()
    candidates = designed(world, job.job_id)
    assert [c.index for c in candidates] == [0, 1]
    assert len(world.requests("design")) == designs_before + 1, "only the unpublished candidate is designed again"
