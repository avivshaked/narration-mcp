"""The ``pronunciation`` job (design sections 7.6, 8, 9.1, 10.2, 10.3, 11.1 and 17.4) on the fake workers, and its
results as ``get_results`` gives them.

Every term, respelling and carrier is invented for these tests.
"""

from __future__ import annotations

from typing import Any

from jsonschema import Draft202012Validator
from rapidfuzz import fuzz

from narration import keys
from narration.audition import SIMILARITY, segment_id
from narration.backend.resources import read_resource
from narration.contracts import codes, names
from narration.contracts.models import JobRecord
from narration.contracts.schemas import TOOLS_BY_NAME
from narration.jobs.pins import call_cap
from narration.jobs.voice import clip_path
from narration.measure.lookup import voice_hash_of
from tests.backend.support import write_wav
from tests.design.support import queue
from tests.jobs.support import VOICE_TRANSCRIPT

from .support import CARRIER, TERM, VARIANTS, AuditionWorld


def completed(world: AuditionWorld, job: JobRecord) -> JobRecord:
    finished = world.job(job.job_id)
    assert finished.status == "completed", (finished.status, finished.error)
    return finished


def audition_results(world: AuditionWorld, job_id: str, **args: Any) -> dict[str, Any]:
    """``get_results``'s ``audition``, after checking the whole answer against the tool's output schema."""
    out = world.results(job_id, **args)
    validator = Draft202012Validator(TOOLS_BY_NAME["get_results"].output_schema)
    assert [e.message for e in validator.iter_errors(out)] == []
    return out["audition"]


def the_voice_hash(world: AuditionWorld) -> str:
    return voice_hash_of(clip_sha256=world.clip_sha256, transcript=VOICE_TRANSCRIPT, config=world.config)


# ======================================================================== the variants (sections 7.6, 9.1)


def test_each_variant_speaks_the_carrier_with_its_respelling_s7_6_s9_1(world: AuditionWorld) -> None:
    job = world.audition()
    world.run()
    assert completed(world, job).outcome == "all_passed"
    engine_texts = [CARRIER.replace(TERM, v["respell"]) for v in VARIANTS]
    assert [r["engine_text"] for r in world.requests("synthesize")] == engine_texts
    audition = audition_results(world, job.job_id)
    assert (audition["term"], audition["carrier"]) == (TERM, CARRIER)
    assert [(v["label"], v["respell"], v["engine_text"]) for v in audition["variants"]] == [
        (v["label"], v["respell"], t) for v, t in zip(VARIANTS, engine_texts, strict=True)
    ]
    for variant in audition["variants"]:
        (take,) = variant["takes"]
        assert take["attempt"] == 0 and take["fresh"] is True
        assert take["qa"]["verdict"] == "pass"
        assert variant["heard"] == [take["qa"]["terms"][0]["heard"]], "what the recogniser wrote for the term"
        assert variant["heard"][0].replace(" ", "") == variant["respell"].replace("-", "").lower()


def test_without_a_carrier_the_term_is_spoken_alone_s7_6(world: AuditionWorld) -> None:
    job = world.audition(carrier=None)
    world.run()
    completed(world, job)
    assert [r["engine_text"] for r in world.requests("synthesize")] == [v["respell"] for v in VARIANTS]
    audition = audition_results(world, job.job_id)
    assert audition["carrier"] is None
    assert [v["engine_text"] for v in audition["variants"]] == [v["respell"] for v in VARIANTS]


def test_the_spoken_text_stays_as_sent_and_is_echoed_s9_1(world: AuditionWorld) -> None:
    job = world.audition()
    world.run()
    for item in completed(world, job).items:
        assert item.segment_id.startswith("variant-")
    out = world.results(job.job_id)
    for variant in out["audition"]["variants"]:
        take = variant["takes"][0]
        assert take["cues"] and take["cues"][0]["start_s"] is not None, "the carrier's cue is placed"


def test_a_respelling_heard_as_written_counts_as_the_term_s11_1(world: AuditionWorld) -> None:
    """The variant's respelling is the term's alias: a take that says the respelling is not a misheard term, even
    when the respelling reads nothing like the term's spelling."""
    far = "Ah-lee-sun"
    assert fuzz.ratio("thorvyn", far.replace("-", "").lower()) / 100 < 0.75, "not close to the term by spelling"
    job = world.audition(variants=[{"label": "far", "respell": far}])
    world.run()
    completed(world, job)
    (variant,) = audition_results(world, job.job_id)["variants"]
    (take,) = variant["takes"]
    assert [t["ok"] for t in take["qa"]["terms"]] == [True]
    assert codes.TERM_UNVERIFIED not in {f["code"] for f in take["qa"]["flags"]}
    assert take["qa"]["wer_adj"] == 0.0


# ======================================================================== keys, seeds and the cache (10.2, 10.3)


def test_a_variant_is_keyed_and_seeded_as_a_generation_take_s10_2_s10_3(world: AuditionWorld) -> None:
    job = world.audition()
    world.run()
    voice_hash = the_voice_hash(world)
    profile = world.store.current_engine_profile("base")
    assert profile is not None
    for item, variant in zip(completed(world, job).items, VARIANTS, strict=True):
        (attempt,) = item.attempts
        engine_text = CARRIER.replace(TERM, variant["respell"])
        seed = keys.seed(voice_hash=voice_hash, engine_text=engine_text, attempt=0)
        assert attempt.seed == seed
        assert attempt.render_key == keys.render_key(
            engine_profile_hash=profile.hash, voice_hash=voice_hash, engine_text=engine_text, seed=seed
        )
    for request in world.requests("synthesize"):
        assert request["max_new_tokens"] == call_cap(profile, request["engine_text"])
        assert request["language"] == names.LANGUAGE


def test_the_segment_id_enters_no_key_s10_3(world: AuditionWorld) -> None:
    """Variant 2 alone is planned as segment ``variant-1``, and still names the same render as before."""
    first = world.audition()
    world.run()
    second = world.audition(variants=[dict(VARIANTS[1])])
    world.run()
    before = completed(world, first).items[1].attempts[0]
    (item,) = completed(world, second).items
    assert item.segment_id == segment_id(0)
    assert item.attempts[0].render_key == before.render_key
    assert item.attempts[0].fresh is False


def test_an_audition_sent_again_is_answered_from_the_cache_s10_2(world: AuditionWorld) -> None:
    first = world.audition()
    world.run()
    rendered = len(world.requests("synthesize"))
    again = queue(world.store, "pronunciation", {**world.request(), "variants": [dict(v) for v in VARIANTS]})
    world.run()
    assert len(world.requests("synthesize")) == rendered, "nothing is rendered again"
    one = audition_results(world, first.job_id)
    two = audition_results(world, again.job_id)
    for a, b in zip(one["variants"], two["variants"], strict=True):
        assert [t["take_id"] for t in a["takes"]] == [t["take_id"] for t in b["takes"]]
        assert [t["analysis_id"] for t in a["takes"]] == [t["analysis_id"] for t in b["takes"]]
        assert [t["fresh"] for t in b["takes"]] == [False]
        assert a["spk_sim_clip"] == b["spk_sim_clip"]


# ======================================================================== what an audition take is judged on (11.1)


def test_an_unmeasured_voice_is_auditioned_s7_6(world: AuditionWorld) -> None:
    assert world.store.get_measurement(the_voice_hash(world), "qwen3-base-1.7b.test") is None
    job = world.audition()
    world.run()
    completed(world, job)


def test_the_verdict_has_no_speaker_or_pace_check_and_names_no_measurement_s11_1(world: AuditionWorld) -> None:
    job = world.audition()
    world.run()
    for item in completed(world, job).items:
        analysis_id = item.attempts[0].analysis_id
        assert analysis_id is not None
        analysis = world.store.get_analysis_by_id(analysis_id)
        assert analysis is not None
        assert analysis.versions.measurement is None
        assert analysis.qa.metrics.spk_sim_anchor is None
        assert analysis.qa.metrics.expected_articulation_cps is None
        assert analysis.embedding is not None, "the embedding is kept, for the similarity report"
        assert not {codes.SPK_SIM_LOW, codes.PACE_FAST, codes.PACE_SLOW} & {f.code for f in analysis.qa.flags}


def test_each_takes_similarity_to_the_clip_is_reported_beside_its_verdict_s7_6_s11_1(world: AuditionWorld) -> None:
    job = world.audition()
    world.run()
    finished = completed(world, job)
    embeds = [r["wav"] for r in world.requests("embed")]
    assert embeds.count(str(clip_path(world.store, world.clip_sha256))) == 1, "the clip, as the store keeps it"
    assert finished.result is not None
    by_take = finished.result[SIMILARITY]
    audition = audition_results(world, job.job_id)
    for variant in audition["variants"]:
        (take,) = variant["takes"]
        assert variant["spk_sim_clip"] == [by_take[take["take_id"]]]
        assert 0.9 < variant["spk_sim_clip"][0] <= 1.0, "a take of the voice is close to its clip"
        assert take["qa"]["spk_sim_anchor"] is None, "the similarity is not the verdict's"


def test_the_clip_is_embedded_once_in_the_qa_load_s4(world: AuditionWorld) -> None:
    job = world.audition()
    world.run()
    completed(world, job)
    ops = [(g, o) for g, o, _ in world.pool.requests]
    loads = [g for g, o in ops if o == "load"]
    assert loads == ["qwen", "qa"], "each model group loads once"
    assert len(world.requests("embed")) == len(VARIANTS) + 1, "one embed per take, and one for the clip"


# ======================================================================== retakes (section 8)


def test_a_variant_take_that_is_a_retake_trigger_is_retaken_s8(world: AuditionWorld) -> None:
    respell = VARIANTS[0]["respell"]
    world.inner.faults({"kind": "head_insertion", "when": {"text_contains": respell}, "times": 1})
    job = world.audition()
    world.run()
    finished = completed(world, job)
    first, second = finished.items
    assert [a.attempt for a in first.attempts] == [0, 1]
    assert [a.attempt for a in second.attempts] == [0]
    assert first.attempts[1].seed == keys.seed(
        voice_hash=the_voice_hash(world), engine_text=CARRIER.replace(TERM, respell), attempt=1
    )
    audition = audition_results(world, job.job_id)
    retaken = audition["variants"][0]
    assert len(retaken["takes"]) == len(retaken["heard"]) == len(retaken["spk_sim_clip"]) == 2
    assert codes.HEAD_INSERTION in {f["code"] for f in retaken["takes"][0]["qa"]["flags"]}
    assert codes.RETAKEN in {f["code"] for f in retaken["takes"][1]["flags"]}
    assert retaken["takes"][1]["qa"]["verdict"] == "pass"
    assert finished.result is not None
    assert finished.result["suggestions"][0]["take_id"] == retaken["takes"][1]["take_id"]


# ======================================================================== the voice (section 17.4)


def test_the_daemon_clones_only_a_synthetic_clip_s17_4(world: AuditionWorld) -> None:
    stranger = world.root / "elsewhere" / "stranger.wav"
    sha = write_wav(stranger, freq=330.0)
    job = world.audition(voice={"path": str(stranger), "sha256": sha, "transcript": VOICE_TRANSCRIPT})
    world.run()
    failed = world.job(job.job_id)
    assert failed.status == "failed"
    assert failed.error is not None and failed.error.code == codes.VOICE_NOT_SYNTHETIC
    assert world.requests("synthesize") == []


# ======================================================================== get_results while it runs (section 7.5)


def test_a_queued_audition_lists_its_variants_with_no_takes_yet_s7_5(world: AuditionWorld) -> None:
    job = world.audition()
    audition = audition_results(world, job.job_id)
    assert [v["label"] for v in audition["variants"]] == [v["label"] for v in VARIANTS]
    assert all(v["takes"] == [] and v["heard"] == [] and v["spk_sim_clip"] == [] for v in audition["variants"])


def test_the_audition_records_no_choice_s7_6(world: AuditionWorld) -> None:
    """The job keeps its request and advice only: no chosen variant, no pronunciation list."""
    job = world.audition()
    world.run()
    result = completed(world, job).result
    assert result is not None
    assert set(result) == {"suggestions", "consistency", SIMILARITY}


def test_the_job_report_reads_for_an_audition_s7_7(world: AuditionWorld) -> None:
    job = world.audition()
    world.run()
    completed(world, job)
    report = read_resource(world.backend, f"narration://jobs/{job.job_id}/report")
    assert report.text is not None and job.job_id in report.text
