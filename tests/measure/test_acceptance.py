"""WP33's acceptance: a measured voice is not flagged across its own calibration takes (plan.md WP33).

- The check itself (``run_acceptance.calibration_check``) and the runner's pieces, on numbers: default suite.
- The full-size measurement on the fake workers (the service's corpus with its design text, three seeds, the
  default ladder), checked the same way, and through a generation of the calibration paragraphs: ``slow``.
- The acceptance on the GPU with the bakeoff's d4 (``run_acceptance``): ``gpu``, ``evidence`` and ``slow``.
  It needs the daemon, the QA models and the engine pins installed, and a configuration
  (``NARRATION_CONFIG``); it skips, saying what is missing, without them. It runs for up to 30 minutes, so it
  is run only with the lead's OK (AGENTS.md section 5).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from pathlib import Path

import pytest

from narration.config import CONFIG_ENV, Config, MeasurementConfig
from narration.contracts.models import CalibrationTake, MeasurementRecord
from narration.jobs.plan import GenerateRequest
from narration.measure import load_corpus, lookup_measurement, voice_hash_of
from tests.jobs.support import ENGINE_ID, VOICE_TRANSCRIPT, measurement, submit

from .run_acceptance import (
    ACCEPTED,
    ALLOW_ENV,
    BAKEOFF_ENV,
    UNAVAILABLE,
    VOICES,
    Unavailable,
    Voice,
    bakeoff_voice,
    calibration_check,
    calibration_request,
    generation_flags,
    main,
    report,
)
from .support import DESIGN_SEGMENT, MeasureWorld, corpus_data, make_world

SETTINGS = MeasurementConfig()
INVENTED = "The night bus hums past the bakery as the first trays come out."


def takes(*sims: float | None) -> tuple[CalibrationTake, ...]:
    return tuple(
        CalibrationTake(paragraph_id="cal-01", seed=1000 + i, attempt=i, take_id=f"tk_{i:016x}", sim_anchor=s)
        for i, s in enumerate(sims)
    )


def measured_with(*sims: float | None) -> MeasurementRecord:
    return dataclasses.replace(measurement("0" * 64, (1.0, 0.0, 0.0)), calibration=takes(*sims))


# ======================================================================== the check, on numbers


def test_takes_at_or_above_the_warn_threshold_are_not_flagged_s11_1() -> None:
    m = measured_with(0.99, 0.97, 0.94)  # anchor_p5 0.95, so warn below 0.94
    check = calibration_check(m, SETTINGS)
    assert (check.warn_below, check.fail_below) == (0.94, 0.90)
    assert check.accepted and check.flagged == ()


def test_a_take_below_the_warn_threshold_or_the_floor_is_flagged_s11_1() -> None:
    check = calibration_check(measured_with(0.99, 0.9399, 0.85), SETTINGS)
    assert not check.accepted
    assert [(t.attempt, t.flag) for t in check.flagged] == [(1, "warn"), (2, "fail")]


def test_a_take_with_no_similarity_or_no_takes_is_not_accepted_s11_1() -> None:
    assert not calibration_check(measured_with(0.99, None), SETTINGS).accepted
    assert not calibration_check(measured_with(), SETTINGS).accepted


def test_the_report_holds_numbers_and_never_the_transcript_s9_3() -> None:
    voice = Voice(name="d4", clip=Path("clip.wav"), sha256="0" * 64, transcript=INVENTED)
    m = measured_with(0.99, 0.97)
    body = json.dumps(report(voice, m, calibration_check(m, SETTINGS)))
    assert INVENTED not in body and "bus" not in body
    assert json.loads(body)["accepted"] is True


def test_the_voice_is_read_from_the_bakeoff_only_when_allowlisted_s17_4(tmp_path: Path) -> None:
    clip = tmp_path / VOICES["d4"]
    clip.parent.mkdir(parents=True)
    clip.write_bytes(b"RIFF not really audio")
    clip.with_suffix(".json").write_text(json.dumps({"text": INVENTED}), encoding="utf-8")
    digest = hashlib.sha256(clip.read_bytes()).hexdigest()
    with pytest.raises(Unavailable, match=BAKEOFF_ENV):
        bakeoff_voice("d4", {})
    with pytest.raises(Unavailable, match=ALLOW_ENV):
        bakeoff_voice("d4", {BAKEOFF_ENV: str(tmp_path)})
    with pytest.raises(Unavailable, match="not in"):
        bakeoff_voice("d4", {BAKEOFF_ENV: str(tmp_path), ALLOW_ENV: "f" * 64})
    voice = bakeoff_voice("d4", {BAKEOFF_ENV: str(tmp_path), ALLOW_ENV: f"{'f' * 64},{digest.upper()}"})
    assert (voice.sha256, voice.transcript, voice.clip) == (digest, INVENTED, clip)
    with pytest.raises(Unavailable, match="cannot be read"):
        bakeoff_voice("d2", {BAKEOFF_ENV: str(tmp_path), ALLOW_ENV: digest})


def test_the_generation_cross_check_asks_for_the_calibration_takes_exactly_s10_3(tmp_path: Path) -> None:
    """The service's corpus, the measurement's paragraphs and attempts, and no retakes: a request the engine reads."""
    base = Config.for_tests(tmp_path / "store", tmp_path / "models")
    config = dataclasses.replace(base, measurement=MeasurementConfig(seeds=3))
    voice = Voice(name="d4", clip=Path("clip.wav"), sha256="0" * 64, transcript=INVENTED)
    body = calibration_request(voice, measured_with(0.99), config)
    parsed = GenerateRequest.parse(body, config.defaults)
    assert [s.segment_id for s in parsed.segments] == ["cal-01"]
    paragraph = next(p for p in load_corpus(config.measurement.corpus).paragraphs if p.segment_id == "cal-01")
    assert parsed.segments[0].cues == paragraph.cues  # the exact spans included
    assert (parsed.takes, parsed.max_retakes) == (3, 0)


# ======================================================================== the full-size measurement, on the fake


@pytest.mark.slow
def test_full_measurement_is_not_flagged_across_its_own_calibration_takes_s3_2(tmp_path: Path) -> None:
    world = make_world(tmp_path, ladder=SETTINGS.length_ladder_spoken_chars, seeds=SETTINGS.seeds, corpus=corpus_data())
    try:
        _measure_fully(world)
    finally:
        world.close()


def _measure_fully(world: MeasureWorld) -> None:
    job = world.measure()
    world.run(max_steps=5000)
    done = world.job(job.job_id)
    assert done.status == "completed", done.error
    assert done.outcome == "all_passed"
    vh = voice_hash_of(clip_sha256=world.clip_sha256, transcript=VOICE_TRANSCRIPT, config=world.config)
    m = lookup_measurement(world.store, voice_hash=vh, engine_profile_id=ENGINE_ID)
    assert m is not None
    assert len(m.calibration) == 12  # the design text and three paragraphs, three seeds each
    assert m.calibration[0].paragraph_id == DESIGN_SEGMENT
    assert [r.passes for r in m.ladder] == [True] * len(SETTINGS.length_ladder_spoken_chars)
    assert m.max_segment_chars == 560 and len(m.pace.curve) == 9
    assert calibration_check(m, world.config.measurement).accepted

    voice = Voice(name="fake", clip=world.clip, sha256=world.clip_sha256, transcript=VOICE_TRANSCRIPT)
    corpus = world.handler.corpus(world.config.measurement.corpus)
    renders = world.pool.calls[("qwen", "synthesize")]
    generation = submit(world.store, calibration_request(voice, m, world.config, corpus))
    world.run()
    found = generation_flags(world.job(generation.job_id))
    assert found["status"] == "completed"
    assert found["spk_sim_low"] == {}
    assert found["takes"] == found["from_cache"] == 12
    assert world.pool.calls[("qwen", "synthesize")] == renders  # nothing rendered again


# ======================================================================== the acceptance, on the GPU


@pytest.mark.gpu
@pytest.mark.evidence
@pytest.mark.slow
@pytest.mark.timeout(35 * 60)
def test_d4_is_not_flagged_across_its_own_calibration_takes_s3_2(tmp_path: Path) -> None:
    if not os.environ.get(CONFIG_ENV):
        pytest.skip(f"set {CONFIG_ENV} to an installed service's configuration (the daemon runs the job)")
    try:
        bakeoff_voice("d4")
    except Unavailable as exc:
        pytest.skip(str(exc))
    out = tmp_path / "wp33-d4.json"
    code = main(["--voice", "d4", "--out", str(out)])
    if code == UNAVAILABLE:
        pytest.skip("the acceptance could not run here (its reason is on stderr)")
    assert code == ACCEPTED, out.read_text(encoding="utf-8") if out.is_file() else f"exit code {code}"
