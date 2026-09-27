"""The ``measure`` job end to end, on the fake workers (design section 3.2; plan.md WP33).

Milestone M1: a caller measures an allowlisted synthetic clip, then narrates paragraphs with it. Every rule of
section 3.2 that the job applies is checked here through the job, and in ``test_ladder`` on numbers. The
world is small (the design text and one corpus paragraph, two seeds, three rungs) so each test is quick; the
full-size measurement is in ``test_acceptance``.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import jsonschema
import pytest

from narration.contracts import codes
from narration.contracts.errors import NarrationError
from narration.contracts.models import MeasurementRecord, PaceTrend
from narration.contracts.schemas import record_schema
from narration.contracts.serial import from_json
from narration.measure import current_measurement, ladder, lookup_measurement, voice_hash_of
from narration.measure.handler import RETRY_AFTER_S
from tests.jobs.support import ENGINE_ID, VOICE_TRANSCRIPT

from .support import DESIGN_SEGMENT, SHORT_LADDER, MeasureWorld, make_world, measure_request, submit_measure

LAMPS = "The lamplighter walks the canal path, counting bridges under her breath."
KETTLE = "A copper kettle hums on the stove while rain draws lines across the window."


@dataclass(frozen=True)
class Measured:
    """A world whose voice one ``measure`` job measured (shared by the tests that only read the result)."""

    world: MeasureWorld
    job_id: str


@pytest.fixture(scope="module")
def measured(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Measured]:
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(os, "fsync", lambda fd: None)  # as the per-test fixture does (conftest)
        world = make_world(tmp_path_factory.mktemp("measured"))
        try:
            job = world.measure()
            world.run()
            assert world.job(job.job_id).status == "completed"
            yield Measured(world=world, job_id=job.job_id)
        finally:
            world.close()


def measurement_of(world: MeasureWorld) -> MeasurementRecord:
    vh = voice_hash_of(clip_sha256=world.clip_sha256, transcript=VOICE_TRANSCRIPT, config=world.config)
    found = lookup_measurement(world.store, voice_hash=vh, engine_profile_id=ENGINE_ID)
    assert found is not None
    return found


def chars(world: MeasureWorld, segment_id: str) -> int:
    return next(p["spoken_chars"] for p in world.corpus["ladder"] if p["segment_id"] == segment_id)


def test_measure_then_generate_m1_s3_2(world: MeasureWorld) -> None:
    """A voice is refused for generation until it is measured, and accepted once it is."""
    refused = world.generate(LAMPS)
    world.run()
    error = world.job(refused.job_id).error
    assert world.job(refused.job_id).status == "failed"
    assert error is not None and error.code == codes.VOICE_NOT_MEASURED

    job = world.measure()
    world.run()
    done = world.job(job.job_id)
    assert done.status == "completed", done.error
    assert done.outcome == "all_passed"
    assert measurement_of(world).max_segment_chars == chars(world, "ladder-350")

    narrated = world.generate(LAMPS, KETTLE, takes=2)
    world.run()
    final = world.job(narrated.job_id)
    assert final.status == "completed", final.error
    assert final.outcome == "all_passed"


def test_measurement_json_is_published_and_matches_its_schema_s3_2(measured: Measured) -> None:
    world, done = measured.world, measured.world.job(measured.job_id)
    assert done.result is not None
    data = json.loads(Path(done.result["path"]).read_text(encoding="utf-8"))
    jsonschema.validate(data, record_schema(MeasurementRecord))
    m = from_json(MeasurementRecord, data)
    assert m == measurement_of(world)
    assert m.measurement_key == done.result["measurement_key"]
    assert m.transcript_check.ok
    assert m.corpus.startswith("narration-en.v1@sha256:")
    assert [r.paragraph_id for r in m.ladder] == [f"ladder-{t:03d}" for t in SHORT_LADDER]
    assert all(len(r.seeds) == world.config.measurement.seeds for r in m.ladder)
    assert [t.paragraph_id for t in m.calibration] == [DESIGN_SEGMENT] * 2 + ["cal-02"] * 2
    assert [t.attempt for t in m.calibration] == [0, 1, 0, 1]
    assert m.anchor.dim == len(m.anchor.embedding)
    assert m.pace.tol >= world.config.measurement.pace_tol_min
    assert [p.chars for p in m.pace.curve] == [r.chars for r in m.ladder]
    assert m.max_segment_seconds is not None and m.max_segment_seconds > 0
    states = [s.state for s in done.items]
    assert states == ["passed"] * len(done.items)


def test_calibration_takes_are_scored_without_a_measurement_key_s10_2(measured: Measured) -> None:
    """A calibration take's analysis names no measurement and has no speaker check; a ladder take's names the
    measurement it built and was judged against the calibration anchor, without a pace check."""
    world = measured.world
    m = measurement_of(world)
    for take in m.calibration:
        (analysis,) = world.store.analyses_of(take.take_id)
        assert analysis.versions.measurement is None
        assert analysis.qa.metrics.spk_sim_anchor is None
        assert take.sim_anchor is not None
    for rung in m.ladder:
        for seed in rung.seeds:
            assert seed.take_id is not None
            (analysis,) = world.store.analyses_of(seed.take_id)
            assert analysis.versions.measurement == m.measurement_key
            assert analysis.qa.metrics.spk_sim_anchor is not None
            assert round(analysis.qa.metrics.spk_sim_anchor, 6) == seed.sim
            assert analysis.qa.thresholds.spk_warn == round(m.similarity.anchor_p5 - 0.01, 6)
            assert analysis.qa.thresholds.pace_tol is None


def test_a_generation_of_a_calibration_paragraph_gets_its_own_analysis_s11_1(measured: Measured) -> None:
    """The same take, asked for by a generation request, is scored again, with the speaker and pace checks."""
    world = measured.world
    m = measurement_of(world)
    body = world.generate(world.text("cal-02"))
    world.run()
    done = world.job(body.job_id)
    assert done.status == "completed", done.error
    attempt = done.items[0].attempts[0]
    calibration = next(t for t in m.calibration if t.paragraph_id == "cal-02" and t.attempt == 0)
    assert attempt.take_id == calibration.take_id  # the same render and take, from the cache
    assert not attempt.fresh
    assert attempt.analysis_id is not None
    analysis = world.store.get_analysis_by_id(attempt.analysis_id)
    assert analysis is not None
    assert analysis.versions.measurement == m.measurement_key
    assert analysis.qa.metrics.spk_sim_anchor is not None
    assert analysis.qa.thresholds.pace_tol is not None


def test_a_current_measurement_completes_the_job_at_once_s7_6(measured: Measured) -> None:
    world = measured.world
    renders = len(world.pool.texts())
    again = world.measure()
    world.run()
    done = world.job(again.job_id)
    assert done.status == "completed"
    assert len(world.pool.texts()) == renders
    assert done.result is not None and done.result["measurement_key"] == measurement_of(world).measurement_key
    profile = world.store.current_engine_profile("base")
    assert profile is not None
    vh = voice_hash_of(clip_sha256=world.clip_sha256, transcript=VOICE_TRANSCRIPT, config=world.config)
    corpus = world.handler.corpus(world.config.measurement.corpus)
    assert current_measurement(world.store, voice_hash=vh, profile=profile, corpus=corpus, config=world.config)


def test_the_transcript_check_refuses_a_transcript_the_clip_does_not_say_s3_2(world: MeasureWorld) -> None:
    world.faults(heard="Bright gulls wheel over the harbour wall while the fishing boats come home at dusk.")
    job = world.measure()
    world.run()
    done = world.job(job.job_id)
    assert done.status == "failed"
    assert done.error is not None
    assert done.error.code == codes.REF_TEXT_MISMATCH
    assert done.error.field == "voice.transcript"
    assert done.error.details is not None and "heard" in done.error.details
    assert world.pool.texts() == []  # nothing was rendered


def test_the_first_failing_rung_ends_the_ladder_and_nothing_above_it_renders_s3_2(tmp_path: Path) -> None:
    world = make_world(tmp_path, ladder=(80, 150, 350, 400, 450))
    try:
        world.faults({"kind": "token_cap", "op": "synthesize", "when": {"text_contains": "Saltmarrow"}})
        job = world.measure()
        world.run()
        done = world.job(job.job_id)
        assert done.status == "completed", done.error
        m = measurement_of(world)
        assert m.max_segment_chars == chars(world, "ladder-350")
        assert [r.passes for r in m.ladder] == [True, True, True, False]
        assert all("TOKEN_CAP_HIT" in s.flags for s in m.ladder[-1].seeds)
        assert not any("Velmora" in t for t in world.pool.texts())  # ladder-450: never rendered
        states = {s.segment_id: s.state for s in done.items}
        assert states["ladder-400"] == "failed_qa"
        assert states["ladder-450"] == "skipped"
        assert done.result is not None and done.result["ladder_stopped_at"]["paragraph_id"] == "ladder-400"
        assert len(m.pace.curve) == 3
    finally:
        world.close()


def test_a_failing_band_rung_ends_the_run_there_s3_2(world: MeasureWorld) -> None:
    """The band is rendered together (its trend needs every band rung), but the run of passing rungs ends at
    the first that fails, and no rung above the band is rendered."""
    world.faults({"kind": "head_insertion", "op": "synthesize", "when": {"text_contains": "summit"}, "words": ["so"]})
    job = world.measure()
    world.run()
    done = world.job(job.job_id)
    assert done.status == "completed", done.error
    m = measurement_of(world)
    assert m.max_segment_chars == chars(world, "ladder-080")
    assert [r.paragraph_id for r in m.ladder] == ["ladder-080", "ladder-150"]
    assert [r.passes for r in m.ladder] == [True, False]
    assert not any("Glacier" in t for t in world.pool.texts())  # ladder-350: above the band, never rendered


def test_no_passing_rung_leaves_no_reliable_length_and_needs_attention_s3_2(world: MeasureWorld) -> None:
    world.faults({"kind": "head_insertion", "op": "synthesize", "when": {"text_contains": "Tarnholm"}, "words": ["so"]})
    job = world.measure()
    world.run()
    done = world.job(job.job_id)
    assert done.status == "completed", done.error
    assert done.outcome == "needs_attention"
    m = measurement_of(world)
    assert m.max_segment_chars is None and m.max_segment_seconds is None
    assert m.pace.curve == ()


def test_a_voice_the_service_may_not_clone_is_refused_s17_4(world: MeasureWorld) -> None:
    other = world.root / "other.wav"
    other.write_bytes(world.clip.read_bytes() + b"\x00\x00")
    sha = hashlib.sha256(other.read_bytes()).hexdigest()
    job = submit_measure(world.store, measure_request(other, sha))
    world.run()
    done = world.job(job.job_id)
    assert done.status == "failed"
    assert done.error is not None and done.error.code == codes.VOICE_NOT_SYNTHETIC


def test_a_missing_corpus_is_backend_not_installed_s3_2(tmp_path: Path) -> None:
    world = make_world(tmp_path, material=tmp_path / "nowhere")
    try:
        job = world.measure()
        world.run()
        done = world.job(job.job_id)
        assert done.status == "failed"
        assert done.error is not None and done.error.code == codes.BACKEND_NOT_INSTALLED
    finally:
        world.close()


def test_a_rung_the_corpus_lacks_is_backend_not_installed_s16(tmp_path: Path) -> None:
    world = make_world(tmp_path, ladder=(80, 150, 275))
    try:
        job = world.measure()
        world.run()
        done = world.job(job.job_id)
        assert done.status == "failed"
        assert done.error is not None and done.error.code == codes.BACKEND_NOT_INSTALLED
        assert "275" in done.error.message
    finally:
        world.close()


def test_a_callers_clip_is_read_through_the_daemons_platform_check_s17_3(world: MeasureWorld) -> None:
    """The engine as the daemon builds it has no path check of its own: the measurement reads the clip through
    the host platform's, as a generation does (``JobEngine.open``)."""
    assert world.engine.parts.check_path is None
    job = world.measure()
    world.run()
    assert world.job(job.job_id).status == "completed", world.job(job.job_id).error
    assert world.host.platform.paths_checked == [str(world.clip)]


def test_an_engine_built_with_its_own_path_check_uses_that_one_s17_3(tmp_path: Path) -> None:
    asked: list[str] = []

    def refuse(path: str) -> Path:
        asked.append(path)
        raise NarrationError(codes.PATH_NOT_ALLOWED, "refused by the engine's own check")

    world = make_world(tmp_path, check_path=refuse)
    try:
        job = world.measure()
        world.run()
        done = world.job(job.job_id)
        assert done.status == "failed" and done.error is not None
        assert (done.error.code, done.error.field) == (codes.PATH_NOT_ALLOWED, "voice.path")
        assert asked == [str(world.clip)] and world.host.platform.paths_checked == []
        assert world.pool.texts() == []
    finally:
        world.close()


def test_a_ladder_with_no_rung_in_the_trend_band_is_refused_s3_2(tmp_path: Path) -> None:
    """With no rung at or under ``trend_band_max_chars`` there is no trend to judge a rung by: refused at once,
    rather than a measurement whose every rung fails."""
    world = make_world(tmp_path, ladder=(350, 400))
    try:
        job = world.measure()
        world.run()
        done = world.job(job.job_id)
        assert done.status == "failed" and done.error is not None
        assert done.error.code == codes.BACKEND_NOT_INSTALLED
        assert "trend_band_max_chars" in done.error.message
        assert world.pool.texts() == []
    finally:
        world.close()


def test_a_trend_band_with_no_measured_pace_fails_retryably_and_publishes_nothing_s3_2(
    world: MeasureWorld, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_pace(rungs: object, band_max_chars: int) -> PaceTrend:
        raise ValueError("no rung in the trend band has a measured pace")

    monkeypatch.setattr(ladder, "fit_trend", no_pace)
    job = world.measure()
    world.run()
    done = world.job(job.job_id)
    assert done.status == "failed" and done.error is not None
    assert done.error.code == codes.INTERNAL
    assert done.error.retryable and done.error.retry_after_s == RETRY_AFTER_S
    assert done.error.details is not None and done.error.details["band"] == ["ladder-080", "ladder-150"]
    vh = voice_hash_of(clip_sha256=world.clip_sha256, transcript=VOICE_TRANSCRIPT, config=world.config)
    assert lookup_measurement(world.store, voice_hash=vh, engine_profile_id=ENGINE_ID) is None
