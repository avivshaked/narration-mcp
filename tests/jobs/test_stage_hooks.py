"""The stages' two hooks (``Stages.measurement_key`` and ``Stages.scoring_facts``): their defaults keep what a
generation job's takes are judged against and keyed by exactly as before (design sections 10.2 and 11.1).

A kind that scores takes without a finished measurement (``measure_voice``, WP33) overrides them; generation
and scoring-only jobs must see no change.
"""

from __future__ import annotations

import dataclasses

import pytest

from narration.jobs.stages import ScoringFacts
from narration.jobs.state import JobRun

from .conftest import World
from .support import ENGINE_ID, LAMPS, voice_hash


def _open(world: World) -> JobRun:
    world.submit(LAMPS)
    claimed = world.store.claim_next_job(world.host.holder)
    assert claimed is not None
    return world.engine.open(world.host, claimed)


def test_hook_measurement_key_defaults_to_the_voices_measurement_s10_2(world: World) -> None:
    run = _open(world)
    assert run.measurement is not None
    assert world.engine.stages.measurement_key(run, run.segments[0]) == run.measurement.measurement_key


def test_hook_scoring_facts_default_is_the_finished_measurement_alone_s11_1(world: World) -> None:
    run = _open(world)
    facts = world.engine.stages.scoring_facts(run, run.segments[0])
    assert facts == ScoringFacts(measurement=run.measurement)
    assert (facts.anchor, facts.similarity, facts.pace) == (None, None, None)


def test_hook_defaults_refuse_a_run_with_no_measurement_s10_2(world: World) -> None:
    run = _open(world)
    bare = dataclasses.replace(run, measurement=None)
    with pytest.raises(RuntimeError, match="no measurement"):
        world.engine.stages.measurement_key(bare, bare.segments[0])
    with pytest.raises(RuntimeError, match="no measurement"):
        world.engine.stages.scoring_facts(bare, bare.segments[0])


def test_hook_defaults_score_a_generation_take_as_before_s11_1(world: World) -> None:
    """End to end: the analysis names the measurement's key, and the speaker and pace checks ran."""
    job = world.submit(LAMPS)
    world.run()
    done = world.job(job.job_id)
    assert done.status == "completed"
    measurement = world.store.get_measurement(voice_hash(world.clip_sha256), ENGINE_ID)
    assert measurement is not None
    analysis_id = done.items[0].attempts[0].analysis_id
    assert analysis_id is not None
    analysis = world.store.get_analysis_by_id(analysis_id)
    assert analysis is not None
    assert analysis.versions.measurement == measurement.measurement_key
    assert analysis.qa.metrics.spk_sim_anchor is not None
    assert analysis.qa.thresholds.spk_warn is not None
    assert analysis.qa.thresholds.pace_tol is not None
