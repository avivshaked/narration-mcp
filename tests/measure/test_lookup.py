"""Finding a voice's measurement (design sections 3.2, 7.6 and 10.2; ``narration.measure.lookup``), as the
front-end (WP36) asks it."""

from __future__ import annotations

import dataclasses

import pytest

from narration.config import MeasurementConfig
from narration.contracts import codes
from narration.contracts.errors import NarrationError
from narration.measure import (
    current_measurement,
    lookup_measurement,
    measurement_key_of,
    require_measured,
    voice_hash_of,
)
from tests.jobs.support import ENGINE_ID, VOICE_TRANSCRIPT, measurement, voice_hash

from .support import MeasureWorld


def test_voice_hash_is_the_job_engines_s10_2(world: MeasureWorld) -> None:
    vh = voice_hash_of(clip_sha256=world.clip_sha256, transcript=VOICE_TRANSCRIPT, config=world.config)
    assert vh == voice_hash(world.clip_sha256)
    other = voice_hash_of(clip_sha256=world.clip_sha256, transcript=VOICE_TRANSCRIPT + " Again.", config=world.config)
    assert other != vh


def test_an_unmeasured_voice_is_not_measured_s3_2(world: MeasureWorld) -> None:
    vh = voice_hash(world.clip_sha256)
    assert lookup_measurement(world.store, voice_hash=vh, engine_profile_id=ENGINE_ID) is None
    with pytest.raises(NarrationError) as caught:
        require_measured(world.store, voice_hash=vh, engine_profile_id=ENGINE_ID)
    assert caught.value.code == codes.VOICE_NOT_MEASURED
    assert caught.value.field == "voice"


def test_a_measurement_is_found_by_voice_and_engine_profile_s14(world: MeasureWorld) -> None:
    stored = world.store.put_measurement(measurement(world.clip_sha256, (1.0, 0.0, 0.0)))
    vh = voice_hash(world.clip_sha256)
    assert lookup_measurement(world.store, voice_hash=vh, engine_profile_id=ENGINE_ID) == stored
    assert require_measured(world.store, voice_hash=vh, engine_profile_id=ENGINE_ID) == stored
    assert lookup_measurement(world.store, voice_hash=vh, engine_profile_id="ep_other") is None


def test_only_a_measurement_with_todays_key_is_current_s7_6(world: MeasureWorld) -> None:
    """A measurement under another corpus or other ladder settings still serves generation, but
    ``measure_voice`` measures again."""
    vh = voice_hash(world.clip_sha256)
    profile = world.store.current_engine_profile("base")
    assert profile is not None
    corpus = world.handler.corpus(world.config.measurement.corpus)
    old = world.store.put_measurement(measurement(world.clip_sha256, (1.0, 0.0, 0.0)))
    assert current_measurement(world.store, voice_hash=vh, profile=profile, corpus=corpus, config=world.config) is None
    key = measurement_key_of(voice_hash=vh, profile=profile, corpus=corpus, config=world.config)
    assert key != old.measurement_key
    now = world.store.put_measurement(dataclasses.replace(old, measurement_key=key))
    found = current_measurement(world.store, voice_hash=vh, profile=profile, corpus=corpus, config=world.config)
    assert found == now
    more_seeds = dataclasses.replace(
        world.config, measurement=MeasurementConfig(seeds=world.config.measurement.seeds + 1)
    )
    assert current_measurement(world.store, voice_hash=vh, profile=profile, corpus=corpus, config=more_seeds) is None
