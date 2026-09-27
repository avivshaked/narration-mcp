"""Finding a voice's measurement (design sections 3.2, 7.6, 10.2 and 14): for the front-end (WP36) and the
job engine alike.

- **Is the voice measured?** ``lookup_measurement`` finds the measurement of (``voice_hash``, engine profile),
  by the rule the job engine applies before it renders (``JobEngine.open``): the voice's measurement under the
  engine profile id in use. ``require_measured`` raises ``VOICE_NOT_MEASURED`` when there is none, so a
  generation request with an unmeasured clip is refused at submit, and measuring never starts inside another
  job (section 3.2).
- **Is this measurement current?** A measurement answers ``measure_voice`` at once only if its key is the one
  a new measurement would have (``current_measurement``): the key covers the voice, the engine profile's hash,
  the corpus version and the ladder settings (section 10.2), so a new corpus or new settings measure again.
  Generation does not ask this: a voice measured under an earlier corpus keeps its measurement until it is
  measured again.

``voice_hash_of`` is the voice's hash exactly as the job engine computes it (section 10.2).
"""

from __future__ import annotations

from narration import keys
from narration.config import Config
from narration.contracts import codes, names
from narration.contracts.errors import NarrationError
from narration.contracts.interfaces import Store
from narration.contracts.models import EngineProfile, MaterialSet, MeasurementRecord

from .corpus import corpus_version


def voice_hash_of(*, clip_sha256: str, transcript: str, config: Config) -> str:
    """The voice's hash (section 10.2): the Base model's repo id, the clip's sha256, the transcript as sent
    (NFC applied inside), the language and ICL mode, as ``JobEngine.open`` computes it."""
    return keys.voice_hash(
        model=names.MODEL_QWEN_BASE,
        clip_sha256=clip_sha256,
        transcript=transcript,
        language=names.LANGUAGE,
        x_vector_only_mode=config.engines.qwen3_base.x_vector_only_mode,
    )


def measurement_key_of(*, voice_hash: str, profile: EngineProfile, corpus: MaterialSet, config: Config) -> str:
    """The key a measurement of this voice would have now (section 10.2): the voice, the engine profile's
    hash, the corpus version and the ladder settings (``[measurement]``)."""
    return keys.measurement_key(
        voice_hash=voice_hash,
        engine_profile_hash=profile.hash,
        corpus_version=corpus_version(corpus),
        settings=config.measurement,
    )


def lookup_measurement(store: Store, *, voice_hash: str, engine_profile_id: str) -> MeasurementRecord | None:
    """The voice's measurement under the engine profile, or None: the one generation is judged against."""
    return store.get_measurement(voice_hash, engine_profile_id)


def require_measured(store: Store, *, voice_hash: str, engine_profile_id: str) -> MeasurementRecord:
    """The voice's measurement under the engine profile; ``VOICE_NOT_MEASURED`` (``field: voice``, with the hint
    to run ``measure_voice`` first) when there is none."""
    found = lookup_measurement(store, voice_hash=voice_hash, engine_profile_id=engine_profile_id)
    if found is None:
        raise NarrationError(
            codes.VOICE_NOT_MEASURED,
            f"voice {voice_hash} has no measurement under engine profile {engine_profile_id}",
            field="voice",
        )
    return found


def current_measurement(
    store: Store, *, voice_hash: str, profile: EngineProfile, corpus: MaterialSet, config: Config
) -> MeasurementRecord | None:
    """The voice's measurement if it is current (its key is the one a new measurement would have), else None:
    what ``measure_voice`` returns at once instead of queuing a job (section 7.6)."""
    found = lookup_measurement(store, voice_hash=voice_hash, engine_profile_id=profile.engine_profile_id)
    if found is None:
        return None
    key = measurement_key_of(voice_hash=voice_hash, profile=profile, corpus=corpus, config=config)
    return found if found.measurement_key == key else None


__all__ = [
    "current_measurement",
    "lookup_measurement",
    "measurement_key_of",
    "require_measured",
    "voice_hash_of",
]
