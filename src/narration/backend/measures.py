"""A voice's measurement as the front-end looks it up (design sections 3.2, 7.6, 10.2): a narrow seam over
``narration.measure`` (WP33), which owns the rules.

- ``voice_hash``: the voice's hash, as the job engine computes it;
- ``require``: the measurement a generation is judged against, or ``VOICE_NOT_MEASURED`` (``field: voice``,
  with the hint to run ``measure_voice`` first), by the same rule as the job engine;
- ``current``: ``measure_voice``'s answer at once: a stored measurement only when its key is the one a new
  measurement would have (the same voice, engine profile, corpus and ladder settings); else None, and a
  ``measure`` job is queued.

``StoreMeasurements`` calls ``narration.measure`` when the build has it. A build from before WP33 merged
answers from the store alone: the same hash and the same refusal, and any stored measurement counts as
current. The rebase onto ``main`` replaces the fallback with direct imports. Tests pass their own
``Measurements``.
"""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Protocol

from narration import keys
from narration.config import Config
from narration.contracts import codes, names
from narration.contracts.errors import MaterialError, NarrationError
from narration.contracts.interfaces import Store
from narration.contracts.models import EngineProfile, MeasurementRecord
from narration.jobs.plan import VoiceSpec

MEASURE_MODULE = "narration.measure"


class Measurements(Protocol):
    """How the front-end asks about a voice's measurement (see the module docstring)."""

    def voice_hash(self, voice: VoiceSpec) -> str:
        """The voice's hash (section 10.2)."""
        ...

    def require(self, voice_hash: str, profile: EngineProfile) -> MeasurementRecord:
        """The voice's measurement under ``profile``; ``VOICE_NOT_MEASURED`` when there is none."""
        ...

    def current(self, voice_hash: str, profile: EngineProfile) -> MeasurementRecord | None:
        """The voice's measurement when it is current, else None (``measure_voice`` then queues a job)."""
        ...


def _measure_module() -> ModuleType | None:
    try:
        return importlib.import_module(MEASURE_MODULE)
    except ModuleNotFoundError as exc:
        if exc.name != MEASURE_MODULE:
            raise
        return None


class StoreMeasurements:
    """``Measurements`` over the store, by ``narration.measure``'s rules (see the module docstring)."""

    def __init__(self, store: Store, config: Config) -> None:
        self.store = store
        self.config = config
        self._module = _measure_module()

    def voice_hash(self, voice: VoiceSpec) -> str:
        if self._module is not None:
            return str(
                self._module.voice_hash_of(clip_sha256=voice.sha256, transcript=voice.transcript, config=self.config)
            )
        return keys.voice_hash(
            model=names.MODEL_QWEN_BASE,
            clip_sha256=voice.sha256,
            transcript=voice.transcript,
            language=names.LANGUAGE,
            x_vector_only_mode=self.config.engines.qwen3_base.x_vector_only_mode,
        )

    def require(self, voice_hash: str, profile: EngineProfile) -> MeasurementRecord:
        if self._module is not None:
            found: MeasurementRecord = self._module.require_measured(
                self.store, voice_hash=voice_hash, engine_profile_id=profile.engine_profile_id
            )
            return found
        measurement = self.store.get_measurement(voice_hash, profile.engine_profile_id)
        if measurement is None:
            raise NarrationError(
                codes.VOICE_NOT_MEASURED,
                f"voice {voice_hash} has no measurement under engine profile {profile.engine_profile_id}",
                field="voice",
            )
        return measurement

    def current(self, voice_hash: str, profile: EngineProfile) -> MeasurementRecord | None:
        if self._module is None:
            return self.store.get_measurement(voice_hash, profile.engine_profile_id)
        try:
            corpus = self._module.load_corpus(self.config.measurement.corpus)
        except MaterialError as exc:
            raise NarrationError(
                codes.BACKEND_NOT_INSTALLED,
                f"the calibration corpus {self.config.measurement.corpus} cannot be read: {exc}",
                hint="Reinstall the service: its material folder is missing or changed.",
            ) from exc
        found: MeasurementRecord | None = self._module.current_measurement(
            self.store, voice_hash=voice_hash, profile=profile, corpus=corpus, config=self.config
        )
        return found


__all__ = ["MEASURE_MODULE", "Measurements", "StoreMeasurements"]
