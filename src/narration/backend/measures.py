"""A voice's measurement as the front-end looks it up (design sections 3.2, 7.6, 10.2): a narrow seam over
``narration.measure`` (WP33), which owns the rules.

- ``voice_hash``: the voice's hash, as the job engine computes it (``voice_hash_of``);
- ``require``: the measurement a generation is judged against, or ``VOICE_NOT_MEASURED`` (``field: voice``,
  with the hint to run ``measure_voice`` first), by the job engine's rule (``require_measured``);
- ``current``: ``measure_voice``'s answer at once: a stored measurement only when its key is the one a new
  measurement would have (the same voice, engine profile, corpus and ladder settings;
  ``current_measurement``); else None, and a ``measure`` job is queued.

``StoreMeasurements`` is the service's; tests pass their own ``Measurements``.
"""

from __future__ import annotations

from typing import Protocol

from narration.config import Config
from narration.contracts import codes
from narration.contracts.errors import MaterialError, NarrationError
from narration.contracts.interfaces import Store
from narration.contracts.models import EngineProfile, MaterialSet, MeasurementRecord
from narration.jobs.plan import VoiceSpec
from narration.measure import current_measurement, load_corpus, require_measured, voice_hash_of


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


class StoreMeasurements:
    """``Measurements`` over the store, by ``narration.measure``'s rules. The calibration corpus (for
    ``current``'s key) is read once, from the service's ``material/``."""

    def __init__(self, store: Store, config: Config) -> None:
        self.store = store
        self.config = config
        self._corpus: MaterialSet | None = None

    def voice_hash(self, voice: VoiceSpec) -> str:
        return voice_hash_of(clip_sha256=voice.sha256, transcript=voice.transcript, config=self.config)

    def require(self, voice_hash: str, profile: EngineProfile) -> MeasurementRecord:
        return require_measured(self.store, voice_hash=voice_hash, engine_profile_id=profile.engine_profile_id)

    def current(self, voice_hash: str, profile: EngineProfile) -> MeasurementRecord | None:
        return current_measurement(
            self.store, voice_hash=voice_hash, profile=profile, corpus=self._corpus_set(), config=self.config
        )

    def _corpus_set(self) -> MaterialSet:
        """The calibration corpus ``[measurement] corpus``; ``BACKEND_NOT_INSTALLED`` when it cannot be read."""
        if self._corpus is None:
            set_id = self.config.measurement.corpus
            try:
                self._corpus = load_corpus(set_id)
            except MaterialError as exc:
                raise NarrationError(
                    codes.BACKEND_NOT_INSTALLED,
                    f"the calibration corpus {set_id} cannot be read: {exc}",
                    hint="Reinstall the service: its material folder is missing or changed.",
                    details={"corpus": set_id},
                ) from exc
        return self._corpus


__all__ = ["Measurements", "StoreMeasurements"]
