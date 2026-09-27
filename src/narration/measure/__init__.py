"""``measure_voice``: measuring a voice once, before generation (design section 3.2; plan.md WP33).

- ``handler``: the ``measure`` job kind's handler (``build_measure_handler``, ``measure_registry``), which the
  runner dispatches a claimed job to: the transcript check, the calibration set, the length ladder, and the
  measurement it publishes.
- ``lookup``: whether a voice is measured (``require_measured``: ``VOICE_NOT_MEASURED``), and whether its
  measurement is current (``current_measurement``), for the front-end (WP36).
- ``transcript``, ``baseline``, ``ladder``: the pure rules: the transcript check, the anchor and similarity
  baseline, the pace trend, ``tol`` and the rung judgement.
- ``corpus``: the calibration corpus, read from ``material/`` and checked against its manifest.
"""

from __future__ import annotations

from .corpus import corpus_version, default_material_root, load_corpus
from .handler import KIND, MeasureHandler, MeasureRun, build_measure_handler, measure_registry
from .lookup import current_measurement, lookup_measurement, measurement_key_of, require_measured, voice_hash_of

__all__ = [
    "KIND",
    "MeasureHandler",
    "MeasureRun",
    "build_measure_handler",
    "corpus_version",
    "current_measurement",
    "default_material_root",
    "load_corpus",
    "lookup_measurement",
    "measure_registry",
    "measurement_key_of",
    "require_measured",
    "voice_hash_of",
]
