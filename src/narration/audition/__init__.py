"""The GENERATE step's ``audition_pronunciation`` (design section 7.6; plan.md WP35).

- ``handler``: the ``pronunciation`` job kind's handler (``build_audition_handler``): each respelling variant
  rendered, post-processed and scored as a generation take, on the job engine's stages, with each take's speaker
  similarity to the clip reported beside its verdict.
- ``variants``: what an audition asks for, one segment and hint per variant, read the same way by the handler and
  by ``get_results``.

``audition_registry`` gives the runner's handlers by kind with ``pronunciation`` added
(``narration.engine.installed`` adds it to the daemon's).
"""

from __future__ import annotations

from narration.jobs.engine import JobEngine
from narration.jobs.handlers import Registry

from .handler import KIND as AUDITION_KIND
from .handler import SIMILARITY, AuditionHandler, AuditionRun, build_audition_handler
from .variants import AuditionAsk, AuditionRequestError, Variant, heard_term, segment_id, variant_hint


def audition_registry(engine: JobEngine, base: Registry | None = None) -> Registry:
    """The runner's handlers by kind: ``base``'s (the job engine's ``generate`` and ``analyse`` when None), with
    ``pronunciation`` on the same ``engine``."""
    base = base if base is not None else Registry.of(engine)
    handlers = {**base.handlers, AUDITION_KIND: build_audition_handler(engine)}
    return Registry(handlers=handlers, residency=base.residency, throughput=base.throughput)


__all__ = [
    "AUDITION_KIND",
    "SIMILARITY",
    "AuditionAsk",
    "AuditionHandler",
    "AuditionRequestError",
    "AuditionRun",
    "Variant",
    "audition_registry",
    "build_audition_handler",
    "heard_term",
    "segment_id",
    "variant_hint",
]
