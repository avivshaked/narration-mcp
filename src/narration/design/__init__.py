"""The DESIGN step: ``design_voice`` and ``profile_voice`` (design sections 3.1, 3.5, 3.6 and 17.4; plan.md WP34).

- ``handler``: the ``design`` job kind's handler (``build_design_handler``): candidates designed on Qwen
  VoiceDesign with derived seeds, each with its clip, its exact transcript checked by speech recognition, the
  positive-only lint and its voice profile; each clip's sha256 appended to the provenance list.
- ``profile``: the ``profile`` job kind's handler (``build_profile_handler``): the voice profile of any WAV the
  owner can read, by path and sha256, on the CPU.
- ``seeds``: the candidates' seeds, derived from the request alone.
- ``work``: what the two handlers share (the run, failure handling, the job record's endings, the profile reply).

``design_registry`` gives the runner's handlers by kind with both added (``narration.engine.installed`` adds
them to the daemon's).
"""

from __future__ import annotations

from narration.jobs.engine import JobEngine
from narration.jobs.handlers import Registry

from .handler import KIND as DESIGN_KIND
from .handler import DesignHandler, DesignRun, build_design_handler, candidate_flags
from .profile import KIND as PROFILE_KIND
from .profile import ProfileHandler, ProfileRun, build_profile_handler
from .seeds import DESIGN_SEED_SCHEME, description_sha256, design_seed


def design_registry(engine: JobEngine, base: Registry | None = None) -> Registry:
    """The runner's handlers by kind: ``base``'s (the job engine's ``generate`` and ``analyse`` when None), with
    ``design`` and ``profile`` on the same ``engine``."""
    base = base if base is not None else Registry.of(engine)
    handlers = {**base.handlers, DESIGN_KIND: build_design_handler(engine), PROFILE_KIND: build_profile_handler(engine)}
    return Registry(handlers=handlers, residency=base.residency, throughput=base.throughput)


__all__ = [
    "DESIGN_KIND",
    "DESIGN_SEED_SCHEME",
    "PROFILE_KIND",
    "DesignHandler",
    "DesignRun",
    "ProfileHandler",
    "ProfileRun",
    "build_design_handler",
    "build_profile_handler",
    "candidate_flags",
    "description_sha256",
    "design_registry",
    "design_seed",
]
