"""The models a job loads, as pinned: an engine profile for Qwen, and the QA models (sections 4, 6, 10.1).

**Qwen** (``EngineProfile``, WP32). Every audio-changing setting is passed explicitly, never left to a
library default (section 10.1): ``load`` gets the profile's snapshot, dtype, attention, determinism switches,
``non_streaming_mode`` and the complete ``generation`` values, whose ``max_new_tokens`` is the ceiling (8192).
Each ``synthesize`` passes its own cap (DC-4), ``names.max_new_tokens_for`` over the text the call speaks,
with the profile's ``max_new_tokens_per_char`` and ``max_new_tokens_floor``.

The profile's ``settings`` are read as WP32 records them (``EngineProfile``'s docstring)::

    {"non_streaming_mode": false, "generation": {... every sampling value, "max_new_tokens": 8192},
     "max_new_tokens_per_char": 2.5, "max_new_tokens_floor": 128}

A profile without one of them cannot run a render as pinned, so it is refused (``ProfileError``) rather than
filled in from a default.

**The QA group** (``QaPins``, WP22). The analysis key names the ASR and speaker models by repo and revision
(section 10.2), and the aligner by its method id, so these are pins of the service, known before any work.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from narration.contracts.models import EngineProfile, Licence
from narration.contracts.names import MIN_MAX_NEW_TOKENS, max_new_tokens_for


class ProfileError(ValueError):
    """The engine profile lacks a setting a render needs; it must be pinned again (``narration-admin engine``)."""


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelPin:
    """One pinned model: its repo, its 40-hex revision, and the local snapshot folder named by it."""

    repo: str
    revision: str
    snapshot_dir: str

    @property
    def name(self) -> str:
        """``repo@revision``: how the analysis key and the analysis record name the model."""
        return f"{self.repo}@{self.revision}"

    def ref(self) -> dict[str, str]:
        """The protocol's ``ModelRef``."""
        return {"repo": self.repo, "revision": self.revision, "snapshot_dir": self.snapshot_dir}


@dataclass(frozen=True, slots=True, kw_only=True)
class QaPins:
    """The QA group's pinned models (Whisper, WavLM-SV, the CTC aligner), the VRAM the group needs, and the
    licences recorded with every analysis (section 18)."""

    asr: ModelPin
    sv: ModelPin
    aligner: ModelPin
    vram_need_mb: int
    licence: Licence = field(default_factory=Licence)

    @property
    def key(self) -> str:
        """What names this set of models, for the scheduler's residency."""
        return "qa:" + "|".join(p.name for p in (self.asr, self.sv, self.aligner))

    def load_payload(self, device: str) -> dict[str, Any]:
        """``load`` for the QA worker: the models by use, and the device (the aligner runs on the CPU)."""
        return {"device": device, "models": {"asr": self.asr.ref(), "sv": self.sv.ref(), "aligner": self.aligner.ref()}}


# ======================================================================== Qwen


def _setting(profile: EngineProfile, name: str) -> Any:
    if name not in profile.settings:
        raise ProfileError(
            f"engine profile {profile.engine_profile_id} has no settings.{name}; every audio-changing setting "
            "must be pinned (section 10.1): pin the engine again"
        )
    return profile.settings[name]


def generation(profile: EngineProfile) -> dict[str, Any]:
    """The profile's complete ``generation`` values (``max_new_tokens`` is the ceiling)."""
    values = _setting(profile, "generation")
    if not isinstance(values, dict) or not isinstance(values.get("max_new_tokens"), int):
        raise ProfileError(f"engine profile {profile.engine_profile_id}: settings.generation.max_new_tokens is missing")
    return dict(values)


def ceiling(profile: EngineProfile) -> int:
    """The loaded ceiling of every call's cap: ``generation.max_new_tokens`` (8192, plan.md section 1.3)."""
    return int(generation(profile)["max_new_tokens"])


def call_cap(profile: EngineProfile, text: str) -> int:
    """The ``max_new_tokens`` of one ``synthesize`` or ``design`` call speaking ``text`` (DC-4)."""
    per_char = _setting(profile, "max_new_tokens_per_char")
    floor = _setting(profile, "max_new_tokens_floor")
    if (
        isinstance(per_char, bool)
        or not isinstance(per_char, int | float)
        or not math.isfinite(per_char)
        or isinstance(floor, bool)
        or not isinstance(floor, int)
    ):
        raise ProfileError(f"engine profile {profile.engine_profile_id}: the per-call cap rule is malformed")
    top = ceiling(profile)
    if floor < MIN_MAX_NEW_TOKENS or top < MIN_MAX_NEW_TOKENS or per_char <= 0:
        raise ProfileError(f"engine profile {profile.engine_profile_id}: the per-call cap rule is out of range")
    return max_new_tokens_for(text, per_char=float(per_char), floor=floor, ceiling=top)


def qwen_load_payload(profile: EngineProfile, device: str) -> dict[str, Any]:
    """``load`` for the Qwen worker, every audio-changing setting explicit (section 10.1, App. A)."""
    mode = _setting(profile, "non_streaming_mode")
    if not isinstance(mode, bool):
        raise ProfileError(f"engine profile {profile.engine_profile_id}: settings.non_streaming_mode is not a boolean")
    determinism = profile.determinism
    return {
        "device": device,
        "model": {"repo": profile.model_repo, "revision": profile.model_revision, "snapshot_dir": profile.snapshot_dir},
        "engine_profile_id": profile.engine_profile_id,
        "dtype": profile.dtype,
        "attn_implementation": determinism.attn_implementation,
        "determinism": {
            "tf32": determinism.tf32,
            "cudnn_deterministic": determinism.cudnn_deterministic,
            "cudnn_benchmark": determinism.cudnn_benchmark,
            "deterministic_algorithms": determinism.deterministic_algorithms,
        },
        "settings": {"non_streaming_mode": mode, "generation": generation(profile)},
    }


__all__ = ["ModelPin", "ProfileError", "QaPins", "call_cap", "ceiling", "generation", "qwen_load_payload"]
