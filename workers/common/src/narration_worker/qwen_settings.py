"""The checks on a Qwen ``load``'s audio-changing settings (design section 10.1), shared by every worker.

The ``qwen3`` worker (``narration_qwen3tts``) and the fake (``narration_worker.fake``) check a Qwen load with
these same functions, so the fake refuses exactly the settings ``qwen3`` refuses, with the same
``details.field``. Every setting that changes audio is passed to qwen-tts **explicitly**, never left to a
library default:

- ``non_streaming_mode``: false for Base (the clone path every piece of evidence used), true for VoiceDesign;
- the ten sampling values of ``QwenSettings.generation``, all of them, including ``max_new_tokens``, which
  is the loaded **ceiling** (``CEILING_FIELD``): each ``synthesize`` or ``design`` call passes its own cap,
  at most that (design section 10.1, DC-4).

qwen-tts 0.1.1 fills a value the caller leaves out from the snapshot's ``generation_config.json`` and then
from a hard-coded fallback (``Qwen3TTSModel._merge_generate_kwargs``). A worker that relied on that would
render differently if the snapshot's file changed, so ``parse_settings`` refuses a ``generation`` that lacks
any of the ten keys. What qwen-tts would have used for a snapshot is the ``qwen3`` worker's business
(``narration_qwen3tts.settings.effective_generation``).

Standard library only, like the rest of ``narration_worker``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Final, cast

from .handler import MIN_MAX_NEW_TOKENS
from .protocol import Generation, QwenSettings

GENERATION_KEYS: Final = (
    "do_sample",
    "top_k",
    "top_p",
    "temperature",
    "repetition_penalty",
    "subtalker_dosample",
    "subtalker_top_k",
    "subtalker_top_p",
    "subtalker_temperature",
    "max_new_tokens",
)
"""The sampling values ``Qwen3TTSModel._merge_generate_kwargs`` merges (qwen-tts 0.1.1), in its order."""

BOOL_KEYS: Final = ("do_sample", "subtalker_dosample")
INT_KEYS: Final = ("top_k", "subtalker_top_k", "max_new_tokens")
PROBABILITY_KEYS: Final = ("top_p", "subtalker_top_p")
POSITIVE_FLOAT_KEYS: Final = ("temperature", "repetition_penalty", "subtalker_temperature")

DTYPES: Final = ("bfloat16", "float16", "float32")
"""The ``dtype`` values a Qwen ``load`` accepts."""
ATTN_IMPLEMENTATIONS: Final = ("sdpa", "eager", "flash_attention_2")
"""The ``attn_implementation`` values a Qwen ``load`` accepts."""

CEILING_FIELD: Final = "settings.generation.max_new_tokens"
"""The loaded ceiling of every call's cap (DC-4), as ``details.field`` names it when a ``load`` is refused."""


class SettingsError(ValueError):
    """A settings value that cannot be used; ``field`` names it (``settings.generation.top_k``)."""

    def __init__(self, field: str, message: str) -> None:
        super().__init__(f"{field}: {message}")
        self.field = field
        self.message = message


def parse_generation(value: object, field: str = "settings.generation") -> Generation:
    """Check a ``generation`` object: exactly the ten keys, each of the right type and range.

    The first problem found is raised as ``SettingsError``, whose ``field`` names the member: a missing key
    (the first in ``GENERATION_KEYS`` order), then an unknown key (the first in sorted order), then the
    first bad value. A boolean is never taken for a number.
    """
    if not isinstance(value, dict):
        raise SettingsError(field, "must be an object with every sampling value (section 10.1)")
    given = cast(dict[str, object], value)
    missing = [key for key in GENERATION_KEYS if key not in given]
    if missing:
        raise SettingsError(
            f"{field}.{missing[0]}",
            "every sampling value must be passed explicitly (section 10.1); missing: " + ", ".join(missing),
        )
    unknown = sorted(set(given) - set(GENERATION_KEYS))
    if unknown:
        raise SettingsError(f"{field}.{unknown[0]}", "unknown sampling value; known: " + ", ".join(GENERATION_KEYS))
    for key in BOOL_KEYS:
        if not isinstance(given[key], bool):
            raise SettingsError(f"{field}.{key}", "must be true or false")
    for key in INT_KEYS:
        item = given[key]
        least = MIN_MAX_NEW_TOKENS if key == "max_new_tokens" else 1  # qwen-tts's min_new_tokens is 2
        if not isinstance(item, int) or isinstance(item, bool) or item < least:
            raise SettingsError(f"{field}.{key}", f"must be an integer of at least {least}")
    for key in PROBABILITY_KEYS:
        number = _finite(given[key], f"{field}.{key}")
        if not 0.0 < number <= 1.0:
            raise SettingsError(f"{field}.{key}", "must be in (0, 1]")
    for key in POSITIVE_FLOAT_KEYS:
        if _finite(given[key], f"{field}.{key}") <= 0.0:
            raise SettingsError(f"{field}.{key}", "must be greater than 0")
    return cast(Generation, {key: given[key] for key in GENERATION_KEYS})


def parse_settings(value: object, field: str = "settings") -> QwenSettings:
    """Check a ``load`` request's ``settings``: ``non_streaming_mode`` and a complete ``generation``.

    In order: ``settings`` is an object with no other key; ``non_streaming_mode`` is given and is a boolean;
    ``generation`` is given and passes ``parse_generation``. Raises ``SettingsError`` naming the member.
    """
    if not isinstance(value, dict):
        raise SettingsError(field, "must be an object with non_streaming_mode and generation")
    given = cast(dict[str, object], value)
    unknown = sorted(set(given) - {"non_streaming_mode", "generation"})
    if unknown:
        raise SettingsError(f"{field}.{unknown[0]}", "unknown setting; known: non_streaming_mode, generation")
    if "non_streaming_mode" not in given:
        raise SettingsError(f"{field}.non_streaming_mode", "must be passed explicitly (section 10.1)")
    mode = given["non_streaming_mode"]
    if not isinstance(mode, bool):
        raise SettingsError(f"{field}.non_streaming_mode", "must be true or false")
    if "generation" not in given:
        raise SettingsError(f"{field}.generation", "must be passed explicitly (section 10.1)")
    generation = parse_generation(given["generation"], f"{field}.generation")
    return QwenSettings(non_streaming_mode=mode, generation=generation)


def generation_kwargs(generation: Generation) -> dict[str, bool | int | float]:
    """The keyword arguments for ``generate_voice_clone`` / ``generate_voice_design``: all ten, explicit."""
    values = cast(Mapping[str, bool | int | float], generation)
    return {key: values[key] for key in GENERATION_KEYS}


def ceiling_of(generation: Generation) -> int:
    """A checked ``generation``'s ``max_new_tokens``: the loaded ceiling of the calls' own caps (DC-4)."""
    return int(generation_kwargs(generation)["max_new_tokens"])


def _finite(value: object, field: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool) or not math.isfinite(value):
        raise SettingsError(field, "must be a finite number")
    return float(value)
