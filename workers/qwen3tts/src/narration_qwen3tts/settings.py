"""The audio-changing Qwen settings, checked before they reach the model (design section 10.1).

Every setting that changes audio is passed to qwen-tts **explicitly**, never left to a library default:

- ``non_streaming_mode``: false for Base (the clone path every piece of evidence used), true for VoiceDesign;
- the ten sampling values of ``QwenSettings.generation``, all of them, including ``max_new_tokens``, which
  is the loaded **ceiling**: each ``synthesize`` or ``design`` call passes its own cap, at most that
  (design section 10.1, DC-4).

qwen-tts 0.1.1 fills a value the caller leaves out from the snapshot's ``generation_config.json`` and then
from a hard-coded fallback (``Qwen3TTSModel._merge_generate_kwargs``). A worker that relied on that would
render differently if the snapshot's file changed, so this module refuses a ``generation`` that lacks any
of the ten keys. ``effective_generation`` computes the values qwen-tts would have used, from a snapshot's
``generation_config.json`` and the library's fallbacks: that is what an engine profile pins (section 6).

Standard library only: the daemon's side can reuse the checks, and the tests run without torch.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast

from narration_worker.handler import MIN_MAX_NEW_TOKENS
from narration_worker.protocol import Generation, QwenSettings

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

LIBRARY_FALLBACKS: Final[Mapping[str, bool | int | float]] = {
    "do_sample": True,
    "top_k": 50,
    "top_p": 1.0,
    "temperature": 0.9,
    "repetition_penalty": 1.05,
    "subtalker_dosample": True,
    "subtalker_top_k": 50,
    "subtalker_top_p": 1.0,
    "subtalker_temperature": 0.9,
    "max_new_tokens": 2048,
}
"""qwen-tts 0.1.1's hard-coded fallbacks (``_merge_generate_kwargs``), used only for a key the snapshot's
``generation_config.json`` does not set. The pinned snapshots set all ten, with ``max_new_tokens`` 8192
(plan.md section 1.3 item 1)."""

GENERATION_CONFIG_FILE: Final = "generation_config.json"
"""The file qwen-tts 0.1.1 reads the sampling defaults from. The design calls it ``generate_config.json``
(section 10.1); the installed code reads ``generation_config.json`` (plan.md section 1.3 item 1)."""

DTYPES: Final = ("bfloat16", "float16", "float32")
ATTN_IMPLEMENTATIONS: Final = ("sdpa", "eager", "flash_attention_2")


class SettingsError(ValueError):
    """A settings value that cannot be used; ``field`` names it (``settings.generation.top_k``)."""

    def __init__(self, field: str, message: str) -> None:
        super().__init__(f"{field}: {message}")
        self.field = field
        self.message = message


class SnapshotUnreadable(SettingsError):
    """A snapshot folder whose ``config.json`` is missing, unreadable or not a JSON object: a broken or
    incomplete install, not a wrong request."""


@dataclass(frozen=True, slots=True)
class ModelKind:
    """What a snapshot holds, from its ``config.json``: ``tts_model_type`` is ``base`` or ``voice_design``."""

    tts_model_type: str
    tokenizer_type: str | None
    tts_model_size: str | None


def parse_generation(value: object, field: str = "settings.generation") -> Generation:
    """Check a ``generation`` object: exactly the ten keys, each of the right type and range."""
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
    """Check a ``load`` request's ``settings``: ``non_streaming_mode`` and a complete ``generation``."""
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


def effective_generation(snapshot_dir: Path) -> Generation:
    """The sampling values qwen-tts 0.1.1 uses for a snapshot when none is passed: the snapshot's
    ``generation_config.json``, then the library's fallbacks. Raises ``SettingsError`` for a value there
    that ``parse_generation`` would refuse."""
    path = snapshot_dir / GENERATION_CONFIG_FILE
    config = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    if not isinstance(config, dict):
        raise SettingsError(GENERATION_CONFIG_FILE, "must hold a JSON object")
    config = cast(dict[str, object], config)
    merged = {key: config.get(key, LIBRARY_FALLBACKS[key]) for key in GENERATION_KEYS}
    return parse_generation(merged, GENERATION_CONFIG_FILE)


def read_model_kind(snapshot_dir: Path) -> ModelKind:
    """``tts_model_type`` (``base`` / ``voice_design`` / ``custom_voice``) and friends from ``config.json``.

    Raises ``SnapshotUnreadable`` when ``config.json`` is missing, unreadable or not a JSON object (a broken
    install), and ``SettingsError`` when it has no ``tts_model_type`` (another model's snapshot).
    """
    path = snapshot_dir / "config.json"
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SnapshotUnreadable("model.snapshot_dir", f"cannot read {path.name}: {exc}") from exc
    if not isinstance(config, dict):
        raise SnapshotUnreadable("model.snapshot_dir", f"{path.name} does not hold a JSON object")
    config = cast(dict[str, object], config)
    if not isinstance(config.get("tts_model_type"), str):
        raise SettingsError("model.snapshot_dir", f"{path.name} has no tts_model_type: not a Qwen3-TTS snapshot")
    return ModelKind(
        tts_model_type=cast(str, config["tts_model_type"]),
        tokenizer_type=_str_or_none(config.get("tokenizer_type")),
        tts_model_size=_str_or_none(config.get("tts_model_size")),
    )


def _finite(value: object, field: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool) or not math.isfinite(value):
        raise SettingsError(field, "must be a finite number")
    return float(value)


def _str_or_none(value: object) -> str | None:
    return value if isinstance(value, str) else None
